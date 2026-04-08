"""
Test-time SLOT optimization + GSM8K evaluation.

Baseline (times=0) uses vLLM for fast batched inference.
SLOT (times>0) uses HF model with per-sample optimization.

Usage:
  # Baseline (fast, vLLM):
  python eval_slot_gsm8k.py --model_path outputs/grpo_qwen3_0.6b_gsm8k --times 0

  # SLOT with 5 iterations:
  python eval_slot_gsm8k.py --model_path outputs/grpo_qwen3_0.6b_gsm8k --times 5 --lr 0.1
"""

import argparse
import json
import os
import random
import re

import torch
import torch.nn as nn
from datasets import load_dataset
from math_verify import ExprExtractionConfig, parse, verify

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Please think step by step and put your final answer in \\boxed{{}}.\n"
)


# ──────────────────────── Reward helpers ────────────────────────

def check_format(text):
    return bool(re.search(r"\\boxed\{[^}]+\}", text))


def check_correct(text, ground_truth):
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if not match:
        return False
    try:
        a = parse(match.group(1).strip(), extraction_config=[ExprExtractionConfig()])
        b = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
        return a is not None and b is not None and verify(a, b)
    except Exception:
        return False


def score_completions(completions, qa_pairs):
    correct, fmt_correct = 0, 0
    for comp, qa in zip(completions, qa_pairs):
        is_fmt = check_format(comp)
        is_correct = check_correct(comp, qa["A"])
        fmt_correct += is_fmt
        correct += is_correct
    total = len(qa_pairs)
    accuracy = correct / total if total > 0 else 0
    format_acc = fmt_correct / total if total > 0 else 0
    return accuracy, format_acc


# ──────────────────── Baseline: vLLM batched ────────────────────

def evaluate_baseline(args, qa_pairs):
    from vllm import LLM, SamplingParams

    prompts = [PROMPT_TEMPLATE.format(question=qa["Q"]) for qa in qa_pairs]

    llm = LLM(model=args.model_path, dtype="bfloat16", gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=0, max_tokens=2048)

    outputs = llm.generate(prompts, sampling_params)
    completions = [o.outputs[0].text for o in outputs]

    return score_completions(completions, qa_pairs)


# ──────────────────── SLOT: test-time delta ─────────────────────

@torch.no_grad()
def get_prompt_hidden_states(model, input_ids):
    outputs = model.model(input_ids=input_ids)
    return outputs[0]


def slot_optimize(model, input_ids, times, lr, optimizer_type="adamw"):
    hidden = get_prompt_hidden_states(model, input_ids).detach()

    delta = nn.Parameter(torch.zeros(1, 1, hidden.shape[-1], device=hidden.device, dtype=hidden.dtype))

    def compute_loss():
        logits = model.lm_head(hidden + delta)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        return nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS([delta], lr=lr, max_iter=times, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            return loss

        optimizer.step(closure)
    else:
        optimizer = torch.optim.AdamW([delta], lr=lr, weight_decay=1e-8, eps=1e-5)
        for _ in range(times):
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()

    return delta.detach()


class SlotHook:
    def __init__(self, model, delta):
        self.delta = delta
        self.handle = model.model.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        hidden = output[0] + self.delta
        return (hidden,) + output[1:]

    def remove(self):
        self.handle.remove()


def evaluate_slot(args, qa_pairs):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()

    correct, fmt_correct, total = 0, 0, len(qa_pairs)
    gen_kwargs = dict(do_sample=False, max_new_tokens=2048)

    for i, qa in enumerate(qa_pairs):
        prompt_text = PROMPT_TEMPLATE.format(question=qa["Q"])
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        with torch.enable_grad():
            delta = slot_optimize(model, input_ids, args.times, args.lr, args.optimizer)
        hook = SlotHook(model, delta)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        completion = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

        hook.remove()

        is_fmt = check_format(completion)
        is_correct = check_correct(completion, qa["A"])
        fmt_correct += is_fmt
        correct += is_correct

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[{i+1}/{total}]  acc={correct/(i+1):.3f}  fmt={fmt_correct/(i+1):.3f}")

    accuracy = correct / total if total > 0 else 0
    format_acc = fmt_correct / total if total > 0 else 0
    return accuracy, format_acc


# ──────────────────────── Main ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--times", type=int, default=5, help="SLOT optimisation steps (0 = vLLM baseline)")
    parser.add_argument("--lr", type=float, default=0.1, help="SLOT learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lbfgs"], help="SLOT optimizer")
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    qa_pairs = [
        {"Q": q, "A": a.split("####")[-1].strip()}
        for q, a in zip(dataset["question"], dataset["answer"])
    ]
    random.seed(args.seed)
    if args.eval_samples and len(qa_pairs) > args.eval_samples:
        qa_pairs = random.sample(qa_pairs, args.eval_samples)

    print(f"Evaluating {len(qa_pairs)} samples, times={args.times}")

    if args.times == 0:
        accuracy, format_acc = evaluate_baseline(args, qa_pairs)
    else:
        accuracy, format_acc = evaluate_slot(args, qa_pairs)

    print(f"\nFinal  ({len(qa_pairs)} samples):  accuracy={accuracy:.4f}  format={format_acc:.4f}")

    # Save results
    result = {
        "model_path": args.model_path,
        "optimizer": args.optimizer if args.times > 0 else "none",
        "times": args.times,
        "lr": args.lr,
        "split": args.split,
        "eval_samples": len(qa_pairs),
        "accuracy": accuracy,
        "format_accuracy": format_acc,
    }
    os.makedirs("results", exist_ok=True)
    tag = "baseline" if args.times == 0 else f"{args.optimizer}_t{args.times}_lr{args.lr}"
    results_path = f"results/{tag}.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
