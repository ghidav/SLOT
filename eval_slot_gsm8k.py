"""
Test-time SLOT optimization + GSM8K evaluation for any causal LM.

Works with the GRPO-trained Qwen3-0.6B (or any HF causal model).

Usage:
  # Baseline (no SLOT):
  python eval_slot_gsm8k.py --model_path outputs/grpo_qwen3_0.6b_gsm8k --times 0

  # SLOT with 5 iterations:
  python eval_slot_gsm8k.py --model_path outputs/grpo_qwen3_0.6b_gsm8k --times 5 --lr 0.1
"""

import argparse
import random
import re

import torch
import torch.nn as nn
from datasets import load_dataset
from math_verify import ExprExtractionConfig, parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a helpful assistant. A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. The Assistant first "
    "thinks about the reasoning process in the mind and then provides the user "
    "with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>."
)


# ──────────────────────── Reward helpers ────────────────────────

def check_format(text):
    return bool(re.match(r"^<think>.*?</think><answer>.*?</answer>$", text, re.DOTALL))


def check_correct(text, ground_truth):
    nums = re.findall(r"\d+\.\d+|\d+/\d+|\d+", text)
    if not nums:
        return False
    try:
        a = parse(nums[-1], extraction_config=[ExprExtractionConfig()])
        b = parse(ground_truth, extraction_config=[ExprExtractionConfig()])
        return a is not None and b is not None and verify(a, b)
    except Exception:
        return False


# ──────────────────── SLOT: test-time delta ─────────────────────

@torch.no_grad()
def get_prompt_hidden_states(model, input_ids):
    """Run a forward pass and grab the last hidden states (before lm_head)."""
    outputs = model.model(input_ids=input_ids)
    return outputs[0]  # (batch, seq, hidden)


def slot_optimize(model, input_ids, times, lr):
    """
    Learn an additive delta on the hidden states that minimises next-token
    prediction loss on the prompt itself (self-supervised at test time).
    Returns the optimised delta tensor.
    """
    hidden = get_prompt_hidden_states(model, input_ids).detach()

    delta = nn.Parameter(torch.zeros(1, 1, hidden.shape[-1], device=hidden.device, dtype=hidden.dtype))
    optimizer = torch.optim.AdamW([delta], lr=lr, weight_decay=1e-8, eps=1e-5)

    for _ in range(times):
        optimizer.zero_grad()
        logits = model.lm_head(hidden + delta)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        optimizer.step()

    return delta.detach()


class SlotHook:
    """Hooks into the model so that `generate()` uses the optimised delta."""

    def __init__(self, model, delta):
        self.delta = delta
        # Hook right after the transformer body, before lm_head
        self.handle = model.model.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # output is BaseModelOutputWithPast; output[0] is hidden_states
        hidden = output[0] + self.delta
        return (hidden,) + output[1:]

    def remove(self):
        self.handle.remove()


# ──────────────────────── Evaluation ────────────────────────────

def evaluate(model, tokenizer, args):
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    qa_pairs = [
        {"Q": q, "A": a.split("####")[-1].strip()}
        for q, a in zip(dataset["question"], dataset["answer"])
    ]
    random.seed(args.seed)
    if args.eval_samples and len(qa_pairs) > args.eval_samples:
        qa_pairs = random.sample(qa_pairs, args.eval_samples)

    correct, fmt_correct, total = 0, 0, len(qa_pairs)
    gen_kwargs = dict(do_sample=False, max_new_tokens=1024)

    for i, qa in enumerate(qa_pairs):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": qa["Q"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        input_ids = inputs["input_ids"]

        # SLOT optimisation
        hook = None
        if args.times > 0:
            with torch.enable_grad():
                delta = slot_optimize(model, input_ids, args.times, args.lr)
            hook = SlotHook(model, delta)

        # Generate
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        completion = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

        if hook is not None:
            hook.remove()

        is_fmt = check_format(completion)
        is_correct = check_correct(completion, qa["A"])
        fmt_correct += is_fmt
        correct += is_correct

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[{i+1}/{total}]  acc={correct/(i+1):.3f}  fmt={fmt_correct/(i+1):.3f}")

    print(f"\nFinal  ({total} samples):  accuracy={correct/total:.4f}  format={fmt_correct/total:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--times", type=int, default=5, help="SLOT optimisation steps (0 = baseline)")
    parser.add_argument("--lr", type=float, default=0.1, help="SLOT learning rate")
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()

    evaluate(model, tokenizer, args)


if __name__ == "__main__":
    main()
