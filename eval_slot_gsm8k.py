"""
GSM8K evaluation with optional SLOT (via patched vLLM).

Usage:
  # Install the SLOT-patched model_runner into vLLM (once):
  python -c "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), 'worker'))" | xargs -I{} cp ./vllm/model_runner.py {}/

  # Baseline:
  python eval_slot_gsm8k.py --model_path outputs/grpo_qwen3_0.6b_gsm8k --times 0

  # SLOT with 5 iterations:
  python eval_slot_gsm8k.py --model_path outputs/grpo_qwen3_0.6b_gsm8k --times 5 --lr 0.1
"""

import argparse
import json
import os
import random
import re

from datasets import load_dataset
from math_verify import ExprExtractionConfig, parse, verify
from vllm import LLM, SamplingParams

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


# ──────────────────────── Evaluation ────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--times", type=int, default=5, help="SLOT steps (0 = baseline)")
    parser.add_argument("--lr", type=float, default=0.1, help="SLOT learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lbfgs"], help="SLOT optimizer")
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set SLOT env vars (read by patched model_runner.py)
    os.environ["CHOT_STEPS"] = str(args.times)
    os.environ["CHOT_LR"] = str(args.lr)
    os.environ["CHOT_OPTIMIZER"] = args.optimizer

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    qa_pairs = [
        {"Q": q, "A": a.split("####")[-1].strip()}
        for q, a in zip(dataset["question"], dataset["answer"])
    ]
    random.seed(args.seed)
    if args.eval_samples and len(qa_pairs) > args.eval_samples:
        qa_pairs = random.sample(qa_pairs, args.eval_samples)

    prompts = [PROMPT_TEMPLATE.format(question=qa["Q"]) for qa in qa_pairs]

    print(f"Evaluating {len(qa_pairs)} samples, CHOT_STEPS={args.times}, CHOT_LR={args.lr}")

    llm = LLM(model=args.model_path, dtype="bfloat16", gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=0, max_tokens=2048)

    outputs = llm.generate(prompts, sampling_params)
    completions = [o.outputs[0].text for o in outputs]

    correct, fmt_correct = 0, 0
    for comp, qa in zip(completions, qa_pairs):
        fmt_correct += check_format(comp)
        correct += check_correct(comp, qa["A"])

    total = len(qa_pairs)
    accuracy = correct / total if total > 0 else 0
    format_acc = fmt_correct / total if total > 0 else 0

    print(f"\nFinal  ({total} samples):  accuracy={accuracy:.4f}  format={format_acc:.4f}")

    # Save results
    result = {
        "model_path": args.model_path,
        "optimizer": args.optimizer if args.times > 0 else "none",
        "slot_steps": args.times,
        "slot_lr": args.lr,
        "split": args.split,
        "eval_samples": total,
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
