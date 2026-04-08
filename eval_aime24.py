"""
AIME 2024 evaluation with optional SLOT (via patched vLLM).

Replicates the SLOT paper's setup for AIME24.

Usage:
  # Baseline:
  python eval_aime24.py --model_path Qwen/Qwen2.5-Math-1.5B --times 0

  # SLOT (paper uses times=1, lr=0.1):
  python eval_aime24.py --model_path Qwen/Qwen2.5-Math-1.5B --times 1 --lr 0.1
"""

import argparse
import json
import os
import re

from datasets import load_dataset
from math_verify import ExprExtractionConfig, parse, verify
from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = """Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{problem}"""


def check_correct(text, ground_truth):
    # Extract from \boxed{...}
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if not match:
        return False
    try:
        a = parse(match.group(1).strip(), extraction_config=[ExprExtractionConfig()])
        b = parse(str(ground_truth), extraction_config=[ExprExtractionConfig()])
        return a is not None and b is not None and verify(a, b)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--times", type=int, default=1, help="SLOT steps (0 = baseline)")
    parser.add_argument("--lr", type=float, default=0.1, help="SLOT learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lbfgs"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    # Set SLOT env vars
    os.environ["CHOT_STEPS"] = str(args.times)
    os.environ["CHOT_LR"] = str(args.lr)
    os.environ["CHOT_OPTIMIZER"] = args.optimizer
    os.environ.setdefault("tensor_parallel_size_my", "1")
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    problems = [{"problem": row["problem"], "answer": row["answer"]} for row in dataset]

    prompts = [PROMPT_TEMPLATE.format(problem=p["problem"]) for p in problems]

    print(f"Evaluating {len(problems)} AIME 2024 problems")
    print(f"Model: {args.model_path}")
    print(f"SLOT: steps={args.times}, lr={args.lr}, optimizer={args.optimizer}")

    llm = LLM(model=args.model_path, dtype="bfloat16", gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    outputs = llm.generate(prompts, sampling_params)
    completions = [o.outputs[0].text for o in outputs]

    correct = 0
    for i, (comp, p) in enumerate(zip(completions, problems)):
        is_correct = check_correct(comp, p["answer"])
        correct += is_correct
        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:2d}/30] {status}  answer={p['answer']}")

    accuracy = correct / len(problems)
    print(f"\nAIME 2024: {correct}/{len(problems)} = {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Save
    result = {
        "model_path": args.model_path,
        "benchmark": "aime24",
        "optimizer": args.optimizer if args.times > 0 else "none",
        "slot_steps": args.times,
        "slot_lr": args.lr,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
    }
    os.makedirs("results", exist_ok=True)
    tag = "aime24_baseline" if args.times == 0 else f"aime24_{args.optimizer}_t{args.times}_lr{args.lr}"
    path = f"results/{tag}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
