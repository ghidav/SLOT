"""
GRPO training for Qwen3-0.6B on GSM8K using TRL + vLLM.

Usage:
  python train_grpo_gsm8k.py

Requires: trl>=0.16, vllm, math-verify, datasets, transformers
"""

import re
from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig
from trl import GRPOConfig, GRPOTrainer

# ──────────────────────────── Config ────────────────────────────

MODEL_NAME = "Qwen/Qwen3-0.6B"

SYSTEM_PROMPT = (
    "You are a helpful assistant. A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. The Assistant first "
    "thinks about the reasoning process in the mind and then provides the user "
    "with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>."
)

# ──────────────────────────── Dataset ───────────────────────────

def build_prompt(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
    }


def extract_answer(text):
    """Extract the ground-truth numeric answer from the GSM8K answer string."""
    return text.split("####")[-1].strip()


dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = dataset.map(build_prompt)

# ──────────────────────────── Rewards ───────────────────────────

def reward_format(completions, **kwargs):
    """Reward for using the correct <think>...</think><answer>...</answer> format."""
    pattern = re.compile(r"^<think>.*?</think><answer>.*?</answer>$", re.DOTALL)
    return [1.0 if pattern.match(c) else 0.0 for c in completions]


def reward_correctness(completions, answer, **kwargs):
    """Reward for producing the correct numerical answer."""
    scores = []
    for completion, gt in zip(completions, answer):
        gt_value = extract_answer(gt)
        # Pull last number from completion
        nums = re.findall(r"\d+\.\d+|\d+/\d+|\d+", completion)
        if not nums:
            scores.append(0.0)
            continue
        try:
            parsed_answer = parse(nums[-1], extraction_config=[ExprExtractionConfig()])
            parsed_gt = parse(gt_value, extraction_config=[ExprExtractionConfig()])
            if parsed_answer is not None and parsed_gt is not None and verify(parsed_answer, parsed_gt):
                scores.append(1.0)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    return scores


# ──────────────────────────── Training ──────────────────────────

training_args = GRPOConfig(
    output_dir="outputs/grpo_qwen3_0.6b_gsm8k",
    run_name="grpo-qwen3-0.6b-gsm8k",

    # GRPO
    num_generations=8,
    max_completion_length=1024,
    max_prompt_length=512,

    # Training
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-6,
    warmup_ratio=0.05,
    bf16=True,

    # vLLM generation backend
    use_vllm=True,
    vllm_gpu_utilization=0.3,

    # Logging
    logging_steps=1,
    save_steps=100,
    save_total_limit=3,
    report_to="wandb",
)

trainer = GRPOTrainer(
    model=MODEL_NAME,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[reward_format, reward_correctness],
)

trainer.train()
trainer.save_model(training_args.output_dir)
