"""
GRPO training for Qwen3-0.6B on GSM8K using TRL + vLLM.

Usage:
  python train_grpo_gsm8k.py

Requires: trl>=0.16, vllm, math-verify, datasets, transformers
"""

import re
from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ──────────────────────────── Config ────────────────────────────

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Please think step by step and put your final answer in \\boxed{{}}.\n"
)

# ──────────────────────────── Dataset ───────────────────────────

def build_prompt(example):
    return {
        "prompt": PROMPT_TEMPLATE.format(question=example["question"]),
    }


def extract_answer(text):
    """Extract the ground-truth numeric answer from the GSM8K answer string."""
    return text.split("####")[-1].strip()


def extract_boxed(text):
    """Extract content from \\boxed{...}."""
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    return match.group(1).strip() if match else None


dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = dataset.map(build_prompt)

# ──────────────────────────── Rewards ───────────────────────────

def reward_format(completions, **kwargs):
    """Reward for containing a \\boxed{} answer."""
    pattern = re.compile(r"\\boxed\{[^}]+\}")
    return [1.0 if pattern.search(c) else 0.0 for c in completions]


def reward_correctness(completions, answer, **kwargs):
    """Reward for producing the correct numerical answer in \\boxed{}."""
    scores = []
    for completion, gt in zip(completions, answer):
        gt_value = extract_answer(gt)
        boxed = extract_boxed(completion)
        if boxed is None:
            scores.append(0.0)
            continue
        try:
            parsed_answer = parse(boxed, extraction_config=[ExprExtractionConfig()])
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
    num_generations=4,
    max_completion_length=2048,

    # Training
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-6,
    lr_scheduler_type="constant",
    warmup_ratio=0.0,
    bf16=True,

    # vLLM generation backend
    use_vllm=True,
    vllm_gpu_memory_utilization=0.4,

    # Logging
    logging_steps=1,
    log_completions=True,
    save_steps=100,
    save_total_limit=3,
    report_to="wandb",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=MODEL_NAME,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[reward_format, reward_correctness],
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(training_args.output_dir)
