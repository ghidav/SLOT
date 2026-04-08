#!/bin/bash
set -e

MODEL_PATH=${1:-"outputs/grpo_qwen3_0.6b_gsm8k"}
LR=${2:-0.1}

echo "=== Baseline ==="
python eval_slot_gsm8k.py --model_path "$MODEL_PATH" --times 0

for t in 2 4 8 16 32; do
    echo "=== AdamW times=$t lr=$LR ==="
    python eval_slot_gsm8k.py --model_path "$MODEL_PATH" --times $t --lr $LR --optimizer adamw
done

for t in 2 4 8 16; do
    echo "=== L-BFGS times=$t lr=$LR ==="
    python eval_slot_gsm8k.py --model_path "$MODEL_PATH" --times $t --lr $LR --optimizer lbfgs
done
