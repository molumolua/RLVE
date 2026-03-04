#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 WANDB_PROJECT"
    exit 1
fi

WANDB_PROJECT=$1

bash scripts/training/Qwen3-1.7B/rlve.sh "${WANDB_PROJECT}" \
    "[Qwen3-1.7B]_[num-environment=1]" \
    "Multiplication"
