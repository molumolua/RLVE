#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 WANDB_PROJECT"
    exit 1
fi

WANDB_PROJECT=$1

bash scripts/training/Qwen3-4B/rlve.sh "${WANDB_PROJECT}" \
    "[Qwen3-4B]_[num-environment=4]" \
    "Division EuclidGame Multiplication Sorting"
