#!/bin/bash
set -euo pipefail

# Qwen2.5-3B instruction tuning. Required inputs: configured PRETRAIN_ADAPTER
# (or SOURCE_STAGE2_CHECKPOINT + STAGE2_START_PHASE_B=True), total MAX_STEPS,
# STAGE2_MAX_PHASE_B_STEPS, STAGE2_REFINE_LR, STAGE2_CE_TOLERANCE and
# STAGE2_TEXT_KL_BUDGET. Defaults never guess a trained adapter or gate budget.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-3B-Instruct}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./playground/data/llava_v1_5_mix665k.json}"
export TRAIN_IMAGE_FOLDER="${TRAIN_IMAGE_FOLDER:-./playground/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/finetune-protected-cka/qwen2.5-3b/llava-finetune}"
export RUN_NAME="${RUN_NAME:-qwen2.5-3b-protected-stage2}"
export GPU_ID="${GPU_ID:-3}"
exec bash "${SCRIPT_DIR}/../finetune_protected_cka.sh" "$@"
