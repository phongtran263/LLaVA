#!/bin/bash
set -euo pipefail

# Qwen2.5-3B Stage 1: CE-only -> plateau or reserved budget -> Protected CKA.
# No Stage-1 checkpoint or separate validation JSON is needed for a fresh run.
# MAX_STEPS is the TOTAL optimizer-step budget across both phases, not epochs.
# Default effective batch: 16 * 16 * one GPU = 256. Choose MAX_STEPS for the
# actual held-out split; reserve defaults require at least 1500 + 350 steps.
# Preview without loading models or launching training: DRY_RUN=1 ...

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-3B-Instruct}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./playground/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}"
export TRAIN_IMAGE_FOLDER="${TRAIN_IMAGE_FOLDER:-./playground/LLaVA-Pretrain/images}"
export OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/pretrain-protected-cka/qwen2.5-3b/llava-pretrain}"
export GPU_ID="${GPU_ID:-3}"
export REFINE_LR="${REFINE_LR:-1e-5}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
# Budget switching is checked every optimizer step, independently of this
# validation cadence. Reference/gate validation is always run at transition.
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export REPORT_TO="${REPORT_TO:-wandb}"
: "${MAX_STEPS:?Set MAX_STEPS to the total Phase A + B optimizer-step budget}"
export MAX_STEPS

# Parent launcher preserves optional adapter/resume/held-out settings and all
# protection defaults. It uses one Python process, not DeepSpeed.
exec bash "${SCRIPT_DIR}/../pretrain_protected_cka.sh" \
    --run_name "${RUN_NAME:-qwen2.5-3b-protected-stage1}" "$@"
