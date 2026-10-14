#!/bin/bash
set -euo pipefail

# Stage 2 only: baseline stage-1 adapter -> projector + final CKA.
# Plain CE + weighted CKA losses: no PCGrad, norm cap, or gradient diagnostics.
# Baseline adapter weights are loaded, not its optimizer/trainer state.
# Usage: GPU_INCLUDE=localhost:3 SEED=42 bash scripts/v1_5/7b/finetune_llama3_2_3b_cka_from_baseline.sh
# DRY_RUN=1 prints the command without training. The output must be fresh.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${RUN_NAME:=llama-3.2-3b}"
: "${SEED:=42}"
: "${PRETRAIN_ADAPTER:=./checkpoints/finetune-cka/${RUN_NAME}/baseline-vs-cka-stage1-stage2-seed${SEED}/baseline/llava-pretrain/mm_projector.bin}"
: "${OUTPUT_ROOT:=./checkpoints/finetune-cka/${RUN_NAME}/baseline-stage1-cka-proj-final-stage2-seed${SEED}}"
: "${WANDB_RUN_GROUP:=${RUN_NAME}-baseline-stage1-cka-proj-final-stage2-seed${SEED}}"
: "${WANDB_RUN_NAME:=${RUN_NAME}-baseline-stage1-cka-proj-final-finetune-seed${SEED}}"

exec env STAGE=finetune FINETUNE_PCGRAD=False \
    RUN_NAME="${RUN_NAME}" SEED="${SEED}" \
    PRETRAIN_ADAPTER="${PRETRAIN_ADAPTER}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    WANDB_RUN_GROUP="${WANDB_RUN_GROUP}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
    bash "${SCRIPT_DIR}/train_llama3_2_3b_cka_pcgrad.sh"
