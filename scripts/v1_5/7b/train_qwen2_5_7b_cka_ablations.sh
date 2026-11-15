#!/bin/bash
set -euo pipefail

# Three repeats: baseline, projector CKA, and projector + final-state CKA.
# Only Trainer is seeded (42). Fresh Stage-1 projectors use ambient RNG.
# Each repeat trains two adapters; both CKA finetunes share the CKA adapter.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Single B200 defaults: effective batch 256 in Stage 1 and 128 in Stage 2.
# Checkpointing and a small Stage-2 microbatch reduce activation memory;
# optimizer states still reside on the single GPU with ZeRO-2.
exec env \
    "MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct" \
    "RUN_NAME=qwen2.5-7b" \
    "MM_PROJECTOR_TYPE=mlp2x_gelu" \
    "PROJECTOR_INIT_SEED_MODE=random" \
    "MM_PROJECTOR_INIT_SEED=" \
    "SETTINGS=baseline cka_proj cka_proj_final" \
    "CKA_WEIGHT=0.1" \
    "REPEATS=${REPEATS:-3}" \
    "PRETRAIN_GPU_INCLUDE=${PRETRAIN_GPU_INCLUDE:-${GPU_INCLUDE:-localhost:3}}" \
    "FINETUNE_GPU_INCLUDE=${FINETUNE_GPU_INCLUDE:-${GPU_INCLUDE:-localhost:3}}" \
    "PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE=${PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE:-16}" \
    "PRETRAIN_GRADIENT_ACCUMULATION_STEPS=${PRETRAIN_GRADIENT_ACCUMULATION_STEPS:-16}" \
    "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-128}" \
    "PRETRAIN_GRADIENT_CHECKPOINTING=${PRETRAIN_GRADIENT_CHECKPOINTING:-False}" \
    "FINETUNE_GRADIENT_CHECKPOINTING=${FINETUNE_GRADIENT_CHECKPOINTING:-True}" \
    "EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-./checkpoints/cka-two-stage/qwen2.5-7b-mlp2x_gelu-random-projector-lambda0.1-seed42}" \
    bash "${SCRIPT_DIR}/train_cka_two_stage.sh" "$@"
