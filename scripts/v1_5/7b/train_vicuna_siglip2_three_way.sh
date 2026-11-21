#!/bin/bash
set -euo pipefail

# One matched seed/init, two Stage-1 producers, three Stage-2 comparisons.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
[[ $# == 0 ]] || { echo 'Use environment overrides, not CLI arguments.' >&2; exit 1; }
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    if [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
        echo 'GPU 0 is occupied; no run started.' >&2
        exit 1
    fi
    export CONDA_PREFIX="${CONDA_PREFIX:-${PWD}/../conda/envs/llava1}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

export HF_HUB_CACHE="${HF_HUB_CACHE:-${PWD}/checkpoints/.hf-cache}"
export MODEL_NAME_OR_PATH=lmsys/vicuna-7b-v1.5
export VISION_TOWER=google/siglip2-so400m-patch14-384
export MM_VISION_SELECT_LAYER=-2
export CONVERSATION_VERSION=v1
export GPU_INCLUDE=localhost:0
export REPEATS=1 REPEAT_START=1 SEED=42
export PROJECTOR_INIT_SEED_MODE=seeded PROJECTOR_INIT_SEED_BASE=41
export SETTINGS='baseline cka_proj cka_proj_final'
export CKA_WEIGHT=0.1
export RUN_NAME=vicuna7b-siglip2
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PWD}/checkpoints/full-ft/vicuna7b-siglip2-seed42-b256-m32}"
export PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE=32 PRETRAIN_GRADIENT_ACCUMULATION_STEPS=8
export PER_DEVICE_TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=32
export FINETUNE_GRADIENT_CHECKPOINTING=False
export DEEPSPEED_CONFIG=./scripts/zero2_b200.json
export OPTIM=adamw_torch_fused
export DATALOADER_NUM_WORKERS=8
export DATALOADER_PREFETCH_FACTOR=4 DATALOADER_PERSISTENT_WORKERS=True
export ACCELERATOR_CONFIG='{"non_blocking":true}'
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec bash "${SCRIPT_DIR}/train_cka_two_stage.sh"
