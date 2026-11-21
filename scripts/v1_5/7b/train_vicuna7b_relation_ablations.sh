#!/bin/bash
set -euo pipefail

# Three visual-relation ablations in place of CKA, each trained from its own
# Stage-1 projector and then full Stage-2 Vicuna-7B. Nothing launches on import.
# Example: PRETRAIN_BS=16 PRETRAIN_ACCUM=8 FINETUNE_BS=2 FINETUNE_ACCUM=32 \
#          bash scripts/v1_5/7b/train_vicuna7b_relation_ablations.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
[[ $# == 0 ]] || { echo "Use environment overrides, not positional arguments." >&2; exit 1; }

: "${STAGE:=all}"                  # all | pretrain | finetune
: "${METHODS:=cosrel cosrel_uka cosrel_uka_rkdd}"
: "${GPU_INCLUDE:=localhost:0,1}"
: "${PRETRAIN_BS:=32}"
: "${PRETRAIN_ACCUM:=4}"          # global Stage-1 batch = 2 * BS * ACCUM
: "${FINETUNE_BS:=4}"
: "${FINETUNE_ACCUM:=16}"         # global Stage-2 batch = 2 * BS * ACCUM
: "${SEED:=42}"
: "${PROJECTOR_INIT_SEED:=42}"
: "${RELATION_COS_WEIGHT:=1.0}"
: "${RELATION_UKA_WEIGHT:=1.0}"
: "${RELATION_RKDD_WEIGHT:=1.0}"
: "${STAGE1_WEIGHT:=0.1}"
: "${STAGE2_PROJECTOR_WEIGHT:=0.1}"
: "${STAGE2_FINAL_WEIGHT:=0.1}"
: "${DRY_RUN:=0}"
: "${EXPERIMENT_ROOT:=${PWD}/checkpoints/full-ft/vicuna7b-relation-ablations-seed${SEED}}"

check_output_available() {
    local output="$1"
    if [[ ! -e "$output" && ! -L "$output" ]]; then
        return
    fi
    # A failed startup can leave behind only the empty output directory.
    # Reuse that safely, but never overwrite a checkpoint or partial run.
    if [[ -d "$output" && ! -L "$output" ]] &&
       [[ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Reusing empty output directory: $output"
        return
    fi
    echo "Output contains data: $output. Use a fresh EXPERIMENT_ROOT or select another METHOD." >&2
    exit 1
}

case "$STAGE" in all|pretrain|finetune) ;; *) echo "Invalid STAGE: $STAGE" >&2; exit 1 ;; esac
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1." >&2; exit 1; }
for value in "$PRETRAIN_BS" "$PRETRAIN_ACCUM" "$FINETUNE_BS" "$FINETUNE_ACCUM"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Batch sizes and accumulation must be positive integers." >&2; exit 1; }
done
read -r -a methods <<< "$METHODS"
(( ${#methods[@]} )) || { echo "METHODS is empty." >&2; exit 1; }
declare -A seen=()
for method in "${methods[@]}"; do
    case "$method" in cosrel|cosrel_uka|cosrel_uka_rkdd) ;; *) echo "Invalid method: $method" >&2; exit 1 ;; esac
    [[ -z "${seen[$method]:-}" ]] || { echo "Duplicate method: $method" >&2; exit 1; }
    seen[$method]=1
    if [[ "$STAGE" != finetune ]]; then
        output="${EXPERIMENT_ROOT}/${method}/llava-pretrain"
        check_output_available "$output"
    fi
    if [[ "$STAGE" != pretrain ]]; then
        output="${EXPERIMENT_ROOT}/${method}/llava-finetune"
        check_output_available "$output"
    fi
done
if [[ "$DRY_RUN" == 0 ]]; then
    export CONDA_PREFIX="${CONDA_PREFIX:-${PWD}/../conda/envs/llava1}"
    [[ -x "${CONDA_PREFIX}/bin/deepspeed" ]] || { echo "Missing DeepSpeed in ${CONDA_PREFIX}" >&2; exit 1; }
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

export MODEL_NAME_OR_PATH=lmsys/vicuna-7b-v1.5
export CONVERSATION_VERSION=v1 VISION_TOWER=openai/clip-vit-large-patch14-336
export MM_VISION_SELECT_LAYER=-2 MM_PROJECTOR_TYPE=mlp2x_gelu
export MM_PROJECTOR_INIT_SEED="$PROJECTOR_INIT_SEED"
export GPU_INCLUDE SEED DRY_RUN RELATION_COS_WEIGHT RELATION_UKA_WEIGHT RELATION_RKDD_WEIGHT
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero2_b200.json}"
export OPTIM="${OPTIM:-adamw_torch_fused}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
export DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-True}"
export ACCELERATOR_CONFIG="${ACCELERATOR_CONFIG:-{\"non_blocking\":true}}"
export PRETRAIN_GRADIENT_CHECKPOINTING="${PRETRAIN_GRADIENT_CHECKPOINTING:-False}"
export FINETUNE_GRADIENT_CHECKPOINTING="${FINETUNE_GRADIENT_CHECKPOINTING:-False}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-vicuna7b-relation-ablations-seed${SEED}}"
export VSP_DIAGNOSTICS=False VSP_PCGRAD=False VSP_NORM_CAP=False
export CKA_LOSS_ENABLED=True CKA_LOSS_START_RATIO=0.0 CKA_LOSS_TAU=0.0
export CKA_ANCHOR_LAYER= CKA_VISION_ANCHOR_LAYER=
export CKA_PROJECTOR_VISION_ANCHOR_LAYER= CKA_FINAL_VISION_ANCHOR_LAYER=
export CKA_FINAL_MLP_HEAD=False SAVE_STRATEGY=no

PRETRAIN_SCRIPT="${SCRIPT_DIR}/pretrain_qwen2_5.sh"
FINETUNE_SCRIPT="${SCRIPT_DIR}/finetune_qwen2_5.sh"
bash -n "$PRETRAIN_SCRIPT"
bash -n "$FINETUNE_SCRIPT"
for method in "${methods[@]}"; do
    export CKA_LOSS_OBJECTIVE="$method"
    root="${EXPERIMENT_ROOT}/${method}"
    if [[ "$STAGE" != finetune ]]; then
        echo "Stage 1: $method; GPU=$GPU_INCLUDE; microbatch=$PRETRAIN_BS; accumulation=$PRETRAIN_ACCUM"
        env OUTPUT_DIR="${root}/llava-pretrain" RUN_NAME="vicuna7b-${method}-pretrain-seed${SEED}" \
            PER_DEVICE_TRAIN_BATCH_SIZE="$PRETRAIN_BS" GRADIENT_ACCUMULATION_STEPS="$PRETRAIN_ACCUM" \
            CKA_PROJECTOR_WEIGHT="$STAGE1_WEIGHT" CKA_FINAL_HIDDEN_WEIGHT=0.0 CKA_LAYERS=-1 \
            bash -s -- < "$PRETRAIN_SCRIPT"
    fi
    if [[ "$STAGE" != pretrain ]]; then
        adapter="${root}/llava-pretrain/mm_projector.bin"
        if [[ "$DRY_RUN" == 0 ]]; then
            [[ -s "$adapter" ]] || { echo "Missing Stage-1 adapter: $adapter" >&2; exit 1; }
        fi
        echo "Stage 2: $method; GPU=$GPU_INCLUDE; microbatch=$FINETUNE_BS; accumulation=$FINETUNE_ACCUM"
        env OUTPUT_DIR="${root}/llava-finetune" PRETRAIN_ADAPTER="$adapter" \
            RUN_NAME="vicuna7b-${method}-finetune-seed${SEED}" \
            WANDB_RUN_NAME="vicuna7b-${method}-finetune-seed${SEED}" \
            PER_DEVICE_TRAIN_BATCH_SIZE="$FINETUNE_BS" GRADIENT_ACCUMULATION_STEPS="$FINETUNE_ACCUM" \
            CKA_PROJECTOR_WEIGHT="$STAGE2_PROJECTOR_WEIGHT" CKA_FINAL_HIDDEN_WEIGHT="$STAGE2_FINAL_WEIGHT" \
            CKA_LAYERS=final bash -s -- < "$FINETUNE_SCRIPT"
    fi
done
