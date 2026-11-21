#!/bin/bash
set -euo pipefail

# For each backbone/arm: one 3-epoch Stage 1, then three independent
# 1-epoch Stage 2 runs from its epoch-1, epoch-2, and epoch-3 adapters.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
[[ $# == 0 ]] || { echo 'Use environment variables, not CLI arguments.' >&2; exit 1; }
DRY_RUN="${DRY_RUN:-0}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || exit 2
GPU_ID="${GPU_ID:?Set GPU_ID explicitly; no GPU is selected by default.}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo 'GPU_ID must be a GPU index.' >&2; exit 1; }
if [[ "$DRY_RUN" == 0 ]]; then
    nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader >/dev/null || {
        echo "GPU ${GPU_ID} does not exist or is unavailable." >&2; exit 1;
    }
    [[ -z "$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || {
        echo "GPU ${GPU_ID} is occupied; no training started." >&2; exit 1;
    }
    export CONDA_PREFIX="${CONDA_PREFIX:-${PWD}/../conda/envs/llava1}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    [[ -x "${CONDA_PREFIX}/bin/deepspeed" ]] || { echo 'Missing llava1 DeepSpeed.' >&2; exit 1; }
fi

ROOT="${OUTPUT_ROOT:-${PWD}/checkpoints/full-ft/llama1b-qwen1p5b-stage1-3e-stage2-1e-seed42}"
LLAMA_PRETRAIN_BATCH_SIZE="${LLAMA_PRETRAIN_BATCH_SIZE:-32}"
LLAMA_FINETUNE_BATCH_SIZE="${LLAMA_FINETUNE_BATCH_SIZE:-16}"
QWEN_PRETRAIN_BATCH_SIZE="${QWEN_PRETRAIN_BATCH_SIZE:-32}"
QWEN_FINETUNE_BATCH_SIZE="${QWEN_FINETUNE_BATCH_SIZE:-8}"
for setting in LLAMA_PRETRAIN_BATCH_SIZE QWEN_PRETRAIN_BATCH_SIZE; do
    value="${!setting}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] && (( 256 % value == 0 )) || {
        echo "${setting} must divide 256." >&2; exit 1;
    }
done
for setting in LLAMA_FINETUNE_BATCH_SIZE QWEN_FINETUNE_BATCH_SIZE; do
    value="${!setting}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] && (( 128 % value == 0 )) || {
        echo "${setting} must divide 128." >&2; exit 1;
    }
done
read -r -a models <<< "${MODELS:-llama-3.2-1b qwen2.5-1.5b}"
(( ${#models[@]} > 0 )) || { echo 'MODELS must not be empty.' >&2; exit 1; }
declare -A seen_models=()
for model in "${models[@]}"; do
    case "$model" in
        llama-3.2-1b|qwen2.5-1.5b) ;;
        *) echo "Unknown model in MODELS: $model" >&2; exit 1 ;;
    esac
    [[ -z "${seen_models[$model]:-}" ]] || { echo "Duplicate model in MODELS: $model" >&2; exit 1; }
    seen_models[$model]=1
done
for model in "${models[@]}"; do
    for variant in baseline cka_proj_final; do
        for output in "${ROOT}/${model}/${variant}/llava-pretrain" \
            "${ROOT}/${model}/${variant}/stage1-epoch-1/llava-finetune" \
            "${ROOT}/${model}/${variant}/stage1-epoch-2/llava-finetune" \
            "${ROOT}/${model}/${variant}/stage1-epoch-3/llava-finetune"; do
            [[ ! -e "$output" && ! -L "$output" ]] || {
                echo "Output already exists: $output" >&2; exit 1;
            }
        done
    done
done

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false

for model in "${models[@]}"; do
    case "$model" in
        llama-3.2-1b)
            model_id=meta-llama/Llama-3.2-1B-Instruct
            pretrain_micro="$LLAMA_PRETRAIN_BATCH_SIZE"
            finetune_micro="$LLAMA_FINETUNE_BATCH_SIZE" ;;
        qwen2.5-1.5b)
            model_id=Qwen/Qwen2.5-1.5B-Instruct
            pretrain_micro="$QWEN_PRETRAIN_BATCH_SIZE"
            finetune_micro="$QWEN_FINETUNE_BATCH_SIZE" ;;
    esac
    for variant in baseline cka_proj_final; do
        enabled=False projector_weight=0.0 final_weight=0.0
        if [[ "$variant" == cka_proj_final ]]; then
            enabled=True projector_weight=0.1 final_weight=0.1
        fi
        parent="${ROOT}/${model}/${variant}"
        pretrain_output="${parent}/llava-pretrain"
        for job in pretrain finetune-1 finetune-2 finetune-3; do
            if [[ "$job" == pretrain ]]; then
                child="${SCRIPT_DIR}/pretrain_qwen2_5.sh"
                output="$pretrain_output"
                micro="$pretrain_micro"
                accum=$((256 / micro))
                epochs=3
                hidden_weight=0.0
                layers=-1
                save_strategy=epoch
                adapter=""
            else
                epoch_index="${job#finetune-}"
                child="${SCRIPT_DIR}/finetune_qwen2_5.sh"
                output="${parent}/stage1-epoch-${epoch_index}/llava-finetune"
                micro="$finetune_micro"
                accum=$((128 / micro))
                epochs=1
                hidden_weight="$final_weight"
                layers=final
                save_strategy=no
                if [[ "$DRY_RUN" == 0 ]]; then
                    if [[ "$epoch_index" == 1 ]]; then
                        shopt -s nullglob
                        checkpoint_dirs=("${pretrain_output}"/checkpoint-*)
                        shopt -u nullglob
                        (( ${#checkpoint_dirs[@]} == 3 )) || {
                            echo "Expected 3 Stage-1 epoch checkpoints in $pretrain_output, found ${#checkpoint_dirs[@]}." >&2; exit 1;
                        }
                        mapfile -t checkpoint_dirs < <(printf '%s\n' "${checkpoint_dirs[@]}" | sort -V)
                    fi
                    adapter="${checkpoint_dirs[$((epoch_index - 1))]}/mm_projector.bin"
                    [[ -s "$adapter" ]] || { echo "Missing Stage-1 adapter: $adapter" >&2; exit 1; }
                else
                    adapter="${pretrain_output}/checkpoint-EPOCH-${epoch_index}/mm_projector.bin"
                fi
            fi
            echo "Starting ${model} ${variant} ${job} (${epochs}e): ${micro} × ${accum} on GPU ${GPU_ID}"
            bash -n "$child" || { echo "Invalid Bash syntax: $child" >&2; exit 1; }
                env -u WANDB_RUN_ID -u WANDB_RESUME \
                    DRY_RUN="$DRY_RUN" MODEL_NAME_OR_PATH="$model_id" \
                    CONVERSATION_VERSION=auto VISION_TOWER=openai/clip-vit-large-patch14-336 \
                    MM_PROJECTOR_TYPE=mlp2x_gelu MM_VISION_SELECT_LAYER=-2 \
                    GPU_INCLUDE="localhost:${GPU_ID}" MASTER_PORT="${MASTER_PORT:-29583}" \
                    DEEPSPEED_CONFIG=./scripts/zero2_b200.json OPTIM=adamw_torch_fused \
                    SEED=42 MM_PROJECTOR_INIT_SEED=42 NUM_TRAIN_EPOCHS="$epochs" \
                    PER_DEVICE_TRAIN_BATCH_SIZE="$micro" GRADIENT_ACCUMULATION_STEPS="$accum" \
                    PRETRAIN_GRADIENT_CHECKPOINTING="${PRETRAIN_GRADIENT_CHECKPOINTING:-False}" FINETUNE_GRADIENT_CHECKPOINTING=False \
                    DATALOADER_NUM_WORKERS=8 DATALOADER_PREFETCH_FACTOR=4 \
                    DATALOADER_PERSISTENT_WORKERS=True ACCELERATOR_CONFIG='{"non_blocking":true}' \
                    SAVE_STRATEGY="$save_strategy" LOGGING_STEPS=10 REPORT_TO=wandb \
                    CKA_LOSS_ENABLED="$enabled" CKA_PROJECTOR_WEIGHT="$projector_weight" \
                    CKA_FINAL_HIDDEN_WEIGHT="$hidden_weight" CKA_LAYERS="$layers" \
                    CKA_LOSS_TAU=0.0 CKA_LOSS_START_RATIO=0.0 CKA_FINAL_MLP_HEAD=False \
                    VSP_DIAGNOSTICS=False VSP_PCGRAD=False VSP_NORM_CAP=False \
                    VSP_APPLY_TO_PROJECTOR_ONLY=False RESUME_FROM_CHECKPOINT= \
                    PRETRAIN_ADAPTER="$adapter" OUTPUT_DIR="$output" \
                    RUN_NAME="${model}-${variant}-${job}-seed42${RUN_TAG:+-${RUN_TAG}}" \
                    WANDB_RUN_NAME="${model}-${variant}-${job}-seed42${RUN_TAG:+-${RUN_TAG}}" \
                    WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-small-backbone-stage1-3e-stage2-1e-seed42}" \
                    bash -s -- < "$child"
            echo "Completed ${model} ${variant} ${job}"
        done
    done
done
