#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/pretrain_qwen2_5_adaptive_pcgrad.sh"

: "${CONDA_PREFIX:?Activate the project environment first}"

run_case() {
    local model_name="$1"
    local run_name="$2"
    local per_device_batch_size="$3"
    local grad_accum_steps="$4"
    local source_adapter="${SOURCE_ROOT}/${run_name}/llava-pretrain/mm_projector.bin"

    if [[ ! -s "${source_adapter}" ]]; then
        echo "Missing source checkpoint used to select this rerun: ${source_adapter}" >&2
        return 1
    fi

    echo "Starting adaptive projector-PCGrad stage 1 for ${run_name}"
    if [[ "${DRY_RUN,,}" == "true" ]]; then
        echo "  model=${model_name} batch=${per_device_batch_size} grad_accum=${grad_accum_steps}"
        echo "  output=${OUTPUT_ROOT}/${run_name}/llava-pretrain"
        return 0
    fi

    MODEL_NAME_OR_PATH="${model_name}" \
    RUN_NAME="${run_name}-adaptive-pcgrad" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${per_device_batch_size}" \
    GRADIENT_ACCUMULATION_STEPS="${grad_accum_steps}" \
    MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE}" \
    OUTPUT_DIR="${OUTPUT_ROOT}/${run_name}/llava-pretrain" \
    GPU_INCLUDE="${GPU_INCLUDE}" \
    FULL_TRAIN="${FULL_TRAIN}" \
    SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS}" \
    ADAPTIVE_PCGRAD_CONFIG="${ADAPTIVE_PCGRAD_CONFIG}" \
    ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE="${ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE}" \
    ADAPTIVE_PCGRAD_PROFILE="${ADAPTIVE_PCGRAD_PROFILE}" \
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
    LOGGING_STEPS="${LOGGING_STEPS}" \
    REPORT_TO="${REPORT_TO}" \
    bash "${RUN_SCRIPT}"
}

GPU_INCLUDE="${GPU_INCLUDE:-localhost:3}"
MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
SOURCE_ROOT="${SOURCE_ROOT:-./checkpoints/pretrain-diag}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints/pretrain-pcgrad}"
FULL_TRAIN="${FULL_TRAIN:-True}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-2}"
ADAPTIVE_PCGRAD_CONFIG="${ADAPTIVE_PCGRAD_CONFIG:-./scripts/adaptive_projector_pcgrad/stage1_defaults.json}"
ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE="${ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE:-8}"
ADAPTIVE_PCGRAD_PROFILE="${ADAPTIVE_PCGRAD_PROFILE:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
REPORT_TO="${REPORT_TO:-wandb}"
DRY_RUN="${DRY_RUN:-False}"

# Rerun every <=1.7B model represented by a completed stage-1 adapter in
# checkpoints/pretrain-diag. The old adapters are only used as an inventory
# check; PCGrad training starts from each base model and writes to OUTPUT_ROOT.
run_case "Qwen/Qwen2.5-0.5B-Instruct" "qwen2.5-0.5b" "64" "4"
run_case "Qwen/Qwen3-0.6B" "qwen3-0.6b" "64" "4"
run_case "google/gemma-3-1b-it" "gemma3-1b" "32" "8"
run_case "meta-llama/Llama-3.2-1B-Instruct" "llama-3.2-1b" "32" "8"
run_case "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "tinyllama-1.1b" "32" "8"
run_case "Qwen/Qwen2.5-1.5B-Instruct" "qwen2.5-1.5b" "32" "8"
run_case "Qwen/Qwen3-1.7B" "qwen3-1.7b" "32" "8"
