#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/finetune_qwen2_5_adaptive_pcgrad.sh"

: "${CONDA_PREFIX:?Activate the project environment first}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

GPU_INCLUDE="${GPU_INCLUDE:-localhost:3}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-./checkpoints/pretrain-pcgrad}"
MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints/finetune-pcgrad}"
FULL_TRAIN="${FULL_TRAIN:-True}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-2}"
ADAPTIVE_PCGRAD_CONFIG="${ADAPTIVE_PCGRAD_CONFIG:-./scripts/adaptive_projector_pcgrad/stage2_defaults.json}"
ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE="${ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE:-4}"
ADAPTIVE_PCGRAD_PROFILE="${ADAPTIVE_PCGRAD_PROFILE:-False}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
REPORT_TO="${REPORT_TO:-wandb}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
SAVE_AT_STEP_RATIO="${SAVE_AT_STEP_RATIO:-0.75}"
SKIP_MISSING_ADAPTERS="${SKIP_MISSING_ADAPTERS:-false}"
DRY_RUN="${DRY_RUN:-false}"

run_case() {
    local model_name="$1"
    local run_name="$2"
    local per_device_batch_size="$3"
    local grad_accum_steps="$4"
    local adapter="${PRETRAIN_ROOT}/${run_name}/llava-pretrain/mm_projector.bin"
    local metadata="${PRETRAIN_ROOT}/${run_name}/llava-pretrain/adaptive_projector_pcgrad_metadata.json"
    local output_dir="${OUTPUT_ROOT}/${run_name}/llava-finetune"

    echo "Adaptive projector-PCGrad stage 2: ${run_name}"
    echo "  model=${model_name} batch=${per_device_batch_size} grad_accum=${grad_accum_steps}"
    echo "  adapter=${adapter}"
    echo "  output=${output_dir}"
    echo "  extra_checkpoint_ratio=${SAVE_AT_STEP_RATIO}"

    if [[ "${DRY_RUN,,}" == "true" ]]; then
        if [[ ! -s "${adapter}" || ! -s "${metadata}" ]]; then
            echo "  status=waiting for completed adaptive-PCGrad stage 1"
        else
            echo "  status=ready"
        fi
        return 0
    fi

    if [[ ! -s "${adapter}" || ! -s "${metadata}" ]]; then
        if [[ "${SKIP_MISSING_ADAPTERS,,}" == "true" ]]; then
            echo "Skipping ${run_name}: stage-1 adapter or metadata is missing"
            return 0
        fi
        echo "Missing adaptive-PCGrad stage-1 handoff for ${run_name}:" >&2
        echo "  ${adapter}" >&2
        echo "  ${metadata}" >&2
        return 1
    fi

    if [[ -s "${output_dir}/config.json" ]] && {
        [[ -s "${output_dir}/model.safetensors" ]] ||
        [[ -s "${output_dir}/model.safetensors.index.json" ]] ||
        [[ -s "${output_dir}/pytorch_model.bin" ]]
    }; then
        echo "Skipping completed run ${run_name}: model exists in ${output_dir}"
        return 0
    fi

    MODEL_NAME_OR_PATH="${model_name}" \
    RUN_NAME="${run_name}-adaptive-pcgrad" \
    GPU_INCLUDE="${GPU_INCLUDE}" \
    PRETRAIN_ADAPTER="${adapter}" \
    MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE}" \
    OUTPUT_DIR="${output_dir}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${per_device_batch_size}" \
    GRADIENT_ACCUMULATION_STEPS="${grad_accum_steps}" \
    FULL_TRAIN="${FULL_TRAIN}" \
    SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS}" \
    ADAPTIVE_PCGRAD_CONFIG="${ADAPTIVE_PCGRAD_CONFIG}" \
    ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE="${ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE}" \
    ADAPTIVE_PCGRAD_PROFILE="${ADAPTIVE_PCGRAD_PROFILE}" \
    MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH}" \
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" \
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
    LOGGING_STEPS="${LOGGING_STEPS}" \
    REPORT_TO="${REPORT_TO}" \
    SAVE_STRATEGY="${SAVE_STRATEGY}" \
    SAVE_AT_STEP_RATIO="${SAVE_AT_STEP_RATIO}" \
    bash "${RUN_SCRIPT}"
}

# Stage 2 for every <=1.7B model selected by pretrain_list.sh.
# run_case "Qwen/Qwen2.5-0.5B-Instruct" "qwen2.5-0.5b" "32" "4"
run_case "Qwen/Qwen3-0.6B" "qwen3-0.6b" "32" "4"
run_case "google/gemma-3-1b-it" "gemma3-1b" "16" "8"
run_case "meta-llama/Llama-3.2-1B-Instruct" "llama-3.2-1b" "16" "8"
run_case "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "tinyllama-1.1b" "16" "8"
run_case "Qwen/Qwen2.5-1.5B-Instruct" "qwen2.5-1.5b" "8" "16"
run_case "Qwen/Qwen3-1.7B" "qwen3-1.7b" "8" "16"
