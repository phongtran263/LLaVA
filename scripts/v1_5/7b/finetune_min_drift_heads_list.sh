#!/bin/bash
set -euo pipefail

# Stage-2 finetuning with the fixed minimum-drift head calibrated at stage 1.
# Layer IDs are 1-based; query/output head IDs are 0-based.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/finetune_qwen2_5.sh"

: "${CONDA_PREFIX:?Activate the llava1 environment first}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

GPU_INCLUDE="${GPU_INCLUDE:-localhost:3}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-./checkpoints/pretrain-diag}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints/finetune-cka}"
MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
CKA_PROJECTOR_WEIGHT="${CKA_PROJECTOR_WEIGHT:-0.1}"
CKA_FINAL_HIDDEN_WEIGHT="${CKA_FINAL_HIDDEN_WEIGHT:-0.1}"
CKA_LOSS_START_RATIO="${CKA_LOSS_START_RATIO:-0.0}"
SKIP_MISSING_ADAPTERS="${SKIP_MISSING_ADAPTERS:-false}"
DRY_RUN="${DRY_RUN:-false}"

run_case() {
    local model_name="$1"
    local run_name="$2"
    local per_device_batch_size="$3"
    local grad_accum_steps="$4"
    local head_ids="$5"

    local adapter="${PRETRAIN_ROOT}/${run_name}/llava-pretrain/mm_projector.bin"
    local experiment_name="top1-min-head-w${CKA_FINAL_HIDDEN_WEIGHT}"
    local output_dir="${OUTPUT_ROOT}/${run_name}/${experiment_name}/llava-finetune"
    local wandb_run_name="${run_name}-${experiment_name}"

    if [ ! -f "${adapter}" ]; then
        if [[ "${SKIP_MISSING_ADAPTERS,,}" == "true" ]]; then
            echo "Skipping ${run_name}: adapter not found at ${adapter}"
            return
        fi
        echo "Missing pretrained adapter for ${run_name}: ${adapter}" >&2
        echo "Set PRETRAIN_ROOT or place the stage-1 adapter at that path." >&2
        return 1
    fi

    if [ -f "${output_dir}/config.json" ] && {
        [ -f "${output_dir}/model.safetensors" ] ||
        [ -f "${output_dir}/model.safetensors.index.json" ] ||
        [ -f "${output_dir}/pytorch_model.bin" ]
    }; then
        echo "Skipping completed run ${run_name}: model exists in ${output_dir}"
        return
    fi

    echo "Starting minimum-drift-head finetune: ${run_name}"
    echo "  model:   ${model_name}"
    echo "  adapter: ${adapter}"
    echo "  output:  ${output_dir}"
    echo "  heads:   ${head_ids}"
    if [[ "${DRY_RUN,,}" == "true" ]]; then
        return
    fi

    MODEL_NAME_OR_PATH="${model_name}" \
    RUN_NAME="${run_name}" \
    GPU_INCLUDE="${GPU_INCLUDE}" \
    PRETRAIN_ADAPTER="${adapter}" \
    OUTPUT_DIR="${output_dir}" \
    WANDB_RUN_NAME="${wandb_run_name}" \
    MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${per_device_batch_size}" \
    GRADIENT_ACCUMULATION_STEPS="${grad_accum_steps}" \
    CKA_LOSS_ENABLED=True \
    CKA_PROJECTOR_WEIGHT="${CKA_PROJECTOR_WEIGHT}" \
    CKA_FINAL_HIDDEN_WEIGHT="${CKA_FINAL_HIDDEN_WEIGHT}" \
    CKA_LOSS_START_RATIO="${CKA_LOSS_START_RATIO}" \
    CKA_RANDOM_HEADS=False \
    CKA_HEAD_IDS="${head_ids}" \
    CKA_ANCHOR_LAYER= \
    CKA_LAYERS=all \
    bash "${RUN_SCRIPT}"
}

run_case \
    "Qwen/Qwen2.5-1.5B-Instruct" \
    "qwen2.5-1.5b" \
    "32" \
    "4" \
    '{"1":[6],"2":[7],"3":[11],"4":[6],"5":[10],"6":[1],"7":[4],"8":[3],"9":[9],"10":[7],"11":[5],"12":[8],"13":[10],"14":[9],"15":[9],"16":[5],"17":[6],"18":[3],"19":[10],"20":[6],"21":[7],"22":[7],"23":[5],"24":[8],"25":[6],"26":[3],"27":[5],"28":[10]}'

run_case \
    "Qwen/Qwen3-0.6B" \
    "qwen3-0.6b" \
    "32" \
    "4" \
    '{"1":[1],"2":[15],"3":[4],"4":[11],"5":[4],"6":[1],"7":[7],"8":[13],"9":[14],"10":[7],"11":[15],"12":[9],"13":[6],"14":[0],"15":[15],"16":[6],"17":[14],"18":[13],"19":[7],"20":[3],"21":[11],"22":[9],"23":[8],"24":[10],"25":[5],"26":[9],"27":[9],"28":[8]}'

run_case \
    "Qwen/Qwen3-1.7B" \
    "qwen3-1.7b" \
    "32" \
    "4" \
    '{"1":[12],"2":[15],"3":[2],"4":[11],"5":[13],"6":[0],"7":[7],"8":[4],"9":[13],"10":[7],"11":[2],"12":[9],"13":[4],"14":[0],"15":[13],"16":[15],"17":[13],"18":[15],"19":[6],"20":[12],"21":[14],"22":[9],"23":[14],"24":[10],"25":[4],"26":[9],"27":[9],"28":[4]}'

run_case \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "llama-3.2-1b" \
    "32" \
    "4" \
    '{"1":[9],"2":[17],"3":[15],"4":[31],"5":[30],"6":[22],"7":[4],"8":[9],"9":[27],"10":[9],"11":[17],"12":[31],"13":[22],"14":[25],"15":[25],"16":[23]}'

run_case \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    "tinyllama-1.1b" \
    "32" \
    "4" \
    '{"1":[23],"2":[5],"3":[17],"4":[5],"5":[16],"6":[25],"7":[12],"8":[7],"9":[21],"10":[9],"11":[12],"12":[4],"13":[31],"14":[9],"15":[24],"16":[2],"17":[26],"18":[7],"19":[22],"20":[24],"21":[3],"22":[15]}'
