#!/bin/bash
set -euo pipefail

# Two independent full pipelines: baseline S1 -> S2, then CKA S1 -> S2.
# Both start from the base LLM and the same seeded projector initialization.
# Usage: GPU_INCLUDE=localhost:3 SEED=42 bash scripts/v1_5/7b/finetune_llama3_2_3b_baseline_cka.sh
# Set DRY_RUN=1 to print commands without starting training.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
TRAIN_SCRIPT="${SCRIPT_DIR}/finetune_qwen2_5.sh"
PRETRAIN_SCRIPT="${SCRIPT_DIR}/../pretrain_backbone.sh"

: "${RUN_NAME:=llama-3.2-3b}"
: "${GPU_INCLUDE:=localhost:3}"
: "${SEED:=42}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=8}"
: "${GRADIENT_ACCUMULATION_STEPS:=16}"
: "${PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE:=32}"
: "${PRETRAIN_GRADIENT_ACCUMULATION_STEPS:=8}"
: "${CKA_PROJECTOR_WEIGHT:=0.1}"
: "${CKA_FINAL_HIDDEN_WEIGHT:=0.1}"
: "${SWEEP_OUTPUT_ROOT:=./checkpoints/finetune-cka/${RUN_NAME}/baseline-vs-cka-stage1-stage2-seed${SEED}}"
: "${WANDB_RUN_GROUP:=${RUN_NAME}-baseline-vs-cka-stage1-stage2-seed${SEED}}"
: "${DRY_RUN:=0}"

if [[ "${DRY_RUN}" != 0 && "${DRY_RUN}" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if [[ "${DRY_RUN}" == 0 ]] && { [[ -z "${CONDA_PREFIX:-}" ]] || [[ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]]; }; then
    echo "Activate the training environment first, e.g. conda activate llava1." >&2
    exit 1
fi
# Trainer automatically resumes checkpoint-* in output_dir. Require fresh paths
# for ALL four jobs before starting; each stage 2 uses only its own stage 1.
for variant in baseline cka-proj-final; do
    for stage in pretrain finetune; do
        output="${SWEEP_OUTPUT_ROOT}/${variant}/llava-${stage}"
        if [[ -e "${output}" || -L "${output}" ]]; then
            echo "Output already exists: ${output}. Choose a new SWEEP_OUTPUT_ROOT." >&2
            exit 1
        fi
    done
done

COMMON_ENV=(
    "MODEL_NAME_OR_PATH=meta-llama/Llama-3.2-3B-Instruct"
    "RUN_NAME=${RUN_NAME}"
    "GPU_INCLUDE=${GPU_INCLUDE}"
    "SEED=${SEED}"
    "MM_PROJECTOR_TYPE=mlp2x_gelu"
    "VISION_TOWER=openai/clip-vit-large-patch14-336"
    "MM_VISION_SELECT_LAYER=-2"
    "PRETRAIN_ADAPTER="
    "NUM_TRAIN_EPOCHS=1"
    "MODEL_MAX_LENGTH=2048"
    "GRADIENT_CHECKPOINTING=False"
    "DATALOADER_NUM_WORKERS=16"
    "WANDB_RUN_GROUP=${WANDB_RUN_GROUP}"
    "CKA_LOSS_TAU=0.0"
    "CKA_LOSS_START_RATIO=0.0"
    # Empty vision overrides select normal V: CLIP hidden_states[-2] = V23.
    "CKA_ANCHOR_LAYER="
    "CKA_VISION_ANCHOR_LAYER="
    "CKA_PROJECTOR_VISION_ANCHOR_LAYER="
    "CKA_FINAL_VISION_ANCHOR_LAYER="
    "RESUME_FROM_CHECKPOINT="
    "STOP_AFTER_STEP_RATIO="
    "SAVE_STRATEGY=no"
    "VSP_DIAGNOSTICS=False"
    "VSP_PCGRAD=False"
    "VSP_NORM_CAP=False"
    "VSP_APPLY_TO_PROJECTOR_ONLY=False"
)

for variant in baseline cka-proj-final; do
    enabled=False
    projector_weight=0.0
    final_weight=0.0
    if [[ "${variant}" == cka-proj-final ]]; then
        enabled=True
        projector_weight="${CKA_PROJECTOR_WEIGHT}"
        final_weight="${CKA_FINAL_HIDDEN_WEIGHT}"
    fi
    adapter="${SWEEP_OUTPUT_ROOT}/${variant}/llava-pretrain/mm_projector.bin"
    for stage in pretrain finetune; do
        output="${SWEEP_OUTPUT_ROOT}/${variant}/llava-${stage}"
        if [[ "${stage}" == pretrain ]]; then
            stage_script="${PRETRAIN_SCRIPT}"
            stage_env=(
                "PER_DEVICE_TRAIN_BATCH_SIZE=${PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE}"
                "GRADIENT_ACCUMULATION_STEPS=${PRETRAIN_GRADIENT_ACCUMULATION_STEPS}"
                "DATA_PATH=./playground/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
                "IMAGE_FOLDER=./playground/LLaVA-Pretrain/images"
                "LEARNING_RATE=1e-3"
                "CKA_LAYERS=-1"
                "CKA_FINAL_HIDDEN_WEIGHT=0.0"
            )
        else
            if [[ "${DRY_RUN}" == 0 && ! -s "${adapter}" ]]; then
                echo "Stage 1 did not save a nonempty adapter: ${adapter}" >&2
                exit 1
            fi
            stage_script="${TRAIN_SCRIPT}"
            stage_env=(
                "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
                "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
                "PRETRAIN_ADAPTER=${adapter}"
                "CKA_LAYERS=final"
                "CKA_FINAL_HIDDEN_WEIGHT=${final_weight}"
            )
        fi
        command=(
            env -u WANDB_RUN_ID -u WANDB_RESUME
            "${COMMON_ENV[@]}"
            "${stage_env[@]}"
            "OUTPUT_DIR=${output}"
            "WANDB_RUN_NAME=${RUN_NAME}-${variant}-${stage}-seed${SEED}"
            "CKA_LOSS=${enabled}"
            "CKA_LOSS_ENABLED=${enabled}"
            "CKA_PROJECTOR_WEIGHT=${projector_weight}"
            bash "${stage_script}"
        )
        echo "${stage}: ${variant}; seed=${SEED}; output=${output}"
        if [[ "${DRY_RUN}" == 1 ]]; then
            printf '%q ' "${command[@]}"
            printf '\n'
        else
            # set -e stops the pipeline if any stage fails.
            "${command[@]}"
        fi
    done
done
