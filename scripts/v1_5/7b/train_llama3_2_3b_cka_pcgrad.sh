#!/bin/bash
set -euo pipefail

# Llama-3.2-3B: CKA S1 -> S2, projector PCGrad defaults on in both stages.
# FINETUNE_PCGRAD=False disables PCGrad in stage 2 only.
# S1: projector CKA. S2: projector + final CKA; final CKA is not projected.
# No norm cap, gradient diagnostics, or baseline run. Uses normal V as anchor.
# Usage: GPU_INCLUDE=localhost:3 SEED=42 bash scripts/v1_5/7b/train_llama3_2_3b_cka_pcgrad.sh
# For stage 2 only: STAGE=finetune PRETRAIN_ADAPTER=/path/to/mm_projector.bin
# DRY_RUN=1 prints the selected commands without starting training.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

: "${RUN_NAME:=llama-3.2-3b}"
: "${GPU_INCLUDE:=localhost:3}"
: "${SEED:=42}"
: "${PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE:=32}"
: "${PRETRAIN_GRADIENT_ACCUMULATION_STEPS:=8}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=8}"
: "${GRADIENT_ACCUMULATION_STEPS:=16}"
: "${CKA_PROJECTOR_WEIGHT:=0.1}"
: "${CKA_FINAL_HIDDEN_WEIGHT:=0.1}"
: "${OUTPUT_ROOT:=./checkpoints/finetune-cka/${RUN_NAME}/cka-proj-final-pcgrad-stage1-stage2-seed${SEED}}"
: "${WANDB_RUN_GROUP:=${RUN_NAME}-cka-proj-final-pcgrad-stage1-stage2-seed${SEED}}"
: "${DRY_RUN:=0}"
: "${STAGE:=all}"

case "${STAGE}" in
    all) stages=(pretrain finetune) ;;
    finetune) stages=(finetune) ;;
    *) echo "STAGE must be all or finetune." >&2; exit 1 ;;
esac

adapter="${OUTPUT_ROOT}/llava-pretrain/mm_projector.bin"
if [[ "${STAGE}" == finetune ]]; then
    adapter="${PRETRAIN_ADAPTER:-${adapter}}"
fi

if [[ "${DRY_RUN}" != 0 && "${DRY_RUN}" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if [[ "${DRY_RUN}" == 0 ]] && { [[ -z "${CONDA_PREFIX:-}" ]] || [[ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]]; }; then
    echo "Activate the training environment first, e.g. conda activate llava1." >&2
    exit 1
fi
# Trainer auto-resumes checkpoint-* inside output_dir. Require selected outputs
# to be fresh; stage-2-only runs may reuse an existing stage-1 adapter.
for stage in "${stages[@]}"; do
    output="${OUTPUT_ROOT}/llava-${stage}"
    if [[ -e "${output}" || -L "${output}" ]]; then
        echo "Output already exists: ${output}. Choose a new OUTPUT_ROOT." >&2
        exit 1
    fi
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
    "CKA_LOSS=True"
    "CKA_LOSS_ENABLED=True"
    "CKA_PROJECTOR_WEIGHT=${CKA_PROJECTOR_WEIGHT}"
    "CKA_LOSS_TAU=0.0"
    "CKA_LOSS_START_RATIO=0.0"
    # Normal V = selected CLIP features, hidden_states[-2] = V23.
    "CKA_ANCHOR_LAYER="
    "CKA_VISION_ANCHOR_LAYER="
    "CKA_PROJECTOR_VISION_ANCHOR_LAYER="
    "CKA_FINAL_VISION_ANCHOR_LAYER="
    "RESUME_FROM_CHECKPOINT="
    "STOP_AFTER_STEP_RATIO="
    "SAVE_STRATEGY=no"
    "VSP_DIAGNOSTICS=False"
    "VSP_NORM_CAP=False"
    "VSP_PCGRAD_THRESHOLD=0.05"
)

for stage in "${stages[@]}"; do
    if [[ "${stage}" == pretrain ]]; then
        pcgrad=True
        stage_script="${SCRIPT_DIR}/../pretrain_backbone.sh"
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
        pcgrad="${FINETUNE_PCGRAD:-True}"
        if [[ "${DRY_RUN}" == 0 && ! -s "${adapter}" ]]; then
            echo "Stage 1 did not save a nonempty adapter (or PRETRAIN_ADAPTER is missing/empty): ${adapter}" >&2
            exit 1
        fi
        stage_script="${SCRIPT_DIR}/finetune_qwen2_5.sh"
        stage_env=(
            "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
            "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
            "PRETRAIN_ADAPTER=${adapter}"
            "CKA_LAYERS=final"
            "CKA_FINAL_HIDDEN_WEIGHT=${CKA_FINAL_HIDDEN_WEIGHT}"
        )
    fi
    command=(
        env -u WANDB_RUN_ID -u WANDB_RESUME
        "${COMMON_ENV[@]}" "${stage_env[@]}"
        "VSP_PCGRAD=${pcgrad}"
        "VSP_APPLY_TO_PROJECTOR_ONLY=${pcgrad}"
        "OUTPUT_DIR=${OUTPUT_ROOT}/llava-${stage}"
        "WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_NAME}-cka-proj-final-pcgrad-stage1-stage2-${stage}-seed${SEED}}"
        bash "${stage_script}"
    )
    echo "${stage}: projector PCGrad=${pcgrad}; seed=${SEED}; output=${OUTPUT_ROOT}/llava-${stage}"
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
done
