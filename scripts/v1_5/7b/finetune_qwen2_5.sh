#!/bin/bash
set -euo pipefail

# Required by Phi-3.5 with sentencepiece 0.1.99 and protobuf 6.x.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

: "${DRY_RUN:=0}"
if [[ "${DRY_RUN}" != 0 && "${DRY_RUN}" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
DEEPSPEED_BIN="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/}deepspeed"
if [[ "${DRY_RUN}" == 0 ]] && { [[ -z "${CONDA_PREFIX:-}" ]] || [[ ! -x "${DEEPSPEED_BIN}" ]]; }; then
    echo "Please activate the Qwen training conda env first, e.g. conda activate llava-qwen" >&2
    exit 1
fi

: "${MODEL_NAME_OR_PATH:=Qwen/Qwen2.5-7B-Instruct}"
: "${RUN_NAME:=qwen2.5-7b}"
: "${GPU_INCLUDE:=localhost:2,3}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=4}"
: "${GRADIENT_ACCUMULATION_STEPS:=16}"
: "${CKA_FINAL_HIDDEN_WEIGHT:=0.01}"
: "${CKA_FINAL_MLP_HEAD:=False}"
: "${CKA_LOSS_START_RATIO:=0.0}"
: "${MM_PROJECTOR_TYPE:=mlp2x_gelu}"
: "${CKA_LOSS_ENABLED:=False}"
# Accept raw CKA loss <= tau; 0.0 preserves the original objective.
: "${CKA_LOSS_TAU:=0.0}"
: "${CKA_PROJECTOR_WEIGHT:=0.01}"
: "${CKA_LAYERS:=-1}"
: "${CKA_ANCHOR_LAYER:=}"
# An explicitly empty value disables the corresponding anchor override.
: "${CKA_VISION_ANCHOR_LAYER=}"
: "${CKA_PROJECTOR_VISION_ANCHOR_LAYER=}"
: "${CKA_FINAL_VISION_ANCHOR_LAYER=}"
: "${SAVE_STRATEGY:=no}"
: "${STOP_AFTER_STEP_RATIO:=}"
: "${RESUME_FROM_CHECKPOINT:=}"
: "${WANDB_RUN_NAME:=${RUN_NAME}-cka-proj}"

PRETRAIN_ADAPTER="${PRETRAIN_ADAPTER:-./checkpoints/pretrain-baseline/${RUN_NAME}-1/llava-pretrain/mm_projector.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/finetune-baseline/${RUN_NAME}-1/llava-finetune}"
OPTIONAL_TRAIN_ARGS=()
if [[ -n "${STOP_AFTER_STEP_RATIO}" ]]; then
    OPTIONAL_TRAIN_ARGS+=(--stop_after_step_ratio "${STOP_AFTER_STEP_RATIO}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    OPTIONAL_TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${CKA_ANCHOR_LAYER}" ]]; then
    OPTIONAL_TRAIN_ARGS+=(--cka_loss_anchor_layer "${CKA_ANCHOR_LAYER}")
fi
if [[ -n "${CKA_VISION_ANCHOR_LAYER}" ]]; then
    OPTIONAL_TRAIN_ARGS+=(--cka_loss_vision_anchor_layer "${CKA_VISION_ANCHOR_LAYER}")
fi
if [[ -n "${CKA_PROJECTOR_VISION_ANCHOR_LAYER}" ]]; then
    OPTIONAL_TRAIN_ARGS+=(--cka_loss_projector_vision_anchor_layer "${CKA_PROJECTOR_VISION_ANCHOR_LAYER}")
fi
if [[ -n "${CKA_FINAL_VISION_ANCHOR_LAYER}" ]]; then
    OPTIONAL_TRAIN_ARGS+=(--cka_loss_final_vision_anchor_layer "${CKA_FINAL_VISION_ANCHOR_LAYER}")
fi
if [[ "${DRY_RUN}" == 0 ]]; then
PYTHONNOUSERSITE=1 "${CONDA_PREFIX}/bin/python" - <<'PY_CHECK'
from packaging import version
import accelerate
import transformers

if version.parse(transformers.__version__) != version.parse("4.51.3"):
    raise SystemExit(
        f"Multibackbone training expects transformers==4.51.3, got {transformers.__version__}. "
        "Activate the llava1 environment."
    )
if version.parse(accelerate.__version__) < version.parse("1.6.0"):
    raise SystemExit(
        f"Multibackbone training expects accelerate>=1.6.0 with transformers 4.51.x, got {accelerate.__version__}. "
        "Run: conda run -n llava-qwen python -m pip install accelerate==1.6.0"
    )
PY_CHECK
fi

COMMAND=(
    env -u WANDB_RUN_ID -u WANDB_RESUME
    PYTHONNOUSERSITE=1
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION}"
    "WANDB_RUN_NAME=${WANDB_RUN_NAME}"
)
if [[ -n "${WANDB_RUN_GROUP:-}" ]]; then
    COMMAND+=("WANDB_RUN_GROUP=${WANDB_RUN_GROUP}")
fi
COMMAND+=(
    "${DEEPSPEED_BIN}" --include "${GPU_INCLUDE}" llava/train/train_mem.py \
    --deepspeed "${DEEPSPEED_CONFIG:-./scripts/zero2.json}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version "${CONVERSATION_VERSION:-auto}" \
    --data_path "${DATA_PATH:-./playground/data/llava_v1_5_mix665k.json}" \
    --image_folder "${IMAGE_FOLDER:-./playground/data}" \
    --vision_tower "${VISION_TOWER:-openai/clip-vit-large-patch14-336}" \
    --pretrain_mm_mlp_adapter "${PRETRAIN_ADAPTER}" \
    --mm_projector_type "${MM_PROJECTOR_TYPE}" \
    --mm_vision_select_layer "${MM_VISION_SELECT_LAYER:--2}" \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type flat \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter False \
    --freeze_backbone False \
    --freeze_mm_mlp_adapter False \
    --lora_enable False \
    --bits 16 \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED:-42}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-16}" \
    --lazy_preprocess True \
    --report_to "${REPORT_TO:-wandb}" \
    --run_name "${WANDB_RUN_NAME}" \
    --cka_loss "${CKA_LOSS_ENABLED}" \
    --cka_loss_final_mlp_head "${CKA_FINAL_MLP_HEAD}" \
    --cka_loss_geometry linear \
    --cka_loss_tau "${CKA_LOSS_TAU}" \
    --cka_loss_projector_weight "${CKA_PROJECTOR_WEIGHT}" \
    --cka_loss_final_hidden_weight "${CKA_FINAL_HIDDEN_WEIGHT}" \
    --cka_loss_start_ratio "${CKA_LOSS_START_RATIO}" \
    --use_pcgrad False \
    --vsp_gradient_diagnostics "${VSP_DIAGNOSTICS:-False}" \
    --vsp_asymmetric_pcgrad "${VSP_PCGRAD:-False}" \
    --vsp_apply_to_projector_only "${VSP_APPLY_TO_PROJECTOR_ONLY:-False}" \
    --vsp_norm_cap "${VSP_NORM_CAP:-False}" \
    --vsp_pcgrad_threshold 0.05 \
    --vsp_proj_max_grad_ratio 0.5 \
    --vsp_llm_max_grad_ratio 0.5 \
    --vsp_grad_log_interval 10 \
    --cka_loss_layers "${CKA_LAYERS}" \
    --protected_cka_refinement False \
    --protected_stage2 False \
    --log_gradient_norms False \
    "${OPTIONAL_TRAIN_ARGS[@]}"
)
if [[ "${DRY_RUN}" == 1 ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
else
    "${COMMAND[@]}"
fi
