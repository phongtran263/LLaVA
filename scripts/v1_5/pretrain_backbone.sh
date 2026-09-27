#!/bin/bash
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Activate the project environment first (transformers==4.51.3)." >&2
    exit 1
fi

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH, e.g. Qwen/Qwen3-0.6B}"
: "${RUN_NAME:?Set RUN_NAME, e.g. qwen3-0.6b}"

GPU_INCLUDE="${GPU_INCLUDE:-localhost:0,1}"
MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE:-coupling1x_gelu}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/${RUN_NAME}-${MM_PROJECTOR_TYPE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/llava-pretrain}"
CKA_LOSS="${CKA_LOSS:-False}"
CKA_PROJECTOR_WEIGHT="${CKA_PROJECTOR_WEIGHT:-0.1}"
CKA_FINAL_HIDDEN_WEIGHT="${CKA_FINAL_HIDDEN_WEIGHT:-0.0}"
CKA_LOSS_START_RATIO="${CKA_LOSS_START_RATIO:-0.0}"
CKA_LAYERS="${CKA_LAYERS:--1}"
CKA_ANCHOR_LAYER="${CKA_ANCHOR_LAYER:-}"
CKA_RANDOM_HEADS="${CKA_RANDOM_HEADS:-False}"
CKA_HEAD_IDS="${CKA_HEAD_IDS:-}"
# Example manual selection: CKA_HEAD_IDS='{"1":[0,3],"4":[2,5,7]}'
CKA_HEAD_FRACTION="${CKA_HEAD_FRACTION:-0.25}"
CKA_HEAD_SEED="${CKA_HEAD_SEED:-42}"
VSP_DIAGNOSTICS="${VSP_DIAGNOSTICS:-False}"
VSP_PCGRAD="${VSP_PCGRAD:-False}"
VSP_NORM_CAP="${VSP_NORM_CAP:-False}"
OPTIONAL_CKA_ARGS=()
if [[ -n "${CKA_HEAD_IDS}" ]]; then
    OPTIONAL_CKA_ARGS+=(--cka_loss_head_ids "${CKA_HEAD_IDS}")
fi
if [[ -n "${CKA_ANCHOR_LAYER}" ]]; then
    OPTIONAL_CKA_ARGS+=(--cka_loss_anchor_layer "${CKA_ANCHOR_LAYER}")
fi

"${CONDA_PREFIX}/bin/deepspeed" --include "${GPU_INCLUDE}" llava/train/train_mem.py \
    --deepspeed "${DEEPSPEED_CONFIG:-./scripts/zero2.json}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version plain \
    --data_path "${DATA_PATH:-./playground/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}" \
    --image_folder "${IMAGE_FOLDER:-./playground/LLaVA-Pretrain/images}" \
    --vision_tower "${VISION_TOWER:-openai/clip-vit-large-patch14-336}" \
    --mm_projector_type "${MM_PROJECTOR_TYPE}" \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer "${MM_VISION_SELECT_LAYER:--2}" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-16}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS:-24000}" \
    --save_total_limit 1 \
    --learning_rate "${LEARNING_RATE:-1e-3}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH:-2048}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-False}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}-pretrain" \
    --cka_loss "${CKA_LOSS}" \
    --cka_loss_projector_weight "${CKA_PROJECTOR_WEIGHT}" \
    --cka_loss_final_hidden_weight "${CKA_FINAL_HIDDEN_WEIGHT}" \
    --cka_loss_start_ratio "${CKA_LOSS_START_RATIO}" \
    --cka_loss_random_heads "${CKA_RANDOM_HEADS}" \
    --cka_loss_head_fraction "${CKA_HEAD_FRACTION}" \
    --cka_loss_head_seed "${CKA_HEAD_SEED}" \
    --cka_loss_layers "${CKA_LAYERS}" \
    --vsp_gradient_diagnostics "${VSP_DIAGNOSTICS}" \
    --vsp_asymmetric_pcgrad "${VSP_PCGRAD}" \
    --vsp_norm_cap "${VSP_NORM_CAP}" \
    --vsp_pcgrad_threshold "${VSP_PCGRAD_THRESHOLD:-0.05}" \
    --vsp_proj_max_grad_ratio "${VSP_PROJ_MAX_GRAD_RATIO:-0.5}" \
    --vsp_llm_max_grad_ratio "${VSP_LLM_MAX_GRAD_RATIO:-0.5}" \
    --vsp_grad_log_interval "${VSP_GRAD_LOG_INTERVAL:-10}" \
    "${OPTIONAL_CKA_ARGS[@]}"
