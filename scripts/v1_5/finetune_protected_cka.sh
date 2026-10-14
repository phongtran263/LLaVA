#!/bin/bash
set -euo pipefail

# Stage 2 instruction CE -> plateau OR reserved Phase B budget -> refinement.
# No DeepSpeed/DDP in this first implementation. Full fine-tuning is the
# default; LORA_ENABLE=True explicitly selects the existing LoRA recipe.
# Stage 2 gate thresholds are required under STAGE2_* names; Stage 1 values
# are never implicitly reused. DRY_RUN=1 prints only, without model loading.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH explicitly}"
: "${TRAIN_DATA_PATH:?Set Stage 2 instruction TRAIN_DATA_PATH explicitly}"
: "${TRAIN_IMAGE_FOLDER:?Set TRAIN_IMAGE_FOLDER explicitly}"
: "${OUTPUT_DIR:?Set a fresh Stage 2 OUTPUT_DIR explicitly}"
: "${MAX_STEPS:?Set the total Stage 2 MAX_STEPS budget}"
: "${GPU_ID:?Set one GPU_ID explicitly}"
if [[ "${GPU_ID}" == *,* || "${WORLD_SIZE:-1}" != 1 ]]; then
    echo "Protected Stage 2 currently supports one GPU/process, not multi-GPU or distributed training." >&2
    exit 1
fi
if [[ -n "${DEEPSPEED_CONFIG:-}" || -n "${DEEPSPEED:-}" || -n "${GPU_INCLUDE:-}" ]]; then
    echo "This launcher does not support DeepSpeed/GPU_INCLUDE. Use one GPU_ID; requested backends are not silently ignored." >&2
    exit 1
fi
if [[ "${DRY_RUN:-0}" != 0 && "${DRY_RUN:-0}" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi

optional_args=()
enabled="${PROTECTED_CKA_STAGE2:-True}"
start_phase_b="${STAGE2_START_PHASE_B:-False}"
if [[ "${enabled,,}" != true && "${enabled,,}" != false ]]; then
    echo "PROTECTED_CKA_STAGE2 must be True or False." >&2
    exit 1
fi
if [[ "${enabled,,}" == false && -n "${SOURCE_STAGE2_CHECKPOINT:-}" ]]; then
    echo "SOURCE_STAGE2_CHECKPOINT requires PROTECTED_CKA_STAGE2=True; use STAGE2_CONTINUATION=ce_only for matched CE continuation." >&2
    exit 1
fi
if [[ -n "${SOURCE_STAGE2_CHECKPOINT:-}" ]]; then
    if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
        echo "SOURCE_STAGE2_CHECKPOINT and RESUME_FROM_CHECKPOINT are mutually exclusive." >&2
        exit 1
    fi
    if [[ "${start_phase_b,,}" != true ]]; then
        echo "SOURCE_STAGE2_CHECKPOINT requires STAGE2_START_PHASE_B=True for a new continuation." >&2
        exit 1
    fi
fi
initial_lr="${LEARNING_RATE:-2e-5}"
if [[ "${enabled,,}" == true ]]; then
    : "${STAGE2_REFINE_LR:?Set STAGE2_REFINE_LR explicitly}"
    : "${STAGE2_MAX_PHASE_B_STEPS:?Set the independent refinement STAGE2_MAX_PHASE_B_STEPS budget}"
    : "${STAGE2_CE_TOLERANCE:?Choose STAGE2_CE_TOLERANCE for instruction validation, not a Stage 1 default}"
    : "${STAGE2_TEXT_KL_BUDGET:?Choose STAGE2_TEXT_KL_BUDGET for supervised answer tokens}"
    if [[ "${start_phase_b,,}" == true ]]; then
        initial_lr="${LEARNING_RATE:-${STAGE2_REFINE_LR}}"
    fi
    optional_args+=(
        --protected_stage2_start_phase_b "${start_phase_b}"
        --protected_stage2_continuation "${STAGE2_CONTINUATION:-protected}"
        --protected_stage2_reference_device "${STAGE2_REFERENCE_DEVICE:-same}"
        --cka_grad_scope "${CKA_GRAD_SCOPE:-existing}"
        --protected_stage2_refine_lr "${STAGE2_REFINE_LR}"
        --protected_stage2_max_phase_b_steps "${STAGE2_MAX_PHASE_B_STEPS}"
        --protected_stage2_ce_tolerance "${STAGE2_CE_TOLERANCE}"
        --protected_stage2_text_kl_budget "${STAGE2_TEXT_KL_BUDGET}"
        --protected_stage2_margin "${STAGE2_CKA_MARGIN:-0.02}"
        --protected_stage2_kl_weight "${STAGE2_KL_WEIGHT:-0.10}"
        --protected_stage2_ema_decay "${STAGE2_EMA_DECAY:-0.99}"
        --protected_stage2_proj_aux_ratio "${STAGE2_PROJ_AUX_RATIO:-0.10}"
        --protected_stage2_llm_aux_ratio "${STAGE2_LLM_AUX_RATIO:-0.10}"
        --protected_stage2_lambda_max "${STAGE2_LAMBDA_MAX:-1.0}"
        --protected_stage2_warmup_steps "${STAGE2_CKA_WARMUP_STEPS:-100}"
        --protected_stage2_ce_gradient_calibration_batches "${STAGE2_CE_GRADIENT_CALIBRATION_BATCHES:-8}"
        --protected_stage2_eval_interval "${STAGE2_EVAL_INTERVAL:-100}"
        --protected_stage2_max_retries "${STAGE2_MAX_RETRIES:-3}"
        --protected_stage2_kl_chunk_size "${STAGE2_KL_CHUNK_SIZE:-128}"
        --protected_stage2_aux_backoff "${STAGE2_AUX_BACKOFF:-0.5}"
        --protected_stage2_eps "${STAGE2_EPS:-1e-8}"
        --protected_stage2_min_phase_a_steps "${STAGE2_MIN_PHASE_A_STEPS:-1500}"
        --protected_stage2_min_phase_b_steps "${STAGE2_MIN_PHASE_B_STEPS:-350}"
        --protected_stage2_plateau_patience "${STAGE2_PLATEAU_PATIENCE:-3}"
        --protected_stage2_plateau_min_delta "${STAGE2_PLATEAU_MIN_DELTA:-0.001}"
        --protected_stage2_plateau_window "${STAGE2_PLATEAU_WINDOW:-3}"
        --protected_stage2_validation_fraction "${STAGE2_VALIDATION_FRACTION:-0.005}"
        --protected_stage2_validation_max_samples "${STAGE2_VALIDATION_MAX_SAMPLES:-2048}"
        --protected_stage2_validation_seed "${STAGE2_VALIDATION_SEED:-42}"
    )
fi
if [[ -n "${PRETRAIN_ADAPTER:-}" && -z "${SOURCE_STAGE2_CHECKPOINT:-}" ]]; then
    optional_args+=(--pretrain_mm_mlp_adapter "${PRETRAIN_ADAPTER}")
elif [[ -z "${SOURCE_STAGE2_CHECKPOINT:-}" && -z "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    echo "Set PRETRAIN_ADAPTER to the configured Stage 1 mm_projector.bin; no checkpoint path is guessed." >&2
    exit 1
fi
if [[ -n "${SOURCE_STAGE2_CHECKPOINT:-}" ]]; then
    optional_args+=(--protected_stage2_source_stage2_checkpoint "${SOURCE_STAGE2_CHECKPOINT}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    optional_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${STAGE2_VALIDATION_DATA_PATH:-}" ]]; then
    optional_args+=(--protected_stage2_validation_data_path "${STAGE2_VALIDATION_DATA_PATH}")
fi
if [[ -n "${STAGE2_VALIDATION_IMAGE_FOLDER:-}" ]]; then
    optional_args+=(--protected_stage2_validation_image_folder "${STAGE2_VALIDATION_IMAGE_FOLDER}")
fi
if [[ -n "${STAGE2_MAX_PHASE_A_STEPS:-}" ]]; then
    optional_args+=(--protected_stage2_max_phase_a_steps "${STAGE2_MAX_PHASE_A_STEPS}")
fi
if [[ -n "${SAVE_TOTAL_LIMIT:-}" ]]; then
    optional_args+=(--save_total_limit "${SAVE_TOTAL_LIMIT}")
fi
if [[ -n "${STAGE2_DRIFT_BUDGET:-}" ]]; then
    optional_args+=(--protected_stage2_drift_budget "${STAGE2_DRIFT_BUDGET}")
fi
if [[ -n "${MM_PROJECTOR_LR:-}" ]]; then
    optional_args+=(--mm_projector_lr "${MM_PROJECTOR_LR}")
fi
if [[ -n "${CKA_ANCHOR_LAYER:-}" ]]; then
    optional_args+=(--cka_loss_anchor_layer "${CKA_ANCHOR_LAYER}")
fi
if [[ -n "${CKA_VISION_ANCHOR_LAYER:-}" ]]; then
    optional_args+=(--cka_loss_vision_anchor_layer "${CKA_VISION_ANCHOR_LAYER}")
fi
if [[ -n "${CKA_PROJECTOR_VISION_ANCHOR_LAYER:-}" ]]; then
    optional_args+=(--cka_loss_projector_vision_anchor_layer "${CKA_PROJECTOR_VISION_ANCHOR_LAYER}")
fi
if [[ -n "${CKA_FINAL_VISION_ANCHOR_LAYER:-}" ]]; then
    optional_args+=(--cka_loss_final_vision_anchor_layer "${CKA_FINAL_VISION_ANCHOR_LAYER}")
fi

command=(
    env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "PYTHONPATH=${PWD}:${PYTHONPATH:-}"
    "${PYTHON_BIN:-python}" llava/train/train_mem.py
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --version "${CONVERSATION_VERSION:-auto}"
    --data_path "${TRAIN_DATA_PATH}"
    --image_folder "${TRAIN_IMAGE_FOLDER}"
    --vision_tower "${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
    --mm_projector_type "${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
    --mm_vision_select_layer "${MM_VISION_SELECT_LAYER:--2}"
    --tune_mm_mlp_adapter False
    --freeze_mm_mlp_adapter "${FREEZE_MM_MLP_ADAPTER:-False}"
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio "${IMAGE_ASPECT_RATIO:-pad}"
    --lora_enable "${LORA_ENABLE:-False}"
    --lora_r "${LORA_R:-64}"
    --lora_alpha "${LORA_ALPHA:-16}"
    --lora_dropout "${LORA_DROPOUT:-0.05}"
    --lora_bias "${LORA_BIAS:-none}"
    --bf16 "${BF16:-True}"
    --fp16 False
    --bits 16
    --output_dir "${OUTPUT_DIR}"
    --seed "${SEED:-42}"
    --max_steps "${MAX_STEPS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
    --group_by_modality_length "${GROUP_BY_MODALITY_LENGTH:-True}"
    --learning_rate "${initial_lr}"
    --weight_decay "${WEIGHT_DECAY:-0.0}"
    --warmup_ratio "${WARMUP_RATIO:-0.03}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}"
    --eval_strategy no
    --save_strategy no
    --logging_steps "${LOGGING_STEPS:-10}"
    --report_to "${REPORT_TO:-none}"
    --run_name "${RUN_NAME:-protected-cka-stage2}"
    --model_max_length "${MODEL_MAX_LENGTH:-2048}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-False}"
    --gradient_checkpointing_kwargs '{"use_reentrant":false}'
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}"
    --lazy_preprocess True
    --cka_loss "${CKA_LOSS_ENABLED:-True}"
    --cka_loss_tau "${CKA_LOSS_TAU:-0.0}"
    --cka_loss_projector_weight "${CKA_PROJECTOR_WEIGHT:-0.1}"
    --cka_loss_final_hidden_weight "${CKA_FINAL_HIDDEN_WEIGHT:-0.1}"
    --cka_loss_start_ratio "${CKA_LOSS_START_RATIO:-0.0}"
    --cka_loss_layers "${CKA_LAYERS:-final}"
    --cka_loss_subset_query_tokens "${CKA_SUBSET_QUERY_TOKENS:-text}"
    --use_pcgrad False
    --vsp_gradient_diagnostics False
    --vsp_asymmetric_pcgrad False
    --vsp_apply_to_projector_only False
    --vsp_norm_cap False
    --protected_cka_refinement False
    --protected_stage2 "${enabled}"
    "${optional_args[@]}" "$@"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    "${command[@]}"
fi
