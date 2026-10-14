#!/bin/bash
set -euo pipefail

# Single-process Stage 1. After MIN_PHASE_A_STEPS, switch on held-out plateau
# OR reserve MIN_PHASE_B_STEPS before the total budget runs out.
# SOURCE_STAGE1_CHECKPOINT initializes a NEW Phase B run with verified CE-only
# weights and the original holdout. CONTINUATION=ce_only is the matched control;
# it stays in the protected runtime for identical data/LR/schedule and gate
# diagnostics; protected checkpoints still require validation-gate acceptance.
# Phase A defaults to the usual Stage-1 LR 1e-3. Direct Phase B and CE-only
# continuation default to REFINE_LR; LEARNING_RATE can override either.
# DRY_RUN=1 prints a command without model loading or training.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH explicitly}"
: "${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH explicitly}"
: "${TRAIN_IMAGE_FOLDER:?Set TRAIN_IMAGE_FOLDER explicitly}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR explicitly}"
: "${REFINE_LR:?Set REFINE_LR explicitly}"
: "${MAX_STEPS:?Set MAX_STEPS explicitly}"
: "${GPU_ID:?Set one GPU_ID explicitly}"
if [[ "${GPU_ID}" == *,* || "${WORLD_SIZE:-1}" != 1 ]]; then
    echo "Protected CKA launcher supports one GPU_ID, not a device list." >&2
    exit 1
fi
if [[ -n "${DEEPSPEED_CONFIG:-}" || -n "${DEEPSPEED:-}" || -n "${GPU_INCLUDE:-}" ]]; then
    echo "Protected Stage 1 uses one GPU_ID; DeepSpeed/GPU_INCLUDE requests are not silently ignored." >&2
    exit 1
fi
if [[ "${DRY_RUN:-0}" != 0 && "${DRY_RUN:-0}" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi

optional_args=()
protected_enabled="${PROTECTED_CKA_REFINEMENT:-True}"
start_phase_b="${START_PHASE_B:-False}"
if [[ "${protected_enabled,,}" != true && "${protected_enabled,,}" != false ]]; then
    echo "PROTECTED_CKA_REFINEMENT must be True or False." >&2
    exit 1
fi
if [[ -n "${SOURCE_STAGE1_CHECKPOINT:-}" ]]; then
    if [[ "${protected_enabled,,}" != true || "${start_phase_b,,}" != true ]]; then
        echo "SOURCE_STAGE1_CHECKPOINT requires PROTECTED_CKA_REFINEMENT=True and START_PHASE_B=True; use CONTINUATION=ce_only for the control." >&2
        exit 1
    fi
    if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
        echo "SOURCE_STAGE1_CHECKPOINT and RESUME_FROM_CHECKPOINT are mutually exclusive." >&2
        exit 1
    fi
    optional_args+=(--protected_cka_source_stage1_checkpoint "${SOURCE_STAGE1_CHECKPOINT}")
fi
initial_lr="${LEARNING_RATE:-1e-3}"
if [[ "${protected_enabled,,}" == false || "${start_phase_b,,}" == true ]]; then
    initial_lr="${LEARNING_RATE:-${REFINE_LR}}"
fi
if [[ -n "${PRETRAIN_ADAPTER:-}" && -z "${SOURCE_STAGE1_CHECKPOINT:-}" ]]; then
    optional_args+=(--pretrain_mm_mlp_adapter "${PRETRAIN_ADAPTER}")
fi
if [[ -n "${VALIDATION_DATA_PATH:-}" ]]; then
    optional_args+=(--protected_cka_validation_data_path "${VALIDATION_DATA_PATH}")
fi
if [[ -n "${VALIDATION_IMAGE_FOLDER:-}" ]]; then
    optional_args+=(--protected_cka_validation_image_folder "${VALIDATION_IMAGE_FOLDER}")
fi
if [[ -n "${MAX_PHASE_A_STEPS:-}" ]]; then
    optional_args+=(--protected_cka_max_phase_a_steps "${MAX_PHASE_A_STEPS}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    optional_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

command=(
    env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "PYTHONPATH=${PWD}:${PYTHONPATH:-}"
    "${PYTHON_BIN:-python}" llava/train/train_mem.py
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --version plain
    --data_path "${TRAIN_DATA_PATH}"
    --image_folder "${TRAIN_IMAGE_FOLDER}"
    --vision_tower "${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
    --mm_projector_type "${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
    --mm_vision_select_layer "${MM_VISION_SELECT_LAYER:--2}"
    --tune_mm_mlp_adapter True
    --freeze_backbone True
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --bf16 "${BF16:-True}"
    --fp16 False
    --bits 16
    --output_dir "${OUTPUT_DIR}"
    --seed "${SEED:-42}"
    --max_steps "${MAX_STEPS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
    --learning_rate "${initial_lr}"
    --weight_decay "${WEIGHT_DECAY:-0.0}"
    --warmup_ratio "${WARMUP_RATIO:-0.03}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}"
    --eval_strategy no
    --save_strategy no
    --logging_steps "${LOGGING_STEPS:-1}"
    --report_to "${REPORT_TO:-none}"
    --model_max_length "${MODEL_MAX_LENGTH:-2048}"
    --gradient_checkpointing False
    --dataloader_num_workers 0
    --lazy_preprocess True
    --cka_loss False
    --use_pcgrad False
    --vsp_gradient_diagnostics False
    --vsp_asymmetric_pcgrad False
    --vsp_apply_to_projector_only False
    --vsp_norm_cap False
    --protected_cka_refinement "${PROTECTED_CKA_REFINEMENT:-True}"
    --protected_cka_start_phase_b "${START_PHASE_B:-False}"
    --protected_cka_continuation "${CONTINUATION:-protected}"
    --protected_cka_refine_lr "${REFINE_LR}"
    --protected_cka_margin "${CKA_MARGIN:-0.02}"
    --protected_cka_kl_weight "${KL_WEIGHT:-0.10}"
    --protected_cka_ema_decay "${EMA_DECAY:-0.99}"
    --protected_cka_aux_ratio "${AUX_RATIO:-0.10}"
    --protected_cka_lambda_max "${LAMBDA_MAX:-1.0}"
    --protected_cka_ce_tolerance "${CE_TOLERANCE:-0.005}"
    --protected_cka_text_kl_budget "${TEXT_KL_BUDGET:-0.01}"
    --protected_cka_drift_budget "${DRIFT_BUDGET:-0.01}"
    --protected_cka_warmup_steps "${CKA_WARMUP_STEPS:-100}"
    --protected_cka_ce_gradient_calibration_batches "${CE_GRADIENT_CALIBRATION_BATCHES:-8}"
    --protected_cka_eval_interval "${EVAL_INTERVAL:-50}"
    --protected_cka_max_retries "${MAX_RETRIES:-3}"
    --protected_cka_kl_chunk_size "${KL_CHUNK_SIZE:-128}"
    --protected_cka_aux_backoff "${AUX_BACKOFF:-0.5}"
    --protected_cka_eps "${CKA_EPS:-1e-8}"
    --protected_cka_min_phase_a_steps "${MIN_PHASE_A_STEPS:-1500}"
    --protected_cka_min_phase_b_steps "${MIN_PHASE_B_STEPS:-350}"
    --protected_cka_plateau_patience "${PLATEAU_PATIENCE:-3}"
    --protected_cka_plateau_min_delta "${PLATEAU_MIN_DELTA:-0.001}"
    --protected_cka_plateau_window "${PLATEAU_WINDOW:-3}"
    --protected_cka_validation_fraction "${VALIDATION_FRACTION:-0.005}"
    --protected_cka_validation_max_samples "${VALIDATION_MAX_SAMPLES:-2048}"
    --protected_cka_validation_seed "${VALIDATION_SEED:-42}"
    "${optional_args[@]}" "$@"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    "${command[@]}"
fi
