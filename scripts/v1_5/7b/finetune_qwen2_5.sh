#!/bin/bash
set -euo pipefail

# Required by Phi-3.5 with sentencepiece 0.1.99 and protobuf 6.x.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Please activate the Qwen training conda env first, e.g. conda activate llava-qwen" >&2
    exit 1
fi

: "${MODEL_NAME_OR_PATH:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${RUN_NAME:=qwen2.5-0.5b}"
: "${GPU_INCLUDE:=localhost:3}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=16}"
: "${GRADIENT_ACCUMULATION_STEPS:=8}"
: "${CKA_FINAL_HIDDEN_WEIGHT:=0.1}"
: "${CKA_LOSS_START_RATIO:=0.0}"
: "${MM_PROJECTOR_TYPE:=mlp2x_gelu}"
: "${CKA_LOSS_ENABLED:=True}"
# Accept raw CKA loss <= tau; 0.0 preserves the original objective.
: "${CKA_LOSS_TAU:=0.0}"
: "${CKA_PROJECTOR_WEIGHT:=0.1}"
: "${CKA_LAYERS:=}"
: "${CKA_ANCHOR_LAYER:=}"
# An explicitly empty value disables the corresponding anchor override.
: "${CKA_VISION_ANCHOR_LAYER=}"
: "${CKA_PROJECTOR_VISION_ANCHOR_LAYER=}"
: "${CKA_FINAL_VISION_ANCHOR_LAYER=}"
: "${SAVE_STRATEGY:=no}"
: "${STOP_AFTER_STEP_RATIO:=}"
: "${RESUME_FROM_CHECKPOINT:=}"
: "${WANDB_RUN_NAME:=${RUN_NAME}-cka-proj-v${CKA_PROJECTOR_VISION_ANCHOR_LAYER}-mid-v${CKA_VISION_ANCHOR_LAYER}-final-v${CKA_FINAL_VISION_ANCHOR_LAYER}}"

PRETRAIN_ADAPTER="${PRETRAIN_ADAPTER:-./checkpoints/pretrain-pcgrad/${RUN_NAME}/llava-pretrain/mm_projector.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/finetune-cka/${RUN_NAME}/cka-proj-grad/llava-finetune}"
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
python - <<'PY_CHECK'
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

"${CONDA_PREFIX}/bin/deepspeed" --include "${GPU_INCLUDE}" llava/train/train_mem.py \
    --deepspeed "${DEEPSPEED_CONFIG:-./scripts/zero2.json}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version auto \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter "${PRETRAIN_ADAPTER}" \
    --mm_projector_type "${MM_PROJECTOR_TYPE}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
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
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${WANDB_RUN_NAME}" \
    --cka_loss "${CKA_LOSS_ENABLED}" \
    --cka_loss_tau "${CKA_LOSS_TAU}" \
    --cka_loss_projector_weight "${CKA_PROJECTOR_WEIGHT}" \
    --cka_loss_final_hidden_weight "${CKA_FINAL_HIDDEN_WEIGHT}" \
    --cka_loss_start_ratio "${CKA_LOSS_START_RATIO}" \
    --cka_loss_subset_query_tokens text \
    --vsp_gradient_diagnostics False \
    --vsp_asymmetric_pcgrad False \
    --vsp_apply_to_projector_only False \
    --vsp_norm_cap False \
    --vsp_pcgrad_threshold 0.05 \
    --vsp_proj_max_grad_ratio 0.5 \
    --vsp_llm_max_grad_ratio 0.5 \
    --vsp_grad_log_interval 10 \
    --cka_loss_layers "${CKA_LAYERS}" \
    "${OPTIONAL_TRAIN_ARGS[@]}"
