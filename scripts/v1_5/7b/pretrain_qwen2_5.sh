#!/bin/bash
set -euo pipefail

# sentencepiece 0.1.99 bundles protobuf bindings that are incompatible with
# protobuf 6.x's C++ runtime. Phi-3.5's tokenizer imports those bindings.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Please activate the Qwen training conda env first, e.g. conda activate llava-qwen" >&2
    exit 1
fi

: "${MODEL_NAME_OR_PATH:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${RUN_NAME:=qwen2.5-0.5b}"
: "${GPU_INCLUDE:=localhost:3}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=64}"
: "${GRADIENT_ACCUMULATION_STEPS:=4}"
: "${CKA_LOSS_START_RATIO:=0.0}"
: "${CKA_LOSS_TAU:=0.0}"
: "${CKA_ANCHOR_LAYER:=}"
: "${CKA_VISION_ANCHOR_LAYER:=}"
: "${CKA_PROJECTOR_VISION_ANCHOR_LAYER:=24}"
: "${CKA_FINAL_VISION_ANCHOR_LAYER:=}"
: "${MM_PROJECTOR_TYPE:=mlp2x_gelu}"
: "${CKA_LOSS_ENABLED:=True}"

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/pretrain-pcgrad/${RUN_NAME}/cka-proj-v${CKA_PROJECTOR_VISION_ANCHOR_LAYER}-mid-v${CKA_VISION_ANCHOR_LAYER}-final-v${CKA_FINAL_VISION_ANCHOR_LAYER}/llava-pretrain}"
OPTIONAL_CKA_ARGS=()
if [[ -n "${CKA_ANCHOR_LAYER}" ]]; then
    OPTIONAL_CKA_ARGS+=(--cka_loss_anchor_layer "${CKA_ANCHOR_LAYER}")
fi
if [[ -n "${CKA_VISION_ANCHOR_LAYER}" ]]; then
    OPTIONAL_CKA_ARGS+=(--cka_loss_vision_anchor_layer "${CKA_VISION_ANCHOR_LAYER}")
fi
if [[ -n "${CKA_PROJECTOR_VISION_ANCHOR_LAYER}" ]]; then
    OPTIONAL_CKA_ARGS+=(--cka_loss_projector_vision_anchor_layer "${CKA_PROJECTOR_VISION_ANCHOR_LAYER}")
fi
if [[ -n "${CKA_FINAL_VISION_ANCHOR_LAYER}" ]]; then
    OPTIONAL_CKA_ARGS+=(--cka_loss_final_vision_anchor_layer "${CKA_FINAL_VISION_ANCHOR_LAYER}")
fi

"${CONDA_PREFIX}/bin/deepspeed" --include "${GPU_INCLUDE}" llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --force_download False \
    --version plain \
    --data_path ./playground/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type "${MM_PROJECTOR_TYPE}" \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED:-42}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_strategy  "no" \
    --save_strategy "no" \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --cka_loss "${CKA_LOSS_ENABLED}" \
    --cka_loss_tau "${CKA_LOSS_TAU}" \
    --cka_loss_weight 0.1 \
    --cka_loss_start_ratio "${CKA_LOSS_START_RATIO}" \
    --vsp_gradient_diagnostics True \
    --vsp_asymmetric_pcgrad True \
    --vsp_apply_to_projector_only True \
    --vsp_norm_cap True \
    --vsp_pcgrad_threshold 0.05 \
    --vsp_proj_max_grad_ratio 0.5 \
    --vsp_llm_max_grad_ratio 0.5 \
    --vsp_grad_log_interval 10 \
    --cka_loss_layers "-1" \
    "${OPTIONAL_CKA_ARGS[@]}"
