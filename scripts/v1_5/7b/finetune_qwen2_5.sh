#!/bin/bash
set -euo pipefail

# Required by Phi-3.5 with sentencepiece 0.1.99 and protobuf 6.x.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Please activate the Qwen training conda env first, e.g. conda activate llava-qwen" >&2
    exit 1
fi

: "${MODEL_NAME_OR_PATH:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${RUN_NAME:=qwen2.5-1.5b}"
: "${GPU_INCLUDE:=localhost:3}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=16}"
: "${GRADIENT_ACCUMULATION_STEPS:=8}"
: "${CKA_FINAL_HIDDEN_WEIGHT:=0.1}"
: "${CKA_LOSS_START_RATIO:=0.0}"
: "${MM_PROJECTOR_TYPE:=coupling1x_gelu}"

PRETRAIN_ADAPTER="${PRETRAIN_ADAPTER:-./checkpoints/pretrain-coup/${RUN_NAME}-${MM_PROJECTOR_TYPE}/llava-pretrain/mm_projector.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/finetune-coup/${RUN_NAME}-${MM_PROJECTOR_TYPE}/base/llava-finetune}"

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
    --deepspeed "${DEEPSPEED_CONFIG:-./scripts/zero3.json}" \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}-finetune" \
    --cka_loss False \
    --cka_loss_tau 0.0 \
    --cka_loss_projector_weight 0.0 \
    --cka_loss_final_hidden_weight "${CKA_FINAL_HIDDEN_WEIGHT}" \
    --cka_loss_start_ratio "${CKA_LOSS_START_RATIO}" \
    --cka_loss_subset_query_tokens text \
    --vsp_gradient_diagnostics False \
    --vsp_asymmetric_pcgrad False \
    --vsp_norm_cap False \
    --vsp_pcgrad_threshold 0.05 \
    --vsp_proj_max_grad_ratio 0.5 \
    --vsp_llm_max_grad_ratio 0.5 \
    --vsp_grad_log_interval 10 \
    --cka_loss_layers "3"
