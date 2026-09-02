#!/bin/bash
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Please activate the Qwen training conda env first, e.g. conda activate llava-qwen" >&2
    exit 1
fi

PRETRAIN_ADAPTER="${PRETRAIN_ADAPTER:-./checkpoints/qwen-3b-cka-grad/llava-pretrain/mm_projector.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen-3b-cka-grad-last/llava-finetune}"

python - <<'PY_CHECK'
from packaging import version
import accelerate
import transformers

if version.parse(transformers.__version__) != version.parse("4.43.1"):
    raise SystemExit(
        f"Qwen2.5 training expects transformers==4.43.1, got {transformers.__version__}. "
        "Activate llava-qwen or downgrade this env."
    )
if version.parse(accelerate.__version__) < version.parse("0.33.0"):
    raise SystemExit(
        f"Qwen2.5 training expects accelerate>=0.33.0 with transformers 4.43.x, got {accelerate.__version__}. "
        "Run: conda run -n llava-qwen python -m pip install accelerate==0.33.0"
    )
PY_CHECK

"${CONDA_PREFIX}/bin/deepspeed" --include localhost:2 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --version qwen2 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter "${PRETRAIN_ADAPTER}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --evaluation_strategy "no" \
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
    --run_name qwen-3b-cka-grad-finetune \
    --cka_loss True \
    --cka_loss_tau 0.0 \
    --cka_loss_projector_weight 0.1 \
    --cka_loss_final_hidden_weight 0.1 \
    --cka_loss_subset_query_tokens text \
    --vsp_gradient_diagnostics False \
    --vsp_asymmetric_pcgrad False \
    --vsp_norm_cap False \
    --vsp_pcgrad_threshold 0.05 \
    --vsp_proj_max_grad_ratio 0.5 \
    --vsp_llm_max_grad_ratio 0.5 \
    --vsp_grad_log_interval 10 \
    --cka_loss_layers "final"
