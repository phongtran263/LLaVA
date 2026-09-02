#!/bin/bash
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Please activate the Qwen training conda env first, e.g. conda activate llava-qwen" >&2
    exit 1
fi

python - <<'PY_CHECK'
from packaging import version
import accelerate
import transformers

if version.parse(transformers.__version__) != version.parse("4.43.1"):
    raise SystemExit(
        f"Qwen2.5 training expects transformers==4.43.1, got {transformers.__version__}. "
        "Activate llava-qwen or upgrade this env."
    )
if version.parse(accelerate.__version__) < version.parse("0.33.0"):
    raise SystemExit(
        f"Qwen2.5 training expects accelerate>=0.33.0 with transformers 4.43.x, got {accelerate.__version__}. "
        "Run: conda run -n llava-qwen python -m pip install accelerate==0.33.0"
    )
PY_CHECK

"${CONDA_PREFIX}/bin/deepspeed" --include localhost:0,1 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --version qwen2 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/qwen-7b-baseline/llava-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/lora-ft/qwen2.5-7b/llava-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name qwen2.5-7b-ft-lora \
    --cka_loss True \
    --use_pcgrad True \
    --cka_loss_weight 0.1 \
    --cka_loss_layers "final" \
    --cka_loss_hidden_channel_drop_indices "0,7,31" \
    --cka_loss_subset_select_layer 3 \
    --cka_loss_subset_min_ratio 0.01 \
    --cka_loss_subset_max_ratio 0.90 \
    --cka_loss_subset_fallback_mass 0.90 \
    --cka_loss_subset_otsu_min_separability 0.30
