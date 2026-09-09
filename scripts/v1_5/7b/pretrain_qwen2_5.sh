#!/bin/bash
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Please activate the Qwen training conda env first, e.g. conda activate llava-qwen" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/pretrain-diag/qwen2.5-3b/llava-pretrain}"

"${CONDA_PREFIX}/bin/deepspeed" --include localhost:3 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --force_download False \
    --version plain \
    --data_path ./playground/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --eval_strategy  "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
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
    --run_name qwen-3b-pretrain \
    --cka_loss False \
    --cka_loss_tau 0.0 \
    --cka_loss_weight 1.0 \
    --vsp_gradient_diagnostics False \
    --vsp_asymmetric_pcgrad False \
    --vsp_apply_to_projector_only False \
    --vsp_norm_cap False \
    --vsp_pcgrad_threshold 0.05 \
    --vsp_proj_max_grad_ratio 0.5 \
    --vsp_llm_max_grad_ratio 0.5 \
    --vsp_grad_log_interval 10 \
    --cka_loss_layers "-1"
