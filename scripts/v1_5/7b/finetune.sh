#!/bin/bash
set -euo pipefail
# lmsys/vicuna-7b-v1.5
# lmsys/vicuna-13b-v1.5
# openlm-research/open_llama_3b_v2
# mtgv/MobileLLaMA-1.4B-Chat
# TinyLlama/TinyLlama-1.1B-Chat-v1.0

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Please activate the LLaVA training conda env first, e.g. conda activate llava" >&2
    echo "Current CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}" >&2
    exit 1
fi

if [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Active conda env does not provide DeepSpeed: ${CONDA_PREFIX}/bin/deepspeed" >&2
    echo "Please activate the LLaVA training conda env first, e.g. conda activate llava" >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
CKA_LOSS_SUBSET_QUERY_TOKENS="${CKA_LOSS_SUBSET_QUERY_TOKENS:-text}"

"${CONDA_PREFIX}/bin/deepspeed" --include localhost:0,1 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/7b-cka-grad/llava-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/7b-cka-grad/llava-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
    --run_name 7b-cka-grad-finetune \
    --cka_loss True \
    --use_pcgrad False \
    --vsp_asymmetric_pcgrad True \
    --vsp_apply_to_projector_only True \
    --cka_loss_projector_weight 0.1 \
    --cka_loss_final_hidden_weight 0.1 \
    --cka_loss_subset_query_tokens "${CKA_LOSS_SUBSET_QUERY_TOKENS}" \
    --cka_loss_layers "-1" \
    # --cka_loss_subset_select_layer 9 \
    # --cka_loss_subset_min_ratio 0.01 \
    # --cka_loss_subset_max_ratio 0.90 \
    # --cka_loss_subset_fallback_mass 0.90 \
    # --cka_loss_subset_otsu_min_separability 0.30 \
    # --cka_loss_subset_ratio 0.75 \
    # --cka_loss_hidden_channel_drop_indices "0,7,31" \
    # --log_gradient_norms True \
    # --gradient_log_steps 10
