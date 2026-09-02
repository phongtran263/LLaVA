#!/bin/bash
# lmsys/vicuna-7b-v1.5
# lmsys/vicuna-13b-v1.5
# openlm-research/open_llama_3b_v2
# mtgv/MobileLLaMA-1.4B-Chat
# TinyLlama/TinyLlama-1.1B-Chat-v1.0
# Qwen/Qwen3-4B-Instruct-2507

deepspeed --include localhost:0,1 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
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
    --output_dir ./checkpoints/7b-cka-grad/llava-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
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
    --run_name 7b-cka-grad-pretrain \
    --cka_loss True \
    --use_pcgrad False \
    --pretrain_projector_pcgrad True \
    --cka_loss_weight 0.1 \
    --cka_loss_layers "-1" \
    --log_gradient_norms False \
    --train_data_fraction ${TRAIN_DATA_FRACTION:-1.0} \
    --train_data_seed ${TRAIN_DATA_SEED:-42} \
    # --gradient_log_steps 50 \
    # --cka_loss_subset_select_layer 8 \
    # --cka_loss_subset_min_ratio 0.05 \
    # --cka_loss_subset_max_ratio 0.75 \
    # --cka_loss_subset_fallback_mass 0.8 \
    # --cka_loss_subset_otsu_min_separability 0.05
