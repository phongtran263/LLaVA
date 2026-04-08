#!/bin/bash
# CLIP: openai/clip-vit-large-patch14-336 * 300M
# Pix2Struct: google/pix2struct-base * 282M
# DINOv2: facebook/dinov2-large * 
# SigLIP: google/siglip2-so400m-patch16-naflex 
# CO-DETR: zongzhuofan/co-detr-vit-large-lvis-instance
# Owl-ViT: google/owlv2-large-patch14-ensemble *
# Med: google/medsiglip-448

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path mtgv/MobileLLaMA-1.4B-Chat \
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
    --output_dir ./checkpoints/llava-v1.5-mobile-pretrain-cka \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
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
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --cka_loss True \
    --cka_loss_weight 0.1
