#!/bin/bash
# CLIP: openai/clip-vit-large-patch14-336 *
# Pix2Struct: google/pix2struct-large *
# DINOv2: facebook/dinov2-with-registers-large *
# SigLIP: google/siglip2-so400m-patch16-naflex 
# CO-DETR: zongzhuofan/co-detr-vit-large-lvis-instance
# Owl-ViT: google/owlv2-large-patch14-ensemble *
# Med: google/medsiglip-448

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path openlm-research/open_llama_3b_v2 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336,google/pix2struct-large,facebook/dinov2-with-registers-large,google/owlv2-large-patch14-ensemble \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-3b-pretrain-openllama-clip/mm_projector.bin,./checkpoints/llava-v1.5-3b-pretrain-openllama-pix/mm_projector.bin,./checkpoints/llava-v1.5-3b-pretrain-openllama-dino/mm_projector.bin,./checkpoints/llava-v1.5-3b-pretrain-openllama-owl/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-3b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
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
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
