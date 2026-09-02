#!/bin/bash
# lmsys/vicuna-7b-v1.5
# lmsys/vicuna-13b-v1.5
# openlm-research/open_llama_3b_v2
# mtgv/MobileLLaMA-1.4B-Chat
# TinyLlama/TinyLlama-1.1B-Chat-v1.0
# conv template:
#   openlm-research/open_llama_3b_v2       -> openllama_v1
#   mtgv/MobileLLaMA-1.4B-Chat            -> mobilellama_v1
#   TinyLlama/TinyLlama-1.1B-Chat-v1.0    -> tinyllama_chat

deepspeed --include localhost:0,1 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/full-ft/cka-proj-7b/llava-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/lora-ft/cka-proj-last-attn-otsu-7b/llava-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
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
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name 7b-ft-lora-cka-proj-last-attn-otsu \
    --cka_loss True \
    --use_pcgrad True \
    --cka_loss_weight 0.1 \
    --cka_loss_layers "final" \
    --cka_loss_hidden_channel_drop_indices "0,7,31" \
    --cka_loss_subset_select_layer 3 \
    --cka_loss_subset_min_ratio 0.01 \
    --cka_loss_subset_max_ratio 0.90 \
    --cka_loss_subset_fallback_mass 0.90 \
    --cka_loss_subset_otsu_min_separability 0.30 \
    # --cka_loss_subset_ratio 0.75 \
