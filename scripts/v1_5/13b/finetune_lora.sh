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
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --version tinyllama_chat \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower facebook/dinov2-large \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-1b-pretrain-dinov2-cka-proj/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-1b-finetune-lora-dinov2-cka-proj \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
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
    --run_name llava-1b-finetune-lora-dinov2-cka-proj \
    --cka_loss True \
    --use_pcgrad True \
    --cka_loss_weight 1.0 \
    --cka_loss_layers "-1" \
    --cka_loss_hidden_channel_drop_indices "0,7,31" \
    --cka_loss_layer_decay 1 \
    # Enable --cka_loss_layers "final" before using attention-selected final CKA.
    # --cka_loss_subset_select_layer 8 \
    # --cka_loss_subset_min_ratio 0.05 \
    # --cka_loss_subset_max_ratio 0.75 \
    # --cka_loss_subset_fallback_mass 0.8 \
    # --cka_loss_subset_otsu_min_separability 0.05
