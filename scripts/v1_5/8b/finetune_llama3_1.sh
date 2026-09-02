#!/bin/bash
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Please activate the Llama 3.1 training conda env first, e.g. conda activate llava-llama31" >&2
    exit 1
fi

"${CONDA_PREFIX}/bin/python" - <<'PY_CHECK'
from packaging import version
import accelerate
import transformers

if version.parse(transformers.__version__) != version.parse("4.43.1"):
    raise SystemExit(
        f"Llama 3.1 training expects transformers==4.43.1, got {transformers.__version__}. "
        "Activate llava-llama31 or install transformers==4.43.1."
    )
if version.parse(accelerate.__version__) < version.parse("0.33.0"):
    raise SystemExit(
        f"Llama 3.1 training expects accelerate>=0.33.0, got {accelerate.__version__}."
    )
PY_CHECK

CKA_LOSS_SUBSET_QUERY_TOKENS="${CKA_LOSS_SUBSET_QUERY_TOKENS:-text}"

"${CONDA_PREFIX}/bin/deepspeed" --include localhost:0,1 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --version llama_3_1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llama-8b-cka/llava-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llama-8b-cka/llava-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
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
    --run_name llama-8b-cka-finetune \
    --cka_loss True \
    --use_pcgrad True \
    --cka_loss_projector_weight 0.1 \
    --cka_loss_final_hidden_weight 0.1 \
    --cka_loss_subset_query_tokens "${CKA_LOSS_SUBSET_QUERY_TOKENS}" \
    --cka_loss_hidden_channel_drop_indices "0,7,31" \
    --cka_loss_layers "-1"
    # --cka_loss_subset_select_layer 23 \
    # --cka_loss_subset_min_ratio 0.01 \
    # --cka_loss_subset_max_ratio 0.90 \
    # --cka_loss_subset_fallback_mass 0.90 \
    # --cka_loss_subset_otsu_min_separability 0.30
