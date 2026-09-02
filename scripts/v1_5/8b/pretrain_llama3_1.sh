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

"${CONDA_PREFIX}/bin/deepspeed" --include localhost:0,1 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
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
    --output_dir ./checkpoints/llama-8b-cka/llava-pretrain \
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
    --run_name llama-8b-cka-pretrain \
    --cka_loss True \
    --use_pcgrad True \
    --cka_loss_weight 0.1 \
    --cka_loss_layers "-1" \
    --train_data_fraction ${TRAIN_DATA_FRACTION:-1.0} \
    --train_data_seed ${TRAIN_DATA_SEED:-42}
