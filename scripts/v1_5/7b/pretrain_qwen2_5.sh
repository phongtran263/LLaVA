#!/bin/bash
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Please activate the Qwen training conda env first, e.g. conda activate llava-qwen" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/qwen-3b-cka-grad/llava-pretrain}"

python - <<'PY_CHECK'
from packaging import version
import accelerate
import transformers

if version.parse(transformers.__version__) != version.parse("4.43.1"):
    raise SystemExit(
        f"Qwen2.5 training expects transformers==4.43.1, got {transformers.__version__}. "
        "Activate llava-qwen or downgrade this env."
    )
if version.parse(accelerate.__version__) < version.parse("0.33.0"):
    raise SystemExit(
        f"Qwen2.5 training expects accelerate>=0.33.0 with transformers 4.43.x, got {accelerate.__version__}. "
        "Run: conda run -n llava-qwen python -m pip install accelerate==0.33.0"
    )
PY_CHECK

"${CONDA_PREFIX}/bin/deepspeed" --include localhost:0 llava/train/train_mem.py \
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
    --run_name qwen-3b-cka-grad-pretrain \
    --cka_loss True \
    --cka_loss_tau 0.0 \
    --cka_loss_weight 1.0 \
    --vsp_gradient_diagnostics True \
    --vsp_asymmetric_pcgrad True \
    --vsp_apply_to_projector_only True \
    --vsp_norm_cap True \
    --vsp_pcgrad_threshold 0.05 \
    --vsp_proj_max_grad_ratio 0.5 \
    --vsp_llm_max_grad_ratio 0.5 \
    --vsp_grad_log_interval 10 \
    --cka_loss_layers "-1"
