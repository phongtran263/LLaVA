#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVE_ROOT="$(cd "${SNAPSHOT_ROOT}/../.." && pwd)"
RECOVERY_OUTPUT_DIR="${RECOVERY_OUTPUT_DIR:-${ACTIVE_ROOT}/checkpoints/qwen-7b-cka-grad-f6-rerun/llava-pretrain}"

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/deepspeed" ]; then
    echo "Activate the historical llava-qwen environment before running." >&2
    exit 1
fi

if [ -e "${RECOVERY_OUTPUT_DIR}" ] && [ -n "$(find "${RECOVERY_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    echo "Refusing to reuse non-empty output directory: ${RECOVERY_OUTPUT_DIR}" >&2
    exit 1
fi

cd "${ACTIVE_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${CONDA_PREFIX}/bin/deepspeed" --include localhost:0,1 \
    "${SNAPSHOT_ROOT}/llava/train/train_mem.py" \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
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
    --output_dir "${RECOVERY_OUTPUT_DIR}" \
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
    --run_name qwen-7b-cka-grad-f6-rerun \
    --cka_loss True \
    --cka_loss_tau 0.0 \
    --cka_loss_weight 1.0 \
    --vsp_gradient_diagnostics True \
    --vsp_asymmetric_pcgrad True \
    --vsp_norm_cap True \
    --vsp_pcgrad_threshold 0.05 \
    --vsp_proj_max_grad_ratio 0.5 \
    --vsp_llm_max_grad_ratio 0.5 \
    --vsp_grad_log_interval 10 \
    --cka_loss_layers "-1"
