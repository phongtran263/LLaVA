#!/usr/bin/env bash
set -euo pipefail

# Stage 1: frozen vision tower + frozen decoder, train only the projector with
# CE-priority adaptive projector PCGrad.  The default run is intentionally a
# two-step smoke test; set FULL_TRAIN=True explicitly for the full epoch.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/deepspeed" ]]; then
    echo "Activate the repository training environment before running this launcher." >&2
    exit 1
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

: "${MODEL_NAME_OR_PATH:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${RUN_NAME:=qwen2.5-0.5b-adaptive-projector-pcgrad}"
: "${GPU_INCLUDE:=localhost:0}"
: "${FULL_TRAIN:=False}"
: "${SMOKE_MAX_STEPS:=2}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=32}"
: "${GRADIENT_ACCUMULATION_STEPS:=8}"
: "${MM_PROJECTOR_TYPE:=mlp2x_gelu}"
: "${MODEL_MAX_LENGTH:=2048}"
: "${DATALOADER_NUM_WORKERS:=16}"
: "${GRADIENT_CHECKPOINTING:=False}"
: "${LOGGING_STEPS:=1}"
: "${REPORT_TO:=wandb}"
: "${DEEPSPEED_CONFIG:=./scripts/zero2.json}"
: "${ADAPTIVE_PCGRAD_CONFIG:=./scripts/adaptive_projector_pcgrad/stage1_defaults.json}"
: "${ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE:=4}"
: "${ADAPTIVE_PCGRAD_PROFILE:=False}"
: "${SAVE_STRATEGY:=no}"
: "${RESUME_FROM_CHECKPOINT:=}"

if is_true "${FULL_TRAIN}"; then
    RUN_MODE=full
    LIMIT_ARGS=()
else
    RUN_MODE=smoke
    if ! [[ "${SMOKE_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "SMOKE_MAX_STEPS must be a positive integer, got: ${SMOKE_MAX_STEPS}" >&2
        exit 1
    fi
    LIMIT_ARGS=(--max_steps "${SMOKE_MAX_STEPS}")
    SAVE_STRATEGY=steps
    SAVE_STEPS=1
fi

: "${PIPELINE_ROOT:=./checkpoints/adaptive-projector-pcgrad/${RUN_NAME}/${RUN_MODE}}"
: "${OUTPUT_DIR:=${PIPELINE_ROOT}/stage1}"
: "${WANDB_RUN_NAME:=${RUN_NAME}-s1-${RUN_MODE}}"

OPTIONAL_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    OPTIONAL_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

for required in "${DEEPSPEED_CONFIG}" "${ADAPTIVE_PCGRAD_CONFIG}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required configuration not found: ${required}" >&2
        exit 1
    fi
done

"${CONDA_PREFIX}/bin/python" - <<'PY_CHECK'
from packaging import version
import accelerate
import deepspeed
import tokenizers
import torch
import transformers

required = {
    "torch": (torch.__version__, "2.7.1"),
    "transformers": (transformers.__version__, "4.51.3"),
    "tokenizers": (tokenizers.__version__, "0.21.2"),
    "accelerate": (accelerate.__version__, "1.6.0"),
    "deepspeed": (deepspeed.__version__, "0.18.9"),
}
for package, (actual, expected) in required.items():
    if version.parse(actual).base_version != version.parse(expected).base_version:
        raise SystemExit(
            f"Adaptive PCGrad requires {package}=={expected} (local build suffix allowed), "
            f"got {actual}. Activate the environment installed from pyproject.toml."
        )

print(
    "adaptive PCGrad environment:",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"tokenizers={tokenizers.__version__}",
    f"accelerate={accelerate.__version__}",
    f"deepspeed={deepspeed.__version__}",
)
PY_CHECK

"${CONDA_PREFIX}/bin/deepspeed" --include "${GPU_INCLUDE}" llava/train/train_mem.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --force_download False \
    --version plain \
    --data_path ./playground/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type "${MM_PROJECTOR_TYPE}" \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_strategy no \
    --save_strategy "${SAVE_STRATEGY}" \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps "${LOGGING_STEPS}" \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}" \
    --run_name "${WANDB_RUN_NAME}" \
    --adaptive_projector_pcgrad True \
    --adaptive_pcgrad_stage 1 \
    --adaptive_pcgrad_config "${ADAPTIVE_PCGRAD_CONFIG}" \
    --adaptive_pcgrad_cka_chunk_size "${ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE}" \
    --adaptive_pcgrad_profile "${ADAPTIVE_PCGRAD_PROFILE}" \
    --cka_loss False \
    --use_pcgrad False \
    --vsp_gradient_diagnostics False \
    --vsp_asymmetric_pcgrad False \
    --vsp_apply_to_projector_only False \
    --vsp_norm_cap False \
    "${LIMIT_ARGS[@]}" \
    "${OPTIONAL_ARGS[@]}"
