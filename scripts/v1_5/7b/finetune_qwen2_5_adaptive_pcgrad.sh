#!/usr/bin/env bash
set -euo pipefail

# Stage 2: load the matching adaptive-PCGrad stage-1 projector and keep the
# repository's ordinary decoder fine-tuning recipe.  Defaults to a two-step
# smoke test; set FULL_TRAIN=True explicitly for the full epoch.

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
: "${GPU_INCLUDE:=localhost:3}"
: "${FULL_TRAIN:=False}"
: "${SMOKE_MAX_STEPS:=2}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=32}"
: "${GRADIENT_ACCUMULATION_STEPS:=4}"
: "${MM_PROJECTOR_TYPE:=mlp2x_gelu}"
: "${MODEL_MAX_LENGTH:=2048}"
: "${DATALOADER_NUM_WORKERS:=16}"
: "${GRADIENT_CHECKPOINTING:=True}"
: "${LOGGING_STEPS:=1}"
: "${REPORT_TO:=wandb}"
: "${DEEPSPEED_CONFIG:=./scripts/zero2.json}"
: "${ADAPTIVE_PCGRAD_CONFIG:=./scripts/adaptive_projector_pcgrad/stage2_defaults.json}"
: "${ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE:=4}"
: "${ADAPTIVE_PCGRAD_PROFILE:=False}"
: "${SAVE_STRATEGY:=no}"
: "${SAVE_STEPS:=}"
: "${SAVE_TOTAL_LIMIT:=2}"
: "${SAVE_AT_STEP_RATIO:=}"
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
: "${STAGE1_OUTPUT_DIR:=${PIPELINE_ROOT}/stage1}"
: "${PRETRAIN_ADAPTER:=${STAGE1_OUTPUT_DIR}/mm_projector.bin}"
STAGE1_METADATA="$(dirname -- "${PRETRAIN_ADAPTER}")/adaptive_projector_pcgrad_metadata.json"
: "${OUTPUT_DIR:=${PIPELINE_ROOT}/stage2}"
: "${WANDB_RUN_NAME:=${RUN_NAME}-s2-${RUN_MODE}}"

OPTIONAL_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    OPTIONAL_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${SAVE_AT_STEP_RATIO}" ]]; then
    OPTIONAL_ARGS+=(--save_at_step_ratio "${SAVE_AT_STEP_RATIO}")
fi

SAVE_ARGS=(
    --save_strategy "${SAVE_STRATEGY}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
)
if [[ -n "${SAVE_STEPS}" ]]; then
    SAVE_ARGS+=(--save_steps "${SAVE_STEPS}")
fi

for required in "${DEEPSPEED_CONFIG}" "${ADAPTIVE_PCGRAD_CONFIG}" "${PRETRAIN_ADAPTER}" "${STAGE1_METADATA}"; do
    if [[ ! -s "${required}" ]]; then
        echo "Required non-empty file not found: ${required}" >&2
        exit 1
    fi
done

"${CONDA_PREFIX}/bin/python" - "${STAGE1_METADATA}" "${MODEL_NAME_OR_PATH}" <<'PY_VALIDATE'
import json
import sys

metadata_path, expected_base = sys.argv[1:]
with open(metadata_path, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)
if metadata.get("stage") != 1:
    raise SystemExit(f"{metadata_path}: expected adaptive PCGrad stage=1 metadata")
if metadata.get("base_model_identifier") != expected_base:
    raise SystemExit(
        f"{metadata_path}: base model mismatch: "
        f"{metadata.get('base_model_identifier')!r} != {expected_base!r}"
    )
if not metadata.get("projector_sha256"):
    raise SystemExit(f"{metadata_path}: missing projector_sha256")
if not metadata.get("projector_signature"):
    raise SystemExit(f"{metadata_path}: missing projector_signature")
resolved = metadata.get("resolved_config")
if not isinstance(resolved, dict) or resolved.get("stage") != 1:
    raise SystemExit(f"{metadata_path}: missing resolved stage-1 controller config")
print(f"Validated adaptive-PCGrad stage-1 handoff: {metadata_path}")
PY_VALIDATE

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
    --version auto \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter "${PRETRAIN_ADAPTER}" \
    --mm_projector_type "${MM_PROJECTOR_TYPE}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_strategy no \
    "${SAVE_ARGS[@]}" \
    --learning_rate 2e-5 \
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
    --adaptive_pcgrad_stage 2 \
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
