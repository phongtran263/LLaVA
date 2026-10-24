#!/bin/bash
set -euo pipefail

# Finetune one saved Vicuna projector with fixed projector CKA and varying
# final-state CKA. No pretraining or projector reinitialization sweep.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
FINETUNE_SCRIPT="${SCRIPT_DIR}/finetune_qwen2_5.sh"

fail() { echo "$*" >&2; exit 1; }
fresh_output() {
    if [[ -e "$1" || -L "$1" ]]; then
        fail "Output already exists: $1. Choose a new EXPERIMENT_ROOT or select unfinished FINAL_CKA_WEIGHTS."
    fi
}

(( $# == 0 )) || fail "Use environment variables; CLI overrides are not accepted."
DRY_RUN="${DRY_RUN:-0}"
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || fail "DRY_RUN must be 0 or 1."
[[ "${SEED-42}" == 42 && "${SEEDS-42}" == 42 ]] || fail "This sweep uses training seed 42 for every final weight."
PRETRAIN_CHECKPOINT_DIR="${PRETRAIN_CHECKPOINT_DIR:-./checkpoints/cka-proj-7b-0.1/llava-pretrain}"
PRETRAIN_ADAPTER="${PRETRAIN_CHECKPOINT_DIR}/mm_projector.bin"
[[ -f "${PRETRAIN_ADAPTER}" && -s "${PRETRAIN_ADAPTER}" ]] || \
    fail "Missing, empty, or non-file pretrain adapter: ${PRETRAIN_ADAPTER}."
FINAL_CKA_WEIGHTS="${FINAL_CKA_WEIGHTS-0.05 0.25 0.5}"
read -r -a requested_weights <<< "${FINAL_CKA_WEIGHTS}"

# The supplied checkpoint is Vicuna-7B, not the Qwen default in the child.
# Pin the backbone/projector recipe and validate the checkpoint before any job.
weight_values="$(python3 - "${PRETRAIN_CHECKPOINT_DIR}/config.json" "${requested_weights[@]}" <<'PY_PREFLIGHT'
import json
import math
from pathlib import Path
import sys

path = Path(sys.argv[1])
try:
    config = json.loads(path.read_text())
except (OSError, ValueError) as exc:
    raise SystemExit(f"Cannot read pretrain config {path}: {exc}")
if not isinstance(config, dict) or config.get("model_type") not in ("llama", "llava_llama"):
    raise SystemExit("Pretrain config must describe a Vicuna/LLaMA checkpoint.")
expected = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "mm_projector_type": "mlp2x_gelu",
    "mm_hidden_size": 1024,
    "mm_vision_tower": "openai/clip-vit-large-patch14-336",
    "mm_vision_select_layer": -2,
    "mm_patch_merge_type": "flat",
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"Incompatible pretrain config {key}: expected {value!r}, got {config.get(key)!r}.")
if not sys.argv[2:]:
    raise SystemExit("FINAL_CKA_WEIGHTS must not be empty.")
weights, seen = [], set()
for raw in sys.argv[2:]:
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"FINAL_CKA_WEIGHTS requires finite positive numbers, got {raw!r}.")
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"FINAL_CKA_WEIGHTS requires finite positive numbers, got {raw!r}.")
    if value in seen:
        raise SystemExit(f"Duplicate final CKA weight: {raw!r}.")
    seen.add(value)
    weights.append(str(value))
print(" ".join(weights))
PY_PREFLIGHT
)"
read -r -a final_weights <<< "${weight_values}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-./checkpoints/cka-final-weight-sweep/vicuna-7b-proj0.1-seed42}"
GPU_INCLUDE="${FINETUNE_GPU_INCLUDE:-${GPU_INCLUDE:-localhost:2,3}}"
[[ -f "${FINETUNE_SCRIPT}" ]] || fail "Missing finetune launcher: ${FINETUNE_SCRIPT}."
for weight in "${final_weights[@]}"; do
    fresh_output "${EXPERIMENT_ROOT}/final-${weight}/llava-finetune"
done
if [[ "${DRY_RUN}" == 0 ]]; then
    [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/deepspeed" ]] || \
        fail "Activate llava1 (or use conda run -n llava1) with DeepSpeed installed."
fi

COMMON_ENV=(
    "MODEL_NAME_OR_PATH=lmsys/vicuna-7b-v1.5"
    "MM_PROJECTOR_TYPE=mlp2x_gelu"
    "VISION_TOWER=openai/clip-vit-large-patch14-336"
    "MM_VISION_SELECT_LAYER=-2"
    "CONVERSATION_VERSION=v1"
    "PRETRAIN_ADAPTER=${PRETRAIN_ADAPTER}"
    "MM_PROJECTOR_INIT_SEED="
    "SEED=42"
    "GPU_INCLUDE=${GPU_INCLUDE}"
    "DRY_RUN=${DRY_RUN}"
    "DATA_PATH=${FINETUNE_DATA_PATH:-./playground/data/llava_v1_5_mix665k.json}"
    "IMAGE_FOLDER=${FINETUNE_IMAGE_FOLDER:-./playground/data}"
    "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
    "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}"
    "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-16}"
    "REPORT_TO=${REPORT_TO:-wandb}"
    "DEEPSPEED_CONFIG=./scripts/zero2.json"
    "SAVE_STRATEGY=no"
    "LOGGING_STEPS=10"
    "CKA_LOSS_ENABLED=True"
    "CKA_PROJECTOR_WEIGHT=0.1"
    "CKA_LAYERS=final"
    "CKA_LOSS_TAU=0.0"
    "CKA_LOSS_START_RATIO=0.0"
    "CKA_FINAL_MLP_HEAD=False"
    "CKA_ANCHOR_LAYER="
    "CKA_VISION_ANCHOR_LAYER="
    "CKA_PROJECTOR_VISION_ANCHOR_LAYER="
    "CKA_FINAL_VISION_ANCHOR_LAYER="
    "VSP_DIAGNOSTICS=False"
    "VSP_PCGRAD=False"
    "VSP_APPLY_TO_PROJECTOR_ONLY=False"
    "VSP_NORM_CAP=False"
    "STOP_AFTER_STEP_RATIO="
    "RESUME_FROM_CHECKPOINT="
)

echo "Plan: ${#final_weights[@]} Vicuna finetunes; projector lambda=0.1; final lambdas=${weight_values}; seed=42"
echo "Shared pretrain adapter: ${PRETRAIN_ADAPTER}"
for weight in "${final_weights[@]}"; do
    output="${EXPERIMENT_ROOT}/final-${weight}/llava-finetune"
    label="vicuna-7b-cka-proj0.1-final${weight}-seed42"
    echo "Run: ${label}; output=${output}"
    fresh_output "${output}"
    env -u WANDB_RUN_ID -u WANDB_RESUME \
        PYTHONNOUSERSITE=1 \
        "${COMMON_ENV[@]}" \
        "OUTPUT_DIR=${output}" \
        "RUN_NAME=${label}" \
        "WANDB_RUN_NAME=${label}" \
        "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-vicuna-7b-cka-final-weight-sweep}" \
        "CKA_FINAL_HIDDEN_WEIGHT=${weight}" \
        bash "${FINETUNE_SCRIPT}"
done
