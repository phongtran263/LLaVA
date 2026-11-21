#!/bin/bash
set -euo pipefail

# Stage 2 only: compare one intermediate CKA target, with/without projector CKA.
# The selected block is 1-based. Block 16 is the midpoint of Vicuna's 32 blocks.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
REPO_ROOT="${PWD}"
fail() { echo "$*" >&2; exit 1; }
(( $# == 0 )) || fail "Configure this launcher using environment variables."

DRY_RUN="${DRY_RUN:-0}"
CKA_LAYER="${CKA_LAYER:-16}"
CKA_WEIGHT="${CKA_WEIGHT:-0.1}"
SETTINGS="${SETTINGS-intermediate_only projector_intermediate}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/full-ft/vicuna-7b-intermediate-cka}"
BASELINE_PRETRAIN_DIR="${BASELINE_PRETRAIN_DIR:-${REPO_ROOT}/checkpoints/full-ft/baseline-7b/llava-pretrain}"
CKA_PRETRAIN_DIR="${CKA_PRETRAIN_DIR:-${REPO_ROOT}/checkpoints/full-ft/cka-proj-7b/llava-pretrain}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail "DRY_RUN must be 0 or 1."
[[ "$CKA_LAYER" =~ ^[1-9][0-9]?$ ]] && (( CKA_LAYER < 32 )) || fail "CKA_LAYER must be an intermediate block from 1 to 31 (1-based)."
read -r -a settings <<< "$SETTINGS"
(( ${#settings[@]} )) || fail "SETTINGS must not be empty."
declare -A seen=()
parents=()
outputs=()
for setting in "${settings[@]}"; do
    [[ -z "${seen[$setting]:-}" ]] || fail "Duplicate setting: $setting"
    seen[$setting]=1
    case "$setting" in
        intermediate_only) parents+=("$BASELINE_PRETRAIN_DIR") ;;
        projector_intermediate) parents+=("$CKA_PRETRAIN_DIR") ;;
        *) fail "Unknown setting: $setting; use intermediate_only or projector_intermediate." ;;
    esac
    outputs+=("${OUTPUT_ROOT}/layer-${CKA_LAYER}/${setting}/llava-finetune")
done
# Check every destination before starting the first potentially long run.
for output in "${outputs[@]}"; do
    [[ ! -e "$output" && ! -L "$output" ]] || fail "Output exists: $output. Choose a fresh OUTPUT_ROOT; automatic resume is disabled."
done

PYTHON_CHECK="${PYTHON_CHECK:-python3}"
"$PYTHON_CHECK" - "$CKA_WEIGHT" "${parents[@]}" <<'PY_CHECK'
import json
import math
from pathlib import Path
import sys

try:
    weight = float(sys.argv[1])
except ValueError:
    raise SystemExit('CKA_WEIGHT must be finite and positive.')
if not math.isfinite(weight) or weight <= 0:
    raise SystemExit('CKA_WEIGHT must be finite and positive.')
expected = {
    '_name_or_path': 'lmsys/vicuna-7b-v1.5',
    'num_hidden_layers': 32, 'hidden_size': 4096, 'mm_hidden_size': 1024,
    'mm_projector_type': 'mlp2x_gelu', 'mm_vision_select_layer': -2,
    'mm_vision_select_feature': 'patch',
    'mm_vision_tower': 'openai/clip-vit-large-patch14-336',
}
for directory in sys.argv[2:]:
    parent = Path(directory)
    try:
        config = json.loads((parent / 'config.json').read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f'Cannot read stage-1 config at {parent}: {exc}')
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(f'Incompatible parent {parent}: {key}={config.get(key)!r}, expected {value!r}.')
    adapter = parent / 'mm_projector.bin'
    if not adapter.is_file() or adapter.stat().st_size == 0:
        raise SystemExit(f'Missing or empty stage-1 adapter: {adapter}')
print(f'Preflight passed: Vicuna-7B, {len(sys.argv) - 2} stage-1 adapter(s), CKA weight={weight}.')
PY_CHECK

if [[ "$DRY_RUN" == 0 ]]; then
    TRAIN_ENV="${TRAIN_ENV:-${REPO_ROOT}/../conda/envs/llava1}"
    [[ -x "${TRAIN_ENV}/bin/python" && -x "${TRAIN_ENV}/bin/deepspeed" ]] || fail "Set TRAIN_ENV to the llava1 training environment."
    export CONDA_PREFIX="$TRAIN_ENV"
    export PATH="${TRAIN_ENV}/bin:${PATH}"
fi
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CHILD_SCRIPT="${SCRIPT_DIR}/finetune_qwen2_5.sh"
bash -n "$CHILD_SCRIPT" || fail "Invalid Bash syntax: $CHILD_SCRIPT"

for index in "${!settings[@]}"; do
    setting="${settings[$index]}"
    output="${outputs[$index]}"
    [[ ! -e "$output" && ! -L "$output" ]] || fail "Output appeared before launch: $output"
    projector_weight=0.0
    [[ "$setting" != projector_intermediate ]] || projector_weight="$CKA_WEIGHT"
    echo "Starting ${setting}: block=${CKA_LAYER}, projector weight=${projector_weight}, hidden weight=${CKA_WEIGHT}"
    echo "Stage-1 adapter: ${parents[$index]}/mm_projector.bin"
    echo "Output: ${output}"
    bash -n "$CHILD_SCRIPT" || fail "Invalid Bash syntax before ${setting}: $CHILD_SCRIPT"
    env -u WANDB_RUN_ID -u WANDB_RESUME \
        DRY_RUN="$DRY_RUN" \
        MODEL_NAME_OR_PATH=lmsys/vicuna-7b-v1.5 CONVERSATION_VERSION=v1 \
        VISION_TOWER=openai/clip-vit-large-patch14-336 MM_VISION_SELECT_LAYER=-2 \
        MM_PROJECTOR_TYPE=mlp2x_gelu MM_PROJECTOR_INIT_SEED= \
        GPU_INCLUDE="${GPU_INCLUDE:-localhost:3}" MASTER_PORT="${MASTER_PORT:-29576}" \
        DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero2.json}" \
        PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
        GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}" \
        FINETUNE_GRADIENT_CHECKPOINTING="${FINETUNE_GRADIENT_CHECKPOINTING:-True}" \
        DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}" LOGGING_STEPS="${LOGGING_STEPS:-1}" \
        PRETRAIN_ADAPTER="${parents[$index]}/mm_projector.bin" OUTPUT_DIR="$output" \
        RUN_NAME="vicuna-7b-${setting}-layer${CKA_LAYER}-w${CKA_WEIGHT}" \
        WANDB_RUN_NAME="vicuna-7b-${setting}-layer${CKA_LAYER}-w${CKA_WEIGHT}" \
        WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-vicuna-7b-intermediate-cka}" \
        CKA_LOSS_ENABLED=True CKA_LAYERS="$CKA_LAYER" \
        CKA_PROJECTOR_WEIGHT="$projector_weight" CKA_FINAL_HIDDEN_WEIGHT="$CKA_WEIGHT" \
        CKA_FINAL_MLP_HEAD=False CKA_LOSS_TAU=0.0 CKA_LOSS_START_RATIO=0.0 \
        CKA_ANCHOR_LAYER= CKA_VISION_ANCHOR_LAYER= \
        CKA_PROJECTOR_VISION_ANCHOR_LAYER= CKA_FINAL_VISION_ANCHOR_LAYER= \
        VSP_DIAGNOSTICS=False VSP_PCGRAD=False VSP_APPLY_TO_PROJECTOR_ONLY=False VSP_NORM_CAP=False \
        RESUME_FROM_CHECKPOINT= STOP_AFTER_STEP_RATIO= SAVE_STRATEGY=no \
        bash -s -- < "$CHILD_SCRIPT"
    echo "Completed ${setting}."
done
