#!/bin/bash
set -euo pipefail

TRAIN_SCRIPT="scripts/v1_5/7b/finetune_qwen2_5.sh"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Please activate the training environment first, e.g. conda activate llava1" >&2
    exit 1
fi

: "${MODEL_NAME_OR_PATH:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${RUN_NAME:=qwen2.5-1.5b}"
: "${GPU_INCLUDE:=localhost:3}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=16}"
: "${GRADIENT_ACCUMULATION_STEPS:=8}"
: "${MM_PROJECTOR_TYPE:=mlp2x_gelu}"
: "${CKA_LAYERS:=3,final}"
: "${CKA_ANCHOR_LAYER:=2}"
: "${CKA_WEIGHTS:=0.01 0.02 0.05 0.1 0.2 0.5 1}"
: "${SWEEP_OUTPUT_ROOT:=./checkpoints/finetune-cka/${RUN_NAME}/late-cka-sweep}"
: "${WANDB_RUN_GROUP:=${RUN_NAME}-late-cka-sweep}"

BASE_RUN_NAME="${RUN_NAME}"
PRETRAIN_ADAPTER_PATH="${PRETRAIN_ADAPTER:-./checkpoints/pretrain-diag/${BASE_RUN_NAME}/llava-pretrain/mm_projector.bin}"
DEEPSPEED_CONFIG_PATH="${DEEPSPEED_CONFIG:-./scripts/zero2.json}"
COMMON_OUTPUT_DIR="${SWEEP_OUTPUT_ROOT}/shared-80/llava-finetune"
BRANCH_NAME_PREFIX="cka-final"
if [[ -n "${CKA_ANCHOR_LAYER}" ]]; then
    BRANCH_NAME_PREFIX="cka-anchor-layer-${CKA_ANCHOR_LAYER}-final"
fi
export WANDB_RUN_GROUP

read -r -a WEIGHTS <<< "${CKA_WEIGHTS}"

run_finetune() {
    env \
        "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}" \
        "RUN_NAME=${BASE_RUN_NAME}" \
        "GPU_INCLUDE=${GPU_INCLUDE}" \
        "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}" \
        "MM_PROJECTOR_TYPE=${MM_PROJECTOR_TYPE}" \
        "PRETRAIN_ADAPTER=${PRETRAIN_ADAPTER_PATH}" \
        "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG_PATH}" \
        "CKA_LAYERS=${CKA_LAYERS}" \
        "CKA_ANCHOR_LAYER=${CKA_ANCHOR_LAYER}" \
        "$@"
}

shopt -s nullglob
common_checkpoints=("${COMMON_OUTPUT_DIR}"/checkpoint-*)

if (( ${#common_checkpoints[@]} == 0 )); then
    echo "Phase 1: CKA off; train and stop at 80% of the full one-epoch schedule."
    run_finetune \
        "OUTPUT_DIR=${COMMON_OUTPUT_DIR}" \
        "WANDB_RUN_NAME=${BASE_RUN_NAME}-no-cka-first80" \
        "CKA_LOSS_ENABLED=False" \
        "CKA_PROJECTOR_WEIGHT=0.0" \
        "CKA_FINAL_HIDDEN_WEIGHT=0.0" \
        "CKA_LOSS_START_RATIO=0.0" \
        "STOP_AFTER_STEP_RATIO=0.8" \
        "RESUME_FROM_CHECKPOINT=" \
        "SAVE_STRATEGY=no" \
        bash "${TRAIN_SCRIPT}"

    common_checkpoints=("${COMMON_OUTPUT_DIR}"/checkpoint-*)
fi

if (( ${#common_checkpoints[@]} != 1 )); then
    echo "Expected exactly one shared checkpoint in ${COMMON_OUTPUT_DIR}, found ${#common_checkpoints[@]}." >&2
    echo "Use a fresh SWEEP_OUTPUT_ROOT or keep only the valid 80% checkpoint." >&2
    exit 1
fi

COMMON_CHECKPOINT="$(realpath -- "${common_checkpoints[0]}")"
COMMON_STATE="${COMMON_CHECKPOINT}/trainer_state.json"

"${CONDA_PREFIX}/bin/python" - "${COMMON_STATE}" <<'PY_VALIDATE'
import json
import math
import sys

state_path = sys.argv[1]
with open(state_path, encoding="utf-8") as handle:
    state = json.load(handle)

global_step = int(state["global_step"])
max_steps = int(state["max_steps"])
expected_step = math.ceil(0.8 * max_steps)
if global_step != expected_step:
    raise SystemExit(
        f"Invalid shared checkpoint: global_step={global_step}, "
        f"expected ceil(0.8 * {max_steps})={expected_step}"
    )
print(f"Shared checkpoint verified: step {global_step}/{max_steps}")
PY_VALIDATE

branch_is_complete() {
    local branch_state="$1"
    local branch_dir="${branch_state%/trainer_state.json}"
    local completion_marker="${branch_dir}/.sweep_complete"
    [[ -f "${branch_state}" && -f "${completion_marker}" ]] || return 1
    "${CONDA_PREFIX}/bin/python" - "${branch_state}" "${COMMON_STATE}" <<'PY_COMPLETE'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    branch = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    common = json.load(handle)

raise SystemExit(
    0 if int(branch["global_step"]) >= int(common["max_steps"]) else 1
)
PY_COMPLETE
}

for weight in "${WEIGHTS[@]}"; do
    branch_output="${SWEEP_OUTPUT_ROOT}/${BRANCH_NAME_PREFIX}-${weight}/llava-finetune"
    branch_state="${branch_output}/trainer_state.json"

    if branch_is_complete "${branch_state}"; then
        echo "Skip completed branch: cka_loss_final_hidden_weight=${weight}"
        continue
    fi

    echo "Phase 2: resume ${COMMON_CHECKPOINT}; cka_loss_final_hidden_weight=${weight}"
    run_finetune \
        "OUTPUT_DIR=${branch_output}" \
        "WANDB_RUN_NAME=${BASE_RUN_NAME}-${BRANCH_NAME_PREFIX}-${weight}" \
        "CKA_LOSS_ENABLED=True" \
        "CKA_PROJECTOR_WEIGHT=0.0" \
        "CKA_FINAL_HIDDEN_WEIGHT=${weight}" \
        "CKA_LOSS_START_RATIO=0.8" \
        "STOP_AFTER_STEP_RATIO=" \
        "RESUME_FROM_CHECKPOINT=${COMMON_CHECKPOINT}" \
        "SAVE_STRATEGY=no" \
        bash "${TRAIN_SCRIPT}"
    touch "${branch_output}/.sweep_complete"
done

echo "Completed CKA late-stage sweep: ${SWEEP_OUTPUT_ROOT}"
