#!/bin/bash
set -euo pipefail

# Independent end-to-end ablations: each repeat/setting trains its own adapter
# and then finetunes from that adapter. The child launchers retain the recipe.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
PRETRAIN_SCRIPT="${SCRIPT_DIR}/pretrain_qwen2_5.sh"
FINETUNE_SCRIPT="${SCRIPT_DIR}/finetune_qwen2_5.sh"

fail() { echo "$*" >&2; exit 1; }
fresh_output() {
    if [[ -e "$1" || -L "$1" ]]; then
        fail "Output already exists: $1. Choose a new EXPERIMENT_ROOT or select unfinished settings."
    fi
}
require_adapter() {
    [[ -f "$1" && -s "$1" ]] || fail "Missing, empty, or non-file Stage-1 adapter: $1. Run STAGE=pretrain first."
}

(( $# == 0 )) || fail "Use environment variables; CLI overrides are not accepted."
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
REPEATS="${REPEATS:-2}"
SETTINGS="${SETTINGS-baseline cka_proj cka_proj_final}"
CKA_WEIGHT="${CKA_WEIGHT-0.1}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"
RUN_NAME="${RUN_NAME:-qwen2.5-7b}"
MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
PRETRAIN_GPU_INCLUDE="${PRETRAIN_GPU_INCLUDE:-${GPU_INCLUDE:-localhost:2,3}}"
FINETUNE_GPU_INCLUDE="${FINETUNE_GPU_INCLUDE:-${GPU_INCLUDE:-localhost:2,3}}"

case "${STAGE}" in all|pretrain|finetune) ;; *) fail "STAGE must be all, pretrain, or finetune." ;; esac
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || fail "DRY_RUN must be 0 or 1."
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || fail "REPEATS must be a positive integer without leading zeroes."
[[ "${SEED-42}" == 42 && "${SEEDS-42}" == 42 && "${STAGE1_SEED-42}" == 42 ]] || \
    fail "This suite uses seed 42 in both stages and every repeat; unset other seed overrides."
# Validate without importing the training stack; one coefficient controls every
# active CKA term in both stages. Disabled terms remain exactly zero.
CKA_WEIGHT="$(python3 - "${CKA_WEIGHT}" <<'PY_WEIGHT'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("CKA_WEIGHT must be a finite positive number.")
if not math.isfinite(value) or value <= 0:
    raise SystemExit("CKA_WEIGHT must be a finite positive number.")
print(str(value))
PY_WEIGHT
)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-./checkpoints/cka-two-stage/${RUN_NAME}-${MM_PROJECTOR_TYPE}-lambda${CKA_WEIGHT}-seed42}"
read -r -a settings <<< "${SETTINGS}"
(( ${#settings[@]} )) || fail "SETTINGS must not be empty."
declare -A seen_settings=()
for setting in "${settings[@]}"; do
    case "${setting}" in baseline|cka_proj|cka_proj_final) ;; *) fail "Unknown setting: ${setting}. Use baseline, cka_proj, cka_proj_final." ;; esac
    [[ -z "${seen_settings[${setting}]:-}" ]] || fail "Duplicate setting: ${setting}."
    seen_settings["${setting}"]=1
done
[[ -f "${PRETRAIN_SCRIPT}" && -f "${FINETUNE_SCRIPT}" ]] || fail "Both training scripts must exist beside this launcher."

# Check the complete plan before starting any long-running job. Existing
# outputs are never silently reused or resumed; each arm owns its checkpoint.
for (( repeat=1; repeat<=REPEATS; repeat++ )); do
    for setting in "${settings[@]}"; do
        run_dir="${EXPERIMENT_ROOT}/repeat-${repeat}/${setting}"
        if [[ "${STAGE}" != finetune ]]; then
            fresh_output "${run_dir}/llava-pretrain"
        fi
        if [[ "${STAGE}" != pretrain ]]; then
            fresh_output "${run_dir}/llava-finetune"
        fi
        if [[ "${STAGE}" == finetune && "${DRY_RUN}" == 0 ]]; then
            require_adapter "${run_dir}/llava-pretrain/mm_projector.bin"
        fi
    done
done
if [[ "${DRY_RUN}" == 0 ]]; then
    [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/deepspeed" ]] || \
        fail "Activate llava1 (or use conda run -n llava1) with DeepSpeed installed."
fi

COMMON_ENV=(
    "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
    "MM_PROJECTOR_TYPE=${MM_PROJECTOR_TYPE}"
    "VISION_TOWER=openai/clip-vit-large-patch14-336"
    "MM_VISION_SELECT_LAYER=-2"
    "CONVERSATION_VERSION=${CONVERSATION_VERSION:-auto}"
    "SEED=42"
    "DRY_RUN=${DRY_RUN}"
    "DEEPSPEED_CONFIG=./scripts/zero2.json"
    "SAVE_STRATEGY=no"
    "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-16}"
    "REPORT_TO=${REPORT_TO:-wandb}"
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

launch() {
    local stage="$1" output="$2" adapter="$3" label="$4" child gpu batch accumulation data images logging
    if [[ "${stage}" == pretrain ]]; then
        child="${PRETRAIN_SCRIPT}"
        gpu="${PRETRAIN_GPU_INCLUDE}"
        batch="${PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
        accumulation="${PRETRAIN_GRADIENT_ACCUMULATION_STEPS:-4}"
        data="${PRETRAIN_DATA_PATH:-./playground/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}"
        images="${PRETRAIN_IMAGE_FOLDER:-./playground/LLaVA-Pretrain/images}"
        logging=1
    else
        child="${FINETUNE_SCRIPT}"
        gpu="${FINETUNE_GPU_INCLUDE}"
        batch="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
        accumulation="${GRADIENT_ACCUMULATION_STEPS:-16}"
        data="${FINETUNE_DATA_PATH:-./playground/data/llava_v1_5_mix665k.json}"
        images="${FINETUNE_IMAGE_FOLDER:-./playground/data}"
        logging=10
    fi
    echo "Run: ${label}; output=${output}"
    if [[ "${DRY_RUN}" == 0 ]]; then
        fresh_output "${output}"
        [[ "${stage}" == pretrain ]] || require_adapter "${adapter}"
    fi
    env -u WANDB_RUN_ID -u WANDB_RESUME \
        PYTHONNOUSERSITE=1 \
        "${COMMON_ENV[@]}" \
        "GPU_INCLUDE=${gpu}" \
        "RUN_NAME=${label}" \
        "WANDB_RUN_NAME=${label}" \
        "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${RUN_NAME}-cka-two-stage}" \
        "OUTPUT_DIR=${output}" \
        "PRETRAIN_ADAPTER=${adapter}" \
        "DATA_PATH=${data}" \
        "IMAGE_FOLDER=${images}" \
        "PER_DEVICE_TRAIN_BATCH_SIZE=${batch}" \
        "GRADIENT_ACCUMULATION_STEPS=${accumulation}" \
        "LOGGING_STEPS=${logging}" \
        "CKA_LOSS_ENABLED=${enabled}" \
        "CKA_PROJECTOR_WEIGHT=${projector_weight}" \
        "CKA_FINAL_HIDDEN_WEIGHT=${final_weight}" \
        "CKA_LAYERS=${layers}" \
        bash "${child}"
}

echo "Plan: ${REPEATS} independent repeats; settings=${SETTINGS}; stage=${STAGE}; seed=42; CKA lambda=${CKA_WEIGHT}"
echo "Seed is set by Trainer after model/projector construction; projector initialization is not seeded."
for (( repeat=1; repeat<=REPEATS; repeat++ )); do
    for setting in "${settings[@]}"; do
        enabled=True
        projector_weight="${CKA_WEIGHT}"
        final_weight=0.0
        layers=-1
        case "${setting}" in
            baseline) enabled=False; projector_weight=0.0 ;;
            cka_proj) ;;
            cka_proj_final) final_weight="${CKA_WEIGHT}"; layers=final ;;
        esac
        run_dir="${EXPERIMENT_ROOT}/repeat-${repeat}/${setting}"
        adapter="${run_dir}/llava-pretrain/mm_projector.bin"
        label="${RUN_NAME}-${setting}-repeat${repeat}-seed42"
        if [[ "${STAGE}" != finetune ]]; then
            launch pretrain "${run_dir}/llava-pretrain" "" "${label}-pretrain"
            if [[ "${DRY_RUN}" == 0 ]]; then
                require_adapter "${adapter}"
            fi
        fi
        if [[ "${STAGE}" != pretrain ]]; then
            launch finetune "${run_dir}/llava-finetune" "${adapter}" "${label}-finetune"
        fi
    done
done
