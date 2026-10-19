#!/bin/bash
set -euo pipefail

# Each repeat trains a CE adapter and a projector-CKA adapter. cka_proj and
# cka_proj_final finetune from the same projector-CKA adapter in that repeat.
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
pretrain_setting_for() {
    if [[ "$1" == baseline ]]; then
        printf '%s\n' baseline
    else
        printf '%s\n' cka_proj
    fi
}
adapter_for() {
    printf '%s/repeat-%s/%s/llava-pretrain/mm_projector.bin\n' \
        "${EXPERIMENT_ROOT}" "$1" "$(pretrain_setting_for "$2")"
}

(( $# == 0 )) || fail "Use environment variables; CLI overrides are not accepted."
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
REPEATS="${REPEATS:-2}"
REPEAT_START="${REPEAT_START-1}"
PROJECTOR_INIT_SEED_BASE="${PROJECTOR_INIT_SEED_BASE-1000}"
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
[[ "${REPEAT_START}" =~ ^[1-9][0-9]*$ ]] || fail "REPEAT_START must be a positive integer without leading zeroes."
[[ "${PROJECTOR_INIT_SEED_BASE}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "PROJECTOR_INIT_SEED_BASE must be a nonnegative integer without leading zeroes."
[[ "${SEED-42}" == 42 && "${SEEDS-42}" == 42 && "${STAGE1_SEED-42}" == 42 ]] || \
    fail "This suite uses seed 42 in both stages and every repeat; unset other seed overrides."
# Validate without importing the training stack; one coefficient controls every
# active CKA term in both stages. Disabled terms remain exactly zero.
CKA_WEIGHT="$(python3 - "${CKA_WEIGHT}" "${PROJECTOR_INIT_SEED_BASE}" "${REPEAT_START}" "${REPEATS}" <<'PY_WEIGHT'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit("CKA_WEIGHT must be a finite positive number.")
if not math.isfinite(value) or value <= 0:
    raise SystemExit("CKA_WEIGHT must be a finite positive number.")
base, start, count = map(int, sys.argv[2:])
if start + count > 2**63 - 1 or base + start + count - 1 > 2**63 - 1:
    raise SystemExit("Repeat range and projector initialization seeds must fit a signed 64-bit integer.")
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

# Plan each required Stage-1 producer once, before its first consumer. This
# also handles final-only selections and reversed CKA setting order.
job_repeats=() job_settings=() job_stages=()
declare -A planned_pretrains=()
for (( repeat=REPEAT_START; repeat<REPEAT_START+REPEATS; repeat++ )); do
    for setting in "${settings[@]}"; do
        producer="$(pretrain_setting_for "${setting}")"
        producer_key="${repeat}/${producer}"
        if [[ "${STAGE}" != finetune && -z "${planned_pretrains[${producer_key}]:-}" ]]; then
            job_repeats+=("${repeat}")
            job_settings+=("${producer}")
            job_stages+=(pretrain)
            planned_pretrains["${producer_key}"]=1
        fi
        if [[ "${STAGE}" != pretrain ]]; then
            job_repeats+=("${repeat}")
            job_settings+=("${setting}")
            job_stages+=(finetune)
        fi
    done
done
# Check every planned output before starting any long-running job. Existing
# outputs are never silently reused or resumed. Only finetune reuses adapters.
for index in "${!job_stages[@]}"; do
    repeat="${job_repeats[${index}]}"
    setting="${job_settings[${index}]}"
    job_stage="${job_stages[${index}]}"
    fresh_output "${EXPERIMENT_ROOT}/repeat-${repeat}/${setting}/llava-${job_stage}"
    if [[ "${STAGE}" == finetune && "${DRY_RUN}" == 0 ]]; then
        require_adapter "$(adapter_for "${repeat}" "${setting}")"
    fi
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
    local stage="$1" output="$2" adapter="$3" label="$4" projector_init_seed="$5" child gpu batch accumulation data images logging
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
        "MM_PROJECTOR_INIT_SEED=${projector_init_seed}" \
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

echo "Plan: ${REPEATS} independent repeats starting at ${REPEAT_START}; settings=${SETTINGS}; stage=${STAGE}; seed=42; CKA lambda=${CKA_WEIGHT}"
echo "Trainer seed=42; projector initialization seed=${PROJECTOR_INIT_SEED_BASE}+repeat (isolated from training RNG)."
for index in "${!job_stages[@]}"; do
    repeat="${job_repeats[${index}]}"
    setting="${job_settings[${index}]}"
    job_stage="${job_stages[${index}]}"
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
    adapter="$(adapter_for "${repeat}" "${setting}")"
    label="${RUN_NAME}-${setting}-repeat${repeat}-seed42-${job_stage}"
    projector_init_seed=$(( PROJECTOR_INIT_SEED_BASE + repeat ))
    if [[ "${job_stage}" == pretrain ]]; then
        launch pretrain "${run_dir}/llava-pretrain" "" "${label}" "${projector_init_seed}"
        if [[ "${DRY_RUN}" == 0 ]]; then
            require_adapter "${adapter}"
        fi
    else
        launch finetune "${run_dir}/llava-finetune" "${adapter}" "${label}" "${projector_init_seed}"
    fi
done
