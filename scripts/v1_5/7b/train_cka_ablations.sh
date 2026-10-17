#!/bin/bash
set -euo pipefail

# Placement-only CKA ablations using the two Qwen training launchers below.
# INIT_MODE=shared: one structural Stage-1 adapter for all Stage-2 variants.
# INIT_MODE=paired: CE Stage 1 -> baseline; structural Stage 1 -> proj/final/full.
# Usage and recipe: scripts/v1_5/7b/CKA_ABLATIONS.md
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
PRETRAIN_SCRIPT="${SCRIPT_DIR}/pretrain_qwen2_5.sh"
FINETUNE_SCRIPT="${SCRIPT_DIR}/finetune_qwen2_5.sh"

fail() { echo "$*" >&2; exit 1; }
fresh_output() {
    if [[ -e "$1" || -L "$1" ]]; then
        fail "Output already exists: $1. Choose a new EXPERIMENT_ROOT or select only unfinished variants/seeds."
    fi
}
require_adapter() {
    if [[ ! -f "$1" || ! -s "$1" ]]; then
        fail "Missing, empty, or non-file Stage-1 adapter: $1. Run STAGE=pretrain first."
    fi
}
adapter_for_variant() {
    if [[ "${INIT_MODE}" == paired && "$1" == baseline ]]; then
        printf '%s\n' "${BASELINE_PRETRAIN_ADAPTER}"
    else
        printf '%s\n' "${CKA_PRETRAIN_ADAPTER}"
    fi
}

if (( $# )); then
    fail "Use environment variables (STAGE, VARIANTS, SEEDS, DRY_RUN); CLI overrides are not accepted."
fi
STAGE="${STAGE:-all}"
INIT_MODE="${INIT_MODE:-shared}"
DRY_RUN="${DRY_RUN:-0}"
VARIANTS="${VARIANTS:-baseline proj final full}"
SEEDS="${SEEDS:-${SEED:-42}}"
STAGE1_SEED="${STAGE1_SEED:-42}"
# Keep Stage-1 initialization strength independent from Stage-2 coefficients.
# An explicitly empty value is invalid rather than silently taking a default.
PRETRAIN_CKA_PROJECTOR_WEIGHT="${PRETRAIN_CKA_PROJECTOR_WEIGHT-0.1}"
CKA_PROJECTOR_WEIGHT="${CKA_PROJECTOR_WEIGHT-0.01}"
CKA_FINAL_HIDDEN_WEIGHT="${CKA_FINAL_HIDDEN_WEIGHT-0.01}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-lmsys/vicuna-7b-v1.5}"
RUN_NAME="${RUN_NAME:-vicuna}"
MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
CONVERSATION_VERSION="${CONVERSATION_VERSION:-auto}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-./checkpoints/cka-location-ablations/${RUN_NAME}-${MM_PROJECTOR_TYPE}}"
PRETRAIN_GPU_INCLUDE="${PRETRAIN_GPU_INCLUDE:-${GPU_INCLUDE:-localhost:1,2}}"
FINETUNE_GPU_INCLUDE="${FINETUNE_GPU_INCLUDE:-${GPU_INCLUDE:-localhost:1,2}}"

case "${STAGE}" in all|pretrain|finetune) ;; *) fail "STAGE must be all, pretrain, or finetune." ;; esac
case "${INIT_MODE}" in shared|paired) ;; *) fail "INIT_MODE must be shared or paired." ;; esac
RUN_ROOT="${EXPERIMENT_ROOT}"
RUN_LABEL="${RUN_NAME}"
if [[ "${INIT_MODE}" == shared ]]; then
    PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-${RUN_ROOT}/stage1-seed${STAGE1_SEED}/llava-pretrain}"
    PRETRAIN_ADAPTER="${PRETRAIN_ADAPTER:-${PRETRAIN_OUTPUT_DIR}/mm_projector.bin}"
    CKA_PRETRAIN_ADAPTER="${PRETRAIN_ADAPTER}"
    pretrain_names=(shared-proj)
    pretrain_outputs=("${PRETRAIN_OUTPUT_DIR}")
    pretrain_adapters=("${PRETRAIN_ADAPTER}")
else
    [[ -z "${PRETRAIN_OUTPUT_DIR:-}" && -z "${PRETRAIN_ADAPTER:-}" ]] || \
        fail "INIT_MODE=paired uses BASELINE_PRETRAIN_* and CKA_PRETRAIN_*; unset PRETRAIN_OUTPUT_DIR/PRETRAIN_ADAPTER."
    RUN_ROOT="${EXPERIMENT_ROOT}/paired"
    RUN_LABEL="${RUN_NAME}-paired"
    BASELINE_PRETRAIN_OUTPUT_DIR="${BASELINE_PRETRAIN_OUTPUT_DIR:-${RUN_ROOT}/stage1-seed${STAGE1_SEED}/baseline/llava-pretrain}"
    CKA_PRETRAIN_OUTPUT_DIR="${CKA_PRETRAIN_OUTPUT_DIR:-${RUN_ROOT}/stage1-seed${STAGE1_SEED}/cka/llava-pretrain}"
    BASELINE_PRETRAIN_ADAPTER="${BASELINE_PRETRAIN_ADAPTER:-${BASELINE_PRETRAIN_OUTPUT_DIR}/mm_projector.bin}"
    CKA_PRETRAIN_ADAPTER="${CKA_PRETRAIN_ADAPTER:-${CKA_PRETRAIN_OUTPUT_DIR}/mm_projector.bin}"
    # realpath also catches relative spellings and symlinked parents before
    # files exist; -ef catches existing hardlinks to the same checkpoint.
    if [[ "$(realpath -m -- "${BASELINE_PRETRAIN_ADAPTER}")" == "$(realpath -m -- "${CKA_PRETRAIN_ADAPTER}")" \
          || "${BASELINE_PRETRAIN_ADAPTER}" -ef "${CKA_PRETRAIN_ADAPTER}" ]]; then
        fail "Paired Stage-1 baseline and CKA adapters must be different files. Use INIT_MODE=shared for one adapter."
    fi
    if [[ "$(realpath -m -- "${BASELINE_PRETRAIN_OUTPUT_DIR}")" == "$(realpath -m -- "${CKA_PRETRAIN_OUTPUT_DIR}")" ]]; then
        fail "Paired Stage-1 output directories must be different."
    fi
    pretrain_names=(baseline cka)
    pretrain_outputs=("${BASELINE_PRETRAIN_OUTPUT_DIR}" "${CKA_PRETRAIN_OUTPUT_DIR}")
    pretrain_adapters=("${BASELINE_PRETRAIN_ADAPTER}" "${CKA_PRETRAIN_ADAPTER}")
fi
[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || fail "DRY_RUN must be 0 or 1."
[[ "${STAGE1_SEED}" =~ ^[0-9]+$ ]] || fail "STAGE1_SEED must be a nonnegative integer."
# Stdlib only: this also validates dry runs without importing the ML stack.
# Canonicalize zero so numeric/scientific spellings all disable the same term.
weight_values="$(python3 - "${PRETRAIN_CKA_PROJECTOR_WEIGHT}" "${CKA_PROJECTOR_WEIGHT}" "${CKA_FINAL_HIDDEN_WEIGHT}" <<'PY_WEIGHTS'
import math
import sys

names = ("PRETRAIN_CKA_PROJECTOR_WEIGHT", "CKA_PROJECTOR_WEIGHT", "CKA_FINAL_HIDDEN_WEIGHT")
values = []
for name, raw in zip(names, sys.argv[1:]):
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a finite nonnegative number, got {raw!r}.")
    if not math.isfinite(value) or value < 0:
        raise SystemExit(f"{name} must be a finite nonnegative number, got {raw!r}.")
    values.append("0.0" if value == 0 else str(value))
print(" ".join(values))
PY_WEIGHTS
)"
read -r PRETRAIN_CKA_PROJECTOR_WEIGHT CKA_PROJECTOR_WEIGHT CKA_FINAL_HIDDEN_WEIGHT <<< "${weight_values}"
read -r -a variants <<< "${VARIANTS}"
read -r -a seeds <<< "${SEEDS}"
(( ${#variants[@]} )) || fail "VARIANTS must not be empty."
(( ${#seeds[@]} )) || fail "SEEDS must not be empty."
declare -A seen_variants=() seen_seeds=()
for variant in "${variants[@]}"; do
    case "${variant}" in baseline|proj|final|full) ;; *) fail "Unknown variant: ${variant}. Use baseline, proj, final, full." ;; esac
    [[ -z "${seen_variants[${variant}]:-}" ]] || fail "Duplicate variant: ${variant}."
    seen_variants["${variant}"]=1
done
for seed in "${seeds[@]}"; do
    [[ "${seed}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "Invalid seed: ${seed}. Use nonnegative integers without leading zeroes."
    [[ -z "${seen_seeds[${seed}]:-}" ]] || fail "Duplicate seed: ${seed}."
    seen_seeds["${seed}"]=1
done
if [[ "${STAGE}" == all ]]; then
    for index in "${!pretrain_outputs[@]}"; do
        [[ "${pretrain_adapters[${index}]}" == "${pretrain_outputs[${index}]}/mm_projector.bin" ]] || \
            fail "STAGE=all must use the adapters it pretrains. Use STAGE=finetune to reuse external adapters."
    done
fi

# Preflight every selected path before launching anything. Trainer otherwise
# automatically resumes checkpoint-* found in its output directory.
if [[ "${STAGE}" != finetune ]]; then
    for output in "${pretrain_outputs[@]}"; do
        fresh_output "${output}"
    done
fi
if [[ "${STAGE}" != pretrain ]]; then
    for seed in "${seeds[@]}"; do
        for variant in "${variants[@]}"; do
            fresh_output "${RUN_ROOT}/seed-${seed}/${variant}/llava-finetune"
        done
    done
fi
[[ -f "${PRETRAIN_SCRIPT}" && -f "${FINETUNE_SCRIPT}" ]] || fail "Both Qwen training scripts must exist beside this launcher."
if [[ "${DRY_RUN}" == 0 ]]; then
    [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/deepspeed" ]] || fail "Activate a Qwen training conda environment containing DeepSpeed first."
    if [[ "${STAGE}" == finetune ]]; then
        # Check all selected arms before the first launch, not midway through
        # a long suite. A subset needs only its own initialization checkpoint.
        for variant in "${variants[@]}"; do
            require_adapter "$(adapter_for_variant "${variant}")"
        done
    fi
fi

# Keep the model/training recipe in the Qwen scripts. Only the listed
# environment settings are forwarded; stale optional-technique settings are
# reset explicitly before invoking either script.
COMMON_ENV=(
    "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
    "MM_PROJECTOR_TYPE=${MM_PROJECTOR_TYPE}"
    "VISION_TOWER=openai/clip-vit-large-patch14-336"
    "MM_VISION_SELECT_LAYER=-2"
    "CONVERSATION_VERSION=${CONVERSATION_VERSION}"
    "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-16}"
    "REPORT_TO=${REPORT_TO:-wandb}"
    "DRY_RUN=${DRY_RUN}"
    "DEEPSPEED_CONFIG=./scripts/zero2.json"
    "SAVE_STRATEGY=no"
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
    local gpu_include="$1" output="$2" run_name="$3" stage="$4" child_script
    shift 4
    if [[ "${stage}" == pretrain ]]; then
        child_script="${PRETRAIN_SCRIPT}"
    else
        child_script="${FINETUNE_SCRIPT}"
    fi
    local -a command=(
        env -u WANDB_RUN_ID -u WANDB_RESUME
        PYTHONNOUSERSITE=1
        "${COMMON_ENV[@]}"
        "GPU_INCLUDE=${gpu_include}"
        "OUTPUT_DIR=${output}"
        "RUN_NAME=${run_name}"
        "WANDB_RUN_NAME=${run_name}"
        "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${RUN_LABEL}-cka-locations}"
        "$@" bash "${child_script}"
    )
    echo "Run: ${run_name}; output=${output}"
    if [[ "${DRY_RUN}" == 0 ]]; then
        fresh_output "${output}"
    fi
    # Child dry runs print the actual DeepSpeed argv without version/GPU checks.
    "${command[@]}"
}

if [[ "${STAGE}" != finetune ]]; then
    # Both paired arms use the same seed and caption recipe. Trainer seeds
    # after projector construction, so their random initial weights may differ.
    for index in "${!pretrain_names[@]}"; do
        enabled=True
        projector_weight="${PRETRAIN_CKA_PROJECTOR_WEIGHT}"
        if [[ "${pretrain_names[${index}]}" == baseline ]]; then
            enabled=False
            projector_weight=0.0
        fi
        [[ "${projector_weight}" != 0.0 ]] || enabled=False
        launch "${PRETRAIN_GPU_INCLUDE}" "${pretrain_outputs[${index}]}" \
            "${RUN_LABEL}-${pretrain_names[${index}]}-pretrain-seed${STAGE1_SEED}" pretrain \
            "DATA_PATH=${PRETRAIN_DATA_PATH:-./playground/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}" \
            "IMAGE_FOLDER=${PRETRAIN_IMAGE_FOLDER:-./playground/LLaVA-Pretrain/images}" \
            "PRETRAIN_ADAPTER=" \
            "SEED=${STAGE1_SEED}" \
            "PER_DEVICE_TRAIN_BATCH_SIZE=${PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE:-32}" \
            "GRADIENT_ACCUMULATION_STEPS=${PRETRAIN_GRADIENT_ACCUMULATION_STEPS:-4}" \
            "LOGGING_STEPS=1" \
            "CKA_LOSS_ENABLED=${enabled}" \
            "CKA_PROJECTOR_WEIGHT=${projector_weight}" \
            "CKA_FINAL_HIDDEN_WEIGHT=0.0" \
            "CKA_LAYERS=-1"
        if [[ "${DRY_RUN}" == 0 ]]; then
            require_adapter "${pretrain_outputs[${index}]}/mm_projector.bin"
        fi
    done
fi

if [[ "${STAGE}" != pretrain ]]; then
    for seed in "${seeds[@]}"; do
        for variant in "${variants[@]}"; do
            enabled=True
            projector_weight=0.0
            final_weight=0.0
            layers=-1
            case "${variant}" in
                baseline) enabled=False ;;
                proj) projector_weight="${CKA_PROJECTOR_WEIGHT}" ;;
                final) final_weight="${CKA_FINAL_HIDDEN_WEIGHT}" ;;
                full) projector_weight="${CKA_PROJECTOR_WEIGHT}"; final_weight="${CKA_FINAL_HIDDEN_WEIGHT}" ;;
            esac
            [[ "${final_weight}" == 0.0 ]] || layers=final
            if [[ "${projector_weight}" == 0.0 && "${final_weight}" == 0.0 ]]; then
                enabled=False
            fi
            # Paired mode routes baseline to CE Stage 1 and the three CKA
            # variants to structural Stage 1. Shared mode reuses one adapter.
            launch "${FINETUNE_GPU_INCLUDE}" \
                "${RUN_ROOT}/seed-${seed}/${variant}/llava-finetune" \
                "${RUN_LABEL}-${variant}-finetune-seed${seed}" finetune \
                "DATA_PATH=${FINETUNE_DATA_PATH:-./playground/data/llava_v1_5_mix665k.json}" \
                "IMAGE_FOLDER=${FINETUNE_IMAGE_FOLDER:-./playground/data}" \
                "PRETRAIN_ADAPTER=$(adapter_for_variant "${variant}")" \
                "SEED=${seed}" \
                "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
                "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}" \
                "LOGGING_STEPS=10" \
                "CKA_LOSS_ENABLED=${enabled}" \
                "CKA_PROJECTOR_WEIGHT=${projector_weight}" \
                "CKA_FINAL_HIDDEN_WEIGHT=${final_weight}" \
                "CKA_LAYERS=${layers}"
        done
    done
fi
