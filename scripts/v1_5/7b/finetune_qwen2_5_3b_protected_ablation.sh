#!/bin/bash
set -euo pipefail

# Three continuations from the SAME Stage-2 CE-only weights/held-out split.
# All modes use the same data, seed, fresh optimizer, refinement LR and budget.
# CONTINUATION_MODES='ce_only protected' runs the requested two-arm comparison.
# Gate failures/retries can reduce accepted steps; inspect protected2/* logs.
# Required thresholds/LR/model settings are passed to the ordinary launcher.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
: "${SOURCE_STAGE2_CHECKPOINT:?Set the Stage-2 CE-only checkpoint, not a Stage-1 adapter}"
: "${MAX_STEPS:?Set an equal continuation step budget for all three runs}"
: "${ABLATION_OUTPUT_ROOT:?Set a fresh directory for the three continuations}"

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    echo "The comparison initializes each run from SOURCE_STAGE2_CHECKPOINT; do not set RESUME_FROM_CHECKPOINT." >&2
    exit 1
fi
# Do not permit trailing CLI overrides (including argparse abbreviations) to
# defeat the per-mode output/source/budget controls. Shared knobs use env vars.
if (( $# )); then
    echo "The ablation wrapper accepts environment settings only, not trailing CLI overrides." >&2
    exit 1
fi
read -r -a modes <<< "${CONTINUATION_MODES:-ce_only ce_kl protected}"
if (( ${#modes[@]} == 0 )); then
    echo "CONTINUATION_MODES must contain at least one continuation mode." >&2
    exit 1
fi
declare -A seen_modes=()
for mode in "${modes[@]}"; do
    case "${mode}" in
        ce_only|ce_kl|protected) ;;
        *) echo "Unknown continuation mode: ${mode}" >&2; exit 1 ;;
    esac
    if [[ -n "${seen_modes[${mode}]:-}" ]]; then
        echo "Duplicate continuation mode: ${mode}" >&2
        exit 1
    fi
    seen_modes[${mode}]=1
    if [[ ( -e "${ABLATION_OUTPUT_ROOT}/${mode}" || -L "${ABLATION_OUTPUT_ROOT}/${mode}" ) && "${DRY_RUN:-0}" != 1 ]]; then
        echo "Refusing to auto-resume/overwrite existing ablation output: ${ABLATION_OUTPUT_ROOT}/${mode}" >&2
        exit 1
    fi
done
export SOURCE_STAGE2_CHECKPOINT MAX_STEPS
export STAGE2_START_PHASE_B=True
export STAGE2_MAX_PHASE_B_STEPS="${MAX_STEPS}"
export PROTECTED_CKA_STAGE2=True
export SEED="${SEED:-42}"

for mode in "${modes[@]}"; do
    STAGE2_CONTINUATION="${mode}" \
    OUTPUT_DIR="${ABLATION_OUTPUT_ROOT}/${mode}" \
    RUN_NAME="${ABLATION_RUN_NAME:-qwen2.5-3b-protected-stage2}-${mode}" \
    bash "${SCRIPT_DIR}/finetune_qwen2_5_3b_protected_cka.sh"
done
