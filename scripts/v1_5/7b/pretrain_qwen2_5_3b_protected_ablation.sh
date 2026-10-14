#!/bin/bash
set -euo pipefail

# Matched NEW continuations of a verified Stage-1 CE-only checkpoint.
# CE-only and protected runs share weights, original split, seed, fresh
# optimizer, LR schedule and budget. A failed protected validation gate never
# yields an accepted refinement. Preview without training using DRY_RUN=1.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
: "${SOURCE_STAGE1_CHECKPOINT:?Set the original protected Stage-1 CE-only checkpoint directory}"
: "${MAX_STEPS:?Set an equal continuation optimizer-step budget}"
: "${ABLATION_OUTPUT_ROOT:?Set a fresh root for the matched continuations}"

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    echo "A new continuation uses SOURCE_STAGE1_CHECKPOINT, not RESUME_FROM_CHECKPOINT." >&2
    exit 1
fi
if (( $# )); then
    echo "The ablation wrapper accepts environment settings only, not trailing CLI overrides." >&2
    exit 1
fi
for mode in ce_only protected; do
    if [[ ( -e "${ABLATION_OUTPUT_ROOT}/${mode}" || -L "${ABLATION_OUTPUT_ROOT}/${mode}" ) && "${DRY_RUN:-0}" != 1 ]]; then
        echo "Refusing to auto-resume/overwrite existing continuation output: ${ABLATION_OUTPUT_ROOT}/${mode}" >&2
        exit 1
    fi
done
export SOURCE_STAGE1_CHECKPOINT MAX_STEPS
export START_PHASE_B=True
export PROTECTED_CKA_REFINEMENT=True
export SEED="${SEED:-42}"
for mode in ce_only protected; do
    CONTINUATION="${mode}" \
    OUTPUT_DIR="${ABLATION_OUTPUT_ROOT}/${mode}" \
    RUN_NAME="${ABLATION_RUN_NAME:-qwen2.5-3b-protected-stage1}-${mode}" \
    bash "${SCRIPT_DIR}/pretrain_qwen2_5_3b_protected_cka.sh"
done
