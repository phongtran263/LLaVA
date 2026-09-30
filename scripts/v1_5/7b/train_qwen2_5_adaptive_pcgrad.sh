#!/usr/bin/env bash
set -euo pipefail

# Run the matching two-stage adaptive projector-PCGrad pipeline.  A successful,
# validated stage 1 is a hard prerequisite for stage 2.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

: "${MODEL_NAME_OR_PATH:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${RUN_NAME:=qwen2.5-0.5b-adaptive-projector-pcgrad}"
: "${FULL_TRAIN:=False}"
: "${SMOKE_MAX_STEPS:=2}"
: "${STAGE1_ADAPTIVE_PCGRAD_CONFIG:=./scripts/adaptive_projector_pcgrad/stage1_defaults.json}"
: "${STAGE2_ADAPTIVE_PCGRAD_CONFIG:=./scripts/adaptive_projector_pcgrad/stage2_defaults.json}"
: "${STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE:=32}"
: "${STAGE1_GRADIENT_ACCUMULATION_STEPS:=8}"
: "${STAGE1_ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE:=4}"
: "${STAGE1_GRADIENT_CHECKPOINTING:=False}"
: "${STAGE1_LOGGING_STEPS:=1}"
: "${STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE:=32}"
: "${STAGE2_GRADIENT_ACCUMULATION_STEPS:=4}"
: "${STAGE2_ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE:=4}"
: "${STAGE2_GRADIENT_CHECKPOINTING:=True}"
: "${STAGE2_LOGGING_STEPS:=1}"

if is_true "${FULL_TRAIN}"; then
    RUN_MODE=full
else
    RUN_MODE=smoke
fi

: "${PIPELINE_ROOT:=./checkpoints/adaptive-projector-pcgrad/${RUN_NAME}/${RUN_MODE}}"
STAGE1_OUTPUT_DIR="${PIPELINE_ROOT}/stage1"
STAGE2_OUTPUT_DIR="${PIPELINE_ROOT}/stage2"
STAGE1_ADAPTER="${STAGE1_OUTPUT_DIR}/mm_projector.bin"
STAGE1_METADATA="${STAGE1_OUTPUT_DIR}/adaptive_projector_pcgrad_metadata.json"

echo "Adaptive projector-PCGrad pipeline mode: ${RUN_MODE}"
if ! is_true "${FULL_TRAIN}"; then
    echo "Smoke limit: ${SMOKE_MAX_STEPS} optimizer steps per stage"
fi

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
RUN_NAME="${RUN_NAME}" \
FULL_TRAIN="${FULL_TRAIN}" \
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS}" \
PIPELINE_ROOT="${PIPELINE_ROOT}" \
OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" \
ADAPTIVE_PCGRAD_CONFIG="${STAGE1_ADAPTIVE_PCGRAD_CONFIG}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${STAGE1_GRADIENT_ACCUMULATION_STEPS}" \
ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE="${STAGE1_ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE}" \
GRADIENT_CHECKPOINTING="${STAGE1_GRADIENT_CHECKPOINTING}" \
LOGGING_STEPS="${STAGE1_LOGGING_STEPS}" \
bash "${SCRIPT_DIR}/pretrain_qwen2_5_adaptive_pcgrad.sh"

# Do not permit a merely successful process exit to masquerade as a valid
# stage-1 handoff.  The stage-2 launcher repeats the metadata validation.
for required in "${STAGE1_ADAPTER}" "${STAGE1_METADATA}"; do
    if [[ ! -s "${required}" ]]; then
        echo "Stage 1 completed without required handoff artifact: ${required}" >&2
        exit 1
    fi
done

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
RUN_NAME="${RUN_NAME}" \
FULL_TRAIN="${FULL_TRAIN}" \
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS}" \
PIPELINE_ROOT="${PIPELINE_ROOT}" \
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" \
PRETRAIN_ADAPTER="${STAGE1_ADAPTER}" \
OUTPUT_DIR="${STAGE2_OUTPUT_DIR}" \
ADAPTIVE_PCGRAD_CONFIG="${STAGE2_ADAPTIVE_PCGRAD_CONFIG}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${STAGE2_GRADIENT_ACCUMULATION_STEPS}" \
ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE="${STAGE2_ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE}" \
GRADIENT_CHECKPOINTING="${STAGE2_GRADIENT_CHECKPOINTING}" \
LOGGING_STEPS="${STAGE2_LOGGING_STEPS}" \
bash "${SCRIPT_DIR}/finetune_qwen2_5_adaptive_pcgrad.sh"
