#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
MATRIX_SCRIPT="${SCRIPT_DIR}/train_small_backbone_epoch_matrix.sh"
ROOT="${PWD}/checkpoints/full-ft"
FIRST_ROOT="${ROOT}/qwen2p5-1p5b-stage1-b64-no-gc-stage2-1e-seed42"
FALLBACK_ROOT="${ROOT}/qwen2p5-1p5b-stage1-b32-no-gc-fallback-stage2-1e-seed42"

[[ $# == 0 ]] || { echo 'This launcher takes no arguments.' >&2; exit 1; }
bash -n "$MATRIX_SCRIPT"
export GPU_ID=1 MODELS=qwen2.5-1.5b PRETRAIN_GRADIENT_CHECKPOINTING=False
export CONDA_PREFIX="${PWD}/../conda/envs/llava1"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export MASTER_PORT=29583 WANDB_RUN_GROUP=qwen1p5b-stage1-no-gc-batch-sweep-seed42

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    QWEN_PRETRAIN_BATCH_SIZE=64 RUN_TAG=b64-no-gc OUTPUT_ROOT="$FIRST_ROOT" \
        bash "$MATRIX_SCRIPT"
    exit 0
fi

LOG_DIR="${RUN_LOG_DIR:?Set RUN_LOG_DIR to an existing log directory.}"
[[ -d "$LOG_DIR" ]] || { echo "Missing log directory: $LOG_DIR" >&2; exit 1; }
for output in "$FIRST_ROOT" "$FALLBACK_ROOT"; do
    [[ ! -e "$output" && ! -L "$output" ]] || {
        echo "Output already exists: $output" >&2; exit 1;
    }
done
[[ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || {
    echo 'GPU 1 is occupied; no training started.' >&2; exit 1;
}

echo "Trying Qwen Stage 1 microbatch 64, accumulation 4, gradient checkpointing off."
if QWEN_PRETRAIN_BATCH_SIZE=64 RUN_TAG=b64-no-gc OUTPUT_ROOT="$FIRST_ROOT" \
        bash "$MATRIX_SCRIPT" >"${LOG_DIR}/b64.log" 2>&1; then
    echo 'Batch-64 queue completed successfully.'
    exit 0
fi

last_job="$(rg '^Starting ' "${LOG_DIR}/b64.log" | tail -1 || true)"
if [[ "$last_job" != *' pretrain ('* ]] || \
        ! rg -qi 'out of memory|OutOfMemoryError|Cuda failure 2' "${LOG_DIR}/b64.log"; then
    echo 'Batch-64 queue failed for a reason other than Stage 1 CUDA OOM; not retrying blindly.' >&2
    tail -25 "${LOG_DIR}/b64.log" >&2
    exit 1
fi

echo "Stage 1 CUDA OOM during: $last_job"
echo 'Restarting all Qwen runs from scratch at microbatch 32, accumulation 8, checkpointing off.'
for attempt in {1..12}; do
    [[ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)" ]] && break
    sleep 5
done
[[ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || {
    echo 'GPU 1 is still occupied after the failed run; fallback not started.' >&2; exit 1;
}
QWEN_PRETRAIN_BATCH_SIZE=32 RUN_TAG=b32-no-gc-fallback OUTPUT_ROOT="$FALLBACK_ROOT" \
    bash "$MATRIX_SCRIPT" >"${LOG_DIR}/b32.log" 2>&1
echo 'Batch-32 fallback queue completed successfully.'
