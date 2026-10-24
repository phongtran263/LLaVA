#!/bin/bash
set -euo pipefail

# Three Qwen3B CKA repeats with Trainer seed 42 and uncontrolled projector
# initialization. Both finetunes share the repeat's projector-CKA pretrain.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec env \
    "MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct" \
    "RUN_NAME=qwen2.5-3b" \
    "MM_PROJECTOR_TYPE=mlp2x_gelu" \
    "PROJECTOR_INIT_SEED_MODE=random" \
    "MM_PROJECTOR_INIT_SEED=" \
    "SETTINGS=cka_proj cka_proj_final" \
    "CKA_WEIGHT=0.1" \
    "REPEATS=${REPEATS:-3}" \
    "EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-./checkpoints/cka-two-stage/qwen2.5-3b-mlp2x_gelu-random-projector-cka-seed42}" \
    bash "${SCRIPT_DIR}/train_cka_two_stage.sh" "$@"
