#!/bin/bash
set -euo pipefail

# Two complete Stage-2 arms, initialized from their respective Stage-1 arms.
# This is an end-to-end pipeline comparison, NOT two Stage-2 continuations
# from the same Stage-2 CE checkpoint. Phase A is exactly 4,832 steps; Phase B
# has 350 steps. Both use the same Stage-2 data, seed, batch and LR schedules.
# Before either launch, require BOTH Stage-1 checkpoint-350 snapshots to be
# complete CE-only/protected continuations of the same verified CE source and
# held-out split. Protected Stage 1 must have passed its validation gate.
# DRY_RUN=1 prints both commands without reading weights or requiring inputs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
: "${STAGE1_PAIR_ROOT:=./checkpoints/pretrain-protected-cka/qwen2.5-3b/continuation-350}"
: "${STAGE2_PAIR_ROOT:=./checkpoints/finetune-protected-cka/qwen2.5-3b/paired-protected-cka}"
: "${GPU_ID:=3}"
: "${DRY_RUN:=0}"
if (( $# )); then
    echo "This paired pipeline accepts environment settings only, not CLI overrides." >&2
    exit 1
fi
if [[ "${DRY_RUN}" != 0 && "${DRY_RUN}" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
for mode in ce_only protected; do
    if [[ "${DRY_RUN}" != 1 && ( -e "${STAGE2_PAIR_ROOT}/${mode}" || -L "${STAGE2_PAIR_ROOT}/${mode}" ) ]]; then
        echo "Refusing existing Stage-2 output: ${STAGE2_PAIR_ROOT}/${mode}" >&2
        exit 1
    fi
done

if [[ "${DRY_RUN}" != 1 ]]; then
    # mmap keeps large reference/EMA tensor storages lazy: only metadata is
    # inspected on CPU, never model tensors or GPU state.
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN:-python}" - "${STAGE1_PAIR_ROOT}" <<'PY_CHECK'
import hashlib
import json
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
states = {}
initializers = {}
for mode in ("ce_only", "protected"):
    checkpoint = root / mode / "checkpoint-350"
    required = ("protected_cka_state.pt", "trainer_state.json", "mm_projector.bin")
    for name in required:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Incomplete Stage-1 {mode} checkpoint: {path}")
    state = torch.load(checkpoint / "protected_cka_state.pt", map_location="cpu",
                       weights_only=False, mmap=True)
    with (checkpoint / "trainer_state.json").open(encoding="utf-8") as handle:
        step = json.load(handle).get("global_step")
    config = state.get("signature", {}).get("config", {})
    if (state.get("version") != 1 or state.get("phase") != "B"
            or state.get("phase_b_start") != 0 or step != 350
            or config.get("max_steps") != 350
            or config.get("protected_cka_continuation") != mode
            or state.get("finished_reason") or not state.get("reference")):
        raise SystemExit(f"Stage-1 {mode} is not a completed, matching 350-step continuation.")
    if bool(state.get("refinement_accepted")) != (mode == "protected"):
        raise SystemExit(f"Stage-1 {mode} refinement acceptance is invalid; protected must pass its gate.")
    provenance = state.get("source_provenance")
    if not isinstance(provenance, dict) or not provenance.get("checkpoint") or not provenance.get("signature"):
        raise SystemExit(f"Stage-1 {mode} is missing verified CE-source provenance.")
    signature = state["signature"]
    if signature.get("model", {}).get("_name_or_path") != "Qwen/Qwen2.5-3B-Instruct":
        raise SystemExit(f"Stage-1 {mode} is not the configured Qwen2.5-3B backbone.")
    if not config.get("protected_cka_split_fingerprint"):
        raise SystemExit(f"Stage-1 {mode} is missing the original split fingerprint.")
    states[mode] = state
    digest = hashlib.sha256()
    with (checkpoint / "mm_projector.bin").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    initializers[mode] = {"adapter": str((checkpoint / "mm_projector.bin").resolve()),
                          "sha256": digest.hexdigest()}

left, right = states["ce_only"], states["protected"]
if left["source_provenance"] != right["source_provenance"]:
    raise SystemExit("Stage-1 arms have different original CE-source provenance.")
for key in ("model", "train_dataset", "eval_dataset"):
    if not left["signature"].get(key) or left["signature"][key] != right["signature"].get(key):
        raise SystemExit(f"Stage-1 arms have different {key} identities.")
if (left["signature"]["config"]["protected_cka_split_fingerprint"]
        != right["signature"]["config"].get("protected_cka_split_fingerprint")):
    raise SystemExit("Stage-1 arms have different held-out split fingerprints.")
shared_configs = [{key: value for key, value in state["signature"]["config"].items()
                   if key != "protected_cka_continuation"} for state in (left, right)]
if shared_configs[0] != shared_configs[1]:
    different = sorted(key for key in shared_configs[0].keys() | shared_configs[1].keys()
                       if shared_configs[0].get(key) != shared_configs[1].get(key))
    raise SystemExit("Stage-1 arms have different recorded training settings: " + ", ".join(different))
print("Stage-1 pair verified: 350 steps each; protected gate passed; common CE source/split/recipe.", flush=True)
print("Stage-2 initializer provenance: " + json.dumps(initializers, sort_keys=True), flush=True)
PY_CHECK
fi

# Ignore stale settings from earlier single-run/continuation experiments.
# The complete paired recipe is intentionally fixed below, not overridden by
# source/resume/anchor/backend variables inherited from the calling shell.
unset SOURCE_STAGE1_CHECKPOINT SOURCE_STAGE2_CHECKPOINT RESUME_FROM_CHECKPOINT PRETRAIN_ADAPTER
unset CKA_ANCHOR_LAYER CKA_VISION_ANCHOR_LAYER CKA_PROJECTOR_VISION_ANCHOR_LAYER CKA_FINAL_VISION_ANCHOR_LAYER
unset DEEPSPEED DEEPSPEED_CONFIG GPU_INCLUDE STAGE2_DRIFT_BUDGET MM_PROJECTOR_LR
unset STAGE2_VALIDATION_DATA_PATH STAGE2_VALIDATION_IMAGE_FOLDER
export DRY_RUN GPU_ID
export MODEL_NAME_OR_PATH="Qwen/Qwen2.5-3B-Instruct"
export TRAIN_DATA_PATH="./playground/data/llava_v1_5_mix665k.json"
export TRAIN_IMAGE_FOLDER="./playground/data"
export CONVERSATION_VERSION=auto IMAGE_ASPECT_RATIO=pad
export VISION_TOWER="openai/clip-vit-large-patch14-336" MM_PROJECTOR_TYPE=mlp2x_gelu MM_VISION_SELECT_LAYER=-2
export PROTECTED_CKA_STAGE2=True STAGE2_START_PHASE_B=False
export MAX_STEPS=5182 STAGE2_MIN_PHASE_A_STEPS=4832 STAGE2_MAX_PHASE_A_STEPS=4832
export STAGE2_MIN_PHASE_B_STEPS=350 STAGE2_MAX_PHASE_B_STEPS=350
export PER_DEVICE_TRAIN_BATCH_SIZE=8 PER_DEVICE_EVAL_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=16
export SEED=42 STAGE2_VALIDATION_SEED=42 STAGE2_VALIDATION_FRACTION=0.005 STAGE2_VALIDATION_MAX_SAMPLES=2002
export GRADIENT_CHECKPOINTING=True DATALOADER_NUM_WORKERS=0 GROUP_BY_MODALITY_LENGTH=True
export BF16=True LORA_ENABLE=False FREEZE_MM_MLP_ADAPTER=False MODEL_MAX_LENGTH=2048
export LEARNING_RATE=2e-5 STAGE2_REFINE_LR=1e-5 LR_SCHEDULER_TYPE=cosine WARMUP_RATIO=0.03
export WEIGHT_DECAY=0.0 MAX_GRAD_NORM=1.0 STAGE2_EVAL_INTERVAL=100 SAVE_TOTAL_LIMIT=2
export CKA_LOSS_ENABLED=True CKA_PROJECTOR_WEIGHT=0.1 CKA_FINAL_HIDDEN_WEIGHT=0.1 CKA_LAYERS=final
export CKA_LOSS_TAU=0.0 CKA_LOSS_START_RATIO=0.0 CKA_SUBSET_QUERY_TOKENS=text CKA_GRAD_SCOPE=existing
export STAGE2_CKA_MARGIN=0.02 STAGE2_KL_WEIGHT=0.1 STAGE2_PROJ_AUX_RATIO=0.1 STAGE2_LLM_AUX_RATIO=0.1
export STAGE2_CKA_WARMUP_STEPS=100 STAGE2_CE_TOLERANCE=0.02 STAGE2_TEXT_KL_BUDGET=0.05
export STAGE2_REFERENCE_DEVICE=same STAGE2_LAMBDA_MAX=1.0 STAGE2_EMA_DECAY=0.99
export STAGE2_CE_GRADIENT_CALIBRATION_BATCHES=8 STAGE2_MAX_RETRIES=3 STAGE2_KL_CHUNK_SIZE=128
export STAGE2_AUX_BACKOFF=0.5 STAGE2_EPS=1e-8 STAGE2_PLATEAU_PATIENCE=3 STAGE2_PLATEAU_WINDOW=3 STAGE2_PLATEAU_MIN_DELTA=0.001
export REPORT_TO="${REPORT_TO:-wandb}" LOGGING_STEPS="${LOGGING_STEPS:-10}"

for mode in ce_only protected; do
    PRETRAIN_ADAPTER="${STAGE1_PAIR_ROOT}/${mode}/checkpoint-350/mm_projector.bin" \
    STAGE2_CONTINUATION="${mode}" \
    OUTPUT_DIR="${STAGE2_PAIR_ROOT}/${mode}" \
    RUN_NAME="qwen2.5-3b-pipeline-stage2-${mode}" \
    bash "${SCRIPT_DIR}/finetune_qwen2_5_3b_protected_cka.sh"
done
