"""Run the two layer-31 experiments after the current layer-16 pair succeeds."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
PREDECESSOR_PID = 4034756
PREDECESSOR_START_TICKS = "1393985158"
PREDECESSOR_LOG = REPO / "logs/vicuna7b-mid16-cka-Nl4IAJxe/train.log"
OUTPUT_ROOT = REPO / "checkpoints/full-ft/vicuna-7b-intermediate-cka"
SETTINGS = ("intermediate_only", "projector_intermediate")


def report(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def predecessor_running():
    try:
        fields = Path(f"/proc/{PREDECESSOR_PID}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == PREDECESSOR_START_TICKS and fields[0] != "Z"


def require_success():
    completed = set(PREDECESSOR_LOG.read_text().splitlines())
    for setting in SETTINGS:
        if f"Completed {setting}." not in completed:
            raise RuntimeError(f"Predecessor did not finish {setting}; layer-31 jobs will not start.")
        output = OUTPUT_ROOT / "layer-16" / setting / "llava-finetune"
        if not (output / "trainer_state.json").is_file():
            raise RuntimeError(f"Missing completed trainer state: {output}")
        weights = list(output.glob("model*.safetensors")) + list(output.glob("pytorch_model*.bin"))
        if not weights or any(path.stat().st_size == 0 for path in weights):
            raise RuntimeError(f"Missing or empty final model weights: {output}")


def gpu_busy():
    result = subprocess.run(
        ["nvidia-smi", "-i", "3", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def main():
    report(f"Queued layer31: waiting for layer16 runner {PREDECESSOR_PID}; GPU3 only.")
    while predecessor_running():
        time.sleep(30)
    require_success()
    report("Both layer16 runs completed successfully; waiting for GPU3 to be idle.")
    while gpu_busy():
        time.sleep(30)
    env = os.environ.copy()
    env.update(
        GPU_INCLUDE="localhost:3", CKA_LAYER="31", CKA_WEIGHT="0.1",
        SETTINGS=" ".join(SETTINGS), OUTPUT_ROOT=str(OUTPUT_ROOT),
        BASELINE_PRETRAIN_DIR=str(REPO / "checkpoints/full-ft/baseline-7b/llava-pretrain"),
        CKA_PRETRAIN_DIR=str(REPO / "checkpoints/full-ft/cka-proj-7b/llava-pretrain"),
        PER_DEVICE_TRAIN_BATCH_SIZE="4", GRADIENT_ACCUMULATION_STEPS="32",
        FINETUNE_GRADIENT_CHECKPOINTING="True", DRY_RUN="0",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    os.chdir(REPO)
    report("Starting intermediate_only then projector_intermediate at layer31.")
    os.execvpe("bash", ["bash", "scripts/v1_5/7b/finetune_vicuna_7b_intermediate_cka.sh"], env)


if __name__ == "__main__":
    main()
