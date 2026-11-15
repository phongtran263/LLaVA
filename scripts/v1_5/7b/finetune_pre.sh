#!/bin/bash
set -euo pipefail

# PRe, LLaVA-1.5/Vicuna-7B: stage 2 only, starting from the baseline projector.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
REPO_ROOT="${PWD}"
fail() { echo "$*" >&2; exit 1; }
(( $# == 0 )) || fail "Configure this recipe with environment variables; CLI overrides are not accepted."

DRY_RUN="${DRY_RUN:-0}"
CHECK_ALL_IMAGES="${CHECK_ALL_IMAGES:-1}"
[[ "$DRY_RUN" =~ ^[01]$ && "$CHECK_ALL_IMAGES" =~ ^[01]$ ]] || fail "DRY_RUN and CHECK_ALL_IMAGES must be 0 or 1."
if [[ -z "${TRAIN_ENV:-}" ]]; then
    if [[ -x "${CONDA_PREFIX:-}/bin/deepspeed" ]]; then
        TRAIN_ENV="${CONDA_PREFIX}"
    else
        TRAIN_ENV="${REPO_ROOT}/../conda/envs/llava1"
    fi
fi
[[ -x "${TRAIN_ENV}/bin/python" && -x "${TRAIN_ENV}/bin/deepspeed" ]] || fail "Set TRAIN_ENV to the LLaVA training conda environment."
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRETRAIN_CHECKPOINT_DIR="${PRETRAIN_CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/full-ft/baseline-7b/llava-pretrain}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/full-ft/pre-7b/llava-finetune}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/playground/data/llava_v1_5_mix665k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}/playground/data}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-lmsys/vicuna-7b-v1.5}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/scripts/zero2_b200.json}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
GPU_INCLUDE="${GPU_INCLUDE:-localhost:3}"
[[ "$GPU_INCLUDE" =~ ^localhost:[0-9]+(,[0-9]+)*$ ]] || fail "GPU_INCLUDE must be localhost:3 or localhost:0,1,..."
IFS=, read -r -a GPU_IDS <<< "${GPU_INCLUDE#localhost:}"
NUM_GPUS="${#GPU_IDS[@]}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
[[ "$PER_DEVICE_TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "PER_DEVICE_TRAIN_BATCH_SIZE must be positive."
(( 128 % (NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE) == 0 )) || fail "GPU count × microbatch must divide the paper's global batch 128."
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$((128 / (NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE)))}"
[[ "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "GRADIENT_ACCUMULATION_STEPS must be positive."
(( NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS == 128 )) || fail "This reproduction requires effective batch size 128."
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || fail "Output exists: $OUTPUT_DIR. Choose a fresh OUTPUT_DIR; this launcher starts a new stage-2 run."

"${TRAIN_ENV}/bin/python" - "$PRETRAIN_CHECKPOINT_DIR" "$DATA_PATH" "$IMAGE_FOLDER" "$CHECK_ALL_IMAGES" "$GPU_INCLUDE" <<'PY_PREFLIGHT'
import importlib.util
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import torch

checkpoint, data_path, image_root = map(Path, sys.argv[1:4])
config = json.loads((checkpoint / 'config.json').read_text())
expected = dict(hidden_size=4096, num_hidden_layers=32, mm_hidden_size=1024,
                mm_projector_type='mlp2x_gelu', mm_vision_select_layer=-2,
                mm_vision_tower='openai/clip-vit-large-patch14-336')
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f'Incompatible stage-1 checkpoint: {key}={config.get(key)!r}; expected {value!r}.')
state = torch.load(checkpoint / 'mm_projector.bin', map_location='cpu', weights_only=True)
expected_shapes = {'0.weight': (4096, 1024), '0.bias': (4096,),
                   '2.weight': (4096, 4096), '2.bias': (4096,)}
actual = {key.split('mm_projector.', 1)[-1]: tuple(value.shape) for key, value in state.items()}
if actual != expected_shapes:
    raise SystemExit(f'Unexpected stage-1 projector tensor shapes: {actual}')
for module in ('deepspeed', 'flash_attn'):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f'Missing training dependency: {module}')
expected_sha256 = 'ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8'
digest = hashlib.sha256()
with data_path.open('rb') as source:
    while chunk := source.read(8 * 1024 * 1024):
        digest.update(chunk)
if digest.hexdigest() != expected_sha256:
    raise SystemExit(f'Dataset SHA256 mismatch: expected {expected_sha256}, got {digest.hexdigest()}.')
data = json.loads(data_path.read_text())
if len(data) != 665298:
    raise SystemExit(f'Expected the complete LLaVA-1.5 mix665k (665298 samples), got {len(data)}.')
images = {item['image'] for item in data if 'image' in item}
counts = Counter(item.get('image', '<text>').split('/')[0] for item in data)
if sys.argv[4] == '1':
    print(f'Checking {len(images)} unique image paths...', flush=True)
    missing = []
    image_names = sorted(images)
    checked = 0
    def check_image(name):
        return name, (image_root / name).is_file()
    # Bound queued work while overlapping metadata reads on the data volume.
    with ThreadPoolExecutor(max_workers=16) as pool:
        for start in range(0, len(image_names), 1000):
            for name, exists in pool.map(check_image, image_names[start:start + 1000]):
                checked += 1
                if not exists:
                    missing.append(name)
                    if len(missing) <= 10:
                        print(f'Missing image: {name}', flush=True)
                if checked % 50000 == 0:
                    print(f'Checked {checked}/{len(images)} image paths; missing={len(missing)}', flush=True)
    if missing:
        raise SystemExit(f'{len(missing)} missing images; first examples: {missing[:10]}')
else:
    print('Full image-path check skipped (CHECK_ALL_IMAGES=0).')
ids = [int(x) for x in sys.argv[5].split(':')[1].split(',')]
if len(ids) != len(set(ids)):
    raise SystemExit('GPU_INCLUDE contains duplicate devices.')
if not torch.cuda.is_available() or max(ids) >= torch.cuda.device_count():
    raise SystemExit(f'Requested GPU IDs {ids} are unavailable.')
print(f'Preflight passed: {len(data)} samples; image/text prefixes={dict(counts)}')
print(f'Dataset SHA256: {digest.hexdigest()}')
print(f'Projector: {checkpoint / "mm_projector.bin"}')
PY_PREFLIGHT

CMD=(
    "${TRAIN_ENV}/bin/deepspeed" --include "$GPU_INCLUDE" --master_port "${MASTER_PORT:-29573}"
    llava/train/train_mem.py --deepspeed "$DEEPSPEED_CONFIG"
    --model_name_or_path "$MODEL_NAME_OR_PATH" --version v1
    --data_path "$DATA_PATH" --image_folder "$IMAGE_FOLDER" --train_data_fraction 1.0
    --vision_tower "$VISION_TOWER" --pretrain_mm_mlp_adapter "${PRETRAIN_CHECKPOINT_DIR}/mm_projector.bin"
    --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 --mm_vision_select_feature patch
    --mm_use_im_start_end False --mm_use_im_patch_token False --image_aspect_ratio pad
    --freeze_backbone False --tune_mm_mlp_adapter False --freeze_mm_mlp_adapter False
    --lora_enable False --bits 16 --cka_loss False --use_pcgrad False
    --protected_cka_refinement False --protected_stage2 False
    --pre_loss True --pre_loss_layer 16 --pre_loss_weight 0.5
    --group_by_modality_length True --bf16 True --tf32 True
    --output_dir "$OUTPUT_DIR" --seed "${SEED:-42}" --num_train_epochs 1
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" --per_device_eval_batch_size 4
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --eval_strategy no --save_strategy steps --save_steps 1000 --save_total_limit 2
    --optim adamw_torch_fused --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8
    --learning_rate 2e-5 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine
    --logging_steps 1 --model_max_length 2048 --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" --dataloader_prefetch_factor 4
    --dataloader_pin_memory True --dataloader_persistent_workers True
    --accelerator_config '{"non_blocking":true}' --lazy_preprocess True
    --report_to "${REPORT_TO:-none}" --run_name "${RUN_NAME:-llava15-vicuna7b-pre-stage2-seed${SEED:-42}}"
)
printf 'Training environment: %s\nGlobal batch: %s GPUs × %s microbatch × %s accumulation = 128\n' "$TRAIN_ENV" "$NUM_GPUS" "$PER_DEVICE_TRAIN_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS"
printf 'Memory config: %s; checkpointing=%s; CUDA allocator=%s\n' "$DEEPSPEED_CONFIG" "$GRADIENT_CHECKPOINTING" "$PYTORCH_CUDA_ALLOC_CONF"
printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
    exit 0
fi
exec "${CMD[@]}"
