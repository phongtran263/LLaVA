#!/bin/bash
set -euo pipefail

# Stage-2-only ROSS under the same controls as the local PRe comparison.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
REPO_ROOT="${PWD}"
fail() { echo "$*" >&2; exit 1; }
(( $# == 0 )) || fail "Configure this recipe with environment variables; CLI overrides are not accepted."
DRY_RUN="${DRY_RUN:-0}"
[[ "$DRY_RUN" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1."
if [[ -z "${TRAIN_ENV:-}" ]]; then
    if [[ -x "${CONDA_PREFIX:-}/bin/deepspeed" ]]; then
        TRAIN_ENV="${CONDA_PREFIX}"
    else
        TRAIN_ENV="${REPO_ROOT}/../conda/envs/llava1"
    fi
fi
[[ -x "${TRAIN_ENV}/bin/python" && -x "${TRAIN_ENV}/bin/deepspeed" ]] || fail "Set TRAIN_ENV to the LLaVA training conda environment."
export TRAIN_ENV
export PATH="${TRAIN_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROSS_EXTRA_PYTHONPATH="${ROSS_EXTRA_PYTHONPATH-${REPO_ROOT}/.deps/ross}"
if [[ -n "$ROSS_EXTRA_PYTHONPATH" ]]; then
    export PYTHONPATH="${ROSS_EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/full-ft/ross-7b/llava-finetune}"
export RUN_NAME="${RUN_NAME:-llava15-vicuna7b-ross-stage2-seed${SEED:-42}}"
export MASTER_PORT="${MASTER_PORT:-29574}"
ROSS_VAE_PATH="${ROSS_VAE_PATH:-${REPO_ROOT}/checkpoints/auxiliary/ross-flux-vae}"
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || fail "Output exists: $OUTPUT_DIR. Choose a fresh OUTPUT_DIR."

"${TRAIN_ENV}/bin/python" - "$ROSS_VAE_PATH" <<'PY_ROSS_PREFLIGHT'
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import sys

for module in ('diffusers', 'timm', 'einops'):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f'Missing ROSS dependency: {module}. Prepare the ROSS dependencies before launching.')
from diffusers import AutoencoderKL
vae = Path(sys.argv[1])
try:
    config = json.loads((vae / 'config.json').read_text())
except (OSError, ValueError) as exc:
    raise SystemExit(f'Cannot read the local FLUX VAE config at {vae}: {exc}')
if config.get('_class_name') != 'AutoencoderKL' or config.get('latent_channels') != 16:
    raise SystemExit('ROSS requires the FLUX AutoencoderKL with 16 latent channels.')
for key, expected in (('shift_factor', 0.1159), ('scaling_factor', 0.3611)):
    value = config.get(key)
    if not isinstance(value, (int, float)) or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-6):
        raise SystemExit(f'FLUX VAE {key}: expected {expected}, got {value!r}.')
weights = vae / 'diffusion_pytorch_model.safetensors'
expected_size = 167666902
expected_sha256 = 'f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3'
if not weights.is_file() or weights.stat().st_size != expected_size:
    raise SystemExit(f'Expected FLUX VAE weights of {expected_size} bytes: {weights}')
digest = hashlib.sha256()
with weights.open('rb') as source:
    while chunk := source.read(8 * 1024 * 1024):
        digest.update(chunk)
if digest.hexdigest() != expected_sha256:
    raise SystemExit(f'FLUX VAE SHA256 mismatch: expected {expected_sha256}, got {digest.hexdigest()}.')
print(f'ROSS VAE preflight passed: {vae}', flush=True)
print(f'ROSS VAE SHA256: {digest.hexdigest()}', flush=True)
PY_ROSS_PREFLIGHT

COMMAND_FILE="$(mktemp "${TMPDIR:-/tmp}/ross-command.XXXXXX")"
trap 'rm -f -- "$COMMAND_FILE"' EXIT
# Reuse the full canonical data/projector checks and the current shared profile.
# Suppress only the intermediate PRe command; the final ROSS command is below.
DRY_RUN=1 bash "${SCRIPT_DIR}/finetune_pre.sh" | \
    "${TRAIN_ENV}/bin/python" -u -c '
from pathlib import Path
import sys
path = Path(sys.argv[1])
for line in sys.stdin:
    if line.startswith("Command:"):
        path.write_text(line[len("Command:"):].strip())
    else:
        print(line, end="", flush=True)
' "$COMMAND_FILE"

"${TRAIN_ENV}/bin/python" - "$COMMAND_FILE" "$ROSS_VAE_PATH" "$DRY_RUN" <<'PY_ROSS_LAUNCH'
import os
from pathlib import Path
import shlex
import sys

command = shlex.split(Path(sys.argv[1]).read_text())
if not command:
    raise SystemExit('The shared launcher did not produce a training command.')
for flag in ('--pre_loss_layer', '--pre_loss_weight'):
    if flag in command:
        index = command.index(flag)
        del command[index:index + 2]
command[command.index('--pre_loss') + 1] = 'False'
command.extend(['--ross_enable', 'True', '--ross_vae_path', sys.argv[2], '--ross_loss_weight', '1.0'])

# TrainingArguments can initialize CUDA while parsing. Isolate that process
# after the shared preflight has checked the physical device identifiers.
previous_visibility = os.environ.get('CUDA_VISIBLE_DEVICES')
validation_visibility = command[command.index('--include') + 1].split(':', 1)[1]
os.environ['CUDA_VISIBLE_DEVICES'] = validation_visibility

# Validate every rendered training argument against this installed framework.
from transformers import HfArgumentParser
from llava.train.train import ModelArguments, DataArguments, TrainingArguments
training_start = command.index('llava/train/train_mem.py') + 1
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=command[training_start:])
if not model_args.ross_enable or model_args.pre_loss or model_args.cka_loss:
    raise SystemExit('Expected ROSS enabled with PRe and CKA disabled.')
print('ROSS final argument-parser validation passed.', flush=True)
print(f'Argument-parser CUDA_VISIBLE_DEVICES={validation_visibility}', flush=True)
print('Command:', shlex.join(command), flush=True)
if sys.argv[3] != '1':
    # DeepSpeed --include uses physical slots. Start its fresh process with the
    # original visibility so its own resource discovery can see those slots.
    if previous_visibility is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = previous_visibility
    os.execvpe(command[0], command, os.environ)
PY_ROSS_LAUNCH
