"""Persistent launcher for the frozen, verified ROSS stage-2 comparison."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

run = Path(__file__).resolve().parent
repo = run.parent.parent
manifest = json.loads((run / 'manifest.json').read_text())
source = run / manifest['source_directory']
assert manifest['validation']['cpu_tests_passed'] == 75
assert manifest['validation']['all_images_exist']
assert manifest['validation']['real_gpu_smoke_passed']
assert manifest['validation']['full_7b_memory_benchmark_passed']
for relative, expected in manifest['source_sha256'].items():
    path = source / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
assert json.loads((source / 'scripts/zero2_b200.json').read_text())['zero_optimization']['stage'] == 2
assert not Path(manifest['output_dir']).exists(), 'Fresh ROSS output already exists'
protocol = manifest['protocol']
overrides = {
    'TRAIN_ENV': str(repo.parent / 'conda/envs/llava1'),
    'PRETRAIN_CHECKPOINT_DIR': str(repo / 'checkpoints/full-ft/baseline-7b/llava-pretrain'),
    'OUTPUT_DIR': manifest['output_dir'],
    'DATA_PATH': str(repo / 'playground/data/llava_v1_5_mix665k.json'),
    'IMAGE_FOLDER': str(repo / 'playground/data'),
    'MODEL_NAME_OR_PATH': 'lmsys/vicuna-7b-v1.5',
    'VISION_TOWER': 'openai/clip-vit-large-patch14-336',
    'ROSS_VAE_PATH': str(repo / 'checkpoints/auxiliary/ross-flux-vae'),
    'ROSS_EXTRA_PYTHONPATH': str(repo / '.deps/ross'),
    'DEEPSPEED_CONFIG': str(source / 'scripts/zero2_b200.json'),
    'GRADIENT_CHECKPOINTING': str(protocol['gradient_checkpointing']),
    'GPU_INCLUDE': 'localhost:3', 'MASTER_PORT': '29574',
    'PER_DEVICE_TRAIN_BATCH_SIZE': str(protocol['microbatch']),
    'GRADIENT_ACCUMULATION_STEPS': str(protocol['accumulation']),
    'DATALOADER_NUM_WORKERS': '8', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
    'CHECK_ALL_IMAGES': '0', 'DRY_RUN': '0', 'REPORT_TO': 'none', 'SEED': '42',
    'PYTHONPATH': str(source), 'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'TOKENIZERS_PARALLELISM': 'false', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'HF_HOME': str(repo.parent / 'cache/huggingface'),
    'TRANSFORMERS_CACHE': str(repo.parent / 'cache/huggingface/transformers'),
    'RUN_NAME': 'llava15-vicuna7b-ross-stage2-seed42',
}
env = os.environ.copy()
env.update(overrides)
env.pop('CUDA_VISIBLE_DEVICES', None)
command = ['bash', str(source / 'scripts/v1_5/7b/finetune_ross.sh')]
record = dict(driver_pid=os.getpid(), command=command, environment=overrides,
    started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='starting')
(run / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL)
record.update(launcher_pid=child.pid, status='running')
(run / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
code = child.wait()
record.update(exit_code=code, status='finished' if code == 0 else 'failed',
    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(run / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
print('ROSS_ZERO2_STAGE2_EXIT', json.dumps(record), flush=True)
sys.exit(code)
