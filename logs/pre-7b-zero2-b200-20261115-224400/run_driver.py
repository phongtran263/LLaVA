"""Run the frozen, benchmarked B200 ZeRO-2 stage-2 recipe."""
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
source = run / manifest.get('source_directory', 'source')
assert manifest['validation']['all_images_exist']
assert manifest['validation'].get('gas32_gpu_gradient_reference') == 'passed'
assert json.loads((source / 'scripts/zero2_b200.json').read_text())['zero_optimization']['stage'] == 2
overrides = {
    'TRAIN_ENV': str(repo.parent / 'conda/envs/llava1'),
    'PRETRAIN_CHECKPOINT_DIR': str(repo / 'checkpoints/full-ft/baseline-7b/llava-pretrain'),
    'OUTPUT_DIR': str(repo / 'checkpoints/full-ft/pre-7b/llava-finetune'),
    'DATA_PATH': str(repo / 'playground/data/llava_v1_5_mix665k.json'),
    'IMAGE_FOLDER': str(repo / 'playground/data'),
    'MODEL_NAME_OR_PATH': 'lmsys/vicuna-7b-v1.5',
    'VISION_TOWER': 'openai/clip-vit-large-patch14-336',
    'DEEPSPEED_CONFIG': str(source / 'scripts/zero2_b200.json'),
    'GRADIENT_CHECKPOINTING': 'False',
    'GPU_INCLUDE': 'localhost:3', 'MASTER_PORT': '29573',
    'PER_DEVICE_TRAIN_BATCH_SIZE': '4', 'GRADIENT_ACCUMULATION_STEPS': '32',
    'DATALOADER_NUM_WORKERS': '8', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
    'CHECK_ALL_IMAGES': '0', 'DRY_RUN': '0', 'REPORT_TO': 'none', 'SEED': '42',
    'PYTHONPATH': str(source), 'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'TOKENIZERS_PARALLELISM': 'false', 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
}
env = os.environ.copy()
env.update(overrides)
command = ['bash', str(source / 'scripts/v1_5/7b/finetune_pre.sh')]
record = {'driver_pid': os.getpid(), 'command': command, 'environment': overrides,
          'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL)
record['launcher_pid'] = child.pid
(run / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
code = child.wait()
record.update(exit_code=code, finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(run / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
print('PRE_ZERO2_STAGE2_EXIT', json.dumps(record), flush=True)
sys.exit(code)
