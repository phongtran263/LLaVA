"""Run the frozen stage-2 recipe and preserve its exit status."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

run = Path(__file__).resolve().parent
repo = run.parent.parent
manifest = json.loads((run / 'manifest.json').read_text())
if not manifest['validation'].get('all_images_exist'):
    raise RuntimeError('The complete image preflight must pass before launch.')
source = run / manifest.get('source_directory', 'source')
env = os.environ.copy()
overrides = {
    'TRAIN_ENV': str(repo.parent / 'conda/envs/llava1'),
    'PRETRAIN_CHECKPOINT_DIR': str(repo / 'checkpoints/full-ft/baseline-7b/llava-pretrain'),
    'OUTPUT_DIR': str(repo / 'checkpoints/full-ft/pre-7b/llava-finetune'),
    'DATA_PATH': str(repo / 'playground/data/llava_v1_5_mix665k.json'),
    'IMAGE_FOLDER': str(repo / 'playground/data'),
    'GPU_INCLUDE': 'localhost:3',
    'MASTER_PORT': '29573',
    'PER_DEVICE_TRAIN_BATCH_SIZE': '4',
    'GRADIENT_ACCUMULATION_STEPS': '32',
    'DATALOADER_NUM_WORKERS': '4',
    'CHECK_ALL_IMAGES': '0',
    'DRY_RUN': '0',
    'REPORT_TO': 'none',
    'SEED': '42',
    'PYTHONPATH': str(source),
    'PYTHONNOUSERSITE': '1',
    'PYTHONUNBUFFERED': '1',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'TOKENIZERS_PARALLELISM': 'false',
    'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',
}
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
print('PRE_STAGE2_EXIT', json.dumps(record), flush=True)
sys.exit(code)
