"""Run the four authorized CKA experiments on GPU 3, sequentially."""
import json
import os
from pathlib import Path
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO / 'checkpoints/full-ft/vicuna-7b-intermediate-cka-b200'


def main():
    # Synthetic CKA optimizer updates were non-finite, so their timing is not
    # accepted as a production result. This launch is monitored on real data.
    # Never attach to old runs or overwrite an existing destination.
    for layer in (16, 31):
        for setting in ('intermediate_only', 'projector_intermediate'):
            target = OUTPUT_ROOT / f'layer-{layer}' / setting / 'llava-finetune'
            if target.exists() or target.is_symlink():
                raise RuntimeError(f'Output already exists: {target}')
    with tarfile.open(LOG_DIR / 'training-source.tar.gz', 'x:gz') as archive:
        for relative in ('llava', 'scripts/v1_5/7b/finetune_qwen2_5.sh',
                         'scripts/v1_5/7b/finetune_vicuna_7b_intermediate_cka.sh',
                         'scripts/zero2_b200.json', 'pyproject.toml'):
            archive.add(REPO / relative, arcname=relative,
                        filter=lambda member: None if '__pycache__' in member.name else member)
    env = dict(os.environ)
    env.update(
        GPU_INCLUDE='localhost:3', MASTER_PORT='29576',
        OUTPUT_ROOT=str(OUTPUT_ROOT), SEED='42',
        SETTINGS='intermediate_only projector_intermediate', CKA_WEIGHT='0.1',
        PER_DEVICE_TRAIN_BATCH_SIZE='4', GRADIENT_ACCUMULATION_STEPS='32',
        FINETUNE_GRADIENT_CHECKPOINTING='False', OPTIM='adamw_torch_fused',
        DEEPSPEED_CONFIG='./scripts/zero2_b200.json',
        DATALOADER_NUM_WORKERS='8', DATALOADER_PREFETCH_FACTOR='4',
        DATALOADER_PERSISTENT_WORKERS='True', ACCELERATOR_CONFIG='{"non_blocking":true}',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
        LOGGING_STEPS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        WANDB_RUN_GROUP='vicuna-7b-intermediate-cka-b200',
    )
    print('B200 sweep configuration: ' + json.dumps({k: env[k] for k in (
        'OUTPUT_ROOT', 'GPU_INCLUDE', 'OPTIM', 'DEEPSPEED_CONFIG',
        'PER_DEVICE_TRAIN_BATCH_SIZE', 'GRADIENT_ACCUMULATION_STEPS',
        'FINETUNE_GRADIENT_CHECKPOINTING')}), flush=True)
    for layer in (16, 31):
        occupied = subprocess.check_output([
            'nvidia-smi', '-i', '3', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        if occupied:
            raise RuntimeError(f'GPU 3 is occupied; no process was stopped: {occupied}')
        env['CKA_LAYER'] = str(layer)
        print(f'Starting layer {layer} pair', flush=True)
        subprocess.run(['bash', 'scripts/v1_5/7b/finetune_vicuna_7b_intermediate_cka.sh'],
                       cwd=REPO, env=env, check=True)
        print(f'Completed layer {layer} pair', flush=True)
    print('ALL FOUR EXPERIMENTS COMPLETED', flush=True)


if __name__ == '__main__':
    main()
