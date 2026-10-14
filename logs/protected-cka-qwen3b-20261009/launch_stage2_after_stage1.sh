#!/bin/bash
set -euo pipefail
cd /home/jovyan/dowser-lora-datavol-1/phong/LLaVA

# This controller is bound to the already-launched Stage-1 pair, not pgrep.
# A reused PID cannot silently become the dependency for this experiment.
stage1_pid=547860
stage1_start_ticks=1059352592
printf 'Waiting for Stage-1 pair PID %s (start ticks %s).\n' "${stage1_pid}" "${stage1_start_ticks}"
while [[ -r "/proc/${stage1_pid}/stat" ]]; do
    current_ticks=$(awk '{print $22}' "/proc/${stage1_pid}/stat" 2>/dev/null || true)
    [[ "${current_ticks}" == "${stage1_start_ticks}" ]] || break
    sleep 30
done

# The Stage-2 launcher performs the full source/gate/signature checks below.
# Also require completed Stage-1 final exports before handing over the GPU.
for mode in ce_only protected; do
    output="checkpoints/pretrain-protected-cka/qwen2.5-3b/continuation-350/${mode}"
    for artifact in mm_projector.bin trainer_state.json config.json; do
        if [[ ! -s "${output}/${artifact}" ]]; then
            printf 'Stage-1 %s final export is incomplete: %s\n' "${mode}" "${artifact}" >&2
            exit 1
        fi
    done
done

printf 'Stage-1 parent ended. Waiting for physical GPU 3 to be idle.\n'
while true; do
    gpu_jobs=$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z "${gpu_jobs//[[:space:]]/}" ]] && break
    sleep 30
done

available_bytes=$(df --output=avail -B1 . | tail -n 1 | tr -d '[:space:]')
if [[ ! "${available_bytes}" =~ ^[0-9]+$ ]] || (( available_bytes < 300 * 1024 * 1024 * 1024 )); then
    printf 'Stage-2 launch requires at least 300 GiB free for this paired run; available bytes: %s\n' "${available_bytes}" >&2
    exit 1
fi

printf 'Checking Stage-1 acceptance and launching the matched Stage-2 pair.\n'
exec env \
    PYTHON_BIN=/home/jovyan/dowser-lora-datavol-1/phong/conda/envs/llava1/bin/python \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false GPU_ID=3 REPORT_TO=wandb \
    STAGE1_PAIR_ROOT=./checkpoints/pretrain-protected-cka/qwen2.5-3b/continuation-350 \
    STAGE2_PAIR_ROOT=./checkpoints/finetune-protected-cka/qwen2.5-3b/paired-protected-cka \
    bash scripts/v1_5/7b/finetune_qwen2_5_3b_protected_pipeline_pair.sh
