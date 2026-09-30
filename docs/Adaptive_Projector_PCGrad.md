# Adaptive projector PCGrad

This repository has an opt-in two-stage launcher for CE-priority, one-sided
PCGrad on the multimodal projector.  The feature is separate from the legacy
CKA/VSP path: the launchers explicitly set `cka_loss=False` and disable every
legacy PCGrad/norm-cap switch, so the adaptive branch owns the only projector
CKA objective.

## Files

- `scripts/adaptive_projector_pcgrad/stage1_defaults.json`
- `scripts/adaptive_projector_pcgrad/stage2_defaults.json`
- `scripts/v1_5/7b/pretrain_qwen2_5_adaptive_pcgrad.sh`
- `scripts/v1_5/7b/finetune_qwen2_5_adaptive_pcgrad.sh`
- `scripts/v1_5/7b/train_qwen2_5_adaptive_pcgrad.sh`

The two JSON files intentionally have the same numerical defaults but different
`stage` values.  `planned_optimizer_steps` is not hard-coded in either file;
the Trainer fills it after resolving the dataset, world size, batch size,
gradient accumulation and `max_steps` for that stage.

## Run a bounded smoke test

The chained launcher defaults to two optimizer steps in each stage and writes
under a `smoke` directory:

```bash
conda activate <training-environment>
bash scripts/v1_5/7b/train_qwen2_5_adaptive_pcgrad.sh
```

Change the bound without enabling a full run:

```bash
SMOKE_MAX_STEPS=5 GPU_INCLUDE=localhost:0,1 \
  bash scripts/v1_5/7b/train_qwen2_5_adaptive_pcgrad.sh
```

No launcher starts a full epoch by default.  A full two-stage run requires the
explicit opt-in below and writes under a separate `full` directory:

```bash
FULL_TRAIN=True GPU_INCLUDE=localhost:0,1 \
  bash scripts/v1_5/7b/train_qwen2_5_adaptive_pcgrad.sh
```

The default model for both stages is
`Qwen/Qwen2.5-0.5B-Instruct`.  Override it once on the chained command so stage
1 metadata and stage 2 loading cannot silently refer to different base models:

```bash
FULL_TRAIN=True \
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct \
RUN_NAME=qwen2.5-7b-adaptive-projector-pcgrad \
STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE=16 \
STAGE1_GRADIENT_ACCUMULATION_STEPS=8 \
STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE=4 \
STAGE2_GRADIENT_ACCUMULATION_STEPS=32 \
GPU_INCLUDE=localhost:0,1 \
  bash scripts/v1_5/7b/train_qwen2_5_adaptive_pcgrad.sh
```

Stage 2 is launched only if stage 1 exits successfully and produces both a
non-empty `mm_projector.bin` and
`adaptive_projector_pcgrad_metadata.json`.  Before training, stage 2 validates
the metadata stage, base-model identifier, resolved stage-1 config, projector
signature and stored projector hash.  The training integration performs the
actual name/shape/hash check against the loaded module.

## Run or resume one stage

The individual launchers use the same smoke/full guard:

```bash
# Stage 1 smoke
bash scripts/v1_5/7b/pretrain_qwen2_5_adaptive_pcgrad.sh

# Resume within stage 1 (controller/optimizer/scheduler state must be present)
FULL_TRAIN=True RESUME_FROM_CHECKPOINT=/path/to/stage1/checkpoint-N \
  bash scripts/v1_5/7b/pretrain_qwen2_5_adaptive_pcgrad.sh

# Stage 2 from a validated stage-1 artifact
PRETRAIN_ADAPTER=/path/to/stage1/mm_projector.bin \
  bash scripts/v1_5/7b/finetune_qwen2_5_adaptive_pcgrad.sh
```

Stage-2 always reads provenance metadata from the same directory as
`PRETRAIN_ADAPTER`; a separate metadata path cannot be supplied accidentally.

Do not pass a stage-1 controller checkpoint as a stage-2 resume checkpoint.
Stage 2 starts a fresh optimizer, scheduler, EMA state and successful-step
counter.  `RESUME_FROM_CHECKPOINT` is only for resuming within the same stage.
The stage config and planned optimizer-step horizon must also remain identical;
in particular, a two-step smoke checkpoint cannot be resumed as a full run.

Useful environment overrides include `PIPELINE_ROOT`, `OUTPUT_DIR`,
`GPU_INCLUDE`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
`GRADIENT_ACCUMULATION_STEPS`, `DATALOADER_NUM_WORKERS`,
`GRADIENT_CHECKPOINTING`, `LOGGING_STEPS`, `REPORT_TO`,
`DEEPSPEED_CONFIG`, `ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE`, and
`ADAPTIVE_PCGRAD_PROFILE`.

The chained launcher additionally accepts independent
`STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE`,
`STAGE1_GRADIENT_ACCUMULATION_STEPS`,
`STAGE1_ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE`,
`STAGE1_GRADIENT_CHECKPOINTING`, `STAGE1_LOGGING_STEPS`,
`STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE`, and
`STAGE2_GRADIENT_ACCUMULATION_STEPS`,
`STAGE2_ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE`,
`STAGE2_GRADIENT_CHECKPOINTING`, `STAGE2_LOGGING_STEPS` values.

For independently tuned stage settings, copy the matching JSON, keep its
`stage` value unchanged, edit the numerical fields, and pass it only to that
stage:

```bash
ADAPTIVE_PCGRAD_CONFIG=/path/to/my_stage1.json \
  bash scripts/v1_5/7b/pretrain_qwen2_5_adaptive_pcgrad.sh
```

For the chained launcher, use `STAGE1_ADAPTIVE_PCGRAD_CONFIG` and
`STAGE2_ADAPTIVE_PCGRAD_CONFIG`; their values are passed to only the matching
stage.

`lambda_max=1.0` is an upper bound on the adaptive gradient coefficient, not a
fixed scalar-loss weight.  Do not add `cka_loss_projector_weight` or another
legacy CKA weight to these commands.

## Compute and memory behavior

The optimized path performs one normal VLM forward and one CE backward per
microbatch.  It captures detached pre-projector vision features and replays
only the deterministic projector in a plain sidecar.  Projector CKA uses FP32
Gram matrices per image/crop with CUDA TF32 disabled locally, obtains gradients only for sidecar projector
parameters, and immediately accumulates detached FP32 gradient sums.  It does
not retain the decoder graph for CKA, repeat the vision encoder/decoder forward,
or materialize all decoder hidden states/attention matrices.

The launchers default to `ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE=4`, avoiding one tiny
projector replay/autograd call per image while keeping the temporary Gram batch
bounded. Set it to `1` to minimize peak CKA memory. A larger value can improve
GPU utilization but increases the temporary Gram-matrix footprint; measure it
on the target hardware rather than assuming it is faster.
The CKA kernel evaluates the equivalent scalar Gram dot/norm form instead of
materializing normalized Gram copies. Boundary statistics are reduced before
the full projector vector, so an effective batch with no valid observations
skips that large collective; local shard and replay buffers are reused across
steps.
Set `ADAPTIVE_PCGRAD_PROFILE=True` to log optimizer-step time and peak allocated
memory.  Compare it with the identical recipe and batch shape with the adaptive
feature disabled; no fixed overhead percentage is claimed without that
measurement.

For throughput tuning, first benchmark `ADAPTIVE_PCGRAD_CKA_CHUNK_SIZE=8`,
`16`, then the full per-device image batch, and keep the largest value that
fits memory.  This reduces projector replay/autograd launches without changing
the global SUM/count objective.  If stage 2 fits without activation
checkpointing, `GRADIENT_CHECKPOINTING=False` removes decoder recomputation;
for full runs, increasing `LOGGING_STEPS` from `1` to `10` or `50` can also
remove avoidable W&B/console overhead.  Keep `ADAPTIVE_PCGRAD_PROFILE=False`
outside short measurements because profiling intentionally synchronizes CUDA.

## Backend support boundary

The supplied launchers intentionally use BF16 DeepSpeed ZeRO-2 without CPU or
NVMe offload (`scripts/zero2.json`) and a deterministic Linear/GELU projector.
They fail fast unless the installed stack matches the repository pins: PyTorch
2.7.1 (a local CUDA build suffix is allowed), Transformers 4.51.3, Tokenizers
0.21.2, Accelerate 1.6.0 and DeepSpeed 0.18.9.
The adaptive integration must fail fast if its runtime validation cannot access
and replace the real optimizer gradient storage.

The optimized replay path rejects projector dropout, BatchNorm/mutable buffers,
tied or parametrized weights, and projector hooks unless exact replay has been
separately implemented and tested.  ZeRO-3, FSDP, optimizer/parameter offload,
pipeline/MoE layouts, and unverified FP16 scaler paths are not enabled by these
launchers.  Use the legacy path, or add backend equivalence tests, rather than
silently falling back to CE-only behavior.

Gradient surgery changes only the raw projector gradient. Stage-2 decoder raw
gradients remain the CE gradients before clipping; global gradient clipping and
AdamW state can still make the final joint update differ from a CE-only run.

## Verification status

The CPU suite covers controller math/state, per-image FP32 CKA, replay/direct
gradient equivalence for linear, MLP and coupling projectors, stage-1 frozen
decoder flow, unchanged stage-2 decoder raw gradients, BF16 replay, accumulation
lifecycle, same-stage resume, ZeRO-2 shard/padding mapping and a two-rank Gloo
global-mean reference:

```bash
python -m unittest -v \
  tests.test_projector_pcgrad \
  tests.test_pretrain_projector_pcgrad \
  tests.test_adaptive_projector_pcgrad_backend
```

The two-rank test uses real collectives with a CPU representation of the pinned
ZeRO-2 shard layout; it is not a substitute for a two-GPU DeepSpeed smoke run.
A real one-GPU B200 run of the chained launcher was also completed with two
optimizer steps per stage, batch size one, BF16 ZeRO-2, and W&B disabled. Both
stages completed, each controller recorded two successful steps, stage-2
handoff validation passed, and the stage-2 projector hash changed from its
stage-1 parent. This validates the concrete single-rank engine path, but not
two-rank ZeRO-2 equivalence. No full training or GPU timing benchmark is run
automatically. Before a costly experiment, run the default two-step chained
smoke launcher on the target node, then compare
`ADAPTIVE_PCGRAD_PROFILE=True` against the identical CE-only recipe.
