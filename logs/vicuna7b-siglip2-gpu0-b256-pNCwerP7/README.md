# Vicuna-7B + SigLIP2 GPU 0 restart

The initial global-128 Stage-1 run was interrupted at about 33 updates at the
user's request. Its log and empty/incomplete output directory were preserved.
This restart uses a new checkpoint root and follows the user's batch targets:

| Stage | Microbatch | Accumulation | Global batch | Checkpointing |
|---|---:|---:|---:|---|
| Stage 1, baseline and projector CKA | 16 | 16 | 256 | off |
| Stage 2, all three settings | 4 | 32 | 128 | off |

Every job uses GPU 0 only, ZeRO-2 (`scripts/zero2_b200.json`), BF16,
Flash Attention 2, fused AdamW, 8 DataLoader workers, prefetch factor 4,
persistent workers, nonblocking transfers, and 8 OMP/MKL threads. Training
seed and isolated projector-initialization seed are both 42. CKA weight is
0.1 per enabled term; PCGrad and norm cap are off. The projector+final Stage 2
job reuses the projector-CKA Stage-1 adapter, just like the projector-only job.

Batch-size evidence, synthetic maximum expanded length 2048:

- Stage 1 CKA microbatch 16: ~157.3 GiB peak allocated; 32: OOM. Therefore
  16 is the largest tested safe microbatch. Accumulation changed to 16 to
  achieve global batch 256 without changing activation memory.
- Stage 2 projector+final CKA microbatch 2, checkpointing off: ~139.6 GiB
  peak allocated, projected 24.6 s/global-128 update.
- Stage 2 projector+final CKA microbatch 4, checkpointing off, isolated
  fresh process: ~161.2 GiB peak allocated, projected 22.7 s/update. On the
  178.35-GiB B200 this leaves about 17 GiB headroom. The switched-case
  benchmark had a numerical artifact, so its timing was not used.

Synthetic timings are only a comparison of execution settings; actual
training throughput can differ with real sequence lengths and image content.
No benchmark weights were used as training checkpoints. `train.log` is the
production log; the checkpoint root is
`checkpoints/full-ft/vicuna7b-siglip2-seed42-b256`.

Production restarted on GPU 0 with W&B run `ak2pk8m5` (Stage 1 baseline).
The first two real-data optimizer steps had finite losses and gradients, no
OOM; GPU utilization reached 99%, with about 70 GiB allocated. The original
run/checkpoint root was not overwritten. `training-source.tar.gz` records the
code used at restart.
