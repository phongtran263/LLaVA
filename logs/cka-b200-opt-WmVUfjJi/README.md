# Vicuna-7B CKA B200 restart — 2026-11-17

The user authorized interrupting GPU 3. No other GPU was used.
The old layer-31 queue and layer-16 training process group were stopped.
Old outputs/logs remain intact; the interrupted run had 186 of 5197 updates,
but no resumable checkpoint. The optimized run starts again from Stage 1.

## Active experiment queue

- Driver PID at launch: 4089009 (`run_sweep.py`).
- Log: `train.log` in this directory.
- First W&B run: https://wandb.ai/phongtran-kaist/huggingface/runs/zh9suut9
- Output root: `checkpoints/full-ft/vicuna-7b-intermediate-cka-b200`.
- Order: layer 16 intermediate-only, layer 16 projector+intermediate,
  layer 31 intermediate-only, layer 31 projector+intermediate.
- Parents remain `baseline-7b/llava-pretrain` and
  `cka-proj-7b/llava-pretrain`, respectively, under `checkpoints/full-ft`.
- Anchor V, weight 0.1 per enabled term, no PCGrad/norm cap, no extra final
  target, no changed sequence length or dataset. Seed 42, global batch 128.

## Execution changes

| Option | Interrupted run | Restart |
|---|---|---|
| Microbatch × accumulation | 4 × 32 | 4 × 32 |
| Gradient checkpointing | True | False |
| Optimizer | adamw_torch | adamw_torch_fused |
| DeepSpeed config | zero2.json | zero2_b200.json |
| OMP/MKL threads | 1 | 8 |
| DataLoader workers | 8 | 8 |
| Prefetch/persistent workers | default / False | 4 / True |
| Nonblocking transfers | False | True |

The optimizer remains AdamW with unchanged LR, betas, epsilon, and weight
decay. Fused kernels can introduce small numerical differences; bitwise
identity is not promised. Generic launcher defaults are unchanged; these
execution options are opt-in environment variables.

## Validation and timing

- 27 CPU launcher tests passed, shell syntax and `git diff --check` passed.
- Real-data steps 1 and 2 match the old reported loss and component losses;
  grad norms differ slightly, as expected from changed execution kernels.
- First 11 updates have finite CE/CKA/gradient norm, with no OOM.
- At update 10: old elapsed 2:32, restarted elapsed 1:40.
- At update 11: restarted smoothed time 9.60 seconds/update, versus roughly
  14–15 seconds/update in the old run. Provisional estimate: 13–15 hours/run,
  not a guarantee for later batches or other target layers.
- GPU 3 used about 159 GiB, with about 19 GiB free at this check.
- Saved PRe trainer_state reports 53654.9925 seconds (~14 h 54 min), not
  11 hours; this may be a different run from the one the user recalled.

Synthetic CKA benchmarks (`benchmark*.log/json`) encountered non-finite
loss/gradients after updates, including the full-window check. Their timing
is **not accepted** as production evidence. The underlying synthetic failure
has not been diagnosed; real-data measurements above are the basis for the
chosen profile. Benchmark updates were discarded and never saved into any
training checkpoint. `training-source.tar.gz` captures code at restart.
