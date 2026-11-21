# Vicuna-7B + SigLIP2 Stage-1 microbatch 32 restart

The user explicitly requested stopping the Stage-1 microbatch-16 run and
restarting Stage 1 with microbatch 32. The prior run (about step 57, no saved
checkpoint) and its logs/output were preserved. This run uses a fresh output
root: `checkpoints/full-ft/vicuna7b-siglip2-seed42-b256-m32`.

Stage 1: 32 microbatch × 8 accumulation × 1 GPU = 256 global batch.
Stage 2: 4 microbatch × 32 accumulation × 1 GPU = 128 global batch.
All five jobs use GPU 0, training seed 42, projector-init seed 42,
ZeRO-2 B200, BF16, Flash Attention 2, fused AdamW, and no gradient
checkpointing. The two Stage-1 producers are baseline and projector CKA;
the three Stage-2 consumers are baseline, projector CKA, and
projector+final CKA.

Important risk: Stage-1 microbatch 32 OOMed in the synthetic expanded
2048-token probe. Real Stage-1 caption sequences in the previous run used
~70 GiB at microbatch 16 and were much shorter. Keeping max length 2048
means a sufficiently long real batch may still OOM later; the user chose
microbatch 32 with this risk. `train.log` records progress and errors.
