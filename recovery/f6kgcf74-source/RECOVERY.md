# f6kgcf74 source recovery

This directory is an isolated reconstruction of the Python runtime source used
by W&B run `f6kgcf74` (`2026-07-20T06:38:28Z`). It does not replace the active
LLaVA worktree.

## Provenance

- Recorded Git base: `0d7b4e664f3be6de466a3a80b08cba0909ecd675`
- W&B run: `wandb/run-20260720_063828-f6kgcf74`
- The run used a dirty worktree. W&B did not record a dirty-tree hash.
- Codex history records no runtime-source edits between completion of the VSP
  implementation and the end of this run.
- The recovered `train.py` has `trainer.train()` at line 1395, matching the
  historical traceback exactly.
- The first post-run edits produced Git pre-image blobs, allowing the runtime
  files below to be restored byte-for-byte.

## Recovered runtime blobs

| Path | Git blob |
| --- | --- |
| `llava/train/train.py` | `5b67bc2bf24d1caf1b857f574dc2fe7a79c65208` |
| `llava/train/llava_trainer.py` | `f665fb82b4bc5659d9480b09c8b114bd8ec8bba5` |
| `llava/train/vsp_gradient_controller.py` | `23cc8ffbdd85d92e2d473ad62469016cf50af433` |
| `llava/model/llava_arch.py` | `aaa62806391859130e053ce6568191ae1939e281` |
| `llava/model/language_model/llava_qwen.py` | `c3d5b51c659e35407a8427406184f92104b71e23` |
| `llava/model/language_model/llava_llama.py` | `00eeb8a04d46f2d0841dfc605928ec62aceb96be` |
| `llava/model/language_model/llava_mistral.py` | `2c3faa27715b1c92466ce1d6f1969672e1e434df` |
| `llava/model/language_model/llava_mpt.py` | `b9272dd9f91cc8da9a18aa774884dfdb6aecc589` |

All other tracked files come from the recorded base commit. The ignored shell
script is not recoverable byte-for-byte, but its complete effective Python argv
is preserved in `provenance/wandb-f6kgcf74/wandb-metadata.json` and
`config.yaml`.

## Important limitations

- The original `mm_projector.bin` is not contained in W&B and is no longer
  present locally. Exact weight recovery requires a GPFS/PVC backup.
- Re-running this source cannot reproduce identical weights: the projector was
  randomly initialized before Hugging Face Trainer applied seed 42, and the run
  used BF16, TF32, FlashAttention, ZeRO-2, and `full_determinism=False`.
- This snapshot intentionally preserves the historical VSP implementation,
  including graph-retention/memory behavior patched after the run. Use it for
  reproduction and inspection, not as a drop-in replacement for current code.
