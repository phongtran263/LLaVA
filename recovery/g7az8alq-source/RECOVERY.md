# g7az8alq finetune source recovery

This directory reconstructs the tracked Python runtime source loaded by the
completed W&B finetune run `g7az8alq` on 2026-07-20. It is kept separately from
the active LLaVA worktree.

## Provenance

- Recorded base commit: `0d7b4e664f3be6de466a3a80b08cba0909ecd675`.
- W&B run: `wandb/run-20260720_092628-g7az8alq`.
- The run completed 5197 optimizer steps and loaded the projector produced at
  `checkpoints/qwen-7b-cka-grad/llava-pretrain/mm_projector.bin`.
- W&B did not archive the dirty source tree or the original ignored shell
  script, but it preserved the complete Python argv and resolved config under
  `provenance/wandb-g7az8alq/`.
- Session edit history and Git pre-image blobs show that this run loaded the
  memory-optimized VSP controller before the later
  `vsp_apply_to_projector_only` option was introduced. The absence of that
  field from the resolved W&B config independently confirms the boundary.

## Recovered runtime blobs

| Path | Git blob |
| --- | --- |
| `llava/train/train.py` | `5b67bc2bf24d1caf1b857f574dc2fe7a79c65208` |
| `llava/train/llava_trainer.py` | `05e833ec30498364b2116285cd3e638e685fe365` |
| `llava/train/vsp_gradient_controller.py` | `08c893e9de3c02e9091d7ae3e56dadd8121d06fe` |
| `llava/model/llava_arch.py` | `aaa62806391859130e053ce6568191ae1939e281` |
| `llava/model/language_model/llava_qwen.py` | `c3d5b51c659e35407a8427406184f92104b71e23` |
| `llava/model/language_model/llava_llama.py` | `00eeb8a04d46f2d0841dfc605928ec62aceb96be` |
| `llava/model/language_model/llava_mistral.py` | `2c3faa27715b1c92466ce1d6f1969672e1e434df` |
| `llava/model/language_model/llava_mpt.py` | `b9272dd9f91cc8da9a18aa774884dfdb6aecc589` |

All other tracked files come from the recorded base commit.

## Active script

The active `scripts/v1_5/7b/finetune_qwen2_5.sh` reproduces the effective
training arguments while using safe rerun paths by default. Set
`PRETRAIN_ADAPTER` and `OUTPUT_DIR` explicitly when another location is needed.
