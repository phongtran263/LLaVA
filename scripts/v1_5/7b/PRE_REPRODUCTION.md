# PRe: LLaVA-1.5 / Vicuna-7B stage 2

This recipe implements the stage-2 comparison in [Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](https://arxiv.org/html/2603.20808v1), sections 4.2, 5.1 and supplementary section 7.

It initializes the frozen CLIP ViT-L/14-336 vision encoder and Vicuna-7B-v1.5 backbone, loads only `mm_projector.bin` from `checkpoints/full-ft/baseline-7b/llava-pretrain`, and trains the LLM, the standard `mlp2x_gelu` projector, and the auxiliary predictor. The supplied stage-1 directory contains a projector checkpoint; it is not a complete model or a stage-2 Trainer checkpoint. Stage 1 is not rerun.

PRe predicts each detached post-projector image embedding from the corresponding visual hidden state after decoder block 16. The objective is `CE + 0.5 * (-mean_patch_cosine)`. Padding and text positions do not contribute. The predictor is used during training only. This recipe disables CKA, PCGrad, LoRA, and protected refinement.

The paper specifies a two-layer MLP predictor without publishing its hidden width, activation, normalization, bias or initialization details in the accessible text. The local predictor is `Linear(4096,4096,bias=True) → GELU → Linear(4096,4096,bias=True)`, without normalization, trained at the shared learning rate. These unspecified details are an implementation choice, so this is a paper-based reproduction rather than a verified copy of unpublished author code. Applying the method only in stage 2 follows the requested comparison protocol; the paper does not explicitly isolate PRe to stage 2.

The data are the entire `playground/data/llava_v1_5_mix665k.json`: 665,298 conversations, with images under `playground/data/{coco,gqa,ocr_vqa,textvqa,vg}`. The launcher validates the canonical dataset SHA256, sample count, stage-1 projector shapes, and every referenced image path before training. No extra samples or held-out split are introduced.

The local mixture matches the [official Hugging Face file](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) exactly: size 1,029,887,963 bytes, SHA256 `ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8`. The checked local model cache references are Vicuna `3321f76e3f527bd14065daf69dad9344000a201d` and CLIP `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`. Set the model variables to those snapshot directories to pin revisions independently of future cache updates.

| Setting | Recipe |
| --- | --- |
| Stage-2 optimization | Fused torch AdamW, LR `2e-5`, weight decay `0`, cosine schedule, warmup ratio `0.03` |
| Duration | One epoch over the full mixture |
| Effective batch | 128 |
| Precision / memory | BF16, ZeRO-2, FlashAttention 2, gradient checkpointing off |
| Sequence / images | 2048 tokens; padded square image; 576 visual patches |
| PRe | Decoder block 16 (1-based), weight 0.5, patch-level negative cosine |

The paper describes ZeRO-3 with eight devices, microbatch 4 and accumulation 4; it does not name the GPU model. The requested B200 profile uses ZeRO-2 on one GPU (`localhost:3`), microbatch 4 and accumulation 32, preserving effective batch 128. It disables activation recomputation, uses fused AdamW, and enables eight data-loader workers, pinned memory, prefetch factor 4, asynchronous transfers, and eight OMP/MKL threads. The PRe objective, model, data and published optimizer hyperparameters and learning-rate schedule match the frozen-CLIP setting. BF16, the 2048-token limit and image preprocessing follow the local LLaVA-1.5 framework defaults. The authors do not publish a complete software-version manifest or the predictor internals, so an exact environment/code match cannot be verified. Hardware, accumulation and optimizer kernels can change floating-point results. GPU availability must be checked before starting.

The real-model B200 benchmark uses frozen CLIP images at 336 pixels, the actual multimodal insertion path, 2048 expanded tokens, and fully allocated optimizer states. It estimates one global-128 update from measured forward/backward microsteps plus one optimizer update; data are synthetic and already on the GPU, so production throughput must be checked separately.

| Microbatch | Gradient checkpointing | Estimated seconds/global 128 | Peak allocated GiB |
| --- | --- | ---: | ---: |
| 4 | On | 19.72 | 140.26 |
| 8 | On | 18.41 | 140.75 |
| 16 | On | 18.13 | 141.73 |
| 32 | On | 18.29 | 160.67 |
| **4** | **Off** | **15.74** | **163.80** |
| 8 | Off | OOM | — |
| 64 | On | OOM | — |

Microbatch 4 without recomputation is the fastest measured profile and leaves approximately 14 GiB below the device capacity in this worst-length probe. Increasing batch size solely to raise GPU utilization is slower in the measured cases. `scripts/zero2_b200.json` uses no offload and disables communication overlap for the single-GPU run. `benchmark_pre_b200.py` reproduces the probe; its optimizer updates are discarded and its OOM trials must run separately from production.

Production additionally sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ([PyTorch 2.7 memory documentation](https://docs.pytorch.org/docs/2.7/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)). Static-shape probes do not cover the image/text sequence-size transitions in the real mixture: ZeRO-2 temporarily casts a large gradient partition to FP32 during optimizer updates, and the default allocator can fragment its cached blocks despite sufficient total free memory. Confirm stability across text-only batches as well as image batches. The PRe logger includes live allocated/reserved memory, peak allocation, and inactive split memory for this check.

PRe's DeepSpeed backward explicitly divides the **combined** objective by the accumulation count and calls `scale_wrt_gas=False`. DeepSpeed 0.18.9 otherwise attaches normalization hooks to both returned loss and logits: the CE branch can be divided twice while the auxiliary branch is divided once. The explicit path preserves `CE + 0.5 * PRe` and avoids changing the public model outputs. It is scoped to PRe; existing CKA/controller paths are unchanged.

From the repository root:

```bash
# Full preflight and print the exact command, without training.
DRY_RUN=1 bash scripts/v1_5/7b/finetune_pre.sh

# Run stage 2; default output: checkpoints/full-ft/pre-7b/llava-finetune.
bash scripts/v1_5/7b/finetune_pre.sh

# Eight-device ZeRO-3 profile; checkpointing follows standard LLaVA defaults.
DEEPSPEED_CONFIG="$PWD/scripts/zero3.json" GRADIENT_CHECKPOINTING=True \
GPU_INCLUDE=localhost:0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/finetune_pre.sh
```

`TRAIN_ENV` selects the training conda prefix. The launcher uses an active environment with DeepSpeed or falls back to `../conda/envs/llava1`. On this workspace that environment has Python 3.10, PyTorch 2.7.1+cu128 and Transformers 4.51.3. Model identifiers use the existing Hugging Face cache configuration. `MODEL_NAME_OR_PATH` and `VISION_TOWER` may point to cached local snapshots of the same backbones. Other supported operational overrides include `OUTPUT_DIR`, `GPU_INCLUDE`, `MASTER_PORT`, `DEEPSPEED_CONFIG`, `GRADIENT_CHECKPOINTING`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `DATALOADER_NUM_WORKERS`, `REPORT_TO`, `RUN_NAME` and `SEED`. The launcher enforces effective batch 128 and refuses an existing output directory. `CHECK_ALL_IMAGES=0` skips the image existence scan for a quick repeated dry run.

The launcher defaults `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` for the installed SentencePiece/protobuf combination, matching the existing Vicuna launcher workaround.

Resumable checkpoints are saved every 1,000 optimizer steps (two retained), and the full final model is saved after the epoch. The default command does not resume an interrupted run: use the training entrypoint's explicit `--resume_from_checkpoint` with its recorded arguments and the saved checkpoint when continuation is needed.

Validation before launch: 45 CPU tests passed across the PRe, training compatibility, final CKA head, and training seed suites. A real GPU BF16/ZeRO-3 smoke test passed with gradient accumulation 1 and 2, image and text-only batches, and checkpoint save/reload. The broader existing `test_cka_backbones` suite has an unrelated pre-existing `channel_keep_ratio` argument incompatibility; this reproduction does not change that CKA code.

The optimized ZeRO-2 profile also passed real-GPU BF16 tests with fused AdamW and checkpointing disabled, gradient accumulation 1 and 2, visual and text-only batches, and exact predictor save/reload. Optimizer states and Trainer state were present in the resumable checkpoint. Pinned memory, persistent workers, prefetch factor 4, and Accelerate's nonblocking data transfer were verified at runtime. The dedicated benchmark probes the full 7B model; the save/reload smoke uses a small model.

After the normalization fix, 50 CPU tests passed, including installed DeepSpeed output-hook regressions at accumulation 1, 2 and 32. A real ZeRO-2/Trainer GPU probe at accumulation 32 matched the manual BF16 accumulated gradients exactly for independent backbone and predictor parameters (maximum absolute error 0). This numerical probe uses an analytic small model so it can distinguish correct CE/PRe scaling from merely nonzero parameter updates.
