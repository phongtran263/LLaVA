# ROSS with the shared LLaVA-1.5 stage-2 controls

This is a controlled, stage-2-only adaptation of [Reconstructive Visual Instruction Tuning](https://arxiv.org/abs/2410.09575), using the [official implementation](https://github.com/Haochen-Wang409/ross/tree/20ffc6b7ba1605a51111ea5f420f9a08e74e22f9). Its purpose is to compare the ROSS objective with PRe and the existing method under the same backbone, alignment checkpoint, instruction data, and optimizer budget.

The shared initialization is Vicuna-7B-v1.5, frozen CLIP ViT-L/14-336, and the standard `mlp2x_gelu` projector loaded from `checkpoints/full-ft/baseline-7b/llava-pretrain/mm_projector.bin`. The auxiliary ROSS denoiser is initialized for this stage-2 run. The LLM, projector, and denoiser are trained; the CLIP encoder and FLUX.1-dev VAE stay frozen. CKA, PRe, PCGrad, LoRA, and protected refinement are disabled.

The official ROSS protocol trains its denoiser during feature alignment and loads that `mm_inv_projector.bin` for instruction tuning. The shared baseline alignment checkpoint contains no ROSS denoiser. Consequently this recipe does **not** reproduce the complete two-stage ROSS protocol or its published benchmark numbers. It preserves the requested common stage-1 initialization. No ROSS-specific stage-1 training is added.

The released ROSS-7B model uses Qwen2/SigLIP, and the official CLIP/Vicuna launcher targets Vicuna-13B, Cambrian737k, and a 32768-token limit. This comparison instead uses Vicuna-7B, the full LLaVA mix665k, and the same 2048-token limit as the local PRe comparison. The installed LLaVA framework and B200 execution profile are shared with PRe, rather than replacing this repository with the authors' older software stack.

## Objective and image targets

The implementation follows the official denoising branch:

1. Gather the 576 visual tokens from the final normalized LLM hidden states. Apply the denoiser's non-affine LayerNorm and arrange them on a 24 × 24 grid.
2. Reuse the existing CLIP-preprocessed image tensor, undo CLIP mean/std normalization, map it to `[-1,1]`, clamp, and bilinearly resize 336 → 384. This is the reconstruction target branch; the LLM's CLIP input remains 336 pixels.
3. Encode that image with the frozen FLUX.1-dev `AutoencoderKL`, sample its posterior, and apply `(sample - shift_factor) * scaling_factor`. Pack each 2 × 2 latent window, producing 64 channels on a 24 × 24 grid.
4. Repeat the conditioning and target four times, sample independent timesteps and Gaussian noise, and predict noise with the official three-block DiT: width 1024, 16 heads, MLP ratio 4, patch size 1, fixed two-dimensional sine/cosine positions, and adaLN-Zero initialization. The diffusion process uses 1000 timesteps, cosine betas, epsilon prediction, and MSE with `learn_sigma=False`.

The combined objective is `CE + ROSS_noise_MSE`, with reconstruction weight 1. Entirely text-only batches contribute zero reconstruction loss; image conditioning requires complete, aligned image-token grids. The denoiser and VAE are training auxiliaries and are unnecessary for language generation.

Training logs expose `loss/ross_text_loss` (unweighted CE), `loss/ross_loss` (unweighted noise MSE), and `ross/num_images`. These auxiliary fields describe the last microbatch; the Trainer's main `loss` averages the combined objective across an optimizer update. They therefore need not sum exactly on the same log line.

The official `compute_vm_loss` averages its reconstruction losses to a scalar before applying the image mask. In mixed image/text batches this includes dummy-image losses. The default `official` reduction preserves that released-source behavior for reproducibility. A `valid_only` reduction is available in the loss implementation as an explicit alternative, but would be a masking change relative to the source. Modality grouping follows the same sampler as PRe.

The frozen VAE retains its exact FP32 checkpoint weights across parent-model BF16 casts. The DiT computes in FP32 through differentiable functional casts while its trainable parameters and optimizer storage follow the BF16 model. Its official Xavier/adaLN-Zero initialization is protected from Hugging Face's missing-weight initialization, and its fixed two-dimensional sine/cosine positions remain frozen. Checkpoint loading rebuilds the auxiliary head for state compatibility; generation skips reconstruction and does not require the VAE asset.

The target images come from the same instruction examples; no extra annotation file, train split, or additional training examples are introduced. The frozen pretrained VAE is the image tokenizer required by the ROSS method. No separate pretrained DiT checkpoint is needed for this stage-2-only variant.

## Shared training controls

| Setting | Value |
| --- | --- |
| Backbone / vision | Vicuna-7B-v1.5 / frozen CLIP ViT-L/14-336 |
| Initialization | Shared baseline stage-1 `mlp2x_gelu` projector |
| Data | All 665,298 LLaVA-1.5 mix665k conversations |
| Epoch / sequence limit | One epoch / 2048 tokens |
| Global batch | 128: microbatch 4 × accumulation 32 × one GPU |
| Optimization | Fused torch AdamW; LR `2e-5`; weight decay 0; warmup ratio 0.03; cosine schedule; seed 42 |
| Precision / memory | BF16; FlashAttention 2; ZeRO-2; activation checkpointing off; expandable CUDA allocator |
| Loader | Eight workers; pinned memory; prefetch 4; persistent workers; nonblocking transfer |
| Reconstruction | Weight 1; four noise/timestep repeats; three-block DiT |

The launch script reuses `finetune_pre.sh` in dry-run mode to perform the same canonical dataset, projector-shape, sample-count, and image-path checks and to derive the shared training command. It then disables PRe and adds the ROSS arguments. Commands are parsed with `shlex`; shell `eval` is not used.

The canonical mixture is [the official LLaVA file](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), SHA256 `ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8`. The data and image folders are the same as PRe, and the launcher refuses an existing output directory.

## Dependencies and invocation

ROSS requires the FLUX.1-dev VAE's `config.json` and `diffusion_pytorch_model.safetensors`, as described in the [official README](https://github.com/Haochen-Wang409/ross/blob/20ffc6b7ba1605a51111ea5f420f9a08e74e22f9/README.md#download-vae-checkpoints). The official model code uses `diffusers==0.30.0`, `timm`, and `einops`. The current LLaVA environment already provides `timm` and `einops`; dependency and weight availability must pass the launcher's preflight. The local dataset reader does not require the original repository's `megfile`, and denoiser training does not require `torchdiffeq`.

The local VAE weight file was obtained from the public [ostris/Flex.1-alpha mirror at revision `5dd0b1bc2a9421b891abcf7c9218993f696dd550`](https://huggingface.co/ostris/Flex.1-alpha/blob/5dd0b1bc2a9421b891abcf7c9218993f696dd550/vae/diffusion_pytorch_model.safetensors). It is byte-identical to the official FLUX VAE artifact: 167,666,902 bytes, SHA256 `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`. The official Git LFS pointer blob `3557b83a132d75a1f71bde9963021e667b1a0d2d` records that same size and SHA256. The launcher enforces both, and `checkpoints/auxiliary/ross-flux-vae/provenance.json` records the acquisition evidence.

```bash
# Validate dependencies, weights, shared data and arguments; print the command.
DRY_RUN=1 bash scripts/v1_5/7b/finetune_ross.sh

# Start the controlled stage-2 comparison on GPU 3.
bash scripts/v1_5/7b/finetune_ross.sh
```

The default output is `checkpoints/full-ft/ross-7b/llava-finetune`. Operational environment overrides from `finetune_pre.sh` remain available, including `TRAIN_ENV`, `GPU_INCLUDE`, `OUTPUT_DIR`, `MASTER_PORT`, data and checkpoint paths, microbatch and accumulation, loader workers, and reporting. The effective batch must remain 128. Use `CHECK_ALL_IMAGES=0` only when reusing an already completed full image-path validation of these same files.

`ROSS_VAE_PATH` selects the local VAE directory (default `./checkpoints/auxiliary/ross-flux-vae`). `ROSS_EXTRA_PYTHONPATH` exposes the isolated dependency directory (default `./.deps/ross`) while retaining the existing source path and shared training environment. Set that variable to an empty string to use already-installed dependencies. The launcher does not download weights or install packages. It imports the required VAE class, checks the FLUX configuration and exact weight hash, and validates the complete final argument list with the installed framework's parser before execution. That parser process sees only the selected physical GPU IDs; the original visibility is restored before DeepSpeed applies its physical `--include` selection. The default distributed port is 29574. Frozen-source runners should pass the absolute VAE and dependency paths from the working repository.

Resumable checkpoints are saved every 1000 optimizer steps, retaining two, with the full final model saved after the epoch. For an interrupted run, use the recorded final training command and the entrypoint's explicit `--resume_from_checkpoint` rather than restarting this fresh-output launcher.

The full ROSS branch adds VAE encoding and denoiser activations beyond the PRe memory profile. Real-model stability and throughput must be measured with that branch enabled before committing to a long run; a PRe-only memory result does not establish ROSS headroom.

Validation so far includes 75 passing CPU tests. GPU validation and production throughput are recorded separately when measured.
