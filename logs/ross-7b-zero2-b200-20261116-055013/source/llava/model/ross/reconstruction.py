"""LLaVA adapter for the official ROSS reconstruction objective.

Derived from ``ross/model/ross_arch.py::compute_vm_loss`` and
``ross/model/pixel_decoder/flux_decoder.py`` at upstream commit
20ffc6b7ba1605a51111ea5f420f9a08e74e22f9 (Apache-2.0; see LICENSE).
The DiT and diffusion math in adjacent files is vendored from upstream, with a
functional FP32 dispatch for BF16 parameter storage. Target/image alignment
remains the caller's responsibility.
"""

import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from .denoiser_dit import RossDenoiser


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class FrozenFluxAutoencoder(nn.Module):
    """The official frozen full FLUX AutoencoderKL with a lazy dependency.

    ``autoencoder`` allows lightweight CPU tests and checkpoint reconstruction
    without downloading weights. Normal training loads the exact model directory
    supplied as ``config.mm_pixel_decoder``.
    """

    def __init__(self, config, *, autoencoder=None, **load_kwargs):
        super().__init__()
        self.config = config
        if autoencoder is None:
            try:
                from diffusers import AutoencoderKL
            except ImportError as exc:
                raise ImportError(
                    'ROSS requires its optional diffusers AutoencoderKL dependency; '
                    'the official implementation pins diffusers==0.30.0. '
                    'Baseline and PRe training do not require diffusers.'
                ) from exc
            autoencoder = AutoencoderKL.from_pretrained(config.mm_pixel_decoder, **load_kwargs)
        self.pixel_decoder = autoencoder
        self.pixel_decoder.requires_grad_(False)
        self.pixel_decoder.float().eval()
        self.train(False)

    def _apply(self, function, recurse=True):
        # DeepSpeed calls parent.bfloat16() on the full MLLM. Probe its target
        # device using an EMPTY tensor, then move the real frozen weights without
        # a BF16 -> FP32 roundtrip that would irreversibly quantize the tokenizer.
        parameter = next(self.pixel_decoder.parameters(), None)
        device = parameter.device if parameter is not None else torch.device('cpu')
        probe = function(torch.empty(0, device=device, dtype=torch.float32))

        def preserve_fp32(tensor):
            if tensor.is_floating_point():
                return tensor.to(device=probe.device, dtype=torch.float32)
            return function(tensor)

        return super()._apply(preserve_fp32, recurse=recurse)

    @property
    def scaling_factor(self):
        return self.pixel_decoder.config.scaling_factor

    @property
    def shift_factor(self):
        return self.pixel_decoder.config.shift_factor

    @property
    def latent_dim(self):
        return self.pixel_decoder.config.latent_channels * 4

    def train(self, mode=True):
        # The target tokenizer is frozen throughout ROSS training.
        super().train(False)
        return self

    def encode(self, images):
        return self.pixel_decoder.encode(images)

    def decode(self, latents):
        return self.pixel_decoder.decode(latents)


def build_ross_denoiser(config):
    """Exact upstream ``denoiser_vitNx`` builder, which selects DiT.

    The published default is denoiser_vit3x: width 1024, depth 3, 16 heads.
    Naming is retained from upstream even though its builder imports DiT.
    """
    architecture = getattr(config, 'mm_inv_projector_type', 'denoiser_vit3x')
    match = re.fullmatch(r'denoiser_vit(\d+)x', architecture)
    if match is None or int(match.group(1)) < 1:
        raise ValueError('ROSS diffusion reconstruction requires mm_inv_projector_type=denoiser_vitNx.')
    depth = int(match.group(1))
    width = 1280 if depth == 8 else 1536 if depth == 12 else 1024
    patches = int(getattr(config, 'image_embed_len', 576))
    if math.isqrt(patches) ** 2 != patches:
        raise ValueError('ROSS requires a square visual-patch grid.')
    denoiser = RossDenoiser(
        x_channel=int(getattr(config, 'mm_inv_hidden_size', 64)),
        z_channel=config.hidden_size,
        embed_dim=width,
        depth=depth,
        timesteps='1000',
        learn_sigma=False,
        n_patches=patches,
    )
    # HF from_pretrained initializes missing auxiliary weights after constructor
    # completion. Preserve DiT's deliberate Xavier/adaLN-Zero initialization
    # instead of letting the LLaMA initializer overwrite these new modules.
    for module in denoiser.modules():
        module._is_hf_initialized = True
    return denoiser


def prepare_ross_vae_images(images, *, image_mean=CLIP_IMAGE_MEAN,
                            image_std=CLIP_IMAGE_STD, decode_image_size=384):
    """Official CLIP unnormalization, clamp to [-1,1], bilinear 336 -> 384."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError('ROSS reconstruction images must have shape [B,3,H,W].')
    mean = images.new_tensor(image_mean).view(1, -1, 1, 1)
    std = images.new_tensor(image_std).view(1, -1, 1, 1)
    images_vae = ((images * std + mean - 0.5) / 0.5).clamp(-1., 1.)
    return F.interpolate(images_vae, size=(decode_image_size, decode_image_size), mode='bilinear')


def pack_flux_latents(latents):
    """Pack each spatial 2x2 FLUX latent window into four channel groups."""
    if latents.ndim != 4 or latents.shape[2] % 2 or latents.shape[3] % 2:
        raise ValueError('ROSS VAE latents must have even spatial dimensions [B,C,H,W].')
    batch, channels, height, width = latents.shape
    windows = latents.unfold(2, 2, 2).unfold(3, 2, 2)
    return windows.permute(0, 1, 4, 5, 2, 3).reshape(
        batch, channels * 4, height // 2, width // 2,
    ).contiguous()


def compute_ross_reconstruction_loss(denoiser, pixel_decoder, visual_hidden, images,
                                     valid_image_mask=None, *, config=None,
                                     reduction='official', eps=1e-6):
    """Official four-repeat latent diffusion loss on aligned visual states.

    ``visual_hidden`` is [B,576,H], including zero rows for invalid/text samples;
    ``images`` is the corresponding CLIP-normalized [B,3,336,336] tensor, including
    the training loader's dummy image for text rows. The caller selects the LLM
    representation and handles any truncated image-token sequences.

    ``official`` deliberately retains the upstream reduction, whose inner
    ``mean()`` is scalar and includes invalid rows whenever any row is valid.
    ``valid_only`` is supplied as an explicit alternative, never the default.
    Both return a graph-connected zero when every mask entry is false.
    """
    if visual_hidden.ndim != 3 or images.shape[0] != visual_hidden.shape[0]:
        raise ValueError('ROSS visual states [B,N,H] and images must have aligned batches.')
    batch, patches, channels = visual_hidden.shape
    side = math.isqrt(patches)
    if batch < 1 or side ** 2 != patches:
        raise ValueError('ROSS requires a nonempty batch with a square visual-patch grid.')
    if valid_image_mask is None:
        valid_image_mask = torch.ones(batch, device=visual_hidden.device, dtype=torch.bool)
    if valid_image_mask.shape != (batch,):
        raise ValueError('ROSS valid_image_mask must have shape [B].')
    if reduction not in ('official', 'valid_only'):
        raise ValueError('ROSS reduction must be official or valid_only.')
    config = config if config is not None else getattr(pixel_decoder, 'config', None)
    image_mean = getattr(config, 'image_mean', CLIP_IMAGE_MEAN)
    image_std = getattr(config, 'image_std', CLIP_IMAGE_STD)
    decode_size = int(getattr(config, 'decode_image_size', side * 16))
    images_vae = prepare_ross_vae_images(
        images, image_mean=image_mean, image_std=image_std,
        decode_image_size=decode_size,
    )
    # The target visual tokenizer is frozen. FP32 inputs also keep this path
    # usable outside the enclosing Trainer autocast context.
    with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
        posterior = pixel_decoder.encode(images_vae.float()).latent_dist
        targets = (posterior.sample() - pixel_decoder.shift_factor) * pixel_decoder.scaling_factor
        targets = pack_flux_latents(targets)
    if targets.shape[2:] != (side, side):
        raise ValueError('Packed VAE targets must align with the LLM visual-patch grid.')
    with torch.autocast(device_type=visual_hidden.device.type, enabled=False):
        normalized = denoiser.ln_pre(visual_hidden.float())
        context = normalized.transpose(1, 2).reshape(batch, channels, side, side).contiguous()
        losses = denoiser(
            z=context.repeat(4, 1, 1, 1).contiguous().float(),
            target=targets.repeat(4, 1, 1, 1).contiguous().float(),
        ).float()
    mask = valid_image_mask.to(device=losses.device, dtype=torch.bool).repeat(4)
    if reduction == 'official':
        # Preserve upstream compute_vm_loss exactly, including scalar mean.
        return (losses.view(batch, -1).mean() * mask).sum() / (mask.sum() + eps)
    return (losses * mask).sum() / (mask.sum() + eps)
