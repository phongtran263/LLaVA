"""CPU checks for the vendored official ROSS reconstruction objective."""

from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from llava.model.ross import (
    FrozenFluxAutoencoder, RossDenoiser, build_ross_denoiser,
    compute_ross_reconstruction_loss, pack_flux_latents,
    prepare_ross_vae_images,
)
from llava.model.ross.diffusion_utils.gaussian_diffusion import (
    LossType, ModelMeanType, ModelVarType,
)
from llava.model.ross.reconstruction import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD


class _FakeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159,
                                      latent_channels=1)
        self.last_images = None

    def encode(self, images):
        self.last_images = images.detach().clone()
        latents = F.avg_pool2d(images.mean(dim=1, keepdim=True), 8) * self.weight
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latents))


def _pixel_decoder():
    config = SimpleNamespace(decode_image_size=32)
    return FrozenFluxAutoencoder(config, autoencoder=_FakeAutoencoder())


class _KnownLossDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_pre = nn.Identity()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.last_context = None
        self.last_target = None

    def forward(self, z, target):
        self.last_context, self.last_target = z, target
        # Repeat-major layout, matching context.repeat(4,1,1,1).
        offsets = torch.tensor([1., 9.], device=z.device).repeat(4)
        return offsets + self.weight * z.flatten(1).mean(1)


class RossReconstructionTests(unittest.TestCase):
    def test_production_builder_matches_official_dit_architecture(self):
        config = SimpleNamespace(hidden_size=4096, image_embed_len=576,
                                 mm_inv_hidden_size=64,
                                 mm_inv_projector_type='denoiser_vit3x')
        with torch.device('meta'):
            denoiser = build_ross_denoiser(config)
        self.assertEqual(len(denoiser.net.blocks), 3)
        self.assertEqual(denoiser.net.x_embedder.proj.out_channels, 1024)
        self.assertEqual(denoiser.net.blocks[0].attn.num_heads, 16)
        self.assertEqual(denoiser.net.z_embedder[1].in_features, 4096)
        self.assertEqual(denoiser.net.final_layer.linear.out_features, 64)
        self.assertFalse(denoiser.net.pos_embed.requires_grad)
        self.assertFalse(denoiser.ln_pre.elementwise_affine)
        self.assertTrue(all(getattr(module, '_is_hf_initialized', False)
                            for module in denoiser.modules()))

    def test_diffusion_is_official_cosine1000_epsilon_mse(self):
        denoiser = RossDenoiser(x_channel=4, z_channel=8, embed_dim=32, depth=1, n_patches=4)
        diffusion = denoiser.train_diffusion
        self.assertEqual(diffusion.num_timesteps, 1000)
        self.assertIs(diffusion.model_mean_type, ModelMeanType.EPSILON)
        self.assertIs(diffusion.model_var_type, ModelVarType.FIXED_LARGE)
        self.assertIs(diffusion.loss_type, LossType.MSE)
        self.assertAlmostEqual(float(diffusion.betas[0]), 4.128422482196914e-05)
        self.assertAlmostEqual(float(diffusion.betas[-1]), 0.999)

    def test_official_zero_initialization_predicts_zero_noise(self):
        denoiser = RossDenoiser(x_channel=4, z_channel=8, embed_dim=32, depth=1, n_patches=4)
        predicted = denoiser.net(torch.randn(2, 4, 2, 2), torch.tensor([0, 999]),
                                  context=torch.randn(2, 8, 2, 2))
        torch.testing.assert_close(predicted, torch.zeros_like(predicted), rtol=0, atol=0)
        self.assertEqual(int(denoiser.net.blocks[0].adaLN_modulation[-1].weight.count_nonzero()), 0)

    def test_fresh_head_after_from_pretrained_preserves_custom_init_on_post_init(self):
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=4)
        with tempfile.TemporaryDirectory() as directory:
            LlamaForCausalLM(config).save_pretrained(directory)
            restored = LlamaForCausalLM.from_pretrained(directory, config=config)
        # Build a fresh denoiser only after HF's no_init_weights context exits,
        # matching the stage-2 setup. Constructor creation is reserved for loading
        # a complete saved ROSS checkpoint, whose tensors replace initialization.
        restored.ross_head = build_ross_denoiser(SimpleNamespace(
            hidden_size=config.hidden_size, image_embed_len=4,
            mm_inv_hidden_size=4, mm_inv_projector_type='denoiser_vit1x',
        ))
        restored.post_init()
        head = restored.ross_head
        self.assertEqual(int(head.net.final_layer.linear.weight.count_nonzero()), 0)
        self.assertEqual(int(head.net.final_layer.linear.bias.count_nonzero()), 0)
        self.assertEqual(int(head.net.final_layer.adaLN_modulation[-1].weight.count_nonzero()), 0)
        self.assertEqual(int(head.net.blocks[0].adaLN_modulation[-1].weight.count_nonzero()), 0)
        self.assertGreater(float(head.net.x_embedder.proj.weight.norm()), 0.)

    def test_clip_unnormalization_and_resize_matches_official_pixels(self):
        pixels = torch.rand(2, 3, 12, 12)
        mean = pixels.new_tensor(CLIP_IMAGE_MEAN).view(1, 3, 1, 1)
        std = pixels.new_tensor(CLIP_IMAGE_STD).view(1, 3, 1, 1)
        normalized = (pixels - mean) / std
        actual = prepare_ross_vae_images(normalized, decode_image_size=32)
        expected = F.interpolate((pixels - 0.5) / 0.5, (32, 32), mode='bilinear')
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        extremes = prepare_ross_vae_images(torch.full((1, 3, 2, 2), 100.), decode_image_size=4)
        torch.testing.assert_close(extremes, torch.ones_like(extremes), rtol=0, atol=0)

    def test_latent_pack_order_is_channel_then_two_by_two_window(self):
        latents = torch.arange(2 * 4 * 4).reshape(1, 2, 4, 4)
        packed = pack_flux_latents(latents)
        self.assertEqual(packed.shape, (1, 8, 2, 2))
        for channel in range(2):
            for row_offset in range(2):
                for col_offset in range(2):
                    output_channel = channel * 4 + row_offset * 2 + col_offset
                    torch.testing.assert_close(packed[0, output_channel],
                                               latents[0, channel, row_offset::2, col_offset::2])

    def test_frozen_vae_stays_eval_and_exposes_flux_scaling(self):
        decoder = _pixel_decoder().train(True)
        self.assertFalse(decoder.training)
        self.assertFalse(decoder.pixel_decoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in decoder.parameters()))
        self.assertEqual(decoder.latent_dim, 4)
        self.assertEqual(decoder.scaling_factor, 0.3611)
        self.assertEqual(decoder.shift_factor, 0.1159)

    def test_parent_bfloat16_preserves_frozen_vae_precision_without_roundtrip(self):
        decoder = _pixel_decoder()
        with torch.no_grad():
            decoder.pixel_decoder.weight.fill_(1.1234567)
        expected = decoder.pixel_decoder.weight.detach().clone()
        parent = nn.ModuleDict({'vae': decoder, 'trainable': nn.Linear(3, 3)})
        parent.bfloat16()
        self.assertEqual(parent['trainable'].weight.dtype, torch.bfloat16)
        self.assertEqual(decoder.pixel_decoder.weight.dtype, torch.float32)
        torch.testing.assert_close(decoder.pixel_decoder.weight, expected, rtol=0, atol=0)
        parent.to(dtype=torch.float16)
        torch.testing.assert_close(decoder.pixel_decoder.weight, expected, rtol=0, atol=0)

    def test_bfloat16_denoiser_storage_uses_fp32_compute_and_backpropagates(self):
        denoiser = RossDenoiser(x_channel=4, z_channel=8, embed_dim=32, depth=1, n_patches=4)
        # Upstream's output/adaLN zero initialization intentionally blocks the
        # initial conditioning gradient. Activate it to check subsequent steps.
        with torch.no_grad():
            denoiser.net.final_layer.linear.weight.normal_(std=.03)
            denoiser.net.final_layer.adaLN_modulation[-1].weight.normal_(std=.03)
        denoiser.bfloat16()
        hidden = torch.randn(2, 4, 8, dtype=torch.bfloat16, requires_grad=True)
        loss = compute_ross_reconstruction_loss(
            denoiser, _pixel_decoder(), hidden, torch.zeros(2, 3, 16, 16),
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(hidden.grad.float().norm()), 0.)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in denoiser.parameters()))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in denoiser.parameters() if parameter.requires_grad))

    def test_four_repeat_target_scaling_and_official_mask_reduction(self):
        decoder = _pixel_decoder()
        denoiser = _KnownLossDenoiser()
        hidden = torch.zeros(2, 4, 8, requires_grad=True)
        images = torch.zeros(2, 3, 16, 16, requires_grad=True)
        mask = torch.tensor([True, False])
        actual = compute_ross_reconstruction_loss(denoiser, decoder, hidden, images, mask)
        expected = torch.tensor(5.) * 4 / (4 + 1e-6)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(denoiser.last_context.shape, (8, 8, 2, 2))
        self.assertEqual(denoiser.last_target.shape, (8, 4, 2, 2))
        encoded = decoder.pixel_decoder.encode(decoder.pixel_decoder.last_images).latent_dist.sample()
        expected_targets = pack_flux_latents((encoded - 0.1159) * 0.3611).repeat(4, 1, 1, 1)
        torch.testing.assert_close(denoiser.last_target, expected_targets)
        actual.backward()
        self.assertIsNone(images.grad)
        self.assertFalse(denoiser.last_target.requires_grad)
        self.assertIsNotNone(hidden.grad)
        self.assertGreater(float(hidden.grad.norm()), 0.)
        self.assertIsNotNone(denoiser.weight.grad)

    def test_valid_only_reduction_is_explicit_and_not_the_default(self):
        decoder, denoiser = _pixel_decoder(), _KnownLossDenoiser()
        hidden, images = torch.zeros(2, 4, 8), torch.zeros(2, 3, 16, 16)
        mask = torch.tensor([True, False])
        official = compute_ross_reconstruction_loss(denoiser, decoder, hidden, images, mask)
        filtered = compute_ross_reconstruction_loss(denoiser, decoder, hidden, images, mask,
                                                    reduction='valid_only')
        self.assertGreater(float(official), 4.)
        torch.testing.assert_close(filtered, torch.tensor(4. / (4. + 1e-6)))

    def test_text_only_batch_produces_graph_connected_zero(self):
        hidden = torch.zeros(2, 4, 8, requires_grad=True)
        denoiser = _KnownLossDenoiser()
        loss = compute_ross_reconstruction_loss(denoiser, _pixel_decoder(), hidden,
                                                torch.zeros(2, 3, 16, 16),
                                                torch.zeros(2, dtype=torch.bool))
        self.assertEqual(float(loss), 0.)
        loss.backward()
        torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden), rtol=0, atol=0)
        self.assertEqual(float(denoiser.weight.grad), 0.)


if __name__ == '__main__':
    unittest.main()
