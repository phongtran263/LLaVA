import unittest
import warnings
from unittest import mock

import torch
import torch.nn as nn
import transformers

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.llava_arch import (
    compute_linear_cka_loss,
)
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import (
    LlavaMistralConfig,
    LlavaMistralForCausalLM,
)
from llava.model.language_model.llava_mpt import LlavaMptConfig, LlavaMptForCausalLM

if hasattr(transformers, "Qwen2Config"):
    # Do not catch project import errors: on a Qwen-capable Transformers version,
    # a regression in llava_qwen must fail this suite rather than silently skip it.
    from llava.model.language_model.llava_qwen import (
        LlavaQwenConfig,
        LlavaQwenForCausalLM,
    )
else:
    LlavaQwenConfig = None
    LlavaQwenForCausalLM = None
if hasattr(transformers, "Qwen3Config"):
    from llava.model.language_model.llava_modern import (
        LlavaQwen3Config,
        LlavaQwen3ForCausalLM,
        LlavaGemma3Config,
        LlavaGemma3ForCausalLM,
        LlavaPhi3Config,
        LlavaPhi3ForCausalLM,
    )
else:
    LlavaQwen3Config = None
    LlavaQwen3ForCausalLM = None
    LlavaGemma3Config = None
    LlavaGemma3ForCausalLM = None
    LlavaPhi3Config = None
    LlavaPhi3ForCausalLM = None



class _FakeVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(101)
        self.register_buffer("features", torch.randn(5, 6, generator=generator))
        self.num_patches_per_side = 1

    def forward(self, images):
        return self.features.unsqueeze(0).expand(images.shape[0], -1, -1)


def _projector():
    projector = nn.Linear(6, 16, bias=False)
    with torch.no_grad():
        projector.weight.zero_()
        projector.weight[:, 0] = torch.linspace(0.25, 1.0, 16)
    return projector


def _configure_multimodal_cka(config):
    config.cka_loss = True
    config.use_pcgrad = True
    config.cka_loss_layers = [-1]
    config.cka_loss_weight = 1.0
    config.cka_loss_projector_weight = 1.0
    config.cka_loss_final_hidden_weight = 1.0
    config.cka_loss_channel_keep_ratio = 1.0
    config.cka_loss_channel_seed = 42
    config.cka_loss_hidden_channel_drop_indices = []
    config.log_gradient_norms = False
    config.mm_patch_merge_type = "flat"
    config.image_aspect_ratio = "square"
    config.tokenizer_padding_side = "right"
    config.tokenizer_model_max_length = 64
    config.tune_mm_mlp_adapter = False
    config.mm_use_im_start_end = False
    config.use_cache = False
    return config


def _backbone_factories():
    factories = [
        (
            "llama",
            LlavaLlamaForCausalLM,
            lambda: LlavaConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=64,
            ),
        ),
        (
            "mistral",
            LlavaMistralForCausalLM,
            lambda: LlavaMistralConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            ),
        ),
        (
            "mpt",
            LlavaMptForCausalLM,
            lambda: LlavaMptConfig(
                vocab_size=64,
                d_model=16,
                n_layers=2,
                n_heads=4,
                expansion_ratio=2,
                max_seq_len=64,
                attn_config={"attn_impl": "torch"},
            ),
        ),
    ]
    if LlavaQwen3ForCausalLM is not None:
        factories.extend([
            (
                "qwen3",
                LlavaQwen3ForCausalLM,
                lambda: LlavaQwen3Config(
                    vocab_size=64, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=4,
                    max_position_embeddings=64,
                ),
            ),
            (
                "gemma3",
                LlavaGemma3ForCausalLM,
                lambda: LlavaGemma3Config(
                    vocab_size=64, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=4,
                    max_position_embeddings=64, sliding_window=32,
                ),
            ),
            (
                "phi3",
                LlavaPhi3ForCausalLM,
                lambda: LlavaPhi3Config(
                    vocab_size=64, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, max_position_embeddings=64,
                    original_max_position_embeddings=64,
                    pad_token_id=0, bos_token_id=1, eos_token_id=2,
                ),
            ),
        ])
    if LlavaQwenForCausalLM is not None:
        factories.append(
            (
                "qwen2",
                LlavaQwenForCausalLM,
                lambda: LlavaQwenConfig(
                    vocab_size=64,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=64,
                ),
            )
        )
    return factories


class CkaBackboneSmokeTests(unittest.TestCase):
    def test_text_forward_and_backward_with_cka_off_all_available_backbones(self):
        for name, model_cls, config_factory in _backbone_factories():
            with self.subTest(backbone=name):
                config = config_factory()
                config.cka_loss = False
                config.use_cache = False
                model = model_cls(config).train()
                input_ids = torch.tensor([[1, 2, 3, 4]])
                output = model(input_ids=input_ids, labels=input_ids)
                self.assertTrue(torch.isfinite(output.loss))
                self.assertIsNone(getattr(output, "projector_cka_loss", None))
                output.loss.backward()

    def test_multimodal_projector_cka_forward_and_backward_all_available_backbones(self):
        for name, model_cls, config_factory in _backbone_factories():
            with self.subTest(backbone=name):
                torch.manual_seed(23)
                config = _configure_multimodal_cka(config_factory())
                model = model_cls(config).train()
                model.get_model().vision_tower = _FakeVisionTower()
                model.get_model().mm_projector = _projector()

                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                labels = input_ids.clone()
                images = torch.randn(1, 3, 2, 2)

                vision_features = model.get_model().vision_tower(images)
                projected_features = model.get_model().mm_projector(vision_features)
                expected_cka = compute_linear_cka_loss(
                    vision_features,
                    projected_features,
                )

                output = model(input_ids=input_ids, labels=labels, images=images)
                self.assertTrue(torch.isfinite(output.loss))
                self.assertTrue(torch.isfinite(output.projector_cka_loss))
                self.assertTrue(
                    torch.allclose(
                        output.projector_cka_loss,
                        expected_cka,
                        atol=1e-6,
                        rtol=1e-6,
                    )
                )
                self.assertGreater(output.projector_cka_loss.item(), 0.0)
                cka_only_grad = torch.autograd.grad(
                    output.projector_cka_loss,
                    model.get_model().mm_projector.weight,
                    retain_graph=True,
                )[0]
                self.assertTrue(torch.isfinite(cka_only_grad).all())
                self.assertGreater(cka_only_grad.abs().sum().item(), 0.0)

                total_loss = (
                    output.loss
                    + output.projector_cka_loss
                    + sum(output.aux_losses or [])
                )
                total_loss.backward()
                projector_grad = model.get_model().mm_projector.weight.grad
                self.assertIsNotNone(projector_grad)
                self.assertTrue(torch.isfinite(projector_grad).all())


    def test_generic_channel_filter_all_available_backbones(self):
        for name, model_cls, config_factory in _backbone_factories():
            with self.subTest(backbone=name):
                torch.manual_seed(27)
                config = _configure_multimodal_cka(config_factory())
                config.cka_loss_channel_keep_ratio = 0.5
                config.cka_loss_channel_seed = 47
                config.cka_loss_hidden_channel_drop_indices = [1, 13]
                model = model_cls(config).train()
                model.get_model().vision_tower = _FakeVisionTower()
                model.get_model().mm_projector = _projector()

                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                labels = input_ids.clone()
                images = torch.randn(1, 3, 2, 2)

                vision_features = model.get_model().vision_tower(images)
                projected_features = model.get_model().mm_projector(vision_features)
                expected_cka = compute_linear_cka_loss(
                    vision_features,
                    projected_features,
                    channel_keep_ratio=0.5,
                    channel_seed=47,
                    y_channel_drop_indices=[1, 13],
                    share_channel_indices=False,
                )

                output = model(input_ids=input_ids, labels=labels, images=images)
                self.assertTrue(
                    torch.allclose(
                        output.projector_cka_loss,
                        expected_cka,
                        atol=1e-6,
                        rtol=1e-6,
                    )
                )


    def test_hidden_vision_reference_regularizes_supported_backbones(self):
        supported = {
            name: (model_cls, config_factory)
            for name, model_cls, config_factory in _backbone_factories()
            if name in {"llama", "mistral", "qwen2", "qwen3", "gemma3", "phi3"}
        }
        for name, (model_cls, config_factory) in supported.items():
            with self.subTest(backbone=name):
                torch.manual_seed(29)
                config = _configure_multimodal_cka(config_factory())
                config.cka_loss_layers = "final"
                config.cka_loss_channel_keep_ratio = 0.5
                config.cka_loss_channel_seed = 53
                config.cka_loss_hidden_channel_drop_indices = [1, 7]
                model = model_cls(config).train()
                model.get_model().vision_tower = _FakeVisionTower()
                model.get_model().mm_projector = _projector()

                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                labels = input_ids.clone()
                images = torch.randn(1, 3, 2, 2)

                legacy_output = model(input_ids=input_ids, labels=labels, images=images)
                self.assertEqual(len(legacy_output.aux_losses), 1)
                hidden_cka = legacy_output.aux_losses[0]
                self.assertGreater(hidden_cka.item(), 0.0)
                hidden_grads = torch.autograd.grad(
                    hidden_cka,
                    [parameter for parameter in model.get_model().parameters() if parameter.requires_grad],
                    allow_unused=True,
                    retain_graph=True,
                )
                nonzero_hidden_grads = [
                    grad
                    for grad in hidden_grads
                    if grad is not None and grad.abs().sum().item() > 0.0
                ]
                self.assertTrue(nonzero_hidden_grads)
                self.assertTrue(
                    all(torch.isfinite(grad).all() for grad in nonzero_hidden_grads)
                )

    def test_all_selected_hiddens_use_same_raw_vision_reference(self):
        supported = {
            name: (model_cls, config_factory)
            for name, model_cls, config_factory in _backbone_factories()
            if name in {"llama", "mistral", "qwen2", "qwen3", "gemma3", "phi3"}
        }
        for name, (model_cls, config_factory) in supported.items():
            with self.subTest(backbone=name):
                torch.manual_seed(30)
                config = _configure_multimodal_cka(config_factory())
                config.cka_loss_layers = "1,2,final"
                config.cka_loss_projector_weight = 0.0
                config.cka_loss_final_hidden_weight = 1.0
                model = model_cls(config).train()
                vision_tower = _FakeVisionTower()
                vision_tower.features.requires_grad_(True)
                model.get_model().vision_tower = vision_tower
                model.get_model().mm_projector = _projector()

                recorded_calls = []

                def record_masked_cka(
                    projected_features,
                    layer_hidden_states,
                    vision_feature_mask,
                    **kwargs,
                ):
                    recorded_calls.append((
                        projected_features,
                        layer_hidden_states,
                        vision_feature_mask,
                    ))
                    # Keep the synthetic loss connected only to the selected
                    # hidden target. The raw vision endpoint must be a detached
                    # reference, not another optimization target.
                    self.assertTrue(projected_features.requires_grad)
                    self.assertFalse(layer_hidden_states.requires_grad)
                    return projected_features.float().square().mean()

                model._compute_masked_linear_cka_loss = record_masked_cka
                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                labels = input_ids.clone()
                images = torch.randn(1, 3, 2, 2)

                output = model(input_ids=input_ids, labels=labels, images=images)
                # "final" aliases the last numeric layer, so H_2 is not
                # regularized twice.

                self.assertEqual(len(recorded_calls), 2)
                target_hiddens = [call[0] for call in recorded_calls]
                vision_references = [call[1] for call in recorded_calls]
                vision_masks = [call[2] for call in recorded_calls]

                self.assertTrue(all(hidden.shape[-1] == 16 for hidden in target_hiddens))
                self.assertTrue(all(hidden.requires_grad for hidden in target_hiddens))
                self.assertTrue(all(reference.shape[-1] == 6 for reference in vision_references))
                self.assertTrue(all(not reference.requires_grad for reference in vision_references))
                self.assertTrue(all(reference.grad_fn is None for reference in vision_references))

                first_reference = vision_references[0]
                first_mask = vision_masks[0]
                self.assertEqual(int(first_mask.sum().item()), vision_tower.features.shape[0])
                self.assertTrue(torch.equal(
                    first_reference[first_mask],
                    vision_tower.features.detach(),
                ))
                self.assertEqual(
                    torch.count_nonzero(first_reference[~first_mask]).item(),
                    0,
                )
                for reference, mask in zip(vision_references[1:], vision_masks[1:]):
                    self.assertEqual(
                        reference.untyped_storage().data_ptr(),
                        first_reference.untyped_storage().data_ptr(),
                    )
                    self.assertTrue(torch.equal(reference, first_reference))
                    self.assertTrue(torch.equal(mask, first_mask))

                target_storage_ptrs = {
                    hidden.untyped_storage().data_ptr()
                    for hidden in target_hiddens
                }
                self.assertNotIn(
                    first_reference.untyped_storage().data_ptr(),
                    target_storage_ptrs,
                )
                self.assertEqual(
                    list(model.last_cka_per_layer_losses),
                    [
                        "vision_encoder_to_layer_1",
                        "vision_encoder_to_layer_2",
                    ],
                )
                self.assertEqual(len(output.aux_losses), 1)

    def test_selected_hidden_cka_rejects_spatial_list_and_5d_images(self):
        input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        labels = input_ids.clone()
        image_inputs = {
            "list": [torch.randn(2, 3, 2, 2)],
            "5d": torch.randn(1, 2, 3, 2, 2),
        }

        for input_kind, images in image_inputs.items():
            with self.subTest(input_kind=input_kind):
                config = _configure_multimodal_cka(LlavaConfig(
                    vocab_size=64,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    max_position_embeddings=64,
                ))
                config.cka_loss_layers = "final"
                config.cka_loss_projector_weight = 0.0
                config.cka_loss_final_hidden_weight = 1.0
                config.mm_patch_merge_type = "spatial"
                model = LlavaLlamaForCausalLM(config).train()
                model.get_model().vision_tower = _FakeVisionTower()
                model.get_model().mm_projector = _projector()

                with self.assertRaisesRegex(
                    ValueError,
                    r"mm_patch_merge_type='flat'",
                ):
                    model(input_ids=input_ids, labels=labels, images=images)

    def test_final_only_mode_skips_projector_cka_for_supported_backbones(self):
        supported = {
            name: (model_cls, config_factory)
            for name, model_cls, config_factory in _backbone_factories()
            if name in {"llama", "mistral", "qwen2", "qwen3", "gemma3", "phi3"}
        }
        for name, (model_cls, config_factory) in supported.items():
            with self.subTest(backbone=name):
                torch.manual_seed(31)
                config = _configure_multimodal_cka(config_factory())
                config.cka_loss_layers = "final"
                config.cka_loss_projector_weight = 0.0
                config.cka_loss_final_hidden_weight = 1.0
                model = model_cls(config).train()
                model.get_model().vision_tower = _FakeVisionTower()
                model.get_model().mm_projector = _projector()

                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                labels = input_ids.clone()
                images = torch.randn(1, 3, 2, 2)

                with mock.patch(
                    "llava.model.llava_arch.compute_linear_cka_loss",
                    side_effect=AssertionError("projector CKA must be skipped"),
                ):
                    output = model(input_ids=input_ids, labels=labels, images=images)

                self.assertEqual(output.projector_cka_loss.item(), 0.0)
                self.assertEqual(len(output.aux_losses), 1)
                self.assertGreater(output.aux_losses[0].item(), 0.0)

    def test_mpt_warns_once_for_unsupported_hidden_layer_config(self):
        projector_only = {
            name: (model_cls, config_factory)
            for name, model_cls, config_factory in _backbone_factories()
            if name == "mpt"
        }
        for name, (model_cls, config_factory) in projector_only.items():
            with self.subTest(backbone=name):
                config = _configure_multimodal_cka(config_factory())
                config.cka_loss_layers = "final"
                model = model_cls(config).train()
                input_ids = torch.tensor([[1, 2, 3, 4]])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    first = model(input_ids=input_ids, labels=input_ids)
                    second = model(input_ids=input_ids, labels=input_ids)
                matching = [
                    warning
                    for warning in caught
                    if "supports projector CKA only" in str(warning.message)
                ]
                self.assertEqual(len(matching), 1)
                self.assertEqual(first.aux_losses, [])
                self.assertEqual(second.aux_losses, [])

    def test_mistral_generate_is_cache_position_compatible(self):
        config = LlavaMistralConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        config.cka_loss = False
        model = LlavaMistralForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = model.generate(
                inputs=input_ids,
                max_new_tokens=1,
                do_sample=False,
            )
        # Transformers versions differ on whether generation from
        # ``inputs_embeds`` prepends the prompt ids to the returned sequence.
        # This smoke test targets the cache-position API compatibility: either
        # contract is valid as long as one token is generated for this batch.
        self.assertEqual(output.shape[0], input_ids.shape[0])
        self.assertGreaterEqual(output.shape[1], 1)
        self.assertLessEqual(output.shape[1], input_ids.shape[1] + 1)

    def test_llama_cache_position_is_version_compatible(self):
        config = LlavaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
        config.cka_loss = False
        config.use_cache = False
        model = LlavaLlamaForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                cache_position=torch.arange(input_ids.shape[1]),
            )
        self.assertEqual(output.logits.shape[:2], input_ids.shape)



    def test_modern_backbone_generate_is_cache_compatible(self):
        for name, model_cls, config_factory in _backbone_factories():
            if name not in {"qwen3", "gemma3", "phi3"}:
                continue
            with self.subTest(backbone=name):
                config = config_factory()
                config.cka_loss = False
                config.use_cache = True
                model = model_cls(config).eval()
                input_ids = torch.tensor([[1, 2, 3]])
                with torch.no_grad():
                    output = model.generate(
                        inputs=input_ids,
                        max_new_tokens=1,
                        do_sample=False,
                    )
                self.assertEqual(output.shape[0], 1)
                self.assertEqual(output.shape[1], 1)


if __name__ == "__main__":
    unittest.main()
