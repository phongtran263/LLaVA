"""PRe objective and LLaVA stage-2 integration regressions, using CPU tiny models."""

import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from llava.model.predictive_regularization import compute_predictive_regularization_loss


class _VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(541)
        self.register_buffer("features", torch.randn(5, 6, generator=generator))

    def forward(self, images):
        return self.features.unsqueeze(0).expand(images.shape[0], -1, -1)


def _config(enabled=True, layer=1):
    config = LlavaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    config.pre_loss = enabled
    config.pre_loss_layer = layer
    config.pre_loss_weight = 0.5
    config.cka_loss = False
    config.use_cache = False
    config.mm_patch_merge_type = "flat"
    config.image_aspect_ratio = "pad"
    config.tokenizer_padding_side = "right"
    config.tokenizer_model_max_length = 64
    config.tune_mm_mlp_adapter = False
    config.mm_use_im_start_end = False
    return config


def _model(enabled=True, layer=1, multimodal=True):
    model = LlavaLlamaForCausalLM(_config(enabled, layer)).train()
    if multimodal:
        model.get_model().vision_tower = _VisionTower()
        model.get_model().mm_projector = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 16))
    return model


def _batch():
    ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
    return dict(input_ids=ids, labels=ids.clone(), images=torch.zeros(1, 3, 2, 2))


class PredictiveRegularizationLossTests(unittest.TestCase):
    def test_negative_patch_cosine_and_detached_anchor(self):
        torch.manual_seed(547)
        predictor = nn.Sequential(nn.Linear(7, 7), nn.GELU(), nn.Linear(7, 7))
        hidden = torch.randn(2, 5, 7, requires_grad=True)
        anchor = torch.randn(2, 5, 7, requires_grad=True)
        mask = torch.tensor([[False, True, True, False, False], [True, True, True, False, False]])
        expected = -F.cosine_similarity(predictor(hidden[mask]), anchor.detach()[mask], dim=-1).mean()
        actual = compute_predictive_regularization_loss(predictor, hidden, anchor, mask)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(anchor.grad)
        self.assertGreater(float(hidden.grad[mask].norm()), 0)
        self.assertEqual(int(torch.count_nonzero(hidden.grad[~mask])), 0)
        for parameter in predictor.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_identical_patches_give_minus_one(self):
        hidden = torch.tensor([[[1.0, 2.0], [-3.0, 4.0]]], requires_grad=True)
        loss = compute_predictive_regularization_loss(nn.Identity(), hidden, hidden.detach(), torch.ones(1, 2, dtype=torch.bool))
        torch.testing.assert_close(loss, torch.tensor(-1.0))

    def test_excluded_text_and_padding_do_not_enter_predictor(self):
        torch.manual_seed(557)
        predictor = nn.Linear(4, 4)
        hidden = torch.randn(2, 4, 4)
        anchor = torch.randn_like(hidden)
        mask = torch.tensor([[False, True, True, False], [False, True, False, False]])
        hidden[~mask] = float("nan")
        anchor[~mask] = float("nan")
        observed = []
        handle = predictor.register_forward_pre_hook(lambda _module, inputs: observed.append(inputs[0].detach().clone()))
        try:
            loss = compute_predictive_regularization_loss(predictor, hidden, anchor, mask)
        finally:
            handle.remove()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(sum(value.numel() for value in observed), int(mask.sum()) * 4)
        self.assertTrue(all(torch.isfinite(value).all() for value in observed))

    def test_empty_mask_has_zero_gradients_for_hidden_and_all_predictor_parameters(self):
        predictor = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        anchor = torch.randn_like(hidden, requires_grad=True)
        loss = compute_predictive_regularization_loss(predictor, hidden, anchor, torch.zeros(2, 3, dtype=torch.bool))
        self.assertEqual(float(loss), 0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNone(anchor.grad)
        torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden), rtol=0, atol=0)
        for parameter in predictor.parameters():
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)

    def test_cosine_reduction_remains_finite_under_bfloat16_autocast(self):
        predictor = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        hidden = torch.randn(1, 4, 8, requires_grad=True)
        anchor = torch.randn_like(hidden, requires_grad=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = compute_predictive_regularization_loss(predictor, hidden, anchor, torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertIsNone(anchor.grad)


class PredictiveRegularizationIntegrationTests(unittest.TestCase):
    def test_predictor_initialization_is_idempotent_and_follows_model_dtype(self):
        model = _model(multimodal=False).to(dtype=torch.bfloat16)
        head = model.get_model().pre_predictor
        self.assertIsInstance(head[0], nn.Linear)
        self.assertIsInstance(head[1], nn.GELU)
        self.assertIsInstance(head[2], nn.Linear)
        self.assertEqual((head[0].in_features, head[0].out_features), (16, 16))
        self.assertEqual((head[2].in_features, head[2].out_features), (16, 16))
        parameters = tuple(head.parameters())
        rng = torch.get_rng_state().clone()
        model.get_model().initialize_pre_predictor()
        self.assertIs(model.get_model().pre_predictor, head)
        self.assertTrue(all(a is b for a, b in zip(parameters, head.parameters())))
        self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in parameters))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_invalid_layer_weight_and_cka_combination_fail_during_setup(self):
        for name, value in (("pre_loss_layer", 0), ("pre_loss_layer", 3), ("pre_loss_layer", True), ("pre_loss_weight", -0.5), ("pre_loss_weight", float("nan")), ("cka_loss", True)):
            with self.subTest(name=name, value=value):
                config = _config()
                setattr(config, name, value)
                with self.assertRaises(ValueError):
                    LlavaLlamaForCausalLM(config)

    def test_tuple_output_retains_combined_loss(self):
        model = _model()
        as_dict = model(**_batch())
        as_tuple = model(**_batch(), return_dict=False)
        self.assertIsInstance(as_tuple, tuple)
        torch.testing.assert_close(as_tuple[0], as_dict.loss)
        torch.testing.assert_close(as_tuple[1], as_dict.logits)

    def test_reentrant_checkpointing_fails_before_silently_dropping_auxiliary_gradients(self):
        model = _model()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        with self.assertRaisesRegex(ValueError, "use_reentrant"):
            model(**_batch())

    def test_disabled_flag_preserves_state_and_rng(self):
        absent_config = _config(False)
        del absent_config.pre_loss
        torch.manual_seed(563)
        absent = LlavaLlamaForCausalLM(absent_config)
        absent_rng = torch.get_rng_state().clone()
        torch.manual_seed(563)
        disabled = LlavaLlamaForCausalLM(_config(False))
        self.assertFalse(hasattr(absent.get_model(), "pre_predictor"))
        self.assertFalse(hasattr(disabled.get_model(), "pre_predictor"))
        self.assertTrue(torch.equal(absent_rng, torch.get_rng_state()))
        self.assertEqual(absent.state_dict().keys(), disabled.state_dict().keys())
        for name, value in absent.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[name], rtol=0, atol=0)

    def test_selected_block_predicts_postprojector_anchor_and_combines_weighted_loss(self):
        for layer in (1, 2):
            with self.subTest(layer=layer):
                torch.manual_seed(569)
                model = _model(layer=layer)
                observed = {}
                def capture_hidden(_module, _inputs, output):
                    observed["hidden"] = output[0] if isinstance(output, (tuple, list)) else output
                def capture_projector(_module, _inputs, output):
                    observed["anchor"] = output
                def capture_predictor(_module, inputs):
                    observed["predictor_input"] = inputs[0].detach().clone()
                handles = [
                    model.get_model().layers[layer - 1].register_forward_hook(capture_hidden),
                    model.get_model().mm_projector.register_forward_hook(capture_projector),
                    model.get_model().pre_predictor.register_forward_pre_hook(capture_predictor),
                ]
                try:
                    output = model(**_batch())
                finally:
                    for handle in handles:
                        handle.remove()
                torch.testing.assert_close(observed["predictor_input"].reshape(-1, 16), observed["hidden"][:, 1:6].reshape(-1, 16))
                expected_pre = -F.cosine_similarity(model.get_model().pre_predictor(observed["hidden"][:, 1:6]), observed["anchor"].detach(), dim=-1).mean()
                torch.testing.assert_close(model.last_pre_loss, expected_pre.detach())
                self.assertEqual(int(model.last_pre_num_patches), 5)
                self.assertFalse(model.last_pre_loss.requires_grad)
                self.assertFalse(model.last_pre_text_loss.requires_grad)
                torch.testing.assert_close(output.loss.detach(), model.last_pre_text_loss + 0.5 * model.last_pre_loss)
                model.config.pre_loss = False
                ce_only = model(**_batch())
                torch.testing.assert_close(output.logits, ce_only.logits, rtol=0, atol=0)
                torch.testing.assert_close(output.loss.detach(), ce_only.loss.detach() + 0.5 * expected_pre.detach())

    def test_combined_loss_trains_predictor_projector_and_decoder(self):
        torch.manual_seed(571)
        model = _model(layer=1)
        model(**_batch()).loss.backward()
        for module in (model.get_model().pre_predictor, model.get_model().mm_projector, model.get_model().layers[0].mlp):
            gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
            self.assertTrue(gradients)
            self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
            self.assertGreater(sum(float(value.norm()) for value in gradients), 0)

    def test_mask_tracks_image_truncation_and_mixed_text_batch_padding(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                model = _model()
                model.config.tokenizer_padding_side = side
                model.config.tokenizer_model_max_length = 5
                ids = torch.tensor([[1, 2, IMAGE_TOKEN_INDEX, 3], [1, 4, 0, 0]])
                observed = []
                handle = model.get_model().pre_predictor.register_forward_pre_hook(lambda _module, inputs: observed.append(inputs[0].detach()))
                try:
                    output = model(input_ids=ids, labels=ids.clone(), attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]), images=torch.zeros(2, 3, 2, 2))
                finally:
                    handle.remove()
                self.assertTrue(torch.isfinite(output.loss))
                self.assertEqual(int(model.last_pre_num_patches), 3)
                self.assertEqual(sum(value.numel() for value in observed), 3 * 16)
                output.loss.backward()

    def test_text_only_batch_keeps_predictor_in_backward_graph(self):
        model = _model()
        ids = torch.tensor([[1, 2, 3, 4]])
        output = model(input_ids=ids, labels=ids.clone())
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(float(model.last_pre_loss), 0)
        self.assertEqual(int(model.last_pre_num_patches), 0)
        output.loss.backward()
        for parameter in model.get_model().pre_predictor.parameters():
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)

    def test_nonreentrant_checkpointing_preserves_predictor_and_projector_gradients(self):
        torch.manual_seed(577)
        model = _model()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model(**_batch()).loss.backward()
        for module in (model.get_model().pre_predictor, model.get_model().mm_projector):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
            self.assertGreater(sum(float(value.norm()) for value in gradients), 0)

    def test_save_pretrained_restores_predictor_and_configuration(self):
        torch.manual_seed(587)
        model = _model(multimodal=False)
        with torch.no_grad():
            model.get_model().pre_predictor[0].weight.add_(0.125)
        with tempfile.TemporaryDirectory(prefix="llava-pre-test-") as directory:
            model.save_pretrained(directory)
            restored = LlavaLlamaForCausalLM.from_pretrained(directory, local_files_only=True)
        self.assertIs(restored.config.pre_loss, True)
        self.assertEqual(restored.config.pre_loss_layer, 1)
        self.assertEqual(restored.config.pre_loss_weight, 0.5)
        for name, value in model.get_model().pre_predictor.state_dict().items():
            torch.testing.assert_close(value, restored.get_model().pre_predictor.state_dict()[name], rtol=0, atol=0)

    def test_evaluation_skips_predictor_and_preserves_language_loss(self):
        model = _model().eval()
        head = model.get_model().pre_predictor
        with mock.patch.object(head, "forward", wraps=head.forward) as forward:
            with torch.no_grad():
                enabled = model(**_batch())
                model.config.pre_loss = False
                disabled = model(**_batch())
        self.assertEqual(forward.call_count, 0)
        torch.testing.assert_close(enabled.loss, disabled.loss, rtol=0, atol=0)
        torch.testing.assert_close(enabled.logits, disabled.logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
