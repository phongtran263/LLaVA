import unittest

import torch

from llava.constants import IMAGE_TOKEN_INDEX
from tests.test_cka_backbones import (
    _FakeVisionTower,
    _backbone_factories,
    _configure_multimodal_cka,
    _projector,
)


class CkaPreFfnSemanticsTests(unittest.TestCase):
    def test_layer_two_uses_detached_pre_ffn_two_and_layer_two_output(self):
        """``2`` means block 2 and computes CKA(sg(pre-FFN_2), H_2)."""
        supported = {
            name: (model_cls, config_factory)
            for name, model_cls, config_factory in _backbone_factories()
            if name in {"llama", "mistral", "qwen2", "qwen3", "gemma3", "phi3"}
        }

        for name, (model_cls, config_factory) in supported.items():
            with self.subTest(backbone=name):
                torch.manual_seed(37)
                config = _configure_multimodal_cka(config_factory())
                config.cka_loss_layers = "2"
                config.cka_loss_projector_weight = 0.0
                config.cka_loss_final_hidden_weight = 1.0
                model = model_cls(config).train()
                model.get_model().vision_tower = _FakeVisionTower()
                model.get_model().mm_projector = _projector()

                layers = model.get_model().layers
                observed_pre_ffn = {}
                observed_outputs = {}
                observer_handles = []

                for one_based_idx, layer in enumerate(layers, start=1):
                    def capture_pre_ffn(module, module_inputs, idx=one_based_idx):
                        observed_pre_ffn[idx] = module_inputs[0]

                    def capture_output(module, module_inputs, module_output, idx=one_based_idx):
                        observed_outputs[idx] = (
                            module_output[0]
                            if isinstance(module_output, (tuple, list))
                            else module_output
                        )

                    observer_handles.append(layer.mlp.register_forward_pre_hook(capture_pre_ffn))
                    observer_handles.append(layer.register_forward_hook(capture_output))

                recorded_calls = []

                def record_masked_cka(
                    projected_features,
                    layer_hidden_states,
                    vision_feature_mask,
                    **kwargs,
                ):
                    recorded_calls.append(
                        (projected_features, layer_hidden_states, vision_feature_mask)
                    )
                    # Keep the replacement loss differentiable regardless of the
                    # argument order chosen by the implementation. Exactly one
                    # endpoint must be trainable; the pre-FFN endpoint is sg(...).
                    endpoints = (projected_features, layer_hidden_states)
                    trainable = [feature for feature in endpoints if feature.requires_grad]
                    self.assertEqual(len(trainable), 1)
                    return trainable[0].float().square().mean()

                model._compute_masked_linear_cka_loss = record_masked_cka
                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                labels = input_ids.clone()
                images = torch.randn(1, 3, 2, 2)

                try:
                    output = model(input_ids=input_ids, labels=labels, images=images)
                finally:
                    for handle in observer_handles:
                        handle.remove()

                self.assertEqual(len(recorded_calls), 1)
                endpoint_a, endpoint_b, feature_mask = recorded_calls[0]
                detached = [x for x in (endpoint_a, endpoint_b) if not x.requires_grad]
                trainable = [x for x in (endpoint_a, endpoint_b) if x.requires_grad]
                self.assertEqual(len(detached), 1)
                self.assertEqual(len(trainable), 1)

                pre_ffn_reference = detached[0]
                hidden_target = trainable[0]
                expected_pre_ffn = observed_pre_ffn[2]
                expected_hidden = observed_outputs[2]

                # Stop-gradient changes autograd metadata, not the represented
                # value. Both endpoints must come from transformer block 2.
                self.assertFalse(pre_ffn_reference.requires_grad)
                self.assertIsNone(pre_ffn_reference.grad_fn)
                self.assertTrue(expected_pre_ffn.requires_grad)
                self.assertEqual(pre_ffn_reference.shape, expected_pre_ffn.shape)
                self.assertTrue(torch.equal(pre_ffn_reference, expected_pre_ffn.detach()))

                self.assertTrue(hidden_target.requires_grad)
                self.assertEqual(hidden_target.shape, expected_hidden.shape)
                self.assertTrue(torch.equal(hidden_target, expected_hidden))

                # This also guards the 1-based contract: "2" must not select
                # the first block's MLP input or output.
                self.assertFalse(torch.equal(pre_ffn_reference, observed_pre_ffn[1].detach()))
                self.assertFalse(torch.equal(hidden_target, observed_outputs[1]))

                self.assertEqual(feature_mask.shape, hidden_target.shape[:2])
                self.assertEqual(int(feature_mask.sum().item()), 5)
                self.assertEqual(len(output.aux_losses), 1)

    def test_real_layer_two_cka_matches_formula_and_backpropagates(self):
        """Exercise the real CKA kernel rather than the argument-recording stub."""
        supported = {
            name: (model_cls, config_factory)
            for name, model_cls, config_factory in _backbone_factories()
            if name in {"llama", "mistral", "qwen2", "qwen3", "gemma3", "phi3"}
        }
        for name, (model_cls, config_factory) in supported.items():
            with self.subTest(backbone=name):
                torch.manual_seed(41)
                config = _configure_multimodal_cka(config_factory())
                config.cka_loss_layers = "2"
                config.cka_loss_projector_weight = 0.0
                config.cka_loss_final_hidden_weight = 1.0
                model = model_cls(config).train()
                model.get_model().vision_tower = _FakeVisionTower()
                model.get_model().mm_projector = _projector()

                selected_layer = model.get_model().layers[1]
                observed = {}

                def capture_pre_ffn(module, module_inputs):
                    observed["pre_ffn"] = module_inputs[0]

                def capture_output(module, module_inputs, module_output):
                    observed["hidden"] = (
                        module_output[0]
                        if isinstance(module_output, (tuple, list))
                        else module_output
                    )

                handles = [
                    selected_layer.mlp.register_forward_pre_hook(capture_pre_ffn),
                    selected_layer.register_forward_hook(capture_output),
                ]
                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                labels = input_ids.clone()
                images = torch.randn(1, 3, 2, 2)
                try:
                    output = model(input_ids=input_ids, labels=labels, images=images)
                finally:
                    for handle in handles:
                        handle.remove()

                self.assertEqual(len(output.aux_losses), 1)
                actual = output.aux_losses[0]
                image_mask = torch.zeros(
                    observed["hidden"].shape[:2],
                    dtype=torch.bool,
                    device=observed["hidden"].device,
                )
                image_mask[:, 1:6] = True
                expected = model._compute_masked_linear_cka_loss(
                    projected_features=observed["hidden"],
                    layer_hidden_states=observed["pre_ffn"].detach(),
                    vision_feature_mask=image_mask,
                )
                self.assertTrue(torch.isfinite(actual))
                self.assertGreater(actual.item(), 0.0)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

                mlp_parameters = [
                    parameter
                    for parameter in selected_layer.mlp.parameters()
                    if parameter.requires_grad
                ]
                mlp_grads = torch.autograd.grad(
                    actual,
                    mlp_parameters,
                    allow_unused=True,
                    retain_graph=True,
                )
                nonzero_mlp_grads = [
                    grad
                    for grad in mlp_grads
                    if grad is not None and grad.abs().sum().item() > 0.0
                ]
                self.assertTrue(nonzero_mlp_grads)
                self.assertTrue(all(torch.isfinite(grad).all() for grad in nonzero_mlp_grads))

                projector_grad = torch.autograd.grad(
                    actual,
                    model.get_model().mm_projector.weight,
                    allow_unused=True,
                )[0]
                self.assertIsNotNone(projector_grad)
                self.assertTrue(torch.isfinite(projector_grad).all())
                self.assertGreater(projector_grad.abs().sum().item(), 0.0)

    def test_final_and_last_aliases_dedupe_the_last_numeric_layer(self):
        for name, model_cls, config_factory in _backbone_factories():
            if name not in {"llama", "mistral", "qwen2", "qwen3", "gemma3", "phi3"}:
                continue
            with self.subTest(backbone=name):
                model = model_cls(_configure_multimodal_cka(config_factory()))
                for raw_layers in (
                    "2,final",
                    "final,2",
                    "2,last",
                    "last,2",
                    "all,final",
                    "final,all",
                ):
                    with self.subTest(layers=raw_layers):
                        model.get_model().config.cka_loss_layers = raw_layers
                        specs = model._get_cka_layer_specs()
                        layer_indices = [spec["layer_idx"] for spec in specs]
                        self.assertEqual(len(layer_indices), len(set(layer_indices)))
                        self.assertEqual(layer_indices.count(2), 1)


if __name__ == "__main__":
    unittest.main()
