import unittest
from types import SimpleNamespace

import torch

from llava.constants import IMAGE_TOKEN_INDEX
from tests.test_cka_backbones import (
    _FakeVisionTower,
    _backbone_factories,
    _configure_multimodal_cka,
    _projector,
)


class CkaVisionAnchorSemanticsTests(unittest.TestCase):
    def test_layer_two_uses_detached_vision_anchor_and_layer_two_output(self):
        """``2`` means block 2 and computes CKA(sg(V), H_2)."""
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
                # An unset anchor is the backward-compatible raw vision feature V.
                config.cka_loss_anchor_layer = None
                config.cka_loss_projector_weight = 0.0
                config.cka_loss_final_hidden_weight = 1.0
                model = model_cls(config).train()
                vision_tower = _FakeVisionTower()
                # Make stop-gradient observable at the anchor endpoint. V also
                # stays on the normal projector -> decoder path, so gradients
                # may still reach the tower through H_2.
                vision_tower.features.requires_grad_(True)
                model.get_model().vision_tower = vision_tower
                model.get_model().mm_projector = _projector()

                layers = model.get_model().layers
                observed_outputs = {}
                observer_handles = []

                for one_based_idx, layer in enumerate(layers, start=1):

                    def capture_output(module, module_inputs, module_output, idx=one_based_idx):
                        observed_outputs[idx] = (
                            module_output[0]
                            if isinstance(module_output, (tuple, list))
                            else module_output
                        )

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
                    # H_2 is the optimization target; raw V is a fixed anchor.
                    self.assertTrue(projected_features.requires_grad)
                    self.assertFalse(layer_hidden_states.requires_grad)
                    return projected_features.float().square().mean()

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
                hidden_target, vision_reference, feature_mask = recorded_calls[0]
                expected_hidden = observed_outputs[2]

                self.assertFalse(vision_reference.requires_grad)
                self.assertIsNone(vision_reference.grad_fn)
                self.assertEqual(vision_reference.shape[-1], 6)

                self.assertTrue(hidden_target.requires_grad)
                self.assertEqual(hidden_target.shape, expected_hidden.shape)
                self.assertTrue(torch.equal(hidden_target, expected_hidden))

                # This guards the 1-based contract: "2" must not select H_1.
                self.assertFalse(torch.equal(hidden_target, observed_outputs[1]))

                self.assertEqual(feature_mask.shape, hidden_target.shape[:2])
                self.assertEqual(int(feature_mask.sum().item()), 5)
                self.assertEqual(len(output.aux_losses), 1)
                self.assertTrue(torch.equal(
                    vision_reference[feature_mask],
                    vision_tower.features.detach(),
                ))
                self.assertEqual(
                    torch.count_nonzero(vision_reference[~feature_mask]).item(),
                    0,
                )

    def test_layer_anchor_is_one_based_detached_and_reused_across_targets(self):
        """Layer anchor 1 means sg(H_1), reused for H_2 and H_3 targets."""
        supported = {
            name: (model_cls, config_factory)
            for name, model_cls, config_factory in _backbone_factories()
            if name in {"llama", "mistral", "qwen2", "qwen3", "gemma3", "phi3"}
        }

        for name, (model_cls, config_factory) in supported.items():
            with self.subTest(backbone=name):
                torch.manual_seed(43)
                config = _configure_multimodal_cka(config_factory())
                config.num_hidden_layers = 3
                config.cka_loss_layers = "2,3"
                config.cka_loss_anchor_layer = 1
                config.cka_loss_projector_weight = 0.0
                config.cka_loss_final_hidden_weight = 1.0
                model = model_cls(config).train()
                model.get_model().vision_tower = _FakeVisionTower()
                model.get_model().mm_projector = _projector()

                observed_outputs = {}
                observer_handles = []
                for one_based_idx, layer in enumerate(model.get_model().layers, start=1):

                    def capture_output(module, module_inputs, module_output, idx=one_based_idx):
                        observed_outputs[idx] = (
                            module_output[0]
                            if isinstance(module_output, (tuple, list))
                            else module_output
                        )

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
                    return projected_features.float().square().mean()

                model._compute_masked_linear_cka_loss = record_masked_cka
                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                labels = input_ids.clone()
                images = torch.randn(1, 3, 2, 2)

                try:
                    output = model(input_ids=input_ids, labels=labels, images=images)
                finally:
                    for handle in observer_handles:
                        handle.remove()

                self.assertEqual(len(recorded_calls), 2)
                target_hiddens = [call[0] for call in recorded_calls]
                anchor_references = [call[1] for call in recorded_calls]
                feature_masks = [call[2] for call in recorded_calls]

                # Targets are the exact 1-based block outputs H_2 and H_3.
                self.assertTrue(torch.equal(target_hiddens[0], observed_outputs[2]))
                self.assertTrue(torch.equal(target_hiddens[1], observed_outputs[3]))
                self.assertFalse(torch.equal(target_hiddens[0], observed_outputs[1]))
                self.assertTrue(all(target.requires_grad for target in target_hiddens))

                # The same stopped-gradient H_1 endpoint is used for every target.
                first_anchor = anchor_references[0]
                self.assertFalse(first_anchor.requires_grad)
                self.assertIsNone(first_anchor.grad_fn)
                self.assertTrue(observed_outputs[1].requires_grad)
                self.assertTrue(torch.equal(first_anchor, observed_outputs[1].detach()))
                self.assertEqual(
                    first_anchor.untyped_storage().data_ptr(),
                    observed_outputs[1].untyped_storage().data_ptr(),
                )
                for anchor, feature_mask in zip(
                    anchor_references[1:], feature_masks[1:]
                ):
                    self.assertFalse(anchor.requires_grad)
                    self.assertIsNone(anchor.grad_fn)
                    self.assertEqual(
                        anchor.untyped_storage().data_ptr(),
                        first_anchor.untyped_storage().data_ptr(),
                    )
                    self.assertTrue(torch.equal(anchor, first_anchor))
                    self.assertTrue(torch.equal(feature_mask, feature_masks[0]))

                self.assertEqual(
                    list(model.last_cka_per_layer_losses),
                    ["layer_1_to_layer_2", "layer_1_to_layer_3"],
                )
                self.assertEqual(len(output.aux_losses), 1)

                # Stop-gradient applies only to the reference branch. The target
                # H_3 branch must remain trainable through block 3.
                target_parameters = [
                    parameter
                    for parameter in model.get_model().layers[2].mlp.parameters()
                    if parameter.requires_grad
                ]
                target_grads = torch.autograd.grad(
                    output.aux_losses[0],
                    target_parameters,
                    allow_unused=True,
                )
                self.assertTrue(any(
                    grad is not None and grad.abs().sum().item() > 0.0
                    for grad in target_grads
                ))

    def test_invalid_layer_anchor_is_rejected_instead_of_falling_back(self):
        backbone_factories = {
            name: (model_cls, config_factory)
            for name, model_cls, config_factory in _backbone_factories()
        }
        model_cls, config_factory = backbone_factories["llama"]
        for invalid_anchor in (-1, 0, 3):
            with self.subTest(anchor=invalid_anchor):
                config = _configure_multimodal_cka(config_factory())
                config.cka_loss_layers = "2"
                config.cka_loss_anchor_layer = invalid_anchor
                config.cka_loss_projector_weight = 0.0
                config.cka_loss_final_hidden_weight = 1.0
                model = model_cls(config).train()
                model.get_model().vision_tower = _FakeVisionTower()
                model.get_model().mm_projector = _projector()

                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                with self.assertRaisesRegex(ValueError, "cka_loss_anchor_layer"):
                    model(
                        input_ids=input_ids,
                        labels=input_ids.clone(),
                        images=torch.randn(1, 3, 2, 2),
                    )

    def test_layer_anchor_supports_spatial_list_and_5d_images(self):
        """A hidden anchor needs no raw-V alignment through spatial merging."""
        image_inputs = {
            "list": [torch.randn(2, 3, 2, 2)],
            "5d": torch.randn(1, 2, 3, 2, 2),
        }
        llama_factory = {
            name: (model_cls, config_factory)
            for name, model_cls, config_factory in _backbone_factories()
        }["llama"]

        for input_kind, images in image_inputs.items():
            with self.subTest(input_kind=input_kind):
                torch.manual_seed(47)
                model_cls, config_factory = llama_factory
                config = _configure_multimodal_cka(config_factory())
                config.cka_loss_layers = "2"
                config.cka_loss_anchor_layer = 1
                config.cka_loss_projector_weight = 0.0
                config.cka_loss_final_hidden_weight = 1.0
                config.mm_patch_merge_type = "spatial"
                config.image_aspect_ratio = "anyres"
                config.image_grid_pinpoints = [(2, 2)]
                model = model_cls(config).train()

                vision_tower = _FakeVisionTower()
                generator = torch.Generator().manual_seed(109)
                vision_tower.features = torch.randn(4, 6, generator=generator)
                vision_tower.num_patches_per_side = 2
                vision_tower.config = SimpleNamespace(image_size=2)
                model.get_model().vision_tower = vision_tower
                model.get_model().mm_projector = _projector()

                recorded_masks = []

                def record_masked_cka(
                    projected_features,
                    layer_hidden_states,
                    vision_feature_mask,
                    **kwargs,
                ):
                    self.assertEqual(projected_features.shape, layer_hidden_states.shape)
                    self.assertFalse(layer_hidden_states.requires_grad)
                    recorded_masks.append(vision_feature_mask)
                    return projected_features.float().square().mean()

                model._compute_masked_linear_cka_loss = record_masked_cka
                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                output = model(
                    input_ids=input_ids,
                    labels=input_ids.clone(),
                    images=images,
                    image_sizes=[(2, 2)],
                )

                self.assertEqual(len(recorded_masks), 1)
                # Four base-image patches plus four patches from the tiled view.
                self.assertEqual(int(recorded_masks[0].sum().item()), 8)
                self.assertEqual(len(output.aux_losses), 1)
                self.assertTrue(torch.isfinite(output.aux_losses[0]))

    def test_real_layer_two_vision_cka_matches_formula_and_backpropagates(self):
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


                def capture_output(module, module_inputs, module_output):
                    observed["hidden"] = (
                        module_output[0]
                        if isinstance(module_output, (tuple, list))
                        else module_output
                    )

                handles = [selected_layer.register_forward_hook(capture_output)]
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
                vision_reference = torch.zeros(
                    (*observed["hidden"].shape[:2], 6),
                    dtype=model.get_model().vision_tower.features.dtype,
                    device=observed["hidden"].device,
                )
                vision_reference[image_mask] = (
                    model.get_model().vision_tower.features.detach()
                )
                expected = model._compute_masked_linear_cka_loss(
                    projected_features=observed["hidden"],
                    layer_hidden_states=vision_reference,
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
