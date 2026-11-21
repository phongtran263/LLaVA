"""CPU regressions for the optional last-decoder-block CKA target MLP."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.llava_arch import compute_linear_cka_loss
from tests.test_cka_backbones import (
    _FakeVisionTower,
    _backbone_factories,
    _configure_multimodal_cka,
    _projector,
)


def _supported_backbones():
    return [entry for entry in _backbone_factories() if entry[0] != "mpt"]


def _tiny_model(name="llama", layers="final", multimodal=True):
    _, model_cls, config_factory = next(
        entry for entry in _supported_backbones() if entry[0] == name
    )
    config = _configure_multimodal_cka(config_factory())
    config.cka_loss_layers = layers
    config.cka_loss_final_mlp_head = True
    model = model_cls(config).train()
    if multimodal:
        model.get_model().vision_tower = _FakeVisionTower()
        model.get_model().mm_projector = _projector()
    return model


def _image_batch():
    input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
    return dict(
        input_ids=input_ids,
        labels=input_ids.clone(),
        images=torch.zeros(1, 3, 2, 2),
    )


def _initialize_from_projector_adapter(backbone, adapter_path):
    tower = _FakeVisionTower()
    tower.hidden_size = 6
    projector = nn.Linear(6, backbone.config.hidden_size).to(dtype=backbone.dtype)
    model_args = SimpleNamespace(
        vision_tower="fake-vision-tower",
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        pretrain_mm_mlp_adapter=str(adapter_path),
        mm_patch_merge_type="flat",
        mm_projector_type="linear",
    )
    with mock.patch("llava.model.llava_arch.build_vision_tower", return_value=tower), mock.patch(
        "llava.model.llava_arch.build_vision_projector", return_value=projector,
    ):
        backbone.initialize_vision_modules(model_args)


class CkaFinalMlpHeadTests(unittest.TestCase):
    def test_disabled_has_no_module_parameters_or_rng_effect(self):
        for name, model_cls, config_factory in _supported_backbones():
            with self.subTest(backbone=name):
                torch.manual_seed(103)
                missing_flag = model_cls(config_factory())
                missing_rng = torch.get_rng_state().clone()
                config = config_factory()
                config.cka_loss_final_mlp_head = False
                torch.manual_seed(103)
                disabled = model_cls(config)
                self.assertFalse(hasattr(missing_flag.get_model(), "cka_final_mlp_head"))
                self.assertFalse(hasattr(disabled.get_model(), "cka_final_mlp_head"))
                self.assertTrue(torch.equal(missing_rng, torch.get_rng_state()))
                self.assertEqual(missing_flag.state_dict().keys(), disabled.state_dict().keys())
                for key, value in missing_flag.state_dict().items():
                    torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)

    def test_enabled_head_shape_and_initialization_is_idempotent(self):
        for name, _, _ in _supported_backbones():
            with self.subTest(backbone=name):
                model = _tiny_model(name)
                backbone = model.get_model()
                head = backbone.cka_final_mlp_head
                self.assertIsInstance(head, nn.Sequential)
                self.assertEqual(len(head), 3)
                self.assertIsInstance(head[0], nn.Linear)
                self.assertIsInstance(head[1], nn.GELU)
                self.assertIsInstance(head[2], nn.Linear)
                self.assertEqual((head[0].in_features, head[0].out_features), (16, 16))
                self.assertEqual((head[2].in_features, head[2].out_features), (16, 16))
                parameters = tuple(head.parameters())
                rng = torch.get_rng_state().clone()
                backbone.initialize_cka_final_mlp_head()
                self.assertIs(backbone.cka_final_mlp_head, head)
                self.assertTrue(all(a is b for a, b in zip(parameters, head.parameters())))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_only_last_block_selectors_route_raw_pre_norm_hidden_through_head(self):
        for name, _, _ in _supported_backbones():
            for selection, expected_blocks in (
                ("2", [2]), ("final", [2]), ("all", [1, 2]), ("1", [1]),
            ):
                with self.subTest(backbone=name, selection=selection):
                    torch.manual_seed(107)
                    model = _tiny_model(name, layers=selection)
                    observed = {}
                    handles = []
                    for block_idx, layer in enumerate(model.get_model().layers, start=1):
                        def capture(_module, _inputs, output, idx=block_idx):
                            observed[idx] = output[0] if isinstance(output, (tuple, list)) else output
                        handles.append(layer.register_forward_hook(capture))

                    def capture_norm(_module, _inputs, output):
                        observed["normalized"] = output

                    handles.append(model.get_model().norm.register_forward_hook(capture_norm))
                    head_inputs = []
                    handles.append(model.get_model().cka_final_mlp_head.register_forward_pre_hook(
                        lambda _module, inputs: head_inputs.append(inputs[0].detach().clone())
                    ))
                    try:
                        with mock.patch.object(
                            model, "_compute_masked_linear_cka_loss",
                            wraps=model._compute_masked_linear_cka_loss,
                        ) as masked_cka:
                            model(**_image_batch())
                    finally:
                        for handle in handles:
                            handle.remove()

                    self.assertEqual(masked_cka.call_count, len(expected_blocks))
                    for call, block_idx in zip(masked_cka.call_args_list, expected_blocks):
                        target = call.kwargs["projected_features"]
                        reference = call.kwargs["layer_hidden_states"]
                        mask = call.kwargs["vision_feature_mask"]
                        self.assertIs(target, observed[block_idx])
                        self.assertFalse(reference.requires_grad)
                        self.assertIsNone(reference.grad_fn)
                        self.assertEqual(int(mask.sum()), 5)
                        if block_idx == 2:
                            self.assertIs(call.kwargs.get("target_head"), model.get_model().cka_final_mlp_head)
                            self.assertFalse(torch.equal(target, observed["normalized"]))
                        else:
                            self.assertIsNone(call.kwargs.get("target_head"))
                    self.assertEqual(len(head_inputs), int(2 in expected_blocks))
                    if head_inputs:
                        # No full-sequence/text or compact-batch padding is fed to the MLP.
                        self.assertEqual(head_inputs[0].numel(), 5 * 16)

    def test_toggling_head_preserves_ce_logits_projector_and_intermediate_cka(self):
        for name, _, _ in _supported_backbones():
            with self.subTest(backbone=name):
                torch.manual_seed(109)
                model = _tiny_model(name, layers="all")
                enabled = model(**_image_batch())
                enabled_layers = dict(model.last_cka_per_layer_losses)
                model.config.cka_loss_final_mlp_head = False
                disabled = model(**_image_batch())
                disabled_layers = dict(model.last_cka_per_layer_losses)
                torch.testing.assert_close(enabled.logits, disabled.logits, rtol=0, atol=0)
                torch.testing.assert_close(enabled.loss, disabled.loss, rtol=0, atol=0)
                torch.testing.assert_close(enabled.projector_cka_loss, disabled.projector_cka_loss, rtol=0, atol=0)
                self.assertEqual(enabled_layers.keys(), disabled_layers.keys())
                torch.testing.assert_close(
                    enabled_layers["vision_encoder_to_layer_1"],
                    disabled_layers["vision_encoder_to_layer_1"], rtol=0, atol=0,
                )
                self.assertGreater(float(abs(
                    enabled_layers["vision_encoder_to_layer_2"]
                    - disabled_layers["vision_encoder_to_layer_2"]
                )), 1e-7)

    def test_masked_head_sees_only_visual_tokens_with_fp32_cka_and_target_gradients(self):
        torch.manual_seed(113)
        model = _tiny_model()
        head = model.get_model().cka_final_mlp_head
        mask = torch.tensor([
            [True, False, True, True, False, True],
            [False, True, True, False, True, False],
        ])
        target = torch.randn(2, 6, 16, requires_grad=True)
        anchor = torch.randn(2, 6, 6, requires_grad=True)
        with torch.no_grad():
            target[~mask] = float("nan")
            anchor[~mask] = float("nan")
        expected = torch.stack([
            compute_linear_cka_loss(
                head(target[idx, keep]).unsqueeze(0),
                anchor[idx, keep].detach().unsqueeze(0),
            )
            for idx, keep in enumerate(mask)
        ]).mean()
        seen = []
        handle = head.register_forward_pre_hook(
            lambda _module, inputs: seen.append(inputs[0].detach().clone())
        )
        try:
            actual = model._compute_masked_linear_cka_loss(
                target, anchor.detach(), mask, target_head=head,
            )
        finally:
            handle.remove()
        self.assertEqual(sum(value.numel() for value in seen), int(mask.sum()) * 16)
        self.assertTrue(all(torch.isfinite(value).all() for value in seen))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        actual.backward()
        self.assertIsNone(anchor.grad)
        self.assertTrue(torch.isfinite(target.grad).all())
        self.assertEqual(int(torch.count_nonzero(target.grad[~mask])), 0)
        self.assertGreater(float(target.grad[mask].norm()), 0)
        for layer in (head[0], head[2]):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.norm()), 0)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            autocast_loss = model._compute_masked_linear_cka_loss(
                target.detach(), anchor.detach(), mask, target_head=head,
            )
        self.assertEqual(autocast_loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(autocast_loss))
        autocast_loss.backward()
        self.assertTrue(torch.isfinite(head[0].weight.grad).all())

    def test_final_cka_backpropagates_to_head_decoder_and_projector_not_text_hidden(self):
        torch.manual_seed(127)
        model = _tiny_model()
        model.config.cka_loss_projector_weight = 0.0
        captured = []

        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            hidden.retain_grad()
            captured.append(hidden)

        handle = model.get_model().layers[-1].register_forward_hook(capture)
        try:
            output = model(**_image_batch())
        finally:
            handle.remove()
        sum(output.aux_losses).backward()
        final_hidden = captured[0]
        self.assertEqual(int(torch.count_nonzero(final_hidden.grad[:, [0, 6, 7]])), 0)
        self.assertGreater(float(final_hidden.grad[:, 1:6].norm()), 0)
        modules = (
            model.get_model().cka_final_mlp_head,
            model.get_model().layers[-1].mlp,
            model.get_model().mm_projector,
        )
        for module in modules:
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.norm()) for gradient in gradients), 0)

    def test_no_image_batch_and_empty_visual_mask_are_safe(self):
        model = _tiny_model()
        head = model.get_model().cka_final_mlp_head
        with mock.patch.object(head, "forward", wraps=head.forward) as forward:
            ids = torch.tensor([[1, 2, 3, 4]])
            output = model(input_ids=ids, labels=ids.clone())
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertEqual(forward.call_count, 0)
            self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))
            target = torch.randn(2, 4, 16, requires_grad=True)
            loss = model._compute_masked_linear_cka_loss(
                target, torch.randn(2, 4, 6), torch.zeros(2, 4, dtype=torch.bool),
                target_head=head,
            )
            self.assertEqual(float(loss), 0)
            loss.backward()
            torch.testing.assert_close(target.grad, torch.zeros_like(target), rtol=0, atol=0)
            self.assertEqual(forward.call_count, 0)

    def test_nonreentrant_checkpointing_trains_head_and_projector_with_frozen_decoder(self):
        torch.manual_seed(131)
        model = _tiny_model()
        model.config.cka_loss_projector_weight = 0.0
        model.requires_grad_(False)
        model.get_model().cka_final_mlp_head.requires_grad_(True)
        model.get_model().mm_projector.requires_grad_(True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        output = model(**_image_batch())
        sum(output.aux_losses).backward()
        for parameter in (
            model.get_model().cka_final_mlp_head[0].weight,
            model.get_model().cka_final_mlp_head[2].weight,
            model.get_model().mm_projector.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()), 0)

    def test_save_pretrained_roundtrip_restores_head_and_config(self):
        torch.manual_seed(137)
        model = _tiny_model(multimodal=False)
        with torch.no_grad():
            model.get_model().cka_final_mlp_head[0].weight.add_(0.125)
        with tempfile.TemporaryDirectory(prefix="llava-final-cka-test-") as directory:
            model.save_pretrained(directory)
            restored = type(model).from_pretrained(directory, local_files_only=True)
        self.assertIs(restored.config.cka_loss_final_mlp_head, True)
        source_head = model.get_model().cka_final_mlp_head.state_dict()
        loaded_head = restored.get_model().cka_final_mlp_head.state_dict()
        self.assertEqual(source_head.keys(), loaded_head.keys())
        for name, value in source_head.items():
            torch.testing.assert_close(value, loaded_head[name], rtol=0, atol=0)

    def test_projector_adapter_restores_saved_head_in_backbone_dtype(self):
        torch.manual_seed(139)
        model = _tiny_model(multimodal=False).to(dtype=torch.bfloat16)
        backbone = model.get_model()
        projector_state = nn.Linear(6, 16).to(dtype=backbone.dtype).state_dict()
        head_state = {
            name: torch.full_like(value, (index + 1) / 8)
            for index, (name, value) in enumerate(backbone.cka_final_mlp_head.state_dict().items())
        }
        adapter_state = {
            **{f"model.mm_projector.{name}": value for name, value in projector_state.items()},
            **{f"model.cka_final_mlp_head.{name}": value for name, value in head_state.items()},
        }
        with tempfile.TemporaryDirectory(prefix="llava-final-cka-adapter-test-") as directory:
            adapter_path = Path(directory) / "mm_projector.bin"
            torch.save(adapter_state, adapter_path)
            _initialize_from_projector_adapter(backbone, adapter_path)
        for name, value in projector_state.items():
            torch.testing.assert_close(backbone.mm_projector.state_dict()[name], value, rtol=0, atol=0)
        for name, value in head_state.items():
            restored = backbone.cka_final_mlp_head.state_dict()[name]
            self.assertEqual(restored.dtype, backbone.dtype)
            self.assertEqual(restored.shape, value.shape)
            torch.testing.assert_close(restored, value, rtol=0, atol=0)

    def test_old_projector_only_adapter_initializes_new_head_without_saved_head_weights(self):
        torch.manual_seed(149)
        _, model_cls, config_factory = _supported_backbones()[0]
        config = _configure_multimodal_cka(config_factory())
        config.cka_loss_final_mlp_head = False
        backbone = model_cls(config).to(dtype=torch.bfloat16).get_model()
        self.assertFalse(hasattr(backbone, "cka_final_mlp_head"))
        backbone.config.cka_loss_final_mlp_head = True
        projector_state = nn.Linear(6, 16).to(dtype=backbone.dtype).state_dict()
        adapter_state = {f"model.mm_projector.{name}": value for name, value in projector_state.items()}
        with tempfile.TemporaryDirectory(prefix="llava-final-cka-old-adapter-test-") as directory:
            adapter_path = Path(directory) / "mm_projector.bin"
            torch.save(adapter_state, adapter_path)
            _initialize_from_projector_adapter(backbone, adapter_path)
        for name, value in projector_state.items():
            torch.testing.assert_close(backbone.mm_projector.state_dict()[name], value, rtol=0, atol=0)
        head = backbone.cka_final_mlp_head
        self.assertEqual(head[0].weight.shape, (16, 16))
        self.assertEqual(head[2].weight.shape, (16, 16))
        self.assertTrue(all(parameter.dtype == backbone.dtype for parameter in head.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in head.parameters()))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in head.parameters()))
        self.assertGreater(float(head[0].weight.norm()), 0)


if __name__ == "__main__":
    unittest.main()
