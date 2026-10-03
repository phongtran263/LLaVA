import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.llava_arch import compute_linear_cka_loss, validate_cka_vision_anchor
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from tests.test_cka_backbones import _backbone_factories, _configure_multimodal_cka


def tiny_tower(num_hidden_layers=3):
    # Entirely local random weights: no downloads or GPU required.
    tower = CLIPVisionTower.__new__(CLIPVisionTower)
    nn.Module.__init__(tower)
    tower.vision_tower = CLIPVisionModel(CLIPVisionConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=num_hidden_layers,
        num_attention_heads=2, image_size=4, patch_size=2,
    )).requires_grad_(False)
    tower.is_loaded = True
    tower.select_layer = -2
    tower.select_feature = "patch"
    return tower


def tiny_vlm(name="llama", layers="1", num_decoder_layers=None):
    model_cls, config_factory = next(
        (cls, factory) for family, cls, factory in _backbone_factories() if family == name
    )
    config = _configure_multimodal_cka(config_factory())
    if num_decoder_layers is not None:
        config.num_hidden_layers = num_decoder_layers
    config.cka_loss_vision_anchor_layer = 1
    config.cka_loss_layers = layers
    config.cka_loss_anchor_layer = None
    model = model_cls(config).train()
    model.get_model().vision_tower = tiny_tower()
    model.get_model().mm_projector = nn.Linear(8, 16)
    return model


class VisionAnchorTests(unittest.TestCase):
    def test_qwen_layer12_and_final_with_v20_v24(self):
        model = tiny_vlm("qwen2", layers="12,final", num_decoder_layers=28)
        model.get_model().vision_tower = tiny_tower(num_hidden_layers=24)
        model.config.cka_loss_vision_anchor_layer = 20
        model.config.cka_loss_projector_vision_anchor_layer = 24
        model.config.cka_loss_final_vision_anchor_layer = 24
        images = torch.randn(1, 3, 4, 4)
        inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        states = model.get_vision_tower().vision_tower(images, output_hidden_states=True).hidden_states
        with mock.patch.object(model, "_compute_masked_linear_cka_loss", wraps=model._compute_masked_linear_cka_loss) as cka:
            output = model(input_ids=inputs, labels=inputs.clone(), images=images)
        self.assertEqual(cka.call_count, 2)
        for call, anchor in zip(cka.call_args_list, (20, 24)):
            mask = call.kwargs["vision_feature_mask"]
            torch.testing.assert_close(call.kwargs["layer_hidden_states"][mask], states[anchor][0, 1:])
        self.assertEqual(set(model.last_cka_per_layer_losses), {
            "vision_encoder_layer_20_to_layer_12", "vision_encoder_layer_24_to_final",
        })
        (output.projector_cka_loss + sum(output.aux_losses)).backward()
        self.assertTrue(torch.isfinite(model.get_model().mm_projector.weight.grad).all())

    def test_projector_v24_mid_v20_final_v24_share_one_clip_forward(self):
        tower = tiny_tower(num_hidden_layers=24)
        images = torch.randn(2, 3, 4, 4)
        states = tower.vision_tower(images, output_hidden_states=True).hidden_states
        with mock.patch.object(tower.vision_tower, "forward", wraps=tower.vision_tower.forward) as forward:
            features, mid, projector, final = tower(
                images, cka_anchor_layer=20, cka_projector_anchor_layer=24,
                cka_final_anchor_layer=24,
            )
        self.assertEqual(forward.call_count, 1)
        for actual, index in ((features, 23), (mid, 20), (projector, 24), (final, 24)):
            torch.testing.assert_close(actual, states[index][:, 1:])
            self.assertFalse(actual.requires_grad)
        self.assertIs(projector, final)
        for kwargs in ({}, {"cka_projector_anchor_layer": 1}):
            listed = tower(list(images), cka_final_anchor_layer=3, **kwargs)
            batched = tower(images, cka_final_anchor_layer=3, **kwargs)
            for values, expected in zip(listed, batched):
                torch.testing.assert_close(torch.cat(values), expected)

    def test_mid_and_final_losses_use_separate_anchors_all_backbones(self):
        for name, _, _ in _backbone_factories():
            with self.subTest(backbone=name):
                torch.manual_seed(75)
                model = tiny_vlm(name, layers="-1" if name == "mpt" else "1,final")
                model.config.cka_loss_vision_anchor_layer = 1
                model.config.cka_loss_projector_vision_anchor_layer = 3
                model.config.cka_loss_final_vision_anchor_layer = 3
                tower = model.get_vision_tower()
                images = torch.randn(1, 3, 4, 4, requires_grad=True)
                inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                features, mid, projector, final = tower(
                    images, cka_anchor_layer=1, cka_projector_anchor_layer=3, cka_final_anchor_layer=3,
                )
                expected_projector = compute_linear_cka_loss(projector, model.get_model().mm_projector(features))
                with mock.patch.object(tower.vision_tower, "forward", wraps=tower.vision_tower.forward) as forward:
                    if name == "mpt":
                        output = model(input_ids=inputs, labels=inputs.clone(), images=images)
                    else:
                        with mock.patch.object(
                            model, "_compute_masked_linear_cka_loss", wraps=model._compute_masked_linear_cka_loss,
                        ) as cka:
                            output = model(input_ids=inputs, labels=inputs.clone(), images=images)
                        self.assertEqual(cka.call_count, 2)
                        for call, expected in zip(cka.call_args_list, (mid, final)):
                            reference = call.kwargs["layer_hidden_states"]
                            mask = call.kwargs["vision_feature_mask"]
                            torch.testing.assert_close(reference[mask], expected[0])
                            self.assertEqual(reference.shape[-1], 8)  # Never concatenate anchors in CKA.
                            self.assertFalse(reference.requires_grad)
                            self.assertTrue(call.kwargs["projected_features"].requires_grad)
                        self.assertEqual(set(model.last_cka_per_layer_losses), {
                            "vision_encoder_layer_1_to_layer_1", "vision_encoder_layer_3_to_final",
                        })
                self.assertEqual(forward.call_count, 1)
                torch.testing.assert_close(output.projector_cka_loss, expected_projector)
                (output.projector_cka_loss + sum(output.aux_losses or [])).backward()
                grad = model.get_model().mm_projector.weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.norm().item(), 0)
                if name != "mpt":
                    for layer in model.get_model().layers:
                        grads = [p.grad for p in layer.parameters() if p.grad is not None]
                        self.assertTrue(grads)
                        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                        self.assertTrue(any(g.norm().item() > 0 for g in grads))
                self.assertIsNone(images.grad)
                self.assertTrue(all(p.grad is None for p in tower.parameters()))

    def test_final_override_numeric_all_and_duplicate_final_targets(self):
        model = tiny_vlm("qwen2")
        model.config.cka_loss_final_vision_anchor_layer = 3
        inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        images = torch.randn(1, 3, 4, 4)
        for layers in ("1,final", "1,2", "all", "1,2,final", "1,final,2", "final", "1"):
            with self.subTest(layers=layers):
                model.config.cka_loss_layers = layers
                _, mid, _, final = model.get_vision_tower()(images, cka_anchor_layer=1, cka_final_anchor_layer=3)
                with mock.patch.object(
                    model, "_compute_masked_linear_cka_loss", wraps=model._compute_masked_linear_cka_loss,
                ) as cka:
                    model(input_ids=inputs, labels=inputs.clone(), images=images)
                specs = model._get_cka_layer_specs()
                self.assertEqual(cka.call_count, len(specs))
                self.assertEqual(len(specs), 1 if layers in ("final", "1") else 2)
                for call, spec in zip(cka.call_args_list, specs):
                    expected = final if spec["layer_idx"] == 2 else mid
                    mask = call.kwargs["vision_feature_mask"]
                    torch.testing.assert_close(call.kwargs["layer_hidden_states"][mask], expected[0])

    def test_packed_mid_final_references_align_through_multicrop_padding_and_truncation(self):
        for padding in ("left", "right"):
            for kind in ("list", "5d"):
                with self.subTest(padding=padding, kind=kind):
                    model = tiny_vlm()
                    model.config.cka_loss_final_vision_anchor_layer = 3
                    model.config.tokenizer_padding_side = padding
                    model.config.tokenizer_model_max_length = 8
                    crops = torch.randn(2, 2, 3, 4, 4)
                    images = list(crops) if kind == "list" else crops
                    _, mid, _, final = model.get_vision_tower()(
                        crops.flatten(0, 1), cka_anchor_layer=1, cka_final_anchor_layer=3,
                    )
                    inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 0], [1, 2, 3, 0]])
                    result = model.prepare_inputs_labels_for_multimodal(
                        inputs, None, inputs.ne(0), None, inputs.clone(), images,
                    )
                    packed, mask = result[-1], result[-3]
                    aligned_mid, aligned_final = packed.split(8, dim=-1)
                    self.assertEqual(mask[0].sum().item(), 7)
                    for actual, expected in ((aligned_mid, mid), (aligned_final, final)):
                        torch.testing.assert_close(actual[0][mask[0]], expected[:2].flatten(0, 1)[:7])
                        self.assertEqual(actual[~mask].count_nonzero().item(), 0)
                        self.assertFalse(actual.requires_grad)
                    self.assertFalse(mask[1].any())

    def test_v20_v24_are_gathered_in_one_forward_without_changing_input(self):
        tower = tiny_tower(num_hidden_layers=24)
        images = torch.randn(2, 3, 4, 4, requires_grad=True)
        states = tower.vision_tower(images, output_hidden_states=True).hidden_states
        with mock.patch.object(tower.vision_tower, "forward", wraps=tower.vision_tower.forward) as forward:
            features, hidden_anchor, projector_anchor = tower(
                images, cka_anchor_layer=24, cka_projector_anchor_layer=20,
            )
        self.assertEqual(forward.call_count, 1)
        for actual, index in ((features, 23), (hidden_anchor, 24), (projector_anchor, 20)):
            torch.testing.assert_close(actual, states[index][:, 1:])
            self.assertFalse(actual.requires_grad)

    def test_split_anchors_list_images(self):
        tower = tiny_tower()
        images = [torch.randn(3, 4, 4), torch.randn(3, 4, 4)]
        features, hidden, projector = tower(images, cka_anchor_layer=3, cka_projector_anchor_layer=1)
        for i, image in enumerate(images):
            states = tower.vision_tower(image.unsqueeze(0), output_hidden_states=True).hidden_states
            for actual, index in ((features[i], 2), (hidden[i], 3), (projector[i], 1)):
                torch.testing.assert_close(actual, states[index][:, 1:])

    def test_indexing_same_forward_and_unchanged_projector_input(self):
        torch.manual_seed(70)
        tower = tiny_tower()
        images = torch.randn(2, 3, 4, 4, requires_grad=True)
        states = tower.vision_tower(images, output_hidden_states=True).hidden_states
        default = tower(images)
        torch.testing.assert_close(default, states[-2][:, 1:])
        for index in (0, 1, 3, -1, -2, -4):
            with self.subTest(index=index), mock.patch.object(
                tower.vision_tower, "forward", wraps=tower.vision_tower.forward
            ) as forward:
                features, anchor = tower(images, cka_anchor_layer=index)
                self.assertEqual(forward.call_count, 1)
                torch.testing.assert_close(features, default)
                torch.testing.assert_close(anchor, states[index][:, 1:])
                self.assertFalse(anchor.requires_grad)
                self.assertFalse(features.requires_grad)

    def test_list_and_cls_patch_selection(self):
        tower = tiny_tower()
        tower.select_feature = "cls_patch"
        images = [torch.randn(3, 4, 4), torch.randn(3, 4, 4)]
        features, anchors = tower(images, cka_anchor_layer=1)
        for image, features_i, anchor in zip(images, features, anchors):
            expected_features, expected_anchor = tower(image.unsqueeze(0), cka_anchor_layer=1)
            self.assertEqual(anchor.shape[1], 5)
            torch.testing.assert_close(features_i, expected_features)
            torch.testing.assert_close(anchor, expected_anchor)

    def test_validation_rejects_ambiguous_invalid_or_unsupported_anchor(self):
        config = SimpleNamespace(cka_loss_vision_anchor_layer=1, cka_loss_anchor_layer=None)
        tower = tiny_tower()
        validate_cka_vision_anchor(config, tower)
        config.cka_loss_anchor_layer = 1
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_cka_vision_anchor(config, tower)
        config.cka_loss_anchor_layer = None
        for invalid in (4, -5, 1.5, True):
            config.cka_loss_vision_anchor_layer = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "integer"):
                validate_cka_vision_anchor(config, tower)
        config.cka_loss_vision_anchor_layer = 1
        with self.assertRaisesRegex(ValueError, "standard CLIP"):
            validate_cka_vision_anchor(config, nn.Identity())
        s2 = CLIPVisionTowerS2.__new__(CLIPVisionTowerS2)
        nn.Module.__init__(s2)
        with self.assertRaisesRegex(ValueError, "S2"):
            validate_cka_vision_anchor(config, s2)
        config.cka_loss_vision_anchor_layer = None
        config.cka_loss_anchor_layer = 1
        config.cka_loss_projector_vision_anchor_layer = 1
        # A projector vision anchor does not conflict with a decoder hidden anchor.
        validate_cka_vision_anchor(config, tower)
        for invalid in (4, -5, 1.5, True):
            config.cka_loss_projector_vision_anchor_layer = invalid
            with self.subTest(projector_index=invalid), self.assertRaisesRegex(
                ValueError, "cka_loss_projector_vision_anchor_layer"
            ):
                validate_cka_vision_anchor(config, tower)
        config.cka_loss_projector_vision_anchor_layer = None
        config.cka_loss_final_vision_anchor_layer = 3
        with self.assertRaisesRegex(ValueError, "cka_loss_final_vision_anchor_layer.*mutually exclusive"):
            validate_cka_vision_anchor(config, tower)
        config.cka_loss_anchor_layer = None
        validate_cka_vision_anchor(config, tower)
        for invalid in (4, -5, 1.5, True):
            config.cka_loss_final_vision_anchor_layer = invalid
            with self.subTest(final_index=invalid), self.assertRaisesRegex(ValueError, "cka_loss_final_vision_anchor_layer"):
                validate_cka_vision_anchor(config, tower)

    def test_split_projector_and_hidden_anchor_losses_all_backbones(self):
        for name, _, _ in _backbone_factories():
            with self.subTest(backbone=name):
                torch.manual_seed(74)
                model = tiny_vlm(name, layers="-1" if name == "mpt" else "final")
                model.config.cka_loss_vision_anchor_layer = 3
                model.config.cka_loss_projector_vision_anchor_layer = 1
                tower = model.get_vision_tower()
                images = torch.randn(1, 3, 4, 4, requires_grad=True)
                inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                features, hidden_anchor, projector_anchor = tower(
                    images, cka_anchor_layer=3, cka_projector_anchor_layer=1,
                )
                expected = compute_linear_cka_loss(projector_anchor, model.get_model().mm_projector(features))
                with mock.patch.object(tower.vision_tower, "forward", wraps=tower.vision_tower.forward) as forward:
                    if name == "mpt":
                        output = model(input_ids=inputs, labels=inputs.clone(), images=images)
                    else:
                        with mock.patch.object(
                            model, "_compute_masked_linear_cka_loss", wraps=model._compute_masked_linear_cka_loss,
                        ) as cka:
                            output = model(input_ids=inputs, labels=inputs.clone(), images=images)
                        self.assertEqual(cka.call_count, 1)
                        target = cka.call_args.kwargs["projected_features"]
                        reference = cka.call_args.kwargs["layer_hidden_states"]
                        mask = cka.call_args.kwargs["vision_feature_mask"]
                        torch.testing.assert_close(reference[mask], hidden_anchor[0])
                        self.assertFalse(reference.requires_grad)
                        self.assertTrue(target.requires_grad)
                        self.assertTrue(any("vision_encoder_layer_3" in k for k in model.last_cka_per_layer_losses))
                self.assertEqual(forward.call_count, 1)
                torch.testing.assert_close(output.projector_cka_loss, expected)
                (output.projector_cka_loss + sum(output.aux_losses or [])).backward()
                grad = model.get_model().mm_projector.weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.norm().item(), 0)
                if name != "mpt":
                    grads = [p.grad for p in model.get_model().layers[0].parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertTrue(any(g.norm().item() > 0 for g in grads))
                self.assertIsNone(images.grad)
                self.assertTrue(all(p.grad is None for p in tower.parameters()))

    def test_projector_override_preserves_default_or_decoder_hidden_anchor(self):
        model = tiny_vlm()
        model.config.cka_loss_vision_anchor_layer = None
        model.config.cka_loss_projector_vision_anchor_layer = 1
        images = torch.randn(1, 3, 4, 4)
        tower = model.get_vision_tower()
        features, anchor = tower(images, cka_anchor_layer=1)
        expected = compute_linear_cka_loss(anchor, model.get_model().mm_projector(features))
        for decoder_anchor in (None, 1):
            with self.subTest(decoder_anchor=decoder_anchor):
                model.config.cka_loss_anchor_layer = decoder_anchor
                projected, loss, reference = model.encode_images(images)
                torch.testing.assert_close(loss, expected)
                torch.testing.assert_close(projected, model.get_model().mm_projector(features))
                if decoder_anchor is None:
                    torch.testing.assert_close(reference, features)
                else:
                    self.assertIsNone(reference)

    def test_projector_and_hidden_cka_share_anchor_all_backbones(self):
        for name, _, _ in _backbone_factories():
            with self.subTest(backbone=name):
                torch.manual_seed(71)
                model = tiny_vlm(name, layers="-1" if name == "mpt" else "1")
                tower = model.get_vision_tower()
                images = torch.randn(1, 3, 4, 4, requires_grad=True)
                inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
                projector_input, anchor = tower(images, cka_anchor_layer=1)
                expected_projector_loss = compute_linear_cka_loss(
                    anchor, model.get_model().mm_projector(projector_input)
                )
                calls = []
                if name != "mpt":
                    original = model._compute_masked_linear_cka_loss

                    def record(projected_features, layer_hidden_states, vision_feature_mask, **kwargs):
                        calls.append((projected_features, layer_hidden_states, vision_feature_mask))
                        return original(projected_features, layer_hidden_states, vision_feature_mask, **kwargs)

                    model._compute_masked_linear_cka_loss = record
                with mock.patch.object(tower.vision_tower, "forward", wraps=tower.vision_tower.forward) as forward:
                    output = model(input_ids=inputs, labels=inputs.clone(), images=images)
                self.assertEqual(forward.call_count, 1)
                torch.testing.assert_close(output.projector_cka_loss, expected_projector_loss)
                if name != "mpt":
                    self.assertEqual(len(calls), 1)
                    target, reference, mask = calls[0]
                    torch.testing.assert_close(reference[mask], anchor[0])
                    self.assertFalse(reference.requires_grad)
                    self.assertTrue(target.requires_grad)
                    self.assertIn("vision_encoder_layer_1_to_layer_1", model.last_cka_per_layer_losses)
                (output.projector_cka_loss + sum(output.aux_losses or [])).backward()
                grad = model.get_model().mm_projector.weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.norm().item(), 0)
                if name != "mpt":
                    decoder_grads = [
                        p.grad for p in model.get_model().layers[0].parameters() if p.grad is not None
                    ]
                    self.assertTrue(decoder_grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in decoder_grads))
                    self.assertTrue(any(g.norm().item() > 0 for g in decoder_grads))
                self.assertIsNone(images.grad)
                self.assertTrue(all(p.grad is None for p in tower.parameters()))

    def test_flat_multicrop_anchor_alignment_after_padding(self):
        for kind in ("list", "5d"):
            with self.subTest(kind=kind):
                model = tiny_vlm()
                model.config.cka_loss_projector_vision_anchor_layer = 3
                model.config.tokenizer_padding_side = "left"
                crops = torch.randn(2, 2, 3, 4, 4)
                images = list(crops) if kind == "list" else crops
                _, anchor = model.get_vision_tower()(crops.flatten(0, 1), cka_anchor_layer=1)
                inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 0], [1, IMAGE_TOKEN_INDEX, 2, 3]])
                result = model.prepare_inputs_labels_for_multimodal(
                    inputs, None, inputs.ne(0), None, inputs.clone(), images,
                )
                reference, mask = result[-1], result[-3]
                self.assertEqual(reference.shape[:2], mask.shape)
                for index in range(2):
                    torch.testing.assert_close(reference[index][mask[index]], anchor[2 * index:2 * index + 2].flatten(0, 1))
                self.assertFalse(reference.requires_grad)

    def test_anchor_does_not_change_ce_or_inference_and_disabled_cka_skips_it(self):
        torch.manual_seed(72)
        model = tiny_vlm()
        model.config.cka_loss_projector_vision_anchor_layer = 3
        model.config.cka_loss_final_vision_anchor_layer = 3
        inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        images = torch.randn(1, 3, 4, 4)
        anchored = model(input_ids=inputs, labels=inputs.clone(), images=images)
        model.config.cka_loss_vision_anchor_layer = None
        model.config.cka_loss_projector_vision_anchor_layer = None
        model.config.cka_loss_final_vision_anchor_layer = None
        default = model(input_ids=inputs, labels=inputs.clone(), images=images)
        torch.testing.assert_close(anchored.logits, default.logits)
        torch.testing.assert_close(anchored.loss, default.loss)
        tower = model.get_vision_tower()
        expected = model.get_model().mm_projector(tower(images))
        model.config.cka_loss_vision_anchor_layer = 999
        model.config.cka_loss_projector_vision_anchor_layer = 999
        model.config.cka_loss_final_vision_anchor_layer = 999
        model.config.cka_loss = False
        torch.testing.assert_close(model.encode_images(images), expected)
        model.config.cka_loss = True
        model.eval()
        torch.testing.assert_close(model.encode_images(images), expected)

    def test_hidden_cka_reaches_projector_through_checkpointed_frozen_decoder(self):
        torch.manual_seed(73)
        model = tiny_vlm("qwen2", layers="1,final")
        model.config.cka_loss_vision_anchor_layer = 1
        model.config.cka_loss_projector_vision_anchor_layer = 1
        model.config.cka_loss_final_vision_anchor_layer = 3
        model.requires_grad_(False)
        model.get_model().mm_projector.requires_grad_(True)
        model.config.cka_loss_projector_weight = 0.0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        output = model(input_ids=inputs, labels=inputs.clone(), images=torch.randn(1, 3, 4, 4))
        sum(output.aux_losses).backward()
        gradient = model.get_model().mm_projector.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
