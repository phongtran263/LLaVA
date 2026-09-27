"""Fixed random and manual attention-head CKA training integration."""

import copy
import math
import tempfile
import unittest
from unittest import mock

import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.head_cka import (
    compute_head_cka_loss,
    parse_head_ids,
    select_head_indices,
    validate_head_count,
)
from llava.model.llava_arch import compute_linear_cka_loss
from tests.test_cka_backbones import (
    _FakeVisionTower,
    _backbone_factories,
    _configure_multimodal_cka,
    _projector,
)


SUPPORTED_BACKBONES = {"llama", "mistral", "qwen2", "qwen3", "gemma3", "phi3"}


def _supported_factories():
    return [factory for factory in _backbone_factories() if factory[0] in SUPPORTED_BACKBONES]


def _direct_head_loss(heads, reference, mask, tau=0.0):
    """Independent oracle: each head is its own CKA, never concatenated."""
    return torch.stack([
        compute_linear_cka_loss(
            heads[sample, mask[sample], head].unsqueeze(0),
            reference[sample, mask[sample]].detach().unsqueeze(0),
            tau=tau,
        )
        for sample in range(heads.shape[0])
        if int(mask[sample].sum()) >= 2
        for head in range(heads.shape[2])
    ]).mean()


class FixedHeadSelectionTests(unittest.TestCase):
    def test_fixed_layer_specific_seeds_do_not_consume_global_rng(self):
        torch.manual_seed(713)
        rng_before = torch.random.get_rng_state().clone()
        selected = [select_head_indices(32, 0.25, 42, layer) for layer in range(1, 5)]
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        for layer, indices in enumerate(selected, start=1):
            expected = torch.randperm(
                32, generator=torch.Generator(device="cpu").manual_seed(42 + layer),
            )[:8].sort().values
            self.assertEqual(indices.device.type, "cpu")
            self.assertEqual(indices.dtype, torch.long)
            torch.testing.assert_close(indices, expected, rtol=0, atol=0)
            # Calls for other layers and arbitrary training randomness cannot
            # change a layer's fixed selection.
            torch.rand(31)
            torch.testing.assert_close(
                select_head_indices(32, 0.25, 42, layer), indices, rtol=0, atol=0,
            )
        self.assertEqual(len({tuple(indices.tolist()) for indices in selected}), 4)

    def test_fraction_rounds_up_and_samples_without_replacement(self):
        for heads, fraction in ((32, 0.25), (7, 0.25), (1, 0.25), (8, 1.0)):
            with self.subTest(heads=heads, fraction=fraction):
                indices = select_head_indices(heads, fraction, 7, 1)
                self.assertEqual(indices.numel(), max(1, math.ceil(heads * fraction)))
                self.assertEqual(indices.unique().numel(), indices.numel())
                self.assertTrue(bool(((indices >= 0) & (indices < heads)).all()))

    def test_exact_count_overrides_fraction_and_remains_deterministic(self):
        first = select_head_indices(14, 0.01, 42, 3, selected_count=4)
        second = select_head_indices(14, 1.0, 42, 3, selected_count=4)
        self.assertEqual(first.numel(), 4)
        self.assertEqual(first.unique().numel(), 4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_invalid_exact_count_is_rejected(self):
        self.assertIsNone(validate_head_count(None))
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_head_count(count)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            select_head_indices(4, selected_count=5)

    def test_invalid_head_selection_parameters_are_rejected(self):
        for fraction in (0, -0.25, 1.01, float("nan"), float("inf"), True):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                select_head_indices(8, fraction, 42, 1)
        for seed in (-1, 1.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                select_head_indices(8, 0.25, seed, 1)


class ManualHeadSelectionTests(unittest.TestCase):
    def test_absent_and_blank_selection_preserve_legacy_mode(self):
        for value in (None, "", "   ", "\t\n"):
            with self.subTest(value=value):
                self.assertIsNone(parse_head_ids(value))

    def test_json_and_dictionary_inputs_are_sorted_without_mutating_input(self):
        expected = {1: [0, 2], 3: [1, 3]}
        original = {"3": [3, 1], 1: [2, 0]}
        before = copy.deepcopy(original)
        for value in (original, '{"3": [3, 1], "1": [2, 0]}'):
            with self.subTest(value=value):
                parsed = parse_head_ids(value)
                self.assertEqual(parsed, expected)
                self.assertEqual(list(parsed), [1, 3])
                self.assertTrue(all(type(key) is int for key in parsed))
                self.assertTrue(all(type(head) is int for heads in parsed.values() for head in heads))
        self.assertEqual(original, before)

    def test_malformed_or_non_object_json_is_rejected(self):
        for value in ('{"1": [0]', '{1: [0]}', 'null', 'false', '1', '[]', '[1, 2]', '"1"'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_head_ids(value)

    def test_empty_and_non_mapping_selections_are_rejected(self):
        for value in ({}, "{}", [], [1], 1, 1.0, False):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_head_ids(value)

    def test_duplicate_json_and_normalized_layer_keys_are_rejected(self):
        for value in ('{"1": [0], "1": [1]}', {1: [0], "1": [1]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_head_ids(value)

    def test_invalid_layer_ids_are_rejected(self):
        for layer in (0, -1, True, False, 1.0, 1.5, None, "0", "-1", "1.0", "all", ""):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                parse_head_ids({layer: [0]})

    def test_head_lists_require_unique_nonnegative_integers(self):
        for heads in ([], [0, 0], [-1], [True], [False], [1.0], [1.5], ["1"], [None],
                      None, 0, "0", (0, 1), {0, 1}):
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                parse_head_ids({1: heads})


class HeadCkaKernelTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(101)

    def test_masks_then_centers_each_head_and_averages_valid_samples(self):
        heads = torch.randn(3, 8, 3, 4, generator=self.generator, requires_grad=True)
        reference = torch.randn(3, 8, 6, generator=self.generator, requires_grad=True)
        mask = torch.tensor([
            [1, 0, 1, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=torch.bool)
        actual = compute_head_cka_loss(heads, reference, mask)
        expected = _direct_head_loss(heads, reference, mask)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)

        concatenated = torch.stack([
            compute_linear_cka_loss(
                heads[sample, mask[sample]].flatten(1).unsqueeze(0),
                reference[sample, mask[sample]].unsqueeze(0),
            )
            for sample in (0, 1)
        ]).mean()
        self.assertGreater(abs(float(actual - concatenated)), 0.01)

        actual_grad, reference_grad = torch.autograd.grad(
            actual, (heads, reference), allow_unused=True, retain_graph=True,
        )
        expected_grad, = torch.autograd.grad(expected, heads)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-5, atol=2e-6)
        self.assertIsNone(reference_grad)
        self.assertTrue(bool(torch.isfinite(actual_grad).all()))
        self.assertEqual(int(torch.count_nonzero(actual_grad[~mask])), 0)
        self.assertEqual(int(torch.count_nonzero(actual_grad[2])), 0)

        changed_heads = heads.detach().clone()
        changed_reference = reference.detach().clone()
        changed_heads[~mask] = 1e6
        changed_reference[~mask] = -1e6
        torch.testing.assert_close(
            compute_head_cka_loss(changed_heads, changed_reference, mask),
            actual.detach(), rtol=0, atol=0,
        )

    def test_hinge_is_applied_per_head_before_averaging(self):
        reference = torch.randn(2, 9, 4, generator=self.generator)
        other = torch.randn(2, 9, 4, generator=self.generator)
        heads = torch.stack((reference, other), dim=2).requires_grad_(True)
        mask = torch.ones(2, 9, dtype=torch.bool)
        actual = compute_head_cka_loss(heads, reference, mask, tau=0.2)
        expected = _direct_head_loss(heads, reference, mask, tau=0.2)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
        self.assertGreater(float(actual), 0)
        actual.backward()
        self.assertEqual(int(torch.count_nonzero(heads.grad[:, :, 0])), 0)
        self.assertGreater(float(heads.grad[:, :, 1].norm()), 0)

    def test_no_valid_tokens_returns_differentiable_zero(self):
        heads = torch.randn(2, 5, 2, 4, generator=self.generator, requires_grad=True)
        reference = torch.randn(2, 5, 6, generator=self.generator, requires_grad=True)
        for token_count in (0, 1):
            with self.subTest(token_count=token_count):
                mask = torch.zeros(2, 5, dtype=torch.bool)
                mask[:, :token_count] = True
                loss = compute_head_cka_loss(heads, reference, mask)
                self.assertEqual(float(loss), 0.0)
                self.assertTrue(loss.requires_grad)
                gradient, = torch.autograd.grad(loss, heads)
                torch.testing.assert_close(gradient, torch.zeros_like(heads), rtol=0, atol=0)

    def test_autocast_preserves_fp32_loss_and_gradient(self):
        heads = torch.randn(2, 7, 3, 4, generator=self.generator, requires_grad=True)
        reference = torch.randn(2, 7, 6, generator=self.generator)
        mask = torch.ones(2, 7, dtype=torch.bool)
        expected = compute_head_cka_loss(heads, reference, mask)
        expected_grad, = torch.autograd.grad(expected, heads)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = compute_head_cka_loss(heads, reference, mask)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_grad, = torch.autograd.grad(actual, heads)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


class HeadCkaBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _model(self, factory, random_heads=True, fraction=0.25, selected_count=None,
               head_ids=None, num_layers=None):
        _, model_cls, config_factory = factory
        torch.manual_seed(23)
        config = _configure_multimodal_cka(config_factory())
        config.cka_loss_random_heads = random_heads
        config.cka_loss_head_fraction = fraction
        config.cka_loss_num_heads = selected_count
        config.cka_loss_head_seed = 42
        config.cka_loss_head_ids = head_ids
        config.cka_loss_anchor_layer = None
        if num_layers is not None:
            config.num_hidden_layers = num_layers
        # Head mode must override even the legacy projector-only setting.
        config.cka_loss_layers = "-1" if random_heads else "all"
        model = model_cls(config).train()
        model.get_model().vision_tower = _FakeVisionTower()
        model.get_model().mm_projector = _projector()
        return model

    def test_exact_count_is_used_by_every_backbone_layer(self):
        for factory in _supported_factories():
            with self.subTest(backbone=factory[0]):
                model = self._model(factory, fraction=0.01, selected_count=2)
                specs = model._get_cka_layer_specs()
                self.assertEqual(len(specs), 2)
                self.assertTrue(all(spec["head_indices"].numel() == 2 for spec in specs))
                # The selection is cached and remains fixed throughout training.
                for first, second in zip(specs, model._get_cka_layer_specs()):
                    torch.testing.assert_close(
                        first["head_indices"], second["head_indices"], rtol=0, atol=0,
                    )

    @staticmethod
    def _inputs():
        input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        return dict(input_ids=input_ids, labels=input_ids.clone(), images=torch.zeros(1, 3, 2, 2))

    @staticmethod
    def _reference(model, sequence_length):
        features = model.get_model().vision_tower.features
        reference = features.new_zeros(1, sequence_length, features.shape[-1])
        mask = torch.zeros(1, sequence_length, dtype=torch.bool)
        reference[:, 1:1 + features.shape[0]] = features
        mask[:, 1:1 + features.shape[0]] = True
        return reference, mask

    def test_all_layers_use_fixed_pre_projection_heads_and_full_projector(self):
        for factory in _supported_factories():
            with self.subTest(backbone=factory[0]):
                model = self._model(factory)
                specs = model._get_cka_layer_specs()
                self.assertEqual([spec["layer_idx"] for spec in specs], [1, 2])
                observed = {}
                handles = []
                for index, layer in enumerate(model.get_model().layers, start=1):
                    def capture_pre_projection(module, inputs, layer_index=index):
                        observed[layer_index] = inputs[0]
                    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(capture_pre_projection))
                try:
                    output = model(**self._inputs())
                finally:
                    for handle in handles:
                        handle.remove()

                reference, mask = self._reference(model, observed[1].shape[1])
                expected_layers = []
                for spec in specs:
                    layer_index = spec["layer_idx"]
                    self.assertEqual(spec["kind"], "heads")
                    self.assertEqual(spec["num_heads"], 4)
                    self.assertEqual(spec["head_indices"].numel(), 1)
                    expected_indices = select_head_indices(4, 0.25, 42, layer_index)
                    torch.testing.assert_close(spec["head_indices"], expected_indices)
                    heads = observed[layer_index].reshape(1, -1, 4, spec["head_dim"])
                    heads = heads.index_select(2, expected_indices)
                    expected_layers.append(_direct_head_loss(heads, reference, mask))

                self.assertEqual(len(output.aux_losses), 1)
                torch.testing.assert_close(
                    output.aux_losses[0], torch.stack(expected_layers).mean(), rtol=2e-5, atol=2e-6,
                )
                vision = model.get_model().vision_tower(self._inputs()["images"])
                projector = model.get_model().mm_projector(vision)
                torch.testing.assert_close(
                    output.projector_cka_loss,
                    compute_linear_cka_loss(vision, projector), rtol=1e-6, atol=1e-6,
                )
                self.assertEqual(len(model.last_cka_per_layer_losses), 2)
                # Observing activations and adding aux outputs must not alter CE.
                model.config.cka_loss = False
                without_hooks = model(**self._inputs())
                torch.testing.assert_close(output.logits, without_hooks.logits, rtol=0, atol=0)
                torch.testing.assert_close(output.loss, without_hooks.loss, rtol=0, atol=0)

    def test_aux_gradient_reaches_projector_and_attention_but_not_final_o_proj(self):
        for factory in _supported_factories():
            with self.subTest(backbone=factory[0]):
                model = self._model(factory)
                final_attention = model.get_model().layers[-1].self_attn
                final_head_outputs = []
                def capture_final_heads(module, inputs):
                    final_head_outputs.append(inputs[0])
                handle = final_attention.o_proj.register_forward_pre_hook(capture_final_heads)
                try:
                    output = model(**self._inputs())
                finally:
                    handle.remove()
                raw_grad, = torch.autograd.grad(
                    output.aux_losses[0], final_head_outputs[0], retain_graph=True,
                )
                spec = model._get_cka_layer_specs()[-1]
                head_grad = raw_grad.reshape(1, -1, spec["num_heads"], spec["head_dim"])
                selected_mask = torch.zeros(spec["num_heads"], dtype=torch.bool)
                selected_mask[spec["head_indices"]] = True
                self.assertEqual(int(torch.count_nonzero(head_grad[:, :, ~selected_mask])), 0)
                self.assertGreater(float(head_grad[:, :, selected_mask].norm()), 0)
                output.aux_losses[0].backward()
                projector_grad = model.get_model().mm_projector.weight.grad
                self.assertIsNotNone(projector_grad)
                self.assertTrue(bool(torch.isfinite(projector_grad).all()))
                self.assertGreater(float(projector_grad.norm()), 0)
                self.assertIsNone(final_attention.o_proj.weight.grad)
                target_grads = [
                    parameter.grad for name, parameter in final_attention.named_parameters()
                    if not name.startswith("o_proj.") and parameter.grad is not None
                ]
                self.assertTrue(target_grads)
                self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in target_grads))
                self.assertTrue(any(float(grad.norm()) > 0 for grad in target_grads))

    def test_explicit_head_dimension_can_differ_from_hidden_size_divided_by_heads(self):
        factories = [factory for factory in _supported_factories() if factory[0] in {"qwen3", "gemma3"}]
        if not factories:
            self.skipTest("Qwen3/Gemma3 are unavailable in this Transformers version")
        for name, model_cls, config_factory in factories:
            with self.subTest(backbone=name):
                def wider_attention_config():
                    config = config_factory()
                    config.head_dim = 8
                    return config
                model = self._model((name, model_cls, wider_attention_config), fraction=0.5)
                observed = {}
                handles = []
                for index, layer in enumerate(model.get_model().layers, start=1):
                    def capture(module, inputs, layer_index=index):
                        observed[layer_index] = inputs[0]
                    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(capture))
                try:
                    output = model(**self._inputs())
                finally:
                    for handle in handles:
                        handle.remove()
                reference, mask = self._reference(model, observed[1].shape[1])
                expected_losses = []
                for spec in model._get_cka_layer_specs():
                    raw = observed[spec["layer_idx"]]
                    self.assertEqual(raw.shape[-1], 32)
                    self.assertEqual(model.config.hidden_size, 16)
                    self.assertEqual(spec["head_dim"], 8)
                    heads = raw.reshape(1, -1, 4, 8).index_select(2, spec["head_indices"])
                    expected_losses.append(_direct_head_loss(heads, reference, mask))
                torch.testing.assert_close(
                    output.aux_losses[0], torch.stack(expected_losses).mean(), rtol=2e-5, atol=2e-6,
                )
                output.aux_losses[0].backward()
                self.assertTrue(bool(torch.isfinite(model.get_model().mm_projector.weight.grad).all()))

    def test_head_selection_survives_config_checkpoint_roundtrip(self):
        factory = next(factory for factory in _supported_factories() if factory[0] == "llama")
        model = self._model(factory)
        specs = model._get_cka_layer_specs()
        original = {str(spec["layer_idx"]): spec["head_indices"].tolist() for spec in specs}
        self.assertEqual(model.config.cka_loss_head_indices, original)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            model.config.save_pretrained(checkpoint_dir)
            restored_config = model.config.__class__.from_pretrained(checkpoint_dir)
        torch.rand(17)
        restored = factory[1](restored_config)
        restored_specs = restored._get_cka_layer_specs()
        self.assertEqual(
            {str(spec["layer_idx"]): spec["head_indices"].tolist() for spec in restored_specs},
            original,
        )
        self.assertEqual(restored.config.cka_loss_head_seed, 42)
        self.assertEqual(restored.config.cka_loss_head_fraction, 0.25)

    def test_hidden_anchor_is_rejected_for_head_mode(self):
        factory = next(factory for factory in _supported_factories() if factory[0] == "llama")
        model = self._model(factory)
        model.config.cka_loss_anchor_layer = 1
        with self.assertRaisesRegex(ValueError, "cka_loss_anchor_layer"):
            model(**self._inputs())

    def test_reentrant_checkpointing_fails_before_silently_losing_head_gradients(self):
        factory = next(factory for factory in _supported_factories() if factory[0] == "llama")
        model = self._model(factory)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
        with self.assertRaisesRegex(ValueError, "use_reentrant=False"):
            model(**self._inputs())
        for layer in model.get_model().layers:
            self.assertFalse(layer.self_attn.o_proj._forward_pre_hooks)

    def test_unsupported_backbone_rejects_head_mode(self):
        factory = next(factory for factory in _backbone_factories() if factory[0] == "mpt")
        model = self._model(factory)
        with self.assertRaisesRegex(ValueError, "not supported"):
            model(**self._inputs())

    def test_non_reentrant_checkpointing_preserves_all_parameter_gradients(self):
        for factory in _supported_factories():
            with self.subTest(backbone=factory[0]):
                ordinary = self._model(factory, fraction=0.5)
                checkpointed = copy.deepcopy(ordinary)
                checkpointed.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                outputs = []
                for model in (ordinary, checkpointed):
                    output = model(**self._inputs())
                    (output.loss + output.projector_cka_loss + output.aux_losses[0]).backward()
                    outputs.append(output)
                torch.testing.assert_close(outputs[0].aux_losses[0], outputs[1].aux_losses[0])
                expected_parameters = dict(ordinary.named_parameters())
                for name, actual in checkpointed.named_parameters():
                    expected = expected_parameters[name]
                    if expected.grad is None:
                        self.assertIsNone(actual.grad, name)
                    else:
                        self.assertIsNotNone(actual.grad, name)
                        torch.testing.assert_close(
                            actual.grad, expected.grad, rtol=3e-5, atol=2e-6,
                            msg=lambda message: f"{factory[0]} {name}: {message}",
                        )

    def test_manual_heads_override_random_controls_and_legacy_layer_selection(self):
        for factory in _supported_factories():
            with self.subTest(backbone=factory[0]):
                model = self._model(
                    factory, random_heads=False, fraction=0, selected_count=99,
                    head_ids='{"3": [3, 1], "1": [2]}', num_layers=3,
                )
                model.config.cka_loss_head_seed = -1
                rng_before = torch.random.get_rng_state().clone()
                for random_heads, layers in ((False, "-1"), (True, "2"), (False, "all")):
                    model.config.cka_loss_random_heads = random_heads
                    model.config.cka_loss_layers = layers
                    specs = model._get_cka_layer_specs()
                    self.assertEqual([spec["layer_idx"] for spec in specs], [1, 3])
                    self.assertEqual([spec["kind"] for spec in specs], ["heads", "heads"])
                    self.assertEqual([spec["head_indices"].tolist() for spec in specs], [[2], [1, 3]])
                    for spec in specs:
                        self.assertEqual(spec["head_indices"].dtype, torch.long)
                        self.assertEqual(spec["head_indices"].device.type, "cpu")
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
                self.assertEqual(model.config.cka_loss_head_indices, {"1": [2], "3": [1, 3]})

    def test_manual_selection_cache_tracks_changes_to_layers_and_heads(self):
        factory = next(factory for factory in _supported_factories() if factory[0] == "llama")
        model = self._model(factory, random_heads=False, head_ids={1: [0], 2: [1, 3]})
        first = model._get_cka_layer_specs()
        model.config.cka_loss_head_ids = {2: [2, 0]}
        second = model._get_cka_layer_specs()
        self.assertEqual([spec["layer_idx"] for spec in first], [1, 2])
        self.assertEqual([spec["layer_idx"] for spec in second], [2])
        self.assertEqual(second[0]["head_indices"].tolist(), [0, 2])
        self.assertEqual(model.config.cka_loss_head_indices, {"2": [0, 2]})
        # In-place edits must invalidate the cache too.
        model.config.cka_loss_head_ids[2].append(3)
        self.assertEqual(model._get_cka_layer_specs()[0]["head_indices"].tolist(), [0, 2, 3])

    def test_manual_layer_and_head_bounds_are_checked_before_hook_registration(self):
        for factory in _supported_factories():
            for head_ids in ({3: [0]}, {1: [4]}):
                with self.subTest(backbone=factory[0], head_ids=head_ids):
                    model = self._model(factory, random_heads=False, head_ids=head_ids)
                    # Bounds follow the instantiated layers, not stale config metadata.
                    model.config.num_hidden_layers = 100
                    with self.assertRaisesRegex(ValueError, "cka_loss_head_ids"):
                        model(**self._inputs())
                    for layer in model.get_model().layers:
                        self.assertFalse(layer._forward_hooks)
                        self.assertFalse(layer.self_attn.o_proj._forward_pre_hooks)

    def test_manual_targets_average_heads_then_layers_and_leave_projector_and_ce_unchanged(self):
        for factory in _supported_factories():
            with self.subTest(backbone=factory[0]):
                model = self._model(
                    factory, random_heads=False, head_ids={3: [3, 1], 1: [2]}, num_layers=3,
                )
                model.config.cka_loss_layers = "2"
                observed, handles = {}, []
                for layer_index in (1, 3):
                    def capture(module, inputs, layer_index=layer_index):
                        observed[layer_index] = inputs[0]
                    handles.append(model.get_model().layers[layer_index - 1].self_attn.o_proj
                                   .register_forward_pre_hook(capture))
                unselected = model.get_model().layers[1]
                try:
                    with mock.patch.object(
                        unselected.self_attn.o_proj, "register_forward_pre_hook",
                        side_effect=AssertionError("Unselected layer must not receive a head hook"),
                    ), mock.patch.object(
                        unselected, "register_forward_hook",
                        side_effect=AssertionError("Manual selection must not add a full-block hook"),
                    ):
                        output = model(**self._inputs())
                finally:
                    for handle in handles:
                        handle.remove()
                reference, mask = self._reference(model, observed[1].shape[1])
                expected_layers = []
                individual_heads = []
                for layer_index, indices in ((1, [2]), (3, [1, 3])):
                    raw_heads = observed[layer_index].reshape(1, -1, 4, 4)
                    chosen = raw_heads[:, :, indices]
                    expected_layers.append(_direct_head_loss(chosen, reference, mask))
                    individual_heads.extend(
                        _direct_head_loss(raw_heads[:, :, [head]], reference, mask)
                        for head in indices
                    )
                expected = torch.stack(expected_layers).mean()
                self.assertGreater(abs(float(expected - torch.stack(individual_heads).mean())), 1e-6)
                self.assertEqual(len(output.aux_losses), 1)
                torch.testing.assert_close(output.aux_losses[0], expected, rtol=2e-5, atol=2e-6)
                self.assertEqual(
                    set(model.last_cka_per_layer_losses),
                    {"vision_encoder_to_layer_1_heads", "vision_encoder_to_layer_3_heads"},
                )
                vision = model.get_model().vision_tower(self._inputs()["images"])
                projector = model.get_model().mm_projector(vision)
                torch.testing.assert_close(
                    output.projector_cka_loss, compute_linear_cka_loss(vision, projector),
                    rtol=1e-6, atol=1e-6,
                )
                for layer in model.get_model().layers:
                    self.assertFalse(layer._forward_hooks)
                    self.assertFalse(layer.self_attn.o_proj._forward_pre_hooks)
                model.config.cka_loss = False
                baseline = model(**self._inputs())
                torch.testing.assert_close(output.logits, baseline.logits, rtol=0, atol=0)
                torch.testing.assert_close(output.loss, baseline.loss, rtol=0, atol=0)

    def test_manual_final_target_gradient_is_zero_for_unselected_heads_and_text_tokens(self):
        for factory in _supported_factories():
            with self.subTest(backbone=factory[0]):
                model = self._model(factory, random_heads=False, head_ids={2: [3, 1]})
                attention = model.get_model().layers[-1].self_attn
                observed = []
                handle = attention.o_proj.register_forward_pre_hook(
                    lambda module, inputs: observed.append(inputs[0])
                )
                try:
                    output = model(**self._inputs())
                finally:
                    handle.remove()
                raw_gradient, = torch.autograd.grad(
                    output.aux_losses[0], observed[0], retain_graph=True,
                )
                gradient = raw_gradient.reshape(1, -1, 4, 4)
                _, vision_mask = self._reference(model, gradient.shape[1])
                self.assertEqual(int(torch.count_nonzero(gradient[:, :, [0, 2]])), 0)
                self.assertEqual(int(torch.count_nonzero(gradient[~vision_mask])), 0)
                for head in (1, 3):
                    self.assertGreater(float(gradient[:, :, head].norm()), 0)
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                output.aux_losses[0].backward()
                self.assertIsNone(attention.o_proj.weight.grad)
                projector_gradient = model.get_model().mm_projector.weight.grad
                self.assertIsNotNone(projector_gradient)
                self.assertTrue(bool(torch.isfinite(projector_gradient).all()))
                self.assertGreater(float(projector_gradient.norm()), 0)
                attention_gradients = [
                    parameter.grad for name, parameter in attention.named_parameters()
                    if not name.startswith("o_proj.") and parameter.grad is not None
                ]
                self.assertTrue(attention_gradients)
                self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in attention_gradients))
                self.assertTrue(any(float(grad.norm()) > 0 for grad in attention_gradients))

    def test_manual_ids_and_actual_choices_survive_config_checkpoint_roundtrip(self):
        factory = next(factory for factory in _supported_factories() if factory[0] == "llama")
        for head_ids in ('{"2": [3, 1], "1": [2]}', {2: [3, 1], 1: [2]}):
            with self.subTest(head_ids=head_ids):
                model = self._model(factory, random_heads=False, head_ids=head_ids)
                model._get_cka_layer_specs()
                expected = {"1": [2], "2": [1, 3]}
                self.assertEqual(model.config.cka_loss_head_indices, expected)
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    model.config.save_pretrained(checkpoint_dir)
                    restored_config = model.config.__class__.from_pretrained(checkpoint_dir)
                self.assertEqual(parse_head_ids(restored_config.cka_loss_head_ids), {1: [2], 2: [1, 3]})
                self.assertEqual(restored_config.cka_loss_head_indices, expected)
                self.assertFalse(restored_config.cka_loss_random_heads)
                restored = factory[1](restored_config)
                self.assertEqual(
                    {str(spec["layer_idx"]): spec["head_indices"].tolist()
                     for spec in restored._get_cka_layer_specs()},
                    expected,
                )

    def test_manual_head_mode_rejects_hidden_anchor_and_reentrant_checkpointing(self):
        factory = next(factory for factory in _supported_factories() if factory[0] == "llama")
        model = self._model(factory, random_heads=False, head_ids={2: [1]})
        model.config.cka_loss_anchor_layer = 1
        with self.assertRaisesRegex(ValueError, "cka_loss_anchor_layer"):
            model(**self._inputs())
        model.config.cka_loss_anchor_layer = None
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        with self.assertRaisesRegex(ValueError, "use_reentrant=False"):
            model(**self._inputs())
        for layer in model.get_model().layers:
            self.assertFalse(layer.self_attn.o_proj._forward_pre_hooks)

    def test_manual_head_mode_is_rejected_for_unsupported_backbone(self):
        factory = next(factory for factory in _backbone_factories() if factory[0] == "mpt")
        model = self._model(factory, random_heads=False, head_ids={1: [0]})
        with self.assertRaisesRegex(ValueError, "not supported"):
            model(**self._inputs())

    def test_manual_non_reentrant_checkpointing_preserves_parameter_gradients(self):
        for factory in _supported_factories():
            with self.subTest(backbone=factory[0]):
                ordinary = self._model(
                    factory, random_heads=False, head_ids={3: [3, 1], 1: [2]}, num_layers=3,
                )
                checkpointed = copy.deepcopy(ordinary)
                checkpointed.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                outputs = []
                for model in (ordinary, checkpointed):
                    output = model(**self._inputs())
                    (output.loss + output.projector_cka_loss + output.aux_losses[0]).backward()
                    outputs.append(output)
                torch.testing.assert_close(outputs[0].aux_losses[0], outputs[1].aux_losses[0])
                expected_parameters = dict(ordinary.named_parameters())
                for name, actual in checkpointed.named_parameters():
                    expected = expected_parameters[name]
                    if expected.grad is None:
                        self.assertIsNone(actual.grad, name)
                    else:
                        self.assertIsNotNone(actual.grad, name)
                        torch.testing.assert_close(
                            actual.grad, expected.grad, rtol=3e-5, atol=2e-6,
                            msg=lambda message: f"{factory[0]} {name}: {message}",
                        )

    def test_missing_head_flag_preserves_full_block_targets_and_sum(self):
        factory = next(factory for factory in _supported_factories() if factory[0] == "llama")
        model = self._model(factory, random_heads=False)
        del model.config.cka_loss_random_heads
        observed = []
        handles = []
        for layer in model.get_model().layers:
            def capture_block(module, inputs, output):
                observed.append(output[0] if isinstance(output, (tuple, list)) else output)
            handles.append(layer.register_forward_hook(capture_block))
        try:
            output = model(**self._inputs())
        finally:
            for handle in handles:
                handle.remove()
        reference, mask = self._reference(model, observed[0].shape[1])
        expected = torch.stack([
            compute_linear_cka_loss(
                block[mask].unsqueeze(0), reference[mask].detach().unsqueeze(0),
            )
            for block in observed
        ]).sum()
        self.assertEqual(len(observed), 2)
        self.assertGreater(float(expected), 0)
        torch.testing.assert_close(output.aux_losses[0], expected, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
