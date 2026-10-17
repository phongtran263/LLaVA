"""CPU regressions for the controlled projector/final-state VSP ablations."""

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import numpy as np
from torch import nn
from transformers import HfArgumentParser, default_data_collator
from transformers.trainer_callback import PrinterCallback

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.llava_arch import (
    compute_linear_cka_loss,
    normalize_cka_token_rows,
    validate_cka_loss_geometry,
)
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.train.llava_trainer import LLaVATrainer
from llava.train.train import ModelArguments, TrainingArguments, set_cka_loss_geometry_config, validate_cka_loss_geometry_args


class _FrozenVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Parameter(torch.randn(5, 6), requires_grad=False)

    def forward(self, images):
        return self.features.unsqueeze(0).expand(images.shape[0], -1, -1)


def _tiny_model(backbone="llama", projector_weight=0.1, final_weight=0.1, geometry="linear"):
    config_cls, model_cls = {
        "llama": (LlavaConfig, LlavaLlamaForCausalLM),
        "qwen2": (LlavaQwenConfig, LlavaQwenForCausalLM),
    }[backbone]
    config = config_cls(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config.cka_loss = projector_weight != 0 or final_weight != 0
    config.cka_loss_projector_weight = projector_weight
    config.cka_loss_final_hidden_weight = final_weight
    config.cka_loss_weight = 0.1
    config.cka_loss_layers = "final" if final_weight else [-1]
    config.cka_loss_subset_select_layer = None
    config.cka_loss_tau = 0.0
    config.cka_loss_geometry = geometry
    config.use_cache = False
    config.log_gradient_norms = False
    config.mm_patch_merge_type = "flat"
    config.image_aspect_ratio = "pad"
    config.tokenizer_padding_side = "right"
    config.tokenizer_model_max_length = 64
    config.tune_mm_mlp_adapter = False
    config.mm_use_im_start_end = False
    torch.manual_seed(271)
    model = model_cls(config).train()
    model.get_model().vision_tower = _FrozenVision()
    model.get_model().mm_projector = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 16))
    return model


def _batch():
    input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3], [1, IMAGE_TOKEN_INDEX, 4, 5]])
    return dict(input_ids=input_ids, labels=input_ids.clone(), images=torch.zeros(2, 3, 2, 2))


def _reference_row_cka(x, y, mask=None):
    """Independent per-image cosine-Gram reference, normalized then centered."""
    losses = []
    for i in range(x.shape[0]):
        keep = torch.ones(x.shape[1], dtype=torch.bool) if mask is None else mask[i]
        if keep.sum() < 2:
            continue
        rows = []
        for features in (x[i, keep].float(), y[i, keep].float()):
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            features = features - features.mean(dim=0, keepdim=True)
            features = features / features.norm().clamp_min(torch.finfo(torch.float32).tiny)
            rows.append(features @ features.T)
        gx, gy = rows
        similarity = (gx * gy).sum() / (gx.square().sum() * gy.square().sum()).clamp_min(1e-8).sqrt()
        losses.append(torch.relu(1.0 - similarity.clamp(0, 1)))
    return torch.stack(losses).mean()


class Cvpr2027CkaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_geometry_config_preserves_old_resume_signatures(self):
        config = SimpleNamespace(cka_loss=True)
        before = vars(config).copy()
        set_cka_loss_geometry_config(config, "linear")
        self.assertEqual(vars(config), before)
        set_cka_loss_geometry_config(config, "row_normalized")
        self.assertEqual(config.cka_loss_geometry, "row_normalized")
        set_cka_loss_geometry_config(config, "linear")
        self.assertEqual(config.cka_loss_geometry, "linear")
        self.assertEqual(ModelArguments().cka_loss_geometry, "linear")
        parsed, = HfArgumentParser(ModelArguments).parse_args_into_dataclasses(
            ["--cka_loss_geometry", "row_normalized"]
        )
        self.assertEqual(parsed.cka_loss_geometry, "row_normalized")
        for invalid in (None, "cosine", "row-normalized", False):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_cka_loss_geometry(invalid)

    def test_linear_default_is_bitwise_identical_in_loss_and_gradient(self):
        torch.manual_seed(280)
        x = torch.randn(2, 5, 6, requires_grad=True)
        y = torch.randn(2, 5, 9, requires_grad=True)
        legacy = compute_linear_cka_loss(x, y)
        explicit = compute_linear_cka_loss(x, y, geometry="linear")
        torch.testing.assert_close(legacy, explicit, rtol=0, atol=0)
        # The independent feature-space HSIC identity verifies the original
        # centered linear geometry without reusing token-Gram implementation.
        centered_x = x - x.mean(dim=1, keepdim=True)
        centered_y = y - y.mean(dim=1, keepdim=True)
        xy = torch.einsum("bld,ble->bde", centered_x, centered_y).square().sum(dim=(1, 2))
        xx = torch.einsum("bld,ble->bde", centered_x, centered_x).square().sum(dim=(1, 2))
        yy = torch.einsum("bld,ble->bde", centered_y, centered_y).square().sum(dim=(1, 2))
        original_definition = (1 - xy / (xx * yy).sqrt()).mean()
        torch.testing.assert_close(legacy, original_definition)
        legacy_grad = torch.autograd.grad(legacy, (x, y), retain_graph=True)
        explicit_grad = torch.autograd.grad(explicit, (x, y), retain_graph=True)
        definition_grad = torch.autograd.grad(original_definition, (x, y))
        for actual, expected in zip(explicit_grad, legacy_grad):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(legacy_grad, definition_grad):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    def test_protected_modes_keep_legacy_default_and_reject_unsupported_geometry(self):
        for protected in ("protected_cka_refinement", "protected_stage2"):
            training = SimpleNamespace(**{protected: True})
            self.assertEqual(validate_cka_loss_geometry_args(ModelArguments(), training), "linear")
            with self.assertRaisesRegex(ValueError, "Protected CKA modes"):
                validate_cka_loss_geometry_args(ModelArguments(cka_loss_geometry="row_normalized"), training)

    def test_row_normalized_matches_independent_per_image_definition(self):
        torch.manual_seed(281)
        x = torch.randn(3, 7, 6)
        y = torch.randn(3, 7, 9)
        actual = compute_linear_cka_loss(x, y, geometry="row_normalized")
        torch.testing.assert_close(actual, _reference_row_cka(x, y))
        individual = torch.stack([
            compute_linear_cka_loss(x[i:i+1], y[i:i+1], geometry="row_normalized")
            for i in range(3)
        ]).mean()
        torch.testing.assert_close(actual, individual)

    def test_row_normalization_removes_token_magnitudes_before_centering(self):
        torch.manual_seed(282)
        x, y = torch.randn(2, 6, 5), torch.randn(2, 6, 7)
        scaled = x * torch.tensor([0.1, 2., 9., 0.4, 5., 12.])[None, :, None]
        normalized = compute_linear_cka_loss(x, y, geometry="row_normalized")
        torch.testing.assert_close(normalized, compute_linear_cka_loss(scaled, y, geometry="row_normalized"))
        self.assertGreater(abs((compute_linear_cka_loss(x, y) - compute_linear_cka_loss(scaled, y)).item()), 1e-3)
        with self.assertRaisesRegex(ValueError, "rank-3"):
            compute_linear_cka_loss(x.flatten(1), x.flatten(1), geometry="row_normalized")

    def test_row_normalization_is_fp32_with_finite_zero_token_gradients(self):
        x = torch.tensor([[[0., 0., 0.], [1., 2., 0.], [0., 1., 3.]]], dtype=torch.bfloat16, requires_grad=True)
        y = torch.tensor([[[0., 0.], [1., 0.], [2., 1.]]], dtype=torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            rows = normalize_cka_token_rows(x)
            loss = compute_linear_cka_loss(x, y, geometry="row_normalized")
        self.assertEqual(rows.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_masked_row_normalization_excludes_text_padding_and_single_token_images(self):
        torch.manual_seed(283)
        model = _tiny_model(geometry="row_normalized")
        x = torch.randn(3, 7, 16, requires_grad=True)
        y = torch.randn(3, 7, 6)
        mask = torch.tensor([
            [True, False, True, False, True, True, False],
            [False, True, True, False, False, True, False],
            [False, False, True, False, False, False, False],
        ])
        actual = model._compute_masked_linear_cka_loss(x, y, mask)
        torch.testing.assert_close(actual, _reference_row_cka(x, y, mask))
        actual.backward()
        self.assertEqual(x.grad[~mask].abs().sum().item(), 0)
        self.assertEqual(x.grad[2].abs().sum().item(), 0)
        changed_x = x.detach().clone()
        changed_y = y.clone()
        changed_x[~mask] = 1e6
        changed_y[~mask] = -1e6
        torch.testing.assert_close(actual.detach(), model._compute_masked_linear_cka_loss(changed_x, changed_y, mask))

    def test_four_variants_disable_terms_and_keep_identical_forward_initialization(self):
        for backbone in ("llama", "qwen2"):
            for geometry in ("linear", "row_normalized"):
                baseline = _tiny_model(backbone, 0, 0, geometry)
                baseline_output = baseline(**_batch())
                for name, proj, final in (
                    ("baseline", 0, 0), ("proj", 0.1, 0), ("final", 0, 0.1), ("full", 0.1, 0.1),
                ):
                    with self.subTest(backbone=backbone, geometry=geometry, variant=name):
                        model = _tiny_model(backbone, proj, final, geometry)
                        for key, value in baseline.state_dict().items():
                            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
                        with mock.patch("llava.model.llava_arch.compute_linear_cka_loss", wraps=compute_linear_cka_loss) as projector_cka, mock.patch.object(
                            model, "_compute_masked_linear_cka_loss", wraps=model._compute_masked_linear_cka_loss,
                        ) as final_cka:
                            output = model(**_batch())
                        self.assertEqual(projector_cka.call_count, int(proj != 0))
                        self.assertEqual(final_cka.call_count, int(final != 0))
                        torch.testing.assert_close(output.loss, baseline_output.loss, rtol=0, atol=0)
                        torch.testing.assert_close(output.logits, baseline_output.logits, rtol=0, atol=0)
                        total = output.loss
                        if name != "baseline":
                            self.assertEqual(len(output.aux_losses), 1)
                            self.assertTrue(torch.isfinite(output.projector_cka_loss))
                            self.assertTrue(torch.isfinite(output.aux_losses[0]))
                            if proj:
                                self.assertGreater(output.projector_cka_loss.item(), 0)
                            else:
                                self.assertEqual(output.projector_cka_loss.item(), 0)
                                self.assertFalse(output.projector_cka_loss.requires_grad)
                            if final:
                                self.assertGreater(output.aux_losses[0].item(), 0)
                            else:
                                self.assertEqual(output.aux_losses[0].item(), 0)
                                self.assertFalse(output.aux_losses[0].requires_grad)
                            total = total + output.projector_cka_loss + sum(output.aux_losses)
                        total.backward()
                        self.assertTrue(torch.isfinite(model.get_model().mm_projector[0].weight.grad).all())
                        self.assertIsNone(model.get_vision_tower().features.grad)

    def test_final_only_reaches_projector_and_decoder_before_final_norm(self):
        for backbone in ("llama", "qwen2"):
            for geometry in ("linear", "row_normalized"):
                for checkpointing in (False, True):
                    with self.subTest(backbone=backbone, geometry=geometry, checkpointing=checkpointing):
                        model = _tiny_model(backbone, 0, 0.1, geometry)
                        if checkpointing:
                            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        captured = {}
                        handles = [
                            model.get_model().norm.register_forward_pre_hook(
                                lambda _module, inputs: captured.update(pre_norm=inputs[0])
                            ),
                            model.get_model().norm.register_forward_hook(
                                lambda _module, _inputs, output: captured.update(post_norm=output)
                            ),
                        ]
                        try:
                            with mock.patch.object(model, "_compute_masked_linear_cka_loss", wraps=model._compute_masked_linear_cka_loss) as final_cka:
                                output = model(**_batch())
                        finally:
                            for handle in handles:
                                handle.remove()
                        target = final_cka.call_args.kwargs["projected_features"]
                        reference = final_cka.call_args.kwargs["layer_hidden_states"]
                        mask = final_cka.call_args.kwargs["vision_feature_mask"]
                        self.assertIs(target, captured["pre_norm"])
                        self.assertFalse(torch.allclose(target, captured["post_norm"]))
                        self.assertTrue(target.requires_grad)
                        self.assertFalse(reference.requires_grad)
                        self.assertIsNone(reference.grad_fn)
                        self.assertTrue(torch.equal(mask.sum(dim=1), torch.tensor([5, 5])))
                        parameters = (
                            model.get_model().mm_projector[0].weight,
                            model.get_model().layers[0].mlp.up_proj.weight,
                            model.get_model().layers[-1].mlp.down_proj.weight,
                        )
                        gradients = torch.autograd.grad(output.aux_losses[0], parameters, retain_graph=True)
                        for gradient in gradients:
                            self.assertTrue(torch.isfinite(gradient).all())
                            self.assertGreater(gradient.abs().sum().item(), 0)
                        output.aux_losses[0].backward()
                        self.assertIsNone(model.get_model().norm.weight.grad)
                        self.assertIsNone(model.lm_head.weight.grad)
                        self.assertIsNone(model.get_vision_tower().features.grad)

    def test_native_trainer_applies_each_coefficient_once_with_gradient_accumulation(self):
        # Unequal weights catch placement swaps and applying either coefficient
        # twice. Compare every parameter gradient against a direct objective,
        # including two microbatches accumulated into one optimizer update.
        batch = _batch()
        for backbone in ("llama", "qwen2"):
            for geometry in ("linear", "row_normalized"):
                for name, projector_weight, final_weight in (
                    ("baseline", 0, 0), ("proj", 0.13, 0),
                    ("final", 0, 0.27), ("full", 0.13, 0.27),
                ):
                    with self.subTest(backbone=backbone, geometry=geometry, variant=name), tempfile.TemporaryDirectory() as directory:
                        initial = _tiny_model(backbone, projector_weight, final_weight, geometry)
                        expected = copy.deepcopy(initial)
                        actual = copy.deepcopy(initial)
                        args = TrainingArguments(
                            output_dir=directory, use_cpu=True, report_to=[],
                            per_device_train_batch_size=1, gradient_accumulation_steps=2,
                            dataloader_pin_memory=False, remove_unused_columns=False,
                        )
                        trainer = LLaVATrainer(model=actual, args=args)
                        trainer.state.max_steps = 1
                        for index in range(2):
                            microbatch = {key: value[index:index + 1] for key, value in batch.items()}
                            projected = []
                            handle = expected.get_model().mm_projector.register_forward_hook(
                                lambda _module, inputs, output: projected.append((inputs[0], output))
                            )
                            try:
                                with mock.patch.object(expected, "_compute_masked_linear_cka_loss", wraps=expected._compute_masked_linear_cka_loss) as masked:
                                    output = expected(**microbatch)
                            finally:
                                handle.remove()
                            total = output.loss
                            if projector_weight:
                                reference, target = projected[0]
                                raw_projector = compute_linear_cka_loss(reference.detach(), target, geometry=geometry)
                                torch.testing.assert_close(output.projector_cka_loss, raw_projector * projector_weight)
                                total = total + raw_projector * projector_weight
                            if final_weight:
                                raw_final = expected._compute_masked_linear_cka_loss(**masked.call_args.kwargs)
                                torch.testing.assert_close(sum(output.aux_losses), raw_final * final_weight)
                                total = total + raw_final * final_weight
                            (total / 2).backward()
                            reported = trainer.training_step(actual, microbatch)
                            torch.testing.assert_close(reported, total.detach() / 2)
                        for parameter_name, parameter in expected.named_parameters():
                            gradient = dict(actual.named_parameters())[parameter_name].grad
                            if parameter.grad is None:
                                self.assertIsNone(gradient)
                            else:
                                torch.testing.assert_close(gradient, parameter.grad, rtol=1e-5, atol=1e-7)

    def test_actual_coupling_projector_recipe_routes_auxiliary_gradients(self):
        # The audited recipe uses coupling1x_gelu, so also exercise that actual
        # projector implementation rather than relying solely on a generic MLP.
        for backbone in ("llama", "qwen2"):
            for geometry in ("linear", "row_normalized"):
                for name, projector_weight, final_weight in (
                    ("baseline", 0, 0), ("proj", 0.13, 0),
                    ("final", 0, 0.27), ("full", 0.13, 0.27),
                ):
                    with self.subTest(backbone=backbone, geometry=geometry, variant=name):
                        model = _tiny_model(backbone, projector_weight, final_weight, geometry)
                        model.config.mm_projector_type = "coupling1x_gelu"
                        model.config.mm_hidden_size = 6
                        model.get_model().mm_projector = build_vision_projector(model.config)
                        output = model(**_batch())
                        projector = model.get_model().mm_projector
                        decoder = model.get_model().layers[-1].mlp.down_proj.weight
                        if projector_weight:
                            stem_grad, decoder_grad = torch.autograd.grad(
                                output.projector_cka_loss, (projector.stem.weight, decoder),
                                allow_unused=True, retain_graph=True,
                            )
                            self.assertGreater(stem_grad.abs().sum().item(), 0)
                            self.assertIsNone(decoder_grad)
                        if final_weight:
                            gradients = torch.autograd.grad(
                                sum(output.aux_losses),
                                (projector.stem.weight, projector.blocks[0].second_update[-1].weight, decoder),
                                retain_graph=True,
                            )
                            for gradient in gradients:
                                self.assertTrue(torch.isfinite(gradient).all())
                                self.assertGreater(gradient.abs().sum().item(), 0)
                        total = output.loss
                        if projector_weight or final_weight:
                            total = total + output.projector_cka_loss + sum(output.aux_losses)
                        total.backward()
                        self.assertTrue(torch.isfinite(projector.stem.weight.grad).all())
                        self.assertIsNone(model.get_vision_tower().features.grad)

    def test_reference_is_detached_even_if_caller_supplies_trainable_reference(self):
        model = _tiny_model()
        target = torch.randn(2, 5, 16, requires_grad=True)
        reference = torch.randn(2, 5, 6, requires_grad=True)
        specs = model._get_cka_layer_specs()
        losses, _ = model._compute_cka_reference_losses(
            specs, {"final": target}, target, None, reference, "vision_encoder",
            torch.ones(2, 5, dtype=torch.bool), torch.device("cpu"),
        )
        sum(losses).backward()
        self.assertIsNone(reference.grad)
        self.assertGreater(target.grad.abs().sum().item(), 0)

    def test_geometry_survives_native_config_save_reload(self):
        for backbone in ("llama", "qwen2"):
            with self.subTest(backbone=backbone), tempfile.TemporaryDirectory() as directory:
                model = _tiny_model(backbone, 0, 0.1, "row_normalized")
                model.config.save_pretrained(directory)
                config = type(model.config).from_pretrained(directory)
                self.assertEqual(config.cka_loss_geometry, "row_normalized")
                self.assertEqual(config.cka_loss_projector_weight, 0)
                self.assertEqual(config.cka_loss_final_hidden_weight, 0.1)
                self.assertEqual(config.cka_loss_layers, "final")

    def test_final_only_native_trainer_checkpoint_resume_restores_weights_and_optimizer(self):
        # This is real three-step CPU training, not an isolated serialization
        # mock. Continuation from checkpoint-2 must equal uninterrupted step 3.
        batch = _batch()
        dataset = [{name: value[i % 2] for name, value in batch.items()} for i in range(8)]

        def trainer_for(model, directory):
            args = TrainingArguments(
                output_dir=str(directory), use_cpu=True, report_to=[], max_steps=3,
                per_device_train_batch_size=2, gradient_accumulation_steps=1,
                learning_rate=0.003, lr_scheduler_type="constant", optim="adamw_torch",
                max_grad_norm=0, save_strategy="steps", save_steps=2,
                logging_steps=1, disable_tqdm=True, dataloader_pin_memory=False,
                seed=291, data_seed=291, remove_unused_columns=False,
            )
            trainer = LLaVATrainer(model=model, args=args, train_dataset=dataset,
                                  data_collator=default_data_collator)
            trainer.remove_callback(PrinterCallback)
            return trainer

        for backbone in ("llama", "qwen2"):
            with self.subTest(backbone=backbone), tempfile.TemporaryDirectory() as directory:
                initial = _tiny_model(backbone, 0, 0.1, "row_normalized")
                full = trainer_for(copy.deepcopy(initial), Path(directory) / "full")
                full.train()
                checkpoint = Path(directory) / "full/checkpoint-2"
                self.assertEqual(json.loads((checkpoint / "trainer_state.json").read_text())["global_step"], 2)
                self.assertTrue((checkpoint / "model.safetensors").is_file())
                self.assertTrue((checkpoint / "optimizer.pt").is_file())
                config = type(initial.config).from_pretrained(checkpoint)
                self.assertEqual(config.cka_loss_geometry, "row_normalized")
                self.assertEqual(config.cka_loss_projector_weight, 0)
                self.assertEqual(config.cka_loss_final_hidden_weight, 0.1)
                resumed = trainer_for(copy.deepcopy(initial), Path(directory) / "resumed")
                # Old Transformers loads NumPy RNG state without an explicit
                # weights_only option. PyTorch >=2.6 needs these NumPy types
                # allowlisted for this checkpoint created by this test itself.
                # The scoped context restores the allowlist afterwards and
                # preserves restricted loading for all other checkpoint data.
                rng_types = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
                rng_context = (torch.serialization.safe_globals(rng_types)
                               if hasattr(torch.serialization, "safe_globals") else nullcontext())
                with rng_context:
                    resumed.train(resume_from_checkpoint=str(checkpoint))
                self.assertEqual(full.state.global_step, 3)
                self.assertEqual(resumed.state.global_step, 3)
                for name, value in full.model.state_dict().items():
                    torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
                full_optimizer = full.optimizer.state_dict()
                resumed_optimizer = resumed.optimizer.state_dict()
                self.assertEqual(full_optimizer["param_groups"], resumed_optimizer["param_groups"])
                self.assertEqual(full_optimizer["state"].keys(), resumed_optimizer["state"].keys())
                self.assertTrue(full_optimizer["state"])
                for parameter, state in full_optimizer["state"].items():
                    for key, value in state.items():
                        torch.testing.assert_close(value, resumed_optimizer["state"][parameter][key], rtol=0, atol=0)
                self.assertIsNone(resumed.model.get_vision_tower().features.grad)


if __name__ == "__main__":
    unittest.main()
