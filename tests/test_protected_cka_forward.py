"""CPU/offline checks against tiny real LLaVA decoder backbones."""

import copy
import unittest
from unittest import mock

import torch
from torch import nn

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.train.protected_cka_forward import ce_only_forward, protected_cka_forward
from tests.test_cka_backbones import _backbone_factories


class CountingVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 6))
        self.dropout = nn.Dropout(0.8)
        self.calls = 0
        self.last_features = None

    def forward(self, images):
        self.calls += 1
        features = images.flatten(2).transpose(1, 2) @ self.weight
        self.last_features = self.dropout(features)
        return self.last_features


def make_protected_model(name="llama"):
    model_cls, factory = next((cls, factory) for family, cls, factory in _backbone_factories()
                              if family == name)
    config = factory()
    config.cka_loss = False
    config.mm_patch_merge_type = "flat"
    config.tokenizer_padding_side = "right"
    config.tokenizer_model_max_length = 64
    config.attention_dropout = 0.5
    config._attn_implementation = "eager"
    model = model_cls(config)
    model.get_model().vision_tower = CountingVision()
    model.get_model().mm_projector = nn.Sequential(nn.Linear(6, 24), nn.GELU(),
                                                  nn.Dropout(0.7), nn.Linear(24, config.hidden_size))
    model.requires_grad_(False)
    model.get_model().mm_projector.requires_grad_(True)
    reference = copy.deepcopy(model.get_model().mm_projector).requires_grad_(False)
    model.train()
    return model, reference


def multimodal_batch():
    return {
        "input_ids": torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3, 4],
                                   [1, IMAGE_TOKEN_INDEX, 5, 6, 0]]),
        "labels": torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 2, 3, 4],
                                [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 6, IGNORE_INDEX]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        "position_ids": torch.arange(5).unsqueeze(0).expand(2, -1),
        "images": torch.randn(2, 3, 2, 2),
    }


def text_kl(result):
    student = result["student_logits"][:, :-1].float().log_softmax(-1)
    reference = result["reference_logits"][:, :-1].float().log_softmax(-1)
    valid = result["expanded_labels"][:, 1:].ne(IGNORE_INDEX)
    return (reference.exp() * (reference - student)).sum(-1)[valid].mean()


class ProtectedForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_identical_reference_reuses_vision_shared_llm_and_disables_dropout(self):
        for name in ("llama", "qwen2"):
            with self.subTest(backbone=name):
                torch.manual_seed(81)
                model, reference = make_protected_model(name)
                batch = multimodal_batch()
                reference_inputs = []
                hook = reference.register_forward_pre_hook(lambda module, args: reference_inputs.append(args[0]))
                original_parameters = tuple(model.parameters())
                flags = [module.training for module in model.modules()]
                with mock.patch.object(model.get_model(), "forward", wraps=model.get_model().forward) as llm:
                    result = protected_cka_forward(model, reference, batch)
                hook.remove()
                self.assertEqual(llm.call_count, 2)
                self.assertEqual(model.get_vision_tower().calls, 1)
                self.assertIs(reference_inputs[0], model.get_vision_tower().last_features)
                self.assertEqual(tuple(model.parameters()), original_parameters)
                self.assertEqual([module.training for module in model.modules()], flags)
                torch.testing.assert_close(result["student_logits"], result["reference_logits"], atol=0, rtol=0)
                self.assertLess(abs(text_kl(result).item()), 1e-7)
                self.assertTrue(result["student_logits"].requires_grad)
                self.assertTrue(all(value.requires_grad for value in result["projected_features"]))
                self.assertFalse(result["reference_logits"].requires_grad)
                self.assertTrue(all(not value.requires_grad for value in result["reference_projected_features"]))
                second = protected_cka_forward(model, reference, batch)
                torch.testing.assert_close(result["student_logits"], second["student_logits"], atol=0, rtol=0)

    def test_ce_and_nonzero_kl_reach_only_student_projector(self):
        for name in ("llama", "qwen2"):
            with self.subTest(backbone=name):
                torch.manual_seed(42)
                model, reference = make_protected_model(name)
                initial = {key: value.detach().clone() for key, value in model.named_parameters()}
                ref_initial = copy.deepcopy(reference.state_dict())
                with torch.no_grad():
                    model.get_model().mm_projector[-1].weight.add_(torch.randn(16, 24) * 0.2)
                result = protected_cka_forward(model, reference, multimodal_batch())
                parameters = tuple(model.get_model().mm_projector.parameters())
                ce_grad = torch.autograd.grad(result["student_ce"], parameters, retain_graph=True)
                kl_grad = torch.autograd.grad(text_kl(result), parameters, retain_graph=True)
                for gradients in (ce_grad, kl_grad):
                    self.assertGreater(sum(gradient.square().sum().item() for gradient in gradients), 0)
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                (result["student_ce"] + text_kl(result)).backward()
                optimizer = torch.optim.SGD(parameters, lr=0.01)
                optimizer.step()
                for key, parameter in model.named_parameters():
                    if "mm_projector" not in key:
                        self.assertIsNone(parameter.grad)
                        torch.testing.assert_close(parameter, initial[key], atol=0, rtol=0)
                for key, value in reference.state_dict().items():
                    torch.testing.assert_close(value, ref_initial[key], atol=0, rtol=0)
                self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))

    def test_all_available_backbones_paired_stage1_ce_and_kl_gradient_smoke(self):
        for name, _, _ in _backbone_factories():
            with self.subTest(backbone=name):
                torch.manual_seed(817)
                model, reference = make_protected_model(name)
                projector = model.get_model().mm_projector
                batch = multimodal_batch()
                # The Stage-1 collator supplies no explicit positions. MPT uses
                # native ALiBi and intentionally has no position_ids argument.
                batch.pop("position_ids")
                with torch.no_grad():
                    projector[-1].weight.add_(torch.randn_like(projector[-1].weight) * 0.2)
                with mock.patch.object(model.get_model(), "forward", wraps=model.get_model().forward) as llm:
                    result = protected_cka_forward(model, reference, batch)
                self.assertEqual(llm.call_count, 2)
                self.assertEqual(model.get_vision_tower().calls, 1)
                self.assertTrue(torch.isfinite(result["student_ce"]))
                self.assertFalse(result["reference_logits"].requires_grad)
                self.assertEqual(result["expanded_labels"].shape, (2, 8))
                kl = text_kl(result)
                self.assertTrue(torch.isfinite(kl))
                self.assertGreater(kl.item(), 0)
                parameters = tuple(projector.parameters())
                for loss in (result["student_ce"], kl):
                    gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                    self.assertGreater(sum(gradient.square().sum().item() for gradient in gradients), 0)
                (result["student_ce"] + kl).backward()
                projector_ids = {id(parameter) for parameter in parameters}
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()
                                    if id(parameter) not in projector_ids))
                self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))

    def test_expanded_labels_causal_shift_padding_and_truncation(self):
        for padding in ("left", "right"):
            with self.subTest(padding=padding):
                model, reference = make_protected_model()
                model.config.tokenizer_padding_side = padding
                model.config.tokenizer_model_max_length = 7
                result = protected_cka_forward(model, reference, multimodal_batch())
                self.assertEqual(result["expanded_labels"].shape, (2, 7))
                # Image replaces one token with four ignored patch labels.
                self.assertEqual(result["expanded_labels"][0].tolist(), [-100] * 5 + [2, 3])
                self.assertEqual(result["expanded_labels"][1].tolist(), [-100] * 6 + [6])
                self.assertEqual(result["expanded_labels"][:, 1:].ne(-100).sum().item(), 3)
                self.assertEqual(result["expanded_attention_mask"].tolist(), [[1] * 7] * 2)
                self.assertEqual(result["expanded_position_ids"].tolist(), [list(range(7))] * 2)
                self.assertEqual(len(result["vision_features"]), 2)
                self.assertEqual(result["vision_features"][0].shape, (4, 6))

    def test_reference_llm_can_be_skipped_but_reference_features_are_present(self):
        model, reference = make_protected_model()
        with mock.patch.object(model.get_model(), "forward", wraps=model.get_model().forward) as llm:
            result = protected_cka_forward(model, reference, multimodal_batch(), need_reference_logits=False)
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(model.get_vision_tower().calls, 1)
        self.assertIsNone(result["reference_logits"])
        self.assertEqual(len(result["reference_projected_features"]), 2)

    def test_ce_only_has_one_forward_no_reference_and_identical_ce_gradients(self):
        for name in ("llama", "qwen2"):
            with self.subTest(backbone=name):
                model, reference = make_protected_model(name)
                batch = multimodal_batch()
                flags = [module.training for module in model.modules()]
                with mock.patch.object(reference, "forward", side_effect=AssertionError("reference used")), \
                        mock.patch.object(model.get_model(), "forward", wraps=model.get_model().forward) as llm:
                    ce = ce_only_forward(model, batch)
                self.assertEqual(llm.call_count, 1)
                self.assertEqual(model.get_vision_tower().calls, 1)
                self.assertEqual([module.training for module in model.modules()], flags)
                protected = protected_cka_forward(model, reference, batch)
                torch.testing.assert_close(ce["student_ce"], protected["student_ce"], atol=0, rtol=0)
                torch.testing.assert_close(ce["expanded_labels"], protected["expanded_labels"], atol=0, rtol=0)
                parameters = tuple(model.get_model().mm_projector.parameters())
                ce_grad = torch.autograd.grad(ce["student_ce"], parameters)
                protected_grad = torch.autograd.grad(protected["student_ce"], parameters)
                for actual, expected in zip(ce_grad, protected_grad):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_ce_only_zero_labels_validation_and_exception_cleanup(self):
        model, _ = make_protected_model()
        batch = multimodal_batch()
        batch["labels"].fill_(IGNORE_INDEX)
        result = ce_only_forward(model, batch)
        self.assertEqual(result["student_ce"].item(), 0)
        self.assertTrue(result["student_ce"].requires_grad)
        with torch.no_grad():
            self.assertFalse(ce_only_forward(model, batch)["student_ce"].requires_grad)
        flags = [module.training for module in model.modules()]
        with mock.patch.object(model.get_model(), "forward", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                ce_only_forward(model, batch)
        self.assertNotIn("prepare_inputs_labels_for_multimodal", model.__dict__)
        self.assertEqual([module.training for module in model.modules()], flags)

    def test_untruncated_left_and_right_padding_alignment(self):
        for padding in ("left", "right"):
            with self.subTest(padding=padding):
                model, reference = make_protected_model()
                model.config.tokenizer_padding_side = padding
                result = protected_cka_forward(model, reference, multimodal_batch())
                mask = [0] + [1] * 7 if padding == "left" else [1] * 7 + [0]
                labels = [-100] * 7 + [6] if padding == "left" else [-100] * 6 + [6, -100]
                self.assertEqual(result["expanded_attention_mask"][1].tolist(), mask)
                self.assertEqual(result["expanded_labels"][1].tolist(), labels)
                self.assertEqual(result["expanded_labels"][:, 1:].ne(-100).sum().item(), 4)

    def test_bf16_cpu_autocast_preserves_student_gradients_and_matching_reference(self):
        for name in ("llama", "qwen2"):
            with self.subTest(backbone=name):
                model, reference = make_protected_model(name)
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    result = protected_cka_forward(model, reference, multimodal_batch())
                torch.testing.assert_close(result["student_logits"], result["reference_logits"], atol=0, rtol=0)
                result["student_ce"].backward()
                gradients = [parameter.grad for parameter in model.get_model().mm_projector.parameters()]
                self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
                self.assertGreater(sum(value.square().sum().item() for value in gradients), 0)

    def test_reference_only_projector_replays_model_forward_native_amp_context(self):
        model, reference = make_protected_model()
        original_forward = model.forward

        # Like native Accelerate AMP: autocast wraps model.forward, not the
        # protected helper or the standalone reference-projector invocation.
        def native_amp_forward(*args, **kwargs):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                return original_forward(*args, **kwargs)

        with mock.patch.object(model, "forward", side_effect=native_amp_forward):
            result = protected_cka_forward(model, reference, multimodal_batch(),
                                           need_reference_logits=False)
        self.assertEqual(result["vision_features"][0].dtype, torch.bfloat16)
        self.assertEqual(next(reference.parameters()).dtype, torch.float32)
        for student, expected in zip(result["projected_features"], result["reference_projected_features"]):
            torch.testing.assert_close(student, expected, atol=0, rtol=0)
        result["student_ce"].backward()
        self.assertTrue(all(parameter.grad is not None
                            for parameter in model.get_model().mm_projector.parameters()))

    def test_text_only_and_dummy_images_have_no_auxiliary_features(self):
        for dummy_images in (False, True):
            with self.subTest(dummy_images=dummy_images):
                model, reference = make_protected_model()
                batch = {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, 3]])}
                if dummy_images:
                    batch["images"] = torch.randn(1, 3, 2, 2)
                result = protected_cka_forward(model, reference, batch)
                for key in ("vision_features", "projected_features", "reference_projected_features"):
                    self.assertEqual(result[key], [])
                self.assertEqual(text_kl(result).item(), 0)

    def test_mixed_batch_excludes_text_dummy_image(self):
        model, reference = make_protected_model()
        batch = multimodal_batch()
        batch["input_ids"][0, 1] = 7
        result = protected_cka_forward(model, reference, batch)
        self.assertEqual(len(result["vision_features"]), 1)
        torch.testing.assert_close(result["vision_features"][0], model.get_vision_tower().last_features[1])

    def test_no_supervised_tokens_returns_differentiable_zero_ce(self):
        model, reference = make_protected_model()
        batch = multimodal_batch()
        batch["labels"].fill_(IGNORE_INDEX)
        result = protected_cka_forward(model, reference, batch)
        self.assertEqual(result["student_ce"].item(), 0)
        result["student_ce"].backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all()
                            for parameter in model.get_model().mm_projector.parameters()))

    def test_cleanup_on_student_and_reference_failure(self):
        for fail_reference in (False, True):
            with self.subTest(reference=fail_reference):
                model, reference = make_protected_model()
                model.get_model().layers[0].eval()
                reference[0].eval()
                modules = list(model.modules()) + list(reference.modules())
                flags = [module.training for module in modules]
                projector = model.get_model().mm_projector
                vision = model.get_vision_tower()
                original = model.get_model().forward
                calls = 0

                def failing(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == (2 if fail_reference else 1):
                        raise RuntimeError("injected failure")
                    return original(*args, **kwargs)

                with mock.patch.object(model.get_model(), "forward", side_effect=failing):
                    with self.assertRaisesRegex(RuntimeError, "injected failure"):
                        protected_cka_forward(model, reference, multimodal_batch())
                self.assertNotIn("forward", projector.__dict__)
                self.assertNotIn("forward", vision.__dict__)
                self.assertNotIn("prepare_inputs_labels_for_multimodal", model.__dict__)
                self.assertEqual(len(projector._forward_hooks), 0)
                self.assertEqual([module.training for module in modules], flags)
                result = protected_cka_forward(model, reference, multimodal_batch())
                self.assertTrue(torch.isfinite(result["student_ce"]))

    def test_clear_rejections_for_unsupported_inputs_or_training_state(self):
        cases = ("list", "5d", "spatial", "legacy", "llm", "reference", "checkpointing", "fp16", "cls")
        for case in cases:
            with self.subTest(case=case):
                model, reference = make_protected_model()
                batch = multimodal_batch()
                if case == "list":
                    batch["images"] = list(batch["images"])
                elif case == "5d":
                    batch["images"] = batch["images"].unsqueeze(1)
                elif case == "spatial":
                    model.config.mm_patch_merge_type = "spatial_unpad"
                elif case == "legacy":
                    model.config.cka_loss = True
                elif case == "llm":
                    model.lm_head.requires_grad_(True)
                elif case == "reference":
                    reference.requires_grad_(True)
                elif case == "checkpointing":
                    model.gradient_checkpointing_enable()
                elif case == "fp16":
                    model.get_model().mm_projector.half()
                elif case == "cls":
                    model.get_vision_tower().select_feature = "cls_patch"
                with self.assertRaisesRegex(ValueError, "Protected CKA"):
                    protected_cka_forward(model, reference, batch)


if __name__ == "__main__":
    unittest.main()
