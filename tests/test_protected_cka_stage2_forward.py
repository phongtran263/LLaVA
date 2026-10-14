"""Tiny CPU/offline Stage-2 forward, teacher, selection, and routing checks."""

import copy
import unittest
from unittest import mock

import torch

from llava.constants import IGNORE_INDEX
from llava.train.protected_cka_math import token_kl
from llava.train.protected_cka_stage2_forward import (
    snapshot_stage2_reference, stage2_deterministic_context, stage2_forward,
)
from llava.train import protected_cka_stage2_forward as stage2_module
from tests.test_protected_cka_forward import make_protected_model, multimodal_batch
from tests.test_cka_backbones import _backbone_factories


def make_stage2(name="llama", *, hidden=True):
    model, _ = make_protected_model(name)
    model.requires_grad_(True)
    model.get_vision_tower().requires_grad_(False)
    model.config.cka_loss = True
    model.config.cka_loss_layers = "1,final" if hidden else "-1"
    model.config.cka_loss_projector_weight = 0.4
    model.config.cka_loss_final_hidden_weight = 0.7 if hidden else 0.0
    model.config.cka_loss_subset_select_layer = 1
    model.config.cka_loss_subset_min_ratio = 0.25
    model.config.cka_loss_subset_max_ratio = 0.75
    return model


class Stage2ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_independent_full_teacher_stays_frozen_after_student_update(self):
        torch.manual_seed(70)
        model = make_stage2()
        reference = snapshot_stage2_reference(model)
        frozen = {name: value.clone() for name, value in reference.state_dict().items()}
        for student, teacher in zip(model.parameters(), reference.parameters()):
            self.assertNotEqual(student.data_ptr(), teacher.data_ptr())
            self.assertFalse(teacher.requires_grad)
        result = stage2_forward(model, multimodal_batch(), reference, need_cka=True)
        kl, tokens = token_kl(result["student_logits"], result["reference_logits"], result["expanded_labels"])
        self.assertGreater(tokens, 0)
        self.assertLess(abs(kl.item()), 1e-7)
        self.assertEqual(model.get_vision_tower().calls, 1)
        self.assertEqual(reference.get_vision_tower().calls, 0)
        self.assertFalse(any(module.training for module in reference.modules()))
        (result["student_ce"] + kl + result["cka_loss"]).backward()
        self.assertGreater(model.lm_head.weight.grad.norm().item(), 0)
        self.assertGreater(model.get_model().mm_projector[-1].weight.grad.norm().item(), 0)
        torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01).step()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, frozen[name], atol=0, rtol=0)

    def test_phase_a_preserves_normal_training_dropout_and_native_ce(self):
        model = make_stage2()
        batch = multimodal_batch()
        before = [module.training for module in model.modules()]
        model.config.cka_loss = False
        torch.manual_seed(10)
        expected = model(**batch, use_cache=False, return_dict=True).loss
        model.config.cka_loss = True
        torch.manual_seed(10)
        actual = stage2_forward(model, batch)
        torch.testing.assert_close(actual["student_ce"], expected, atol=0, rtol=0)
        self.assertEqual([module.training for module in model.modules()], before)
        self.assertTrue(model.config.cka_loss)
        self.assertIsNone(actual["reference_logits"])
        self.assertEqual(actual["cka_loss"].item(), 0)
        self.assertFalse(actual["cka_loss"].requires_grad)

    def test_same_student_selection_and_configured_layers_used_for_teacher(self):
        torch.manual_seed(12)
        model = make_stage2()
        reference = snapshot_stage2_reference(model)
        with mock.patch.object(model, "_select_vision_feature_subset_from_attention_inputs",
                               wraps=model._select_vision_feature_subset_from_attention_inputs) as selection, \
                mock.patch.object(reference, "_select_vision_feature_subset_from_attention_inputs",
                                  side_effect=AssertionError("teacher must not select different patches")):
            result = stage2_forward(model, multimodal_batch(), reference, need_cka=True)
        self.assertEqual(selection.call_count, 1)
        self.assertEqual(set(result["cka_terms"]), {"projector", "layer_1", "final"})
        first = result["cka_terms"]["layer_1"]["mask"]
        torch.testing.assert_close(first, result["cka_terms"]["final"]["mask"])
        self.assertFalse(first.requires_grad)
        self.assertTrue(bool(first.any()))
        self.assertLess(first.sum().item(), 8)
        self.assertEqual(result["cka_metrics"]["valid_images"], 2)

    def test_projector_only_vjp_routes_aux_without_blocking_main_llm(self):
        model = make_stage2()
        reference = snapshot_stage2_reference(model)
        result = stage2_forward(model, multimodal_batch(), reference, need_cka=True)
        parameters = tuple(model.get_model().mm_projector.parameters())
        auxiliary = torch.autograd.grad(result["cka_loss"], parameters, retain_graph=True)
        self.assertGreater(sum(gradient.square().sum().item() for gradient in auxiliary), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        result["student_ce"].backward()
        self.assertGreater(model.lm_head.weight.grad.norm().item(), 0)

    def test_projector_method_does_not_fake_connect_unrelated_llm_parameters(self):
        model = make_stage2(hidden=False)
        reference = snapshot_stage2_reference(model)
        result = stage2_forward(model, multimodal_batch(), reference, need_cka=True)
        grad, = torch.autograd.grad(result["cka_loss"], [model.lm_head.weight], allow_unused=True)
        self.assertIsNone(grad)

    def test_text_only_and_mixed_batches_exclude_dummy_images(self):
        for dummy in (False, True):
            with self.subTest(dummy=dummy):
                model = make_stage2()
                reference = snapshot_stage2_reference(model)
                batch = dict(input_ids=torch.tensor([[1, 2, 3, 4]]),
                             labels=torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4]]))
                if dummy:
                    batch["images"] = torch.randn(1, 3, 2, 2)
                result = stage2_forward(model, batch, reference, need_cka=True)
                self.assertEqual(result["cka_loss"].item(), 0)
                self.assertFalse(result["cka_loss"].requires_grad)
                self.assertEqual(result["cka_metrics"]["valid_images"], 0)
                self.assertEqual(result["cka_metrics"]["invalid_images"], 0)
                self.assertEqual(result["projected_features"], [])
                self.assertFalse(result["image_rows"].any())
                self.assertLess(abs(token_kl(result["student_logits"], result["reference_logits"],
                                            result["expanded_labels"])[0].item()), 1e-7)
                result["student_ce"].backward()
                self.assertGreater(model.lm_head.weight.grad.norm().item(), 0)
        model = make_stage2()
        reference = snapshot_stage2_reference(model)
        batch = multimodal_batch()
        batch["input_ids"][0, 1] = 9
        result = stage2_forward(model, batch, reference, need_cka=True)
        self.assertEqual(result["image_rows"].tolist(), [False, True])
        self.assertEqual(result["cka_metrics"]["valid_images"], 1)
        self.assertEqual(len(result["projected_features"]), 1)
        for term in result["cka_terms"].values():
            self.assertFalse(term["mask"][0].any())

    def test_nonreentrant_checkpointing_preserves_gradients_through_both_vjps(self):
        torch.manual_seed(90)
        model = make_stage2()
        reference = snapshot_stage2_reference(model)
        baseline = copy.deepcopy(model)
        batch = multimodal_batch()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        with self.assertRaisesRegex(ValueError, "BOTH forward"):
            stage2_forward(model, batch, reference, need_cka=True)
        gradients = []
        for student in (baseline, model):
            parameters = tuple(parameter for parameter in student.parameters() if parameter.requires_grad)
            with stage2_deterministic_context(student):
                result = stage2_forward(student, batch, reference, need_cka=True)
                kl, _ = token_kl(result["student_logits"], result["reference_logits"], result["expanded_labels"])
                auxiliary = torch.autograd.grad(result["cka_loss"], parameters, allow_unused=True, retain_graph=True)
                main = torch.autograd.grad(result["student_ce"] + kl, parameters, allow_unused=True)
                gradients.append((main, auxiliary))
        for reference_group, checkpoint_group in zip(*gradients):
            for expected, actual in zip(reference_group, checkpoint_group):
                self.assertEqual(expected is None, actual is None)
                if expected is not None:
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_snapshot_lora_preserves_adapters_and_other_trainables_independently(self):
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(make_stage2(), LoraConfig(r=2, lora_alpha=4, lora_dropout=0.4,
                                                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        model.get_model().mm_projector.requires_grad_(True)
        model.get_base_model().lm_head.requires_grad_(True)
        reference = snapshot_stage2_reference(model, placement="cpu")
        before = {name: value.clone() for name, value in reference.state_dict().items()}
        result = stage2_forward(model, multimodal_batch(), reference, need_cka=True)
        result["student_ce"].backward()
        adapters = [parameter for name, parameter in model.named_parameters() if "lora_B" in name]
        self.assertTrue(any(parameter.grad is not None and parameter.grad.norm().item() > 0 for parameter in adapters))
        torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1).step()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_bf16_checkpointed_full_ft_and_lora_keep_real_attention_gradients(self):
        from peft import LoraConfig, get_peft_model
        for lora in (False, True):
            with self.subTest(lora=lora):
                model = make_stage2()
                if lora:
                    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.4,
                                                           target_modules=["q_proj", "v_proj"],
                                                           task_type="CAUSAL_LM"))
                    model.get_model().mm_projector.requires_grad_(True)
                reference = snapshot_stage2_reference(model)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
                with stage2_deterministic_context(model):
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        result = stage2_forward(model, multimodal_batch(), reference, need_cka=True)
                    auxiliary = torch.autograd.grad(result["cka_loss"], [p for _, p in named],
                                                    allow_unused=True, retain_graph=True)
                    main = torch.autograd.grad(result["student_ce"], [p for _, p in named], allow_unused=True)
                self.assertTrue(any(gradient is not None and gradient.norm().item() > 0
                                    for gradient in auxiliary))
                self.assertTrue(all(gradient is None or torch.isfinite(gradient).all() for gradient in main))
                query = [(name, gradient) for (name, _), gradient in zip(named, main)
                         if "q_proj" in name and ("lora_B" in name if lora else name.endswith("weight"))]
                self.assertTrue(query)
                self.assertTrue(any(gradient is not None and gradient.norm().item() > 0 for _, gradient in query))

    def test_cpu_storage_and_forward_only_bf16_amp(self):
        model = make_stage2(hidden=False)
        original = model.forward
        model._original_forward = original

        def amp_forward(*args, **kwargs):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return original(*args, **kwargs)

        model.forward = amp_forward
        reference = snapshot_stage2_reference(model, placement="cpu")
        self.assertIs(model.forward, amp_forward)
        self.assertNotIn("_original_forward", reference.__dict__)
        result = stage2_forward(model, multimodal_batch(), reference, need_cka=True,
                                need_reference_logits=False)
        self.assertIsNone(result["reference_logits"])
        self.assertEqual(next(reference.parameters()).device.type, "cpu")
        for student, teacher in zip(result["projected_features"], result["reference_projected_features"]):
            torch.testing.assert_close(student, teacher, atol=0, rtol=0)

    def test_ce_only_control_does_not_execute_or_transfer_reference_decoder(self):
        model = make_stage2()
        reference = snapshot_stage2_reference(model, placement="cpu")
        with mock.patch.object(reference, "forward", side_effect=AssertionError("teacher decoder executed")), \
                mock.patch.object(reference, "to", side_effect=AssertionError("whole teacher transferred")):
            result = stage2_forward(model, multimodal_batch(), reference, need_cka=False,
                                    need_reference_logits=False)
        self.assertIsNone(result["reference_logits"])
        self.assertEqual(len(result["reference_projected_features"]), 2)
        self.assertFalse(result["cka_loss"].requires_grad)

    def test_existing_configured_vision_anchor_layers_are_preserved(self):
        from tests.test_cka_vision_anchor import tiny_tower
        model = make_stage2()
        model.get_model().vision_tower = tiny_tower()
        model.get_model().mm_projector = torch.nn.Linear(8, 16)
        model.config.cka_loss_vision_anchor_layer = 1
        model.config.cka_loss_projector_vision_anchor_layer = 3
        model.config.cka_loss_final_vision_anchor_layer = 3
        reference = snapshot_stage2_reference(model)
        batch = multimodal_batch()
        batch["images"] = torch.randn(2, 3, 4, 4)
        states = model.get_vision_tower().vision_tower(batch["images"], output_hidden_states=True).hidden_states
        with mock.patch.object(stage2_module, "_term", wraps=stage2_module._term) as terms:
            result = stage2_forward(model, batch, reference, need_cka=True)
        self.assertEqual(len(terms.call_args_list), 3)
        projector_anchor, middle_anchor, final_anchor = [call.args[0] for call in terms.call_args_list]
        torch.testing.assert_close(projector_anchor, states[3][:, 1:])
        torch.testing.assert_close(middle_anchor[:, 1:5], states[1][:, 1:])
        torch.testing.assert_close(final_anchor[:, 1:5], states[3][:, 1:])
        self.assertEqual(set(result["cka_terms"]), {"projector", "layer_1", "final"})

    def test_existing_llm_anchor_uses_separate_frozen_teacher_anchor(self):
        model = make_stage2()
        model.config.cka_loss_projector_weight = 0.0
        model.config.cka_loss_layers = "final"
        model.config.cka_loss_anchor_layer = 1
        reference = snapshot_stage2_reference(model)
        with torch.no_grad():
            model.get_model().layers[0].mlp.down_proj.weight.add_(0.5)
        with mock.patch.object(stage2_module, "_term", wraps=stage2_module._term) as terms:
            result = stage2_forward(model, multimodal_batch(), reference, need_cka=True)
        self.assertEqual(len(terms.call_args_list), 1)
        student_anchor, _, teacher_anchor = terms.call_args.args[:3]
        self.assertIsNot(student_anchor, teacher_anchor)
        self.assertFalse(student_anchor.requires_grad)
        self.assertFalse(teacher_anchor.requires_grad)
        self.assertFalse(torch.equal(student_anchor, teacher_anchor))
        self.assertTrue(torch.isfinite(result["cka_loss"]))

    def test_all_degenerate_terms_have_no_auxiliary_graph(self):
        model = make_stage2(hidden=False)
        reference = snapshot_stage2_reference(model)
        batch = multimodal_batch()
        batch["images"].fill_(1.0)
        result = stage2_forward(model, batch, reference, need_cka=True)
        self.assertEqual(result["cka_metrics"]["valid_images"], 0)
        self.assertEqual(result["cka_metrics"]["invalid_images"], 2)
        self.assertFalse(result["cka_loss"].requires_grad)

    def test_cleanup_on_forward_failure_and_reference_guards(self):
        model = make_stage2()
        reference = snapshot_stage2_reference(model)
        flags = [module.training for module in model.modules()]
        with mock.patch.object(model.get_model(), "forward", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                stage2_forward(model, multimodal_batch(), reference, need_cka=True)
        self.assertEqual([module.training for module in model.modules()], flags)
        self.assertNotIn("prepare_inputs_labels_for_multimodal", model.__dict__)
        self.assertNotIn("_compute_masked_linear_cka_loss", model.__dict__)
        self.assertEqual(len(model.get_model().mm_projector._forward_hooks), 0)
        with self.assertRaisesRegex(ValueError, "independent frozen"):
            stage2_forward(model, multimodal_batch(), model, need_cka=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        with self.assertRaisesRegex(ValueError, "use_reentrant=False"):
            with stage2_deterministic_context(model):
                pass

    def test_all_backbones_stage2_paired_gradient_smoke(self):
        for name, _, _ in _backbone_factories():
            with self.subTest(backbone=name):
                model = make_stage2(name, hidden=name != "mpt")
                reference = snapshot_stage2_reference(model)
                batch = multimodal_batch()
                batch.pop("position_ids")
                result = stage2_forward(model, batch, reference, need_cka=True)
                self.assertTrue(torch.isfinite(result["student_ce"]))
                self.assertTrue(torch.isfinite(result["cka_loss"]))
                (result["student_ce"] + result["cka_loss"]).backward()
                self.assertGreater(model.lm_head.weight.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
