import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import HfArgumentParser

from llava.model.llava_arch import compute_relation_loss, compute_visual_structure_loss
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from tests.test_cvpr2027_cka import _batch, _tiny_model
from llava.train.train import ModelArguments, validate_relation_loss_args


class _Model:
    def __init__(self, mode):
        self.model = SimpleNamespace(config=SimpleNamespace(cka_loss_objective=mode))

    def get_model(self):
        return self.model


class RelationLossTests(unittest.TestCase):
    def test_cli_and_reject_silent_fallback(self):
        parsed, = HfArgumentParser(ModelArguments).parse_args_into_dataclasses(
            ["--cka_loss", "True", "--cka_loss_objective", "cosrel_uka_rkdd"])
        self.assertEqual(parsed.cka_loss_objective, "cosrel_uka_rkdd")
        validate_relation_loss_args(parsed, SimpleNamespace(protected_cka_refinement=False,
                                                           protected_stage2=False))
        parsed.cka_loss = False
        with self.assertRaisesRegex(ValueError, "require --cka_loss True"):
            validate_relation_loss_args(parsed, SimpleNamespace())

    def test_identical_features_and_backward(self):
        teacher = torch.randn(2, 7, 5)
        for mode in ("cosrel", "cosrel_uka", "cosrel_uka_rkdd"):
            with self.subTest(mode=mode):
                student = teacher.detach().clone().requires_grad_()
                loss = compute_relation_loss(teacher.detach(), student, mode)
                self.assertLess(abs(loss.item()), 1e-5)
                grad, = torch.autograd.grad(loss, student)
                self.assertTrue(torch.isfinite(grad).all())

    def test_cosrel_is_off_diagonal_per_image_mse(self):
        teacher = torch.randn(2, 5, 3)
        student = torch.randn(2, 5, 4, requires_grad=True)
        tx = F.normalize(teacher, dim=-1)
        sy = F.normalize(student, dim=-1)
        idx = torch.triu_indices(5, 5, 1)
        manual = ((tx @ tx.transpose(1, 2))[:, idx[0], idx[1]]
                  - (sy @ sy.transpose(1, 2))[:, idx[0], idx[1]]).square().mean(dim=1).mean()
        torch.testing.assert_close(compute_relation_loss(teacher, student, "cosrel"), manual)

    def test_combination_is_additive_and_width_independent(self):
        teacher = torch.randn(1, 6, 3)
        student = torch.randn(1, 6, 9, requires_grad=True)
        cos = compute_relation_loss(teacher, student, "cosrel")
        cos_uka = compute_relation_loss(teacher, student, "cosrel_uka")
        uka_only = compute_relation_loss(teacher, student, "cosrel_uka", cos_weight=0)
        torch.testing.assert_close(cos_uka, cos + uka_only)
        all_terms = compute_relation_loss(teacher, student, "cosrel_uka_rkdd")
        rkd_only = compute_relation_loss(teacher, student, "cosrel_uka_rkdd",
                                         cos_weight=0, uka_weight=0)
        torch.testing.assert_close(all_terms, cos_uka + rkd_only)
        self.assertGreaterEqual(rkd_only.item(), 0)

    def test_masked_visual_tokens_only_and_detached_anchor(self):
        teacher = torch.randn(2, 6, 3, requires_grad=True)
        student = torch.randn(2, 6, 4, requires_grad=True)
        mask = torch.tensor([[True, True, False, True, False, False],
                             [False, False, True, False, False, False]])
        model = _Model("cosrel_uka_rkdd")
        loss = LlavaLlamaForCausalLM._compute_masked_linear_cka_loss(
            model, student, teacher.detach(), mask)
        expected = compute_relation_loss(teacher.detach()[0:1, mask[0]],
                                         student[0:1, mask[0]], "cosrel_uka_rkdd")
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad[~mask].abs().sum().item(), 0.0)

    def test_projector_dispatch(self):
        teacher = torch.randn(2, 5, 3)
        student = torch.randn(2, 5, 4)
        config = SimpleNamespace(cka_loss_objective="cosrel_uka", relation_cos_weight=0.5,
                                 relation_uka_weight=2.0)
        actual = compute_visual_structure_loss(teacher, student, config)
        expected = compute_relation_loss(teacher, student, "cosrel_uka", 0.5, 2.0)
        torch.testing.assert_close(actual, expected)

    def test_stage1_and_stage2_forward_keep_ce_and_route_gradients(self):
        for mode in ("cosrel", "cosrel_uka", "cosrel_uka_rkdd"):
            for final_weight in (0.0, 0.1):
                with self.subTest(mode=mode, final_weight=final_weight):
                    model = _tiny_model("llama", 0.1, final_weight)
                    model.config.cka_loss_objective = mode
                    output = model(**_batch())
                    self.assertTrue(torch.isfinite(output.loss))
                    self.assertTrue(torch.isfinite(output.projector_cka_loss))
                    self.assertTrue(torch.isfinite(output.aux_losses[0]))
                    total = output.loss + output.projector_cka_loss + sum(output.aux_losses)
                    total.backward()
                    self.assertTrue(torch.isfinite(model.get_model().mm_projector[0].weight.grad).all())
                    self.assertIsNone(model.get_vision_tower().features.grad)
                    if final_weight:
                        decoder_grad = model.get_model().layers[-1].mlp.down_proj.weight.grad
                        self.assertTrue(torch.isfinite(decoder_grad).all())


if __name__ == "__main__":
    unittest.main()
