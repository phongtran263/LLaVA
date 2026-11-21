"""Regression tests for the accepted-loss threshold, independent of legacy channel APIs."""
import math
import unittest
from types import SimpleNamespace

import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.llava_arch import (
    cka_similarity_to_loss,
    compute_linear_cka_loss,
    validate_cka_loss_tau,
)
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from tests.test_cka_vision_anchor import tiny_vlm


class CkaTauTests(unittest.TestCase):
    def test_validation(self):
        for value in (0, 0.05, 1, "0.125"):
            self.assertEqual(validate_cka_loss_tau(value), float(value))
        for value in (-0.01, 1.01, math.nan, math.inf, -math.inf, "bad", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_cka_loss_tau(value)

    def test_hinge_has_zero_gradient_at_and_below_raw_loss_threshold(self):
        cka = torch.tensor([0.75, 0.875, 0.9375, 1.0], requires_grad=True)
        losses = cka_similarity_to_loss(cka, tau=0.125)
        torch.testing.assert_close(losses, torch.tensor([0.125, 0.0, 0.0, 0.0]))
        losses.sum().backward()
        torch.testing.assert_close(cka.grad, torch.tensor([-1.0, 0.0, 0.0, 0.0]))

    def test_hinge_is_per_sample_before_mean_and_weight(self):
        cka = torch.tensor([0.875, 0.375], requires_grad=True)
        loss = 0.1 * cka_similarity_to_loss(cka, tau=0.25).mean()
        self.assertAlmostEqual(loss.item(), 0.01875, places=7)
        loss.backward()
        torch.testing.assert_close(cka.grad, torch.tensor([0.0, -0.05]))

    def test_global_and_masked_cka_use_same_per_sample_tolerance(self):
        a = torch.tensor([1.0, -1.0, 0.0, 0.0]).view(1, 4, 1)
        b = torch.tensor([0.0, 0.0, 1.0, -1.0]).view(1, 4, 1)
        x = torch.cat((a, a), dim=0)
        y = torch.cat((a, b), dim=0).requires_grad_(True)
        mask = torch.ones(2, 4, dtype=torch.bool)
        dummy = SimpleNamespace(get_model=lambda: SimpleNamespace(config=SimpleNamespace(cka_loss_tau=0.25)))
        loss = compute_linear_cka_loss(x, y, tau=0.25)
        masked = LlavaLlamaForCausalLM._compute_masked_linear_cka_loss(dummy, y, x, mask)
        self.assertAlmostEqual(loss.item(), 0.375, places=6)
        torch.testing.assert_close(loss, masked)
        self.assertAlmostEqual(compute_linear_cka_loss(x, y).item(), 0.5, places=6)

    def test_split_vision_anchors_obey_tau_and_preserve_ce(self):
        torch.manual_seed(83)
        model = tiny_vlm("qwen2", layers="1,final")
        model.config.cka_loss_vision_anchor_layer = 1
        model.config.cka_loss_projector_vision_anchor_layer = 3
        model.config.cka_loss_final_vision_anchor_layer = 3
        model.config.cka_loss_projector_weight = 0.1
        model.config.cka_loss_final_hidden_weight = 0.2
        inputs = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3]])
        images = torch.randn(1, 3, 4, 4)
        model.config.cka_loss_tau = 0.0
        baseline = model(input_ids=inputs, labels=inputs.clone(), images=images)
        raw_projector = model.last_cka_projector_loss.clone()
        raw_layers = dict(model.last_cka_per_layer_losses)
        self.assertEqual(len(raw_layers), 2)
        for tau in (0.125, 1.0):
            with self.subTest(tau=tau):
                model.config.cka_loss_tau = tau
                output = model(input_ids=inputs, labels=inputs.clone(), images=images)
                torch.testing.assert_close(output.loss, baseline.loss)
                torch.testing.assert_close(output.logits, baseline.logits)
                expected_projector = torch.relu(raw_projector - tau)
                expected_layers = {key: torch.relu(value - tau) for key, value in raw_layers.items()}
                torch.testing.assert_close(model.last_cka_projector_loss, expected_projector, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(output.projector_cka_loss, expected_projector * 0.1, atol=2e-6, rtol=1e-5)
                for key, expected in expected_layers.items():
                    torch.testing.assert_close(model.last_cka_per_layer_losses[key], expected, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(sum(output.aux_losses), sum(expected_layers.values()) * 0.2, atol=2e-6, rtol=1e-5)
                if tau == 1.0:
                    aux = output.projector_cka_loss + sum(output.aux_losses)
                    grad, = torch.autograd.grad(aux, model.get_model().mm_projector.weight, retain_graph=True)
                    self.assertEqual(grad.count_nonzero().item(), 0)
                    output.loss.backward()
                    ce_grad = model.get_model().mm_projector.weight.grad
                    self.assertTrue(torch.isfinite(ce_grad).all())
                    self.assertGreater(ce_grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
