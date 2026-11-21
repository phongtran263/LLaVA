"""Parameter-space regressions for the current VSP controller (not removed replay APIs)."""
import unittest
from unittest import mock

import torch
from torch import nn

from llava.train.vsp_gradient_controller import VSPGradientController
from tests.test_vsp_gradient_controller import make_config


class ProjectorPcGradTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(16)
        model = nn.Module()
        model.mm_projector = nn.Linear(2, 3)
        model.decoder = nn.Linear(3, 2)
        return model

    def test_weighted_parameter_projection_matches_independent_reference(self):
        for weight in (0.1, 1., 3.):
            model = self.model()
            z = model.mm_projector(torch.randn(4, 2))
            output = model.decoder(z)
            main, proj, final = output.square().mean(), -weight*z.square().mean(), -0.3*output.sum()
            expected = []
            for params in (list(model.mm_projector.parameters()), list(model.decoder.parameters())):
                vectors = []
                for loss in (main, proj, final):
                    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    vectors.append(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                              for p, g in zip(params, grads)]))
                m = vectors[0]
                total = m.clone()
                for aux in vectors[1:]:
                    dot = torch.dot(m, aux)
                    if dot < 0:
                        aux = aux - dot / (m.square().sum()+1e-12) * m
                    total += aux
                expected.append(total)
            controller = VSPGradientController(model, make_config())
            with mock.patch('torch.autograd.grad', wraps=torch.autograd.grad) as grad:
                controller.compute_and_assign_gradients(main, proj, final, log_stats=False)
            self.assertEqual(grad.call_count, 3)  # Not three traversals per group.
            for params, wanted in zip((model.mm_projector.parameters(), model.decoder.parameters()), expected):
                actual = torch.cat([p.grad.flatten() for p in params])
                torch.testing.assert_close(actual, wanted, atol=1e-6, rtol=1e-5)

    def test_diagnostics_preserves_pending_gradients_and_graph(self):
        model = self.model()
        z = model.mm_projector(torch.ones(2, 2))
        main, aux = model.decoder(z).square().mean(), z.square().mean()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        controller = VSPGradientController(model, make_config(vsp_asymmetric_pcgrad=False))
        controller.compute_diagnostics(main, aux)
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters()))
        (main+aux).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_zero_main_gradient_preserves_auxiliary_without_cap(self):
        model = self.model()
        param = model.mm_projector.weight
        VSPGradientController(model, make_config()).compute_and_assign_gradients(
            param.sum()*0., param.sum(), log_stats=False,
        )
        torch.testing.assert_close(param.grad, torch.ones_like(param))
        self.assertIsNone(model.decoder.weight.grad)

    def test_unused_auxiliary_matches_plain_backward(self):
        model = self.model()
        loss = model.decoder(model.mm_projector(torch.ones(2, 2))).square().sum()
        expected = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True)
        VSPGradientController(model, make_config()).compute_and_assign_gradients(loss, log_stats=False)
        for p, grad in zip(model.parameters(), expected):
            torch.testing.assert_close(p.grad, grad)


if __name__ == '__main__':
    unittest.main()
