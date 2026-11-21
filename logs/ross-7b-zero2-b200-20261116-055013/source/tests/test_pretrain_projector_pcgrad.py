"""Pretraining and supported-backend checks for the current VSP controller."""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from llava.train.llava_trainer import LLaVATrainer
from llava.train.vsp_gradient_controller import VSPGradientController
from tests.test_vsp_gradient_controller import make_config


class PretrainProjectorPCGradTest(unittest.TestCase):
    def model(self):
        model = nn.Module()
        model.mm_projector = nn.Linear(2, 1, bias=False)
        model.decoder = nn.Linear(1, 1, bias=False).requires_grad_(False)
        model.decoder.weight.data.fill_(2.)
        model.config = make_config()
        return model

    def test_opposing_aux_gradient_is_projected(self):
        model = self.model()
        param = model.mm_projector.weight
        VSPGradientController(model, model.config).compute_and_assign_gradients(param.sum(), -param.sum())
        torch.testing.assert_close(param.grad, torch.ones_like(param))

    def test_aligned_aux_gradient_is_preserved(self):
        model = self.model()
        param = model.mm_projector.weight
        VSPGradientController(model, model.config).compute_and_assign_gradients(param.sum(), 0.5*param.sum())
        torch.testing.assert_close(param.grad, torch.full_like(param, 1.5))

    def test_frozen_decoder_still_backpropagates_to_projector(self):
        model = self.model()
        z = model.mm_projector(torch.ones(1, 2))
        main = model.decoder(z).sum()
        VSPGradientController(model, model.config).compute_and_assign_gradients(main, -z.sum())
        torch.testing.assert_close(model.mm_projector.weight.grad, torch.full_like(model.mm_projector.weight, 2.))
        self.assertIsNone(model.decoder.weight.grad)

    def test_projector_only_keeps_hidden_auxiliary_unprojected(self):
        model = self.model()
        model.config.vsp_apply_to_projector_only = True
        param = model.mm_projector.weight
        VSPGradientController(model, model.config).compute_and_assign_gradients(param.sum(), -param.sum(), -2*param.sum())
        torch.testing.assert_close(param.grad, -torch.ones_like(param))

    def test_unsupported_zero3_and_nonzero_accumulation_fail_early(self):
        trainer = object.__new__(LLaVATrainer)
        trainer.model = self.model()
        trainer.args = SimpleNamespace(world_size=1, gradient_accumulation_steps=2)
        with mock.patch.object(trainer, '_get_deepspeed_zero_stage', return_value=3):
            with self.assertRaisesRegex(RuntimeError, 'ZeRO-3'):
                trainer._validate_vsp_gradient_backend(trainer.model)
        with mock.patch.object(trainer, '_get_deepspeed_zero_stage', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'gradient_accumulation_steps=1'):
                trainer._validate_vsp_gradient_backend(trainer.model)


if __name__ == '__main__':
    unittest.main()
