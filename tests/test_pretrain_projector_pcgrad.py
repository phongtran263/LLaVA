import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from llava.train.llava_trainer import LLaVATrainer


class _ProjectorOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_projector = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.mm_projector.weight.fill_(1.0)


def _trainer():
    trainer = object.__new__(LLaVATrainer)
    trainer.args = SimpleNamespace(tune_mm_mlp_adapter=True)
    return trainer


class PretrainProjectorPCGradTest(unittest.TestCase):
    def test_opposing_aux_gradient_is_projected(self):
        model = _ProjectorOnlyModel()
        param = model.mm_projector.weight
        text_loss = param.sum()
        projector_loss = -param.sum()

        objective, backward_loss = _trainer()._build_pretrain_projector_pcgrad_loss(
            model,
            text_loss,
            projector_loss,
            [],
        )
        self.assertIsNone(param.grad)
        backward_loss.backward()

        self.assertEqual(objective.item(), 0.0)
        self.assertTrue(torch.allclose(param.grad, torch.ones_like(param)))

    def test_aligned_aux_gradient_is_preserved(self):
        model = _ProjectorOnlyModel()
        param = model.mm_projector.weight
        text_loss = param.sum()
        projector_loss = 0.5 * param.sum()

        objective, backward_loss = _trainer()._build_pretrain_projector_pcgrad_loss(
            model,
            text_loss,
            projector_loss,
            [],
        )
        backward_loss.backward()

        self.assertEqual(objective.item(), 3.0)
        self.assertTrue(torch.allclose(param.grad, torch.full_like(param, 1.5)))

    def test_rejects_other_trainable_parameters(self):
        model = _ProjectorOnlyModel()
        model.extra = nn.Parameter(torch.ones(1))
        param = model.mm_projector.weight

        with self.assertRaisesRegex(RuntimeError, 'projector-only training'):
            _trainer()._build_pretrain_projector_pcgrad_loss(
                model,
                param.sum(),
                -param.sum(),
                [],
            )

    def test_rejects_active_non_projector_aux(self):
        model = _ProjectorOnlyModel()
        param = model.mm_projector.weight

        with self.assertRaisesRegex(RuntimeError, 'projector CKA only'):
            _trainer()._build_pretrain_projector_pcgrad_loss(
                model,
                param.sum(),
                -param.sum(),
                [param.square().sum()],
            )

    def test_requires_pretrain_adapter_mode(self):
        model = _ProjectorOnlyModel()
        param = model.mm_projector.weight
        trainer = _trainer()
        trainer.args.tune_mm_mlp_adapter = False

        with self.assertRaisesRegex(RuntimeError, 'tune_mm_mlp_adapter'):
            trainer._build_pretrain_projector_pcgrad_loss(
                model,
                param.sum(),
                -param.sum(),
                [],
            )


if __name__ == '__main__':
    unittest.main()
