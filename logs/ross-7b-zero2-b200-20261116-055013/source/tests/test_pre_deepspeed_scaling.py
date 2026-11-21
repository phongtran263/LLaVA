"""PRe CE/auxiliary gradients must be scaled once across DeepSpeed outputs."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from deepspeed.runtime.utils import register_output_backward_hooks
from llava.train import llava_trainer as training


class _DeepSpeedOutputAccelerator:
    """Use installed DS output hooks and its scale_wrt_gas switch on CPU."""

    def __init__(self, steps):
        self.steps = steps
        self.scale_wrt_gas = True
        self.calls = []

    def attach(self, outputs):
        def scale_output_gradient(gradient):
            return gradient / self.steps if self.scale_wrt_gas else gradient

        register_output_backward_hooks(outputs, lambda: None, scale_output_gradient)
        return outputs

    def backward(self, loss, **kwargs):
        self.calls.append(kwargs)
        self.scale_wrt_gas = kwargs.get("scale_wrt_gas", True)
        loss.backward()


class _NestedLossModel(nn.Module):
    def __init__(self, accelerator=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.predictor_weight = nn.Parameter(torch.tensor(0.25))
        self.config = SimpleNamespace(cka_loss=False, pre_loss=True)
        self.output_accelerator = accelerator

    def forward(self):
        # Both loss and logits are returned, and CE depends on those logits.
        # PRe also updates the backbone and has an independent predictor path.
        logits = 3 * self.weight
        ce = logits.square()
        pre = (self.weight - 1).square() + self.predictor_weight.square()
        outputs = {"loss": ce + 0.5 * pre, "logits": logits}
        if self.output_accelerator is not None:
            outputs = self.output_accelerator.attach(outputs)
        return outputs


class PreDeepSpeedScalingTests(unittest.TestCase):
    def test_installed_output_hooks_reproduce_ce_auxiliary_weight_distortion(self):
        for steps in (1, 2, 32):
            with self.subTest(steps=steps):
                accelerator = _DeepSpeedOutputAccelerator(steps)
                model = _NestedLossModel(accelerator)
                accelerator.backward(model()["loss"])
                # A default per-output hook divides the CE path at loss AND
                # logits; the PRe path passes through the loss hook once.
                expected = 36.0 / steps ** 2 + 1.0 / steps
                self.assertAlmostEqual(float(model.weight.grad), expected, places=7)
                self.assertAlmostEqual(float(model.predictor_weight.grad), 0.25 / steps, places=7)
                if steps > 1:
                    self.assertNotAlmostEqual(float(model.weight.grad), 37.0 / steps, places=7)

    def test_explicit_combined_scaling_matches_manual_reference(self):
        for steps in (1, 2, 32):
            with self.subTest(steps=steps):
                reference = _NestedLossModel()
                (reference()["loss"] / steps).backward()
                accelerator = _DeepSpeedOutputAccelerator(steps)
                model = _NestedLossModel(accelerator)
                training.backward_pre_deepspeed_loss(accelerator, model()["loss"], steps)
                self.assertEqual(accelerator.calls, [{"scale_wrt_gas": False}])
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)

    def test_complete_accumulation_has_same_ce_and_pre_gradient_for_all_gas(self):
        expected = _NestedLossModel()
        expected()["loss"].backward()
        for steps in (1, 2, 32):
            with self.subTest(steps=steps):
                accelerator = _DeepSpeedOutputAccelerator(steps)
                model = _NestedLossModel(accelerator)
                for _ in range(steps):
                    training.backward_pre_deepspeed_loss(accelerator, model()["loss"], steps)
                for actual, reference in zip(model.parameters(), expected.parameters()):
                    torch.testing.assert_close(actual.grad, reference.grad, rtol=0, atol=0)

    def test_pre_training_step_uses_once_scaled_deepspeed_path(self):
        for steps in (1, 2, 32):
            with self.subTest(steps=steps):
                accelerator = _DeepSpeedOutputAccelerator(steps)
                model = _NestedLossModel(accelerator)
                trainer = object.__new__(training.LLaVATrainer)
                trainer.model = model
                trainer.args = SimpleNamespace(n_gpu=1, gradient_accumulation_steps=steps)
                trainer.accelerator = accelerator
                trainer.is_deepspeed_enabled = True
                trainer.use_apex = False
                trainer.compute_loss_context_manager = nullcontext
                trainer.compute_loss = lambda current_model, _inputs: current_model()["loss"]
                trainer._collect_gradient_norm_logs = mock.Mock()
                trainer._clear_gradient_log_tensors = mock.Mock()
                trainer._vsp_controller_requested = lambda: False
                with mock.patch.object(
                    training, "backward_pre_deepspeed_loss",
                    wraps=training.backward_pre_deepspeed_loss,
                ) as backward:
                    result = trainer._training_step_with_cka_runtime_state(model, {})
                backward.assert_called_once()
                self.assertEqual(backward.call_args.args[2], steps)
                self.assertAlmostEqual(float(model.weight.grad), 37.0 / steps, places=7)
                self.assertAlmostEqual(float(model.predictor_weight.grad), 0.25 / steps, places=7)
                self.assertTrue(torch.isfinite(result))

    def test_invalid_accumulation_is_rejected_before_backward(self):
        accelerator = mock.Mock()
        for steps in (0, -1):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                training.backward_pre_deepspeed_loss(accelerator, torch.tensor(1.0), steps)
        accelerator.backward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
