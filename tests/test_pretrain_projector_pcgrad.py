import copy
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from llava.model.multimodal_projector.builder import build_vision_projector
from llava.train.adaptive_projector_pcgrad import (
    AdaptiveProjectorPCGradConfig,
    AdaptiveProjectorPCGradController,
)
from llava.train.projector_replay_reference import (
    ProjectorReplayAccumulator,
    _strict_fp32_matmul,
    projector_cka_sum,
    validate_replayable_projector,
)


class ProjectorCkaTests(unittest.TestCase):
    def test_strict_fp32_matmul_temporarily_disables_tf32(self):
        previous = bool(torch.backends.cuda.matmul.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            with _strict_fp32_matmul(torch.device("cuda")):
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def test_cka_is_computed_per_observation_and_anchor_is_detached(self):
        torch.manual_seed(10)
        vision = torch.randn(3, 7, 4, requires_grad=True)
        projected = torch.randn(3, 7, 5, requires_grad=True)

        combined = projector_cka_sum(vision, projected)
        individual = [
            projector_cka_sum(vision[index:index + 1], projected[index:index + 1])
            for index in range(vision.shape[0])
        ]
        self.assertEqual(combined.valid_count.item(), 3.0)
        torch.testing.assert_close(
            combined.loss_sum,
            sum(item.loss_sum for item in individual),
        )

        combined.loss_sum.backward()
        self.assertIsNone(vision.grad)
        self.assertIsNotNone(projected.grad)
        self.assertTrue(torch.isfinite(projected.grad).all())

    def test_mixed_invalid_observations_are_counted_and_excluded(self):
        torch.manual_seed(11)
        vision = torch.randn(3, 6, 4)
        projected = torch.randn(3, 6, 5, requires_grad=True)
        vision[0, 0, 0] = float("nan")
        vision[1].fill_(1.0)

        result = projector_cka_sum(vision, projected)
        self.assertIsNotNone(result.loss_sum)
        self.assertEqual(result.valid_count.item(), 1.0)
        self.assertEqual(result.invalid_counts["nonfinite_input"].item(), 1.0)
        self.assertEqual(result.invalid_counts["zero_gram_norm"].item(), 1.0)
        result.loss_sum.backward()
        self.assertTrue(torch.isfinite(projected.grad).all())
        torch.testing.assert_close(projected.grad[0], torch.zeros_like(projected.grad[0]))
        torch.testing.assert_close(projected.grad[1], torch.zeros_like(projected.grad[1]))

    def test_all_invalid_and_too_few_patch_batches_are_safe(self):
        projected = torch.randn(2, 5, 3, requires_grad=True)
        all_constant = projector_cka_sum(torch.ones(2, 5, 4), projected)
        self.assertEqual(all_constant.loss_sum.item(), 0.0)
        self.assertEqual(all_constant.valid_count.item(), 0.0)
        self.assertEqual(all_constant.invalid_counts["zero_gram_norm"].item(), 2.0)

        too_short = projector_cka_sum(
            torch.randn(4, 1, 4),
            torch.randn(4, 1, 3, requires_grad=True),
        )
        self.assertEqual(too_short.loss_sum.item(), 0.0)
        self.assertEqual(too_short.valid_count.item(), 0.0)
        self.assertEqual(too_short.invalid_counts["too_few_patches"].item(), 4.0)

    def test_empty_observation_batch_is_safe(self):
        result = projector_cka_sum(torch.empty(0, 5, 4), torch.empty(0, 5, 3))
        self.assertEqual(result.loss_sum.item(), 0.0)
        self.assertEqual(result.valid_count.item(), 0.0)
        self.assertTrue(all(value.item() == 0.0 for value in result.invalid_counts.values()))

    def test_large_identity_cka_accepts_only_fp32_reduction_roundoff(self):
        torch.manual_seed(12)
        vision = torch.randn(4, 576, 64)
        projected = vision.detach().clone().requires_grad_(True)

        result = projector_cka_sum(vision, projected, score_rtol=1e-7)

        self.assertEqual(result.valid_count.item(), 4.0)
        self.assertEqual(result.invalid_counts["score_out_of_range"].item(), 0.0)
        self.assertLessEqual(abs(float(result.loss_sum.item())), 1e-4)
        result.loss_sum.backward()
        self.assertTrue(torch.isfinite(projected.grad).all())

    def test_finite_inputs_with_nonfinite_gram_are_severe_invalid(self):
        vision = torch.tensor(
            [[[1e20, -1e20], [1e20, 1e20], [-1e20, 1e20]]],
            dtype=torch.float32,
        )
        projected = vision.detach().clone().requires_grad_(True)

        result = projector_cka_sum(vision, projected)

        self.assertEqual(result.valid_count.item(), 0.0)
        self.assertEqual(result.invalid_counts["nonfinite_input"].item(), 0.0)
        self.assertEqual(result.invalid_counts["nonfinite_gram"].item(), 1.0)
        self.assertEqual(result.invalid_counts["zero_gram_norm"].item(), 0.0)


class ProjectorReplayTests(unittest.TestCase):
    @staticmethod
    def _build(projector_type):
        return build_vision_projector(
            SimpleNamespace(
                mm_projector_type=projector_type,
                mm_hidden_size=4,
                hidden_size=6,
            )
        )

    def test_replay_value_and_gradient_match_direct_autograd(self):
        for index, projector_type in enumerate(("linear", "mlp2x_gelu", "coupling2x_gelu")):
            with self.subTest(projector_type=projector_type):
                torch.manual_seed(100 + index)
                projector = self._build(projector_type)
                features = torch.randn(3, 8, 4)
                projected = projector(features)
                direct = projector_cka_sum(features, projected)
                parameters = [parameter for parameter in projector.parameters() if parameter.requires_grad]
                direct_gradients = torch.autograd.grad(direct.loss_sum, parameters)

                replay = ProjectorReplayAccumulator(projector, chunk_size=2)
                replay.begin_window(stage=1, optimizer_step=0)
                replay.verify_equivalence(features[:1], atol=1e-7, rtol=1e-6)
                replay.add([features[:1], features[1:]])
                gradients, loss_sum, valid_count, invalid = replay.consume()

                torch.testing.assert_close(loss_sum.float(), direct.loss_sum.detach())
                self.assertEqual(valid_count.item(), direct.valid_count.item())
                self.assertTrue(all(value.item() == 0.0 for value in invalid.values()))
                for actual, expected in zip(gradients, direct_gradients):
                    torch.testing.assert_close(actual, expected.float(), atol=2e-5, rtol=2e-5)

    def test_strict_fp32_scope_covers_replay_backward(self):
        projector = self._build("linear")
        replay = ProjectorReplayAccumulator(projector, chunk_size=1)
        replay.begin_window(stage=1, optimizer_step=0)
        depth = [0]
        backward_depths = []

        @contextmanager
        def tracked_scope(device):
            depth[0] += 1
            try:
                yield
            finally:
                depth[0] -= 1

        handle = replay.sidecar_params[0].register_hook(
            lambda gradient: backward_depths.append(depth[0]) or gradient
        )
        try:
            with mock.patch(
                "llava.train.projector_replay_reference._strict_fp32_matmul",
                tracked_scope,
            ):
                replay.add([torch.randn(1, 7, 4)])
        finally:
            handle.remove()

        self.assertTrue(backward_depths)
        self.assertTrue(all(value > 0 for value in backward_depths))
        self.assertEqual(depth[0], 0)

    def test_all_invalid_replay_returns_zero_gradient_sums(self):
        projector = self._build("linear")
        replay = ProjectorReplayAccumulator(projector, chunk_size=1)
        replay.begin_window(stage=1, optimizer_step=0)
        replay.add([torch.ones(2, 5, 4)])
        gradients, loss_sum, valid_count, invalid = replay.consume()

        self.assertEqual(loss_sum.item(), 0.0)
        self.assertEqual(valid_count.item(), 0.0)
        self.assertEqual(invalid["zero_gram_norm"].item(), 2.0)
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_bfloat16_replay_and_manual_scaled_copy_match(self):
        torch.manual_seed(211)
        projector = self._build("linear").to(torch.bfloat16)
        features = torch.randn(2, 7, 4, dtype=torch.bfloat16)
        direct = projector_cka_sum(features, projector(features))
        parameters = [parameter for parameter in projector.parameters() if parameter.requires_grad]
        expected = torch.autograd.grad(direct.loss_sum, parameters)

        replay = ProjectorReplayAccumulator(projector, chunk_size=2)
        replay.begin_window(stage=2, optimizer_step=0)
        replay.add([features], loss_scale=128.0)
        actual, loss_sum, valid_count, _ = replay.consume()

        self.assertEqual(valid_count.item(), 2.0)
        torch.testing.assert_close(loss_sum.float(), direct.loss_sum.detach(), atol=2e-5, rtol=2e-5)
        for copied, reference in zip(actual, expected):
            torch.testing.assert_close(copied, reference.float(), atol=3e-3, rtol=3e-3)

    def test_flat_buffer_is_recycled_across_accumulation_windows(self):
        projector = self._build("linear")
        features = torch.randn(2, 6, 4)
        replay = ProjectorReplayAccumulator(projector, chunk_size=2)
        replay.begin_window(stage=1, optimizer_step=0)
        replay.add([features])
        owned, _, count, _ = replay.consume_flat()
        self.assertEqual(count.item(), 2.0)
        self.assertIsNone(replay.gradient_sums_flat)

        replay.recycle_flat_buffer(owned)
        self.assertIs(replay.gradient_sums_flat, owned)
        self.assertEqual(replay.gradient_sums_flat.count_nonzero().item(), 0)
        replay.begin_window(stage=1, optimizer_step=1)
        replay.add([features])
        _, _, next_count, _ = replay.consume()
        self.assertEqual(next_count.item(), 2.0)

    def test_replay_rejects_stateful_or_parameterless_projector(self):
        with self.assertRaisesRegex(RuntimeError, "stochastic/stateful"):
            validate_replayable_projector(nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1)))
        with self.assertRaisesRegex(RuntimeError, "trainable projector"):
            validate_replayable_projector(nn.Identity())

    def test_replay_rejects_module_and_parameter_hooks(self):
        module_hooked = nn.Linear(4, 4)
        module_handle = module_hooked.register_full_backward_pre_hook(
            lambda module, grad_output: grad_output
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "rejects hooks"):
                validate_replayable_projector(module_hooked)
        finally:
            module_handle.remove()

        parameter_hooked = nn.Linear(4, 4)
        parameter_handle = parameter_hooked.weight.register_hook(lambda gradient: gradient)
        try:
            with self.assertRaisesRegex(RuntimeError, "parameter weight"):
                validate_replayable_projector(parameter_hooked)
        finally:
            parameter_handle.remove()


class _TinyVlm(nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_projector = nn.Sequential(
            nn.Linear(4, 6),
            nn.GELU(),
            nn.Linear(6, 3),
        )
        self.decoder = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))

    def forward(self, features):
        return self.decoder(self.mm_projector(features))


class ProjectorOnlyGradientIntegrationTests(unittest.TestCase):
    def test_frozen_decoder_still_backpropagates_ce_to_projector(self):
        torch.manual_seed(21)
        model = _TinyVlm()
        for parameter in model.decoder.parameters():
            parameter.requires_grad_(False)
        features = torch.randn(2, 7, 4)
        target = torch.randn(2, 7, 2)

        loss = torch.nn.functional.mse_loss(model(features), target)
        loss.backward()

        projector_gradients = [parameter.grad for parameter in model.mm_projector.parameters()]
        self.assertTrue(all(gradient is not None for gradient in projector_gradients))
        self.assertGreater(sum(float(gradient.norm()) for gradient in projector_gradients), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.decoder.parameters()))

    def test_sidecar_and_projector_merge_leave_stage2_decoder_ce_gradient_unchanged(self):
        torch.manual_seed(22)
        baseline = _TinyVlm()
        adaptive = copy.deepcopy(baseline)
        features = torch.randn(2, 7, 4)
        target = torch.randn(2, 7, 2)

        torch.nn.functional.mse_loss(baseline(features), target).backward()
        torch.nn.functional.mse_loss(adaptive(features), target).backward()
        baseline_decoder_gradients = [
            parameter.grad.detach().clone() for parameter in baseline.decoder.parameters()
        ]
        adaptive_decoder_before = [
            parameter.grad.detach().clone() for parameter in adaptive.decoder.parameters()
        ]
        ce_projector_gradients = [
            parameter.grad.detach().clone() for parameter in adaptive.mm_projector.parameters()
        ]

        replay = ProjectorReplayAccumulator(adaptive.mm_projector, chunk_size=1)
        replay.begin_window(stage=2, optimizer_step=0)
        replay.add([features.detach()])
        auxiliary_sums, _, valid_count, _ = replay.consume()

        # Sidecar replay is isolated: it cannot mutate live CE gradients.
        for parameter, expected in zip(adaptive.mm_projector.parameters(), ce_projector_gradients):
            torch.testing.assert_close(parameter.grad, expected)
        for parameter, expected in zip(adaptive.decoder.parameters(), adaptive_decoder_before):
            torch.testing.assert_close(parameter.grad, expected)

        auxiliary_means = [gradient / valid_count.float() for gradient in auxiliary_sums]
        controller = AdaptiveProjectorPCGradController(
            AdaptiveProjectorPCGradConfig(
                stage=2,
                planned_optimizer_steps=1,
                max_aux_ratio=0.5,
                warmup_ratio=0.0,
                lambda_max=10.0,
            )
        )
        merged, _ = controller.prepare(
            ce_projector_gradients,
            auxiliary_means,
            valid_count=float(valid_count.item()),
            reference_tensor=ce_projector_gradients[0],
        )
        with torch.no_grad():
            for parameter, gradient in zip(adaptive.mm_projector.parameters(), merged):
                parameter.grad.copy_(gradient)
        controller.finish(optimizer_stepped=True)

        for actual, expected in zip(adaptive.decoder.parameters(), baseline_decoder_gradients):
            torch.testing.assert_close(actual.grad, expected)


if __name__ == "__main__":
    unittest.main()
