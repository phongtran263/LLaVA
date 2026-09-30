import copy
import unittest

import torch

from llava.train.adaptive_projector_pcgrad import (
    AdaptiveProjectorPCGradConfig,
    AdaptiveProjectorPCGradController,
)


def _config(**overrides):
    values = {
        "stage": 1,
        "planned_optimizer_steps": 10,
        "max_aux_ratio": 0.5,
        "warmup_ratio": 0.0,
        "norm_ema_beta": 0.9,
        "lambda_max": 100.0,
        "ce_norm_floor": 1e-12,
        "aux_norm_floor": 1e-12,
        "residual_rtol": 1e-6,
    }
    values.update(overrides)
    return AdaptiveProjectorPCGradConfig(**values)


def _prepare(controller, main, auxiliary, *, valid_count=1.0):
    main = [torch.as_tensor(main, dtype=torch.float32)]
    auxiliary = [torch.as_tensor(auxiliary, dtype=torch.float32)]
    merged, logs = controller.prepare(
        main,
        auxiliary,
        valid_count=valid_count,
        reference_tensor=main[0],
    )
    update = merged[0] - main[0]
    return main[0], auxiliary[0], merged[0], update, logs


class AdaptiveProjectorPCGradMathTests(unittest.TestCase):
    def test_antiparallel_auxiliary_is_removed_without_changing_main(self):
        controller = AdaptiveProjectorPCGradController(_config())
        main, _, merged, update, logs = _prepare(controller, [1.0, 0.0], [-2.0, 0.0])

        torch.testing.assert_close(update, torch.zeros_like(update), atol=1e-7, rtol=0.0)
        torch.testing.assert_close(merged, main, atol=1e-7, rtol=0.0)
        self.assertEqual(logs["conflict"], 1.0)
        self.assertGreater(logs["lambda"], 0.0)
        self.assertLessEqual(logs["effective_aux_ratio"], controller.rho + 1e-7)

    def test_two_dimensional_conflict_projects_to_feasible_half_space(self):
        controller = AdaptiveProjectorPCGradController(_config())
        main, _, merged, update, logs = _prepare(controller, [1.0, 0.0], [-1.0, 1.0])

        expected_y = controller.rho / (2.0 ** 0.5)
        torch.testing.assert_close(
            update,
            torch.tensor([0.0, expected_y]),
            atol=2e-7,
            rtol=2e-7,
        )
        self.assertGreaterEqual(float(torch.dot(main, update)), -1e-7)
        self.assertLessEqual(float(torch.linalg.vector_norm(update)), controller.rho + 1e-7)
        torch.testing.assert_close(merged, main + update)
        self.assertEqual(logs["cap_scale"], 1.0)

    def test_projection_does_not_reinflate_discarded_conflicting_component(self):
        controller = AdaptiveProjectorPCGradController(_config())
        _, _, _, update, logs = _prepare(controller, [1.0, 0.0], [-1.0, 1.0])

        # The raw scaled auxiliary has norm rho. Orthogonal projection removes
        # one component, and the cap must not stretch the remainder back to rho.
        self.assertAlmostEqual(logs["projection_retention"], 2.0 ** -0.5, places=6)
        self.assertAlmostEqual(logs["cap_scale"], 1.0, places=7)
        self.assertLess(float(torch.linalg.vector_norm(update)), controller.rho)

    def test_hard_cap_uses_current_ce_norm_when_ema_ratio_is_stale(self):
        controller = AdaptiveProjectorPCGradController(
            _config(max_aux_ratio=0.25, norm_ema_beta=0.9)
        )
        _prepare(controller, [10.0, 0.0], [1.0, 0.0])
        controller.finish(optimizer_stepped=True)

        main, _, _, update, logs = _prepare(controller, [1.0, 0.0], [10.0, 0.0])
        allowed = controller.rho * float(torch.linalg.vector_norm(main))
        self.assertLess(logs["cap_scale"], 1.0)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(update)), allowed, places=6)
        self.assertLessEqual(logs["effective_aux_ratio"], controller.rho + 1e-7)

    def test_missing_gradient_entries_are_supported(self):
        controller = AdaptiveProjectorPCGradController(_config())
        main = [torch.tensor([1.0]), None]
        auxiliary = [None, torch.tensor([2.0])]
        merged, logs = controller.prepare(
            main,
            auxiliary,
            valid_count=1.0,
            reference_tensor=main[0],
        )

        # Main and auxiliary live in disjoint coordinates. The auxiliary is
        # capped but retained, while the original main coordinate is untouched.
        torch.testing.assert_close(merged[0], main[0])
        self.assertIsNotNone(merged[1])
        self.assertLessEqual(float(merged[1].norm()), controller.rho + 1e-7)
        self.assertEqual(logs["conflict"], 0.0)

    def test_invalid_valid_count_is_rejected(self):
        controller = AdaptiveProjectorPCGradController(_config())
        with self.assertRaisesRegex(ValueError, "valid_count"):
            _prepare(controller, [1.0], [1.0], valid_count=float("nan"))

    def test_fp32_boundary_roundoff_is_corrected_before_invariant_check(self):
        controller = AdaptiveProjectorPCGradController(
            _config(
                max_aux_ratio=0.1,
                norm_ema_beta=0.95,
                lambda_max=1.0,
                residual_rtol=1e-7,
            )
        )
        main, _, _, update, logs = _prepare(
            controller,
            [1.5409960746765137, -0.293428897857666],
            [-2.3328890800476074, 0.5977741479873657],
        )

        tolerance = 2e-7 * float(main.norm()) * max(float(update.norm()), 1e-30)
        self.assertGreaterEqual(float(torch.dot(main, update)), -tolerance)
        self.assertLessEqual(float(update.norm()), controller.rho * float(main.norm()) + 1e-7)
        self.assertEqual(logs["conflict"], 1.0)


class AdaptiveProjectorPCGradStateTests(unittest.TestCase):
    def test_ema_is_proposed_then_committed_only_after_successful_step(self):
        controller = AdaptiveProjectorPCGradController(_config())
        _prepare(controller, [3.0, 4.0], [0.0, 2.0])

        self.assertIsNone(controller.ema_g)
        self.assertIsNone(controller.ema_b)
        self.assertEqual(controller.successful_steps, 0)
        controller.finish(optimizer_stepped=False)
        self.assertIsNone(controller.ema_g)
        self.assertIsNone(controller.ema_b)
        self.assertEqual(controller.successful_steps, 0)
        self.assertEqual(controller.overflow_skips, 1)

        _prepare(controller, [3.0, 4.0], [0.0, 2.0])
        controller.finish(optimizer_stepped=True)
        self.assertEqual(controller.ema_g, 5.0)
        self.assertEqual(controller.ema_b, 2.0)
        self.assertEqual(controller.successful_steps, 1)
        self.assertEqual(controller.valid_proposals, 1)

    def test_skipped_optimizer_step_does_not_advance_warmup(self):
        controller = AdaptiveProjectorPCGradController(
            _config(planned_optimizer_steps=10, warmup_ratio=0.5)
        )
        initial_rho = controller.rho
        _prepare(controller, [1.0], [1.0])
        controller.finish(optimizer_stepped=False)
        self.assertEqual(controller.rho, initial_rho)

        _prepare(controller, [1.0], [1.0])
        controller.finish(optimizer_stepped=True)
        self.assertGreater(controller.rho, initial_rho)

    def test_all_invalid_window_advances_successful_step_but_not_ema(self):
        controller = AdaptiveProjectorPCGradController(_config())
        _, _, _, update, logs = _prepare(
            controller,
            [1.0, 2.0],
            [7.0, 8.0],
            valid_count=0.0,
        )
        torch.testing.assert_close(update, torch.zeros_like(update))
        self.assertEqual(logs["lambda"], 0.0)
        controller.finish(optimizer_stepped=True)
        self.assertEqual(controller.successful_steps, 1)
        self.assertIsNone(controller.ema_g)
        self.assertIsNone(controller.ema_b)

    def test_resume_round_trip_preserves_adaptive_state(self):
        controller = AdaptiveProjectorPCGradController(_config(stage=2))
        _prepare(controller, [3.0, 4.0], [0.0, 2.0])
        controller.finish(optimizer_stepped=True)
        state = copy.deepcopy(controller.state_dict())

        restored = AdaptiveProjectorPCGradController(_config(stage=2))
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)

        _, _, merged_a, _, logs_a = _prepare(controller, [1.0, 2.0], [-2.0, 1.0])
        _, _, merged_b, _, logs_b = _prepare(restored, [1.0, 2.0], [-2.0, 1.0])
        torch.testing.assert_close(merged_a, merged_b)
        self.assertEqual(logs_a, logs_b)

    def test_resume_rejects_stage_or_planned_step_mismatch(self):
        source = AdaptiveProjectorPCGradController(_config(stage=1))
        state = source.state_dict()
        with self.assertRaisesRegex(RuntimeError, "another stage"):
            AdaptiveProjectorPCGradController(_config(stage=2)).load_state_dict(state)
        with self.assertRaisesRegex(RuntimeError, "planned[_ ]optimizer[_ ]steps"):
            AdaptiveProjectorPCGradController(
                _config(planned_optimizer_steps=11)
            ).load_state_dict(state)

    def test_resume_rejects_other_controller_config_mismatch(self):
        source = AdaptiveProjectorPCGradController(_config(max_aux_ratio=0.5))
        state = source.state_dict()
        with self.assertRaisesRegex(RuntimeError, "config"):
            AdaptiveProjectorPCGradController(
                _config(max_aux_ratio=0.25)
            ).load_state_dict(state)

    def test_resume_rejects_nonfinite_ema_and_invalid_counters(self):
        state = AdaptiveProjectorPCGradController(_config()).state_dict()
        bad_ema = copy.deepcopy(state)
        bad_ema["ema_g"] = float("nan")
        bad_ema["ema_b"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "invalid ema_g"):
            AdaptiveProjectorPCGradController(_config()).load_state_dict(bad_ema)

        bad_counter = copy.deepcopy(state)
        bad_counter["valid_proposals"] = 1
        with self.assertRaisesRegex(RuntimeError, "more valid proposals"):
            AdaptiveProjectorPCGradController(_config()).load_state_dict(bad_counter)


if __name__ == "__main__":
    unittest.main()
