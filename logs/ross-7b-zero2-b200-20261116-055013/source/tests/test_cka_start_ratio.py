"""Focused tests for optimizer-step-based delayed CKA activation."""

import unittest
from types import SimpleNamespace

from llava.train.llava_trainer import (
    LLaVATrainer,
    get_cka_loss_schedule_state,
    validate_cka_loss_start_ratio,
)


class CkaLossStartRatioTests(unittest.TestCase):
    def test_ratio_validation_accepts_closed_interval(self):
        for value, expected in ((0, 0.0), ("0.8", 0.8), (1, 1.0)):
            with self.subTest(value=value):
                self.assertEqual(validate_cka_loss_start_ratio(value), expected)

    def test_ratio_validation_rejects_invalid_values(self):
        for value in (None, "invalid", -0.01, 1.01, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "cka_loss_start_ratio"):
                    validate_cka_loss_start_ratio(value)

    def test_default_ratio_is_active_immediately(self):
        active, start_step = get_cka_loss_schedule_state(
            enabled=True,
            start_ratio=0.0,
            global_step=0,
            max_steps=10,
        )
        self.assertTrue(active)
        self.assertEqual(start_step, 0)

    def test_eighty_percent_boundary_uses_optimizer_steps(self):
        before, start_step = get_cka_loss_schedule_state(True, 0.8, 7, 10)
        at_boundary, repeated_start_step = get_cka_loss_schedule_state(True, 0.8, 8, 10)
        after, _ = get_cka_loss_schedule_state(True, 0.8, 9, 10)

        self.assertEqual(start_step, 8)
        self.assertEqual(repeated_start_step, 8)
        self.assertFalse(before)
        self.assertTrue(at_boundary)
        self.assertTrue(after)

    def test_fractional_boundary_rounds_up(self):
        before, start_step = get_cka_loss_schedule_state(True, 0.8, 5, 7)
        at_boundary, _ = get_cka_loss_schedule_state(True, 0.8, 6, 7)

        self.assertEqual(start_step, 6)
        self.assertFalse(before)
        self.assertTrue(at_boundary)

    def test_ratio_one_has_no_active_training_update(self):
        last_training_step, start_step = get_cka_loss_schedule_state(True, 1.0, 9, 10)
        boundary_after_training, _ = get_cka_loss_schedule_state(True, 1.0, 10, 10)

        self.assertEqual(start_step, 10)
        self.assertFalse(last_training_step)
        self.assertTrue(boundary_after_training)

    def test_disabled_cka_stays_disabled_after_boundary(self):
        active, start_step = get_cka_loss_schedule_state(False, 0.8, 9, 10)
        self.assertFalse(active)
        self.assertEqual(start_step, 8)

    def test_resume_uses_restored_global_step(self):
        active, start_step = get_cka_loss_schedule_state(True, 0.8, 8, 10)
        self.assertTrue(active)
        self.assertEqual(start_step, 8)

    def test_gradient_accumulation_microbatches_share_optimizer_step_state(self):
        # With gradient_accumulation_steps=2, Trainer presents the same
        # global_step to both microbatches in an accumulation window.
        microbatch_global_steps = (7, 7, 8, 8)
        active_states = [
            get_cka_loss_schedule_state(True, 0.8, step, 10)[0]
            for step in microbatch_global_steps
        ]
        self.assertEqual(active_states, [False, False, True, True])

    def test_invalid_step_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "global_step"):
            get_cka_loss_schedule_state(True, 0.8, -1, 10)
        with self.assertRaisesRegex(ValueError, "max_steps"):
            get_cka_loss_schedule_state(True, 0.8, 0, 0)

    @staticmethod
    def _make_runtime_context_fixture(global_step=7):
        config = SimpleNamespace(cka_loss=True, cka_loss_start_ratio=0.8)
        configured_model = SimpleNamespace(config=config)
        runtime_model = SimpleNamespace(
            last_cka_loss="stale",
            last_cka_projector_loss="stale",
            _aux_losses=["stale"],
        )
        trainer = object.__new__(LLaVATrainer)
        trainer.model = configured_model
        trainer.state = SimpleNamespace(global_step=global_step, max_steps=10)

        # The production helper traverses wrapped models. This focused test
        # only needs to verify which values the context asks it to publish.
        trainer._set_model_attr = lambda model, name, value: setattr(model, name, value)
        return trainer, config, runtime_model

    def test_runtime_context_restores_config_after_normal_exit(self):
        trainer, config, runtime_model = self._make_runtime_context_fixture()

        with trainer._cka_loss_runtime_context(runtime_model) as active:
            self.assertFalse(active)
            self.assertFalse(config.cka_loss)
            self.assertIsNone(runtime_model.last_cka_loss)
            self.assertIsNone(runtime_model.last_cka_projector_loss)
            self.assertEqual(runtime_model._aux_losses, [])

        self.assertTrue(config.cka_loss)
        self.assertFalse(trainer._last_cka_schedule_active)
        self.assertEqual(trainer._last_cka_schedule_start_step, 8)
        self.assertEqual(trainer._last_cka_schedule_start_ratio, 0.8)

    def test_runtime_context_restores_config_after_exception(self):
        trainer, config, runtime_model = self._make_runtime_context_fixture()

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with trainer._cka_loss_runtime_context(runtime_model) as active:
                self.assertFalse(active)
                self.assertFalse(config.cka_loss)
                raise RuntimeError("forward failed")

        self.assertTrue(config.cka_loss)

    def test_runtime_context_is_active_at_resume_boundary(self):
        trainer, config, runtime_model = self._make_runtime_context_fixture(global_step=8)

        with trainer._cka_loss_runtime_context(runtime_model) as active:
            self.assertTrue(active)
            self.assertTrue(config.cka_loss)

        self.assertTrue(config.cka_loss)
        self.assertTrue(trainer._last_cka_schedule_active)


if __name__ == "__main__":
    unittest.main()
