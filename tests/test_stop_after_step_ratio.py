import unittest
from types import SimpleNamespace

from llava.train.llava_trainer import (
    StopAfterStepRatioCallback,
    get_stop_after_step,
    validate_stop_after_step_ratio,
)


class StopAfterStepRatioTests(unittest.TestCase):
    def test_validation(self):
        self.assertIsNone(validate_stop_after_step_ratio(None))
        self.assertEqual(validate_stop_after_step_ratio(0.8), 0.8)
        for invalid in (0.0, 1.0, -0.1, 1.1, float("inf"), "invalid"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_stop_after_step_ratio(invalid)

    def test_stop_step_uses_full_horizon_and_rounds_up(self):
        self.assertEqual(get_stop_after_step(0.8, 5197), 4158)
        self.assertEqual(get_stop_after_step(0.8, 10), 8)

    def test_callback_saves_and_stops_at_boundary(self):
        callback = StopAfterStepRatioCallback()
        args = SimpleNamespace(stop_after_step_ratio=0.8)
        control = SimpleNamespace(should_save=False, should_training_stop=False)

        callback.on_step_end(
            args,
            SimpleNamespace(global_step=4157, max_steps=5197),
            control,
        )
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_training_stop)

        callback.on_step_end(
            args,
            SimpleNamespace(global_step=4158, max_steps=5197),
            control,
        )
        self.assertTrue(control.should_save)
        self.assertTrue(control.should_training_stop)

    def test_callback_is_disabled_by_default(self):
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        result = StopAfterStepRatioCallback().on_step_end(
            SimpleNamespace(stop_after_step_ratio=None),
            SimpleNamespace(global_step=8, max_steps=10),
            control,
        )
        self.assertIs(result, control)
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_training_stop)


if __name__ == "__main__":
    unittest.main()
