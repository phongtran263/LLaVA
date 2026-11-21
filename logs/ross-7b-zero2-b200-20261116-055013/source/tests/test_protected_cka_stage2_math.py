"""CPU/offline active-set and per-group Stage 2 gradient-protection tests."""

import io
import unittest
from unittest import mock

import torch

from llava.train.protected_cka_math import protected_projector_gradient
from llava.train.protected_cka_stage2_math import group_gradient_norms, protected_group_gradient


def protect(main, cka, ema=None, **overrides):
    options = dict(group_names=["projector"] * len(main), ce_grad_refs={"projector": 100.},
                   aux_ratios={"projector": 1.}, ema_decay=0.5, lambda_t=1., stat_chunk_size=3)
    options.update(overrides)
    return protected_group_gradient(main, cka, ema, **options)


class ProtectedStage2GroupMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(86)

    def test_complete_single_group_matches_stage1_for_both_projection_signs(self):
        main = [torch.tensor([[1., 2.], [0., 1.]]), torch.tensor([3., 4.])]
        for sign in (-1., 1.):
            cka = [sign * torch.tensor([[3., 1.], [2., 4.]]), sign * torch.tensor([-1., 3.])]
            old = [torch.ones_like(main[0]), torch.tensor([2., 3.])]
            final, ema, logs = protect(main, cka, old)
            expected, expected_ema, expected_logs = protected_projector_gradient(
                torch.cat([value.flatten() for value in main]),
                torch.cat([value.flatten() for value in cka]),
                torch.cat([value.flatten() for value in old]),
                ce_grad_ref=100., aux_ratio=1., ema_decay=0.5,
            )
            torch.testing.assert_close(torch.cat([value.flatten() for value in final]), expected)
            torch.testing.assert_close(torch.cat([value.flatten() for value in ema]), expected_ema)
            self.assertAlmostEqual(logs["projector"]["aux_norm"], expected_logs["aux_norm"], places=5)
            dot = sum(torch.dot((value - original).flatten(), average.flatten())
                      for value, original, average in zip(final, main, ema))
            self.assertAlmostEqual(float(dot), 0., places=5)

    def test_none_cka_cannot_receive_auxiliary_despite_nonzero_main_and_ema(self):
        main = [torch.tensor([1., 2.]), torch.tensor([10., 20.]), None]
        cka = [torch.tensor([3., -1.]), None, None]
        old = [torch.tensor([4., 1.]), torch.tensor([500., 600.]), torch.tensor([700.])]
        old_before = [value.clone() for value in old]
        final, ema, logs = protect(main, cka, old)
        self.assertIs(final[1], main[1])
        self.assertIsNone(final[2])
        self.assertGreater(float((final[0] - main[0]).norm()), 0.)
        self.assertAlmostEqual(float(torch.dot(final[0] - main[0], ema[0])), 0., places=5)
        self.assertEqual(logs["projector"]["active_tensors"], 1)
        self.assertEqual(logs["projector"]["active_parameters"], 2)
        for actual, before in zip(old, old_before):
            torch.testing.assert_close(actual, before, rtol=0, atol=0)

    def test_all_text_only_has_exact_zero_auxiliary_even_after_multimodal_history(self):
        main = [torch.tensor([1., 2.]), None, torch.tensor([3.])]
        old = [torch.tensor([7., -2.]), torch.tensor([4., 5.]), None]
        final, ema, logs = protect(main, [None, None, None], old)
        for actual, original in zip(final, main):
            self.assertIs(actual, original)
        torch.testing.assert_close(ema[0], 0.5 * old[0] + 0.5 * main[0])
        torch.testing.assert_close(ema[1], 0.5 * old[1])
        self.assertEqual(logs["projector"]["active_tensors"], 0)
        self.assertEqual(logs["projector"]["scaled_aux_norm"], 0.)
        self.assertEqual(logs["projector"]["cka_norm"], 0.)

    def test_projector_and_llm_have_independent_projection_and_fixed_caps(self):
        updates = []
        for llm_magnitude in (100., 10000.):
            main = [torch.tensor([1., 0.]), torch.tensor([0.]), torch.tensor([llm_magnitude, 0.])]
            cka = [torch.tensor([2., 10.]), torch.tensor([3.]), torch.tensor([1., 20.])]
            final, ema, logs = protect(
                main, cka, group_names=["projector", "projector", "llm"],
                ce_grad_refs={"projector": 2., "llm": 20.}, aux_ratios={"projector": 0.1, "llm": 0.2},
            )
            updates.append([final[i] - main[i] for i in (0, 1)])
            self.assertLessEqual(logs["projector"]["scaled_aux_norm"], 0.2 + 1e-7)
            self.assertAlmostEqual(logs["projector"]["scaled_aux_norm"], 0.2, places=6)
            self.assertAlmostEqual(logs["llm"]["scaled_aux_norm"], 4., places=6)
            self.assertEqual(logs["projector"]["aux_cap"], 0.2)
            self.assertEqual(logs["llm"]["aux_cap"], 4.)
            self.assertAlmostEqual(float(torch.dot(final[2] - main[2], ema[2])), 0., places=5)
        for first, second in zip(*updates):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_projector_only_routing_leaves_llm_main_and_none_mask_untouched(self):
        main = [torch.tensor([1., 0.]), torch.tensor([7., 8.]), None]
        cka = [torch.tensor([2., 3.]), torch.tensor([10., -30.]), torch.tensor([4.])]
        final, ema, logs = protect(
            main, cka, group_names=["projector", "llm", "llm"],
            allowed_cka_mask=[True, False, False],
            ce_grad_refs={"projector": 100., "llm": 1000.}, aux_ratios={"projector": 1., "llm": 1.},
        )
        torch.testing.assert_close(final[0], torch.tensor([1., 3.]))
        self.assertIs(final[1], main[1])
        self.assertIsNone(final[2])
        self.assertIsNone(ema[1])
        self.assertIsNone(ema[2])
        self.assertEqual(logs["llm"]["active_tensors"], 0)
        self.assertEqual(logs["llm"]["allowed_tensors"], 0)
        self.assertEqual(logs["llm"]["scaled_aux_norm"], 0.)

    def test_small_ema_zero_cka_and_zero_strength_leave_main_exactly_unchanged(self):
        for main, cka, options in (
            ([torch.tensor([1e-10, 0.])], [torch.tensor([2., 3.])], {}),
            ([torch.tensor([1., 0.])], [torch.zeros(2)], {}),
            ([torch.tensor([1., 0.])], [torch.tensor([2., 3.])], {"lambda_t": 0.}),
            ([torch.tensor([1., 0.])], [torch.tensor([2., 3.])], {"aux_ratios": {"projector": 0.}}),
            ([torch.tensor([1., 0.])], [torch.tensor([2., 3.])], {"ce_grad_refs": {"projector": 0.}}),
        ):
            final, _, logs = protect(main, cka, **options)
            self.assertIs(final[0], main[0])
            self.assertEqual(logs["projector"]["scaled_aux_norm"], 0.)

    def test_missing_main_can_receive_aux_only_if_current_cka_graph_uses_parameter(self):
        main = [None, torch.tensor([1., 0.]), None]
        cka = [torch.tensor([2., 3.]), torch.tensor([3., 4.]), None]
        old = [torch.tensor([1., 0.]), None, torch.tensor([50.])]
        final, ema, _ = protect(main, cka, old)
        self.assertIsNotNone(final[0])
        self.assertIsNone(final[2])
        dot = torch.dot(final[0], ema[0]) + torch.dot(final[1] - main[1], ema[1])
        self.assertAlmostEqual(float(dot), 0., places=5)

    def test_none_everywhere_does_not_allocate_zero_gradient_or_ema(self):
        final, ema, logs = protect([None, None], [None, None])
        self.assertEqual(final, [None, None])
        self.assertEqual(ema, [None, None])
        self.assertEqual(logs["projector"]["main_norm"], 0.)
        self.assertEqual(logs["projector"]["aux_norm"], 0.)

    def test_group_calibration_norms_are_chunked_separate_and_masked(self):
        grads = [torch.tensor([3., 4.], dtype=torch.bfloat16), None, torch.tensor([12.]), torch.tensor([5.])]
        names = ["projector", "projector", "llm", "llm"]
        with mock.patch("torch.cat", side_effect=AssertionError("must not concatenate gradients")), \
                mock.patch("torch.dot", wraps=torch.dot) as dot:
            norms = group_gradient_norms(grads, names, chunk_size=1)
        self.assertEqual(norms, {"projector": 5., "llm": 13.})
        self.assertTrue(all(call.args[0].numel() <= 1 for call in dot.call_args_list))
        self.assertTrue(all(call.args[0].dtype == torch.float32 for call in dot.call_args_list))
        self.assertEqual(group_gradient_norms(grads, names, [True, True, True, False]),
                         {"projector": 5., "llm": 12.})

    def test_bfloat16_group_math_never_concatenates_and_bounds_fp32_dot_buffers(self):
        main = [torch.randn(3, 5, dtype=torch.bfloat16), torch.randn(7, dtype=torch.bfloat16)]
        cka = [torch.randn_like(value) for value in main]
        before = [value.clone() for value in main + cka]
        with mock.patch("torch.cat", side_effect=AssertionError("must not concatenate gradients")), \
                mock.patch("torch.dot", wraps=torch.dot) as dot:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                final, ema, _ = protect(main, cka, stat_chunk_size=2)
        self.assertTrue(all(call.args[0].numel() <= 2 for call in dot.call_args_list))
        self.assertTrue(all(call.args[0].dtype == torch.float32 for call in dot.call_args_list))
        self.assertTrue(all(value.dtype == torch.float32 for value in final + ema))
        for actual, original in zip(main + cka, before):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_serialized_ema_replays_identical_next_step_without_mutation(self):
        main = [torch.randn(3, 4), torch.randn(5), None]
        cka = [torch.randn(3, 4), None, None]
        _, ema, _ = protect(main, cka)
        snapshot = [None if value is None else value.clone() for value in ema]
        buffer = io.BytesIO()
        torch.save(ema, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        next_main = [torch.randn(3, 4), torch.randn(5), None]
        next_cka = [torch.randn(3, 4), torch.randn(5), None]
        first = protect(next_main, next_cka, ema)
        second = protect(next_main, next_cka, restored)
        for left_values, right_values in zip(first[:2], second[:2]):
            for left, right in zip(left_values, right_values):
                if left is None:
                    self.assertIsNone(right)
                else:
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(first[2], second[2])
        for actual, original in zip(ema, snapshot):
            if actual is not None:
                torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_noncontiguous_parameter_gradients_preserve_shapes(self):
        main = [torch.randn(2, 5).transpose(0, 1)]
        cka = [torch.randn(2, 5).transpose(0, 1)]
        final, ema, _ = protect(main, cka)
        self.assertEqual(final[0].shape, (5, 2))
        self.assertEqual(ema[0].shape, (5, 2))
        self.assertAlmostEqual(float(((final[0] - main[0]) * ema[0]).sum()), 0., places=5)

    def test_nonfinite_gradients_and_overflow_fail_before_returning_updates(self):
        for location in ("main", "cka", "ema"):
            for bad in (float("nan"), float("inf"), 1e30):
                values = {"main": [torch.ones(2)], "cka": [torch.ones(2)], "ema": [torch.ones(2)]}
                values[location][0][0] = bad
                with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
                    protect(values["main"], values["cka"], values["ema"])
        with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
            group_gradient_norms([torch.tensor([float("nan")])], ["llm"])

    def test_invalid_configuration_and_gradient_layout_are_rejected(self):
        main, cka = [torch.ones(2)], [torch.ones(2)]
        for options in (
            {"ema_decay": -0.1}, {"ema_decay": 1.1}, {"lambda_t": -1.},
            {"eps": 0.}, {"eps": float("nan")}, {"stat_chunk_size": 0},
            {"stat_chunk_size": True}, {"group_names": []}, {"group_names": [None]},
            {"allowed_cka_mask": []}, {"allowed_cka_mask": [1]},
            {"ce_grad_refs": {}}, {"aux_ratios": {"projector": float("inf")}},
            {"ce_grad_refs": {"projector": -1.}},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                protect(main, cka, **options)
        with self.assertRaisesRegex(ValueError, "lengths"):
            protect(main, [])
        with self.assertRaisesRegex(ValueError, "shapes"):
            protect(main, [torch.ones(3)])
        with self.assertRaisesRegex(ValueError, "floating"):
            protect([torch.ones(2, dtype=torch.long)], cka)


if __name__ == "__main__":
    unittest.main()
