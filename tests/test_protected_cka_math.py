"""Offline CPU tests for protected Stage 1 KL, CKA, and gradient surgery."""

import unittest

import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX
from llava.train.protected_cka_math import (
    protected_projector_gradient,
    target_margin_cka,
    token_kl,
)


def naive_token_kl(student, reference, labels):
    log_s = F.log_softmax(student[:, :-1].float(), dim=-1)
    log_r = F.log_softmax(reference.detach()[:, :-1].float(), dim=-1)
    per_token = (log_r.exp() * (log_r - log_s)).sum(dim=-1)
    return per_token[labels[:, 1:] != IGNORE_INDEX].mean()


class ProtectedTokenKLTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)

    def test_exact_full_vocabulary_loss_and_gradient_for_all_chunk_sizes(self):
        values = torch.randn(2, 8, 31)
        reference = torch.randn_like(values, requires_grad=True)
        labels = torch.tensor([
            [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 2, 3, 7, IGNORE_INDEX, 9],
            [IGNORE_INDEX, IGNORE_INDEX, 5, 4, IGNORE_INDEX, 8, 9, IGNORE_INDEX],
        ])
        naive_student = values.clone().requires_grad_()
        expected_loss = naive_token_kl(naive_student, reference, labels)
        expected_gradient, = torch.autograd.grad(2.7 * expected_loss, naive_student)
        for chunk_size in (1, 3, 128):
            with self.subTest(chunk_size=chunk_size):
                student = values.clone().requires_grad_()
                loss, count = token_kl(student, reference, labels, chunk_size)
                self.assertEqual(count, 8)
                self.assertEqual(loss.dtype, torch.float32)
                torch.testing.assert_close(loss, expected_loss)
                (2.7 * loss).backward()
                torch.testing.assert_close(student.grad, expected_gradient)
                self.assertIsNone(reference.grad)

    def test_mask_is_expanded_and_causally_shifted(self):
        student = torch.zeros(1, 6, 2, requires_grad=True)
        reference = torch.tensor([[[8., -8.], [7., -7.], [2., -2.], [6., -6.], [-2., 2.], [9., -9.]]])
        # Expanded image positions and ignored instructions/padding contribute
        # nothing. Labels 3 and 5 supervise logits 2 and 4, not logits 3 and 5.
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 1, IGNORE_INDEX, 0]])
        loss, count = token_kl(student, reference, labels, chunk_size=1)
        self.assertEqual(count, 2)
        loss.backward()
        self.assertGreater(float(student.grad[0, 2].abs().sum()), 0.)
        self.assertGreater(float(student.grad[0, 4].abs().sum()), 0.)
        torch.testing.assert_close(student.grad[0, [0, 1, 3, 5]], torch.zeros(4, 2))

    def test_identical_logits_have_zero_loss_and_gradient(self):
        student = torch.randn(2, 5, 7, requires_grad=True)
        reference = student.detach().clone().requires_grad_()
        loss, count = token_kl(student, reference, torch.zeros(2, 5, dtype=torch.long), chunk_size=3)
        self.assertEqual(count, 8)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))
        self.assertIsNone(reference.grad)

    def test_no_supervision_is_finite_differentiable_zero(self):
        for length in (0, 1, 5):
            student = torch.full((2, length, 7), float("nan"), requires_grad=True)
            reference = torch.full_like(student, float("inf"), requires_grad=True)
            loss, count = token_kl(student, reference, torch.full((2, length), IGNORE_INDEX))
            self.assertEqual(count, 0)
            self.assertEqual(float(loss), 0.)
            loss.backward()
            torch.testing.assert_close(student.grad, torch.zeros_like(student))
            self.assertIsNone(reference.grad)

    def test_chunking_saves_logits_not_fp32_probability_buffers(self):
        student = torch.randn(2, 8, 17, dtype=torch.bfloat16, requires_grad=True)
        reference = torch.randn_like(student)
        saved = []

        def save(tensor):
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
            loss, _ = token_kl(student, reference, torch.zeros(2, 8, dtype=torch.long), chunk_size=2)
        self.assertEqual(len(saved), 3)
        self.assertEqual(saved[0].data_ptr(), student.data_ptr())
        self.assertEqual(saved[1].data_ptr(), reference.data_ptr())
        self.assertEqual(saved[0].dtype, torch.bfloat16)
        self.assertEqual(saved[1].dtype, torch.bfloat16)
        loss.backward()
        expected_student = student.detach().clone().requires_grad_()
        expected_loss = naive_token_kl(expected_student, reference, torch.zeros(2, 8, dtype=torch.long))
        expected_loss.backward()
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(student.grad, expected_student.grad)

    def test_rejects_unexpanded_labels_and_invalid_chunks(self):
        logits = torch.zeros(1, 4, 3)
        with self.assertRaisesRegex(ValueError, "expanded_labels"):
            token_kl(logits, logits, torch.zeros(1, 2, dtype=torch.long))
        for invalid_chunk in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                token_kl(logits, logits, torch.zeros(1, 4, dtype=torch.long), invalid_chunk)


class ProtectedTargetCKATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)

    def test_target_reached_has_zero_loss(self):
        vision = torch.randn(2, 7, 4)
        student = vision.clone().requires_grad_()
        reference = torch.randn(2, 7, 6, requires_grad=True)
        loss, stats = target_margin_cka(vision, student, reference, margin=0.02)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(stats["target_met"], 1.)
        self.assertEqual(stats["valid_images"], 2)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))
        self.assertIsNone(reference.grad)

    def test_reference_is_detached_and_target_margin_matches_formula(self):
        vision = torch.randn(2, 9, 5, requires_grad=True)
        student = torch.randn(2, 9, 7, requires_grad=True)
        reference = student.detach().clone().requires_grad_()
        loss, stats = target_margin_cka(vision, student, reference, margin=0.02)
        self.assertAlmostEqual(float(loss), 0.02 ** 2, places=7)
        self.assertAlmostEqual(stats["cka_ref"], stats["cka_current"], places=7)
        self.assertEqual(stats["target_met"], 0.)
        loss.backward()
        self.assertGreater(float(student.grad.norm()), 0.)
        self.assertIsNone(vision.grad)
        self.assertIsNone(reference.grad)

    def test_centering_is_per_image_not_across_batch(self):
        vision = torch.randn(3, 7, 5)
        student = torch.randn(3, 7, 4, requires_grad=True)
        reference = student.detach().clone()
        batch_loss, batch_stats = target_margin_cka(vision, student, reference)
        individuals = [target_margin_cka(v[None], s[None], r[None])[0] for v, s, r in zip(vision, student, reference)]
        torch.testing.assert_close(batch_loss, torch.stack(individuals).mean())
        offsets_v = torch.randn(3, 1, 5) * 5.
        offsets_s = torch.randn(3, 1, 4) * 5.
        offsets_r = torch.randn(3, 1, 4) * 5.
        translated_loss, translated_stats = target_margin_cka(
            vision + offsets_v, student + offsets_s, reference + offsets_r,
        )
        torch.testing.assert_close(batch_loss, translated_loss)
        self.assertAlmostEqual(batch_stats["cka_current"], translated_stats["cka_current"], places=6)
        self.assertAlmostEqual(batch_stats["cka_ref"], translated_stats["cka_ref"], places=6)

    def test_degenerate_and_nonfinite_images_are_excluded_from_mean(self):
        vision = torch.randn(6, 8, 4)
        student = torch.randn(6, 8, 5)
        reference = student.clone()
        vision[1].fill_(3.)
        student[2].fill_(0.)
        reference[3].fill_(2.)
        student[4, 0, 0] = float("nan")
        reference[5, 0, 0] = float("inf")
        student.requires_grad_()
        loss, stats = target_margin_cka(vision, student, reference)
        expected, _ = target_margin_cka(vision[:1], student[:1], reference[:1])
        torch.testing.assert_close(loss, expected)
        self.assertEqual(stats["valid_images"], 1)
        self.assertEqual(stats["invalid_images"], 5)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(student.grad).all()))
        torch.testing.assert_close(student.grad[1:], torch.zeros_like(student.grad[1:]))

    def test_all_degenerate_including_overflow_returns_safe_zero(self):
        vision = torch.zeros(4, 6, 3)
        student = torch.zeros(4, 6, 4)
        reference = torch.zeros_like(student)
        student[1].fill_(float("nan"))
        student[2].fill_(float("inf"))
        # Finite features whose Gram/norm calculation overflows are invalid too.
        vision[3] = torch.randn(6, 3) * 1e20
        student[3] = torch.randn(6, 4) * 1e20
        reference[3] = torch.randn(6, 4) * 1e20
        student.requires_grad_()
        loss, stats = target_margin_cka(vision, student, reference)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(stats["invalid_images"], 4)
        self.assertEqual(stats["valid_images"], 0)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_cka_stays_fp32_under_bfloat16_autocast(self):
        vision = torch.randn(2, 8, 4)
        student = torch.randn(2, 8, 5, requires_grad=True)
        reference = student.detach().clone()
        expected, expected_stats = target_margin_cka(vision, student, reference)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual, actual_stats = target_margin_cka(vision, student, reference)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_stats, expected_stats)

    def test_near_zero_gram_is_excluded(self):
        vision = torch.randn(1, 6, 4)
        student = (torch.randn(1, 6, 4) * 1e-8).requires_grad_()
        loss, stats = target_margin_cka(vision, student, vision)
        self.assertEqual(stats["valid_images"], 0)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))


class ProtectedProjectorGradientTest(unittest.TestCase):
    def test_whole_vector_projection_removes_both_dot_signs(self):
        main = torch.tensor([1., 10.])  # Think of these as two separate layers.
        for first in (-2., 2.):
            cka = torch.tensor([first, 0.])
            main_before, cka_before = main.clone(), cka.clone()
            final, ema, stats = protected_projector_gradient(
                main, cka, ce_grad_ref=100., aux_ratio=1., ema_decay=0.5,
            )
            expected_aux = cka - torch.dot(cka, ema) / torch.dot(ema, ema) * ema
            torch.testing.assert_close(ema, main * 0.5)
            torch.testing.assert_close(final, main + expected_aux)
            self.assertAlmostEqual(float(torch.dot(final - main, ema)), 0., places=5)
            self.assertGreater(float((final - main).norm()), 0.)
            torch.testing.assert_close(main, main_before, rtol=0, atol=0)
            torch.testing.assert_close(cka, cka_before, rtol=0, atol=0)
            self.assertEqual(stats["scale"], 1.)

    def test_projection_uses_updated_ema_not_current_gradient(self):
        main = torch.tensor([2., 0., 1.])
        previous = torch.tensor([0., 4., 1.])
        previous_before = previous.clone()
        cka = torch.tensor([5., -3., 2.])
        final, ema, _ = protected_projector_gradient(
            main, cka, previous, ema_decay=0.75, ce_grad_ref=100., aux_ratio=1.,
        )
        expected_ema = 0.75 * previous + 0.25 * main
        torch.testing.assert_close(ema, expected_ema)
        self.assertAlmostEqual(float(torch.dot(final - main, ema)), 0., places=5)
        self.assertGreater(abs(float(torch.dot(final - main, main))), 0.1)
        torch.testing.assert_close(previous, previous_before)

    def test_cap_uses_fixed_reference_not_instantaneous_main_norm(self):
        updates = []
        for main_magnitude in (0.1, 100.):
            main = torch.tensor([main_magnitude, 0.])
            final, _, stats = protected_projector_gradient(
                main, torch.tensor([4., 10.]), ce_grad_ref=2., aux_ratio=0.1,
                ema_decay=0.0, lambda_t=1.,
            )
            updates.append(final - main)
            self.assertLessEqual(stats["scaled_aux_norm"], 0.2 + 1e-7)
            self.assertAlmostEqual(stats["scaled_aux_norm"], 0.2, places=6)
            self.assertEqual(stats["aux_cap"], 0.2)
        torch.testing.assert_close(*updates)

    def test_warmup_scale_and_disabled_auxiliary_preserve_main(self):
        main = torch.tensor([1., 0.])
        cka = torch.tensor([2., 4.])
        final, _, stats = protected_projector_gradient(
            main, cka, ce_grad_ref=100., aux_ratio=1., lambda_t=0.25, ema_decay=0.,
        )
        torch.testing.assert_close(final, torch.tensor([1., 1.]))
        self.assertEqual(stats["scale"], 0.25)
        for overrides in ({"lambda_t": 0.}, {"aux_ratio": 0.}, {"ce_grad_ref": 0.}):
            kwargs = {"ce_grad_ref": 100., "aux_ratio": 1., "lambda_t": 1.}
            kwargs.update(overrides)
            final, _, _ = protected_projector_gradient(main, cka, **kwargs)
            torch.testing.assert_close(final, main, rtol=0, atol=0)

    def test_small_ema_and_zero_auxiliary_are_safe(self):
        for main, cka in (
            (torch.zeros(3), torch.ones(3)),
            (torch.tensor([1e-10, 0., 0.]), torch.ones(3)),
            (torch.tensor([1., 0., 0.]), torch.zeros(3)),
            (torch.tensor([1., 0., 0.]), torch.tensor([-4., 0., 0.])),
        ):
            final, _, stats = protected_projector_gradient(main, cka, ce_grad_ref=1., ema_decay=0.)
            torch.testing.assert_close(final, main, rtol=0, atol=0)
            self.assertEqual(stats["scaled_aux_norm"], 0.)
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))

    def test_bfloat16_input_returns_fp32_without_mutation(self):
        main = torch.tensor([1., 0.], dtype=torch.bfloat16)
        cka = torch.tensor([2., 3.], dtype=torch.bfloat16)
        final, ema, _ = protected_projector_gradient(main, cka, ce_grad_ref=100.)
        self.assertEqual(final.dtype, torch.float32)
        self.assertEqual(ema.dtype, torch.float32)
        self.assertEqual(main.dtype, torch.bfloat16)
        torch.testing.assert_close(final, torch.tensor([1., 3.]))

    def test_nonfinite_gradients_or_configuration_fail_before_update(self):
        main, cka = torch.ones(2), torch.zeros(2)
        for name in ("ema_decay", "aux_ratio", "ce_grad_ref", "lambda_t", "eps"):
            kwargs = {"ce_grad_ref": 1., name: float("nan")}
            with self.assertRaisesRegex(ValueError, "finite"):
                protected_projector_gradient(main, cka, **kwargs)
        for kwargs in ({"ema_decay": 1.01}, {"aux_ratio": -1.}, {"lambda_t": -1.}, {"eps": 0.}):
            with self.assertRaises(ValueError):
                protected_projector_gradient(main, cka, ce_grad_ref=1., **kwargs)
        for which in ("main", "cka", "ema"):
            bad = torch.tensor([float("inf"), 0.])
            with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
                protected_projector_gradient(
                    bad if which == "main" else main,
                    bad if which == "cka" else cka,
                    bad if which == "ema" else None,
                    ce_grad_ref=1.,
                )


if __name__ == "__main__":
    unittest.main()
