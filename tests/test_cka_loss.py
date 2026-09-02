import math
import unittest
from types import SimpleNamespace

import torch

from llava.model.llava_arch import (
    cka_similarity_to_loss,
    compute_linear_cka_loss,
    parse_cka_channel_drop_indices,
    select_cka_feature_channels,
    validate_cka_channel_keep_ratio,
    validate_cka_eps,
)
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


def _reference_masked_cka_loss(x, y, mask, eps=1e-8):
    losses = []
    for batch_idx in range(x.shape[0]):
        keep = mask[batch_idx].bool()
        if int(keep.sum()) < 2:
            continue

        x_i = x[batch_idx, keep].float()
        y_i = y[batch_idx, keep].float()
        x_i = x_i - x_i.mean(dim=0, keepdim=True)
        y_i = y_i - y_i.mean(dim=0, keepdim=True)
        tiny = torch.finfo(x_i.dtype).tiny
        x_i = x_i / torch.linalg.vector_norm(x_i).clamp_min(tiny)
        y_i = y_i / torch.linalg.vector_norm(y_i).clamp_min(tiny)
        xx = x_i @ x_i.T
        yy = y_i @ y_i.T
        hsic_xy = (xx * yy).sum()
        hsic_xx = xx.square().sum()
        hsic_yy = yy.square().sum()
        denom = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=eps))
        cka = (hsic_xy / denom).clamp(0.0, 1.0)
        losses.append(1.0 - cka)

    if not losses:
        return x.float().sum() * 0.0
    return torch.stack(losses).mean()


class _DummyCkaModel:
    def __init__(self):
        self.model = SimpleNamespace(config=SimpleNamespace())

    def get_model(self):
        return self.model


class CkaLossTests(unittest.TestCase):

    def test_channel_keep_ratio_validation(self):
        for valid in (1.0, 0.5, 0.01, "0.25"):
            with self.subTest(valid=valid):
                self.assertEqual(
                    validate_cka_channel_keep_ratio(valid),
                    float(valid),
                )

        for invalid in (0.0, -0.1, 1.1, math.nan, math.inf, -math.inf, "bad", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_cka_channel_keep_ratio(invalid)
                with self.assertRaises(ValueError):
                    compute_linear_cka_loss(
                        torch.randn(1, 4, 6),
                        torch.randn(1, 4, 8),
                        channel_keep_ratio=invalid,
                    )

    def test_explicit_channel_drop_indices_validation_and_selection(self):
        self.assertEqual(parse_cka_channel_drop_indices(None), ())
        self.assertEqual(parse_cka_channel_drop_indices(""), ())
        self.assertEqual(parse_cka_channel_drop_indices("3, 1, 3"), (1, 3))
        self.assertEqual(parse_cka_channel_drop_indices([5, 2]), (2, 5))
        self.assertEqual(parse_cka_channel_drop_indices(4), (4,))

        for invalid in ("-1", "1,,2", "1.5", "bad", [True], [-1], [1.0]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_cka_channel_drop_indices(invalid)

        features = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6)
        features.requires_grad_(True)
        selected = select_cka_feature_channels(
            features,
            drop_indices="1,4,4",
        )
        self.assertTrue(torch.equal(
            selected.detach(),
            features.detach()[..., [0, 2, 3, 5]],
        ))
        selected.sum().backward()
        self.assertEqual(features.grad[..., [1, 4]].abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(
            features.grad[..., [0, 2, 3, 5]],
            torch.ones_like(features.grad[..., [0, 2, 3, 5]]),
        ))

        no_op_features = features.detach()
        self.assertIs(
            select_cka_feature_channels(no_op_features, drop_indices=""),
            no_op_features,
        )
        with self.assertRaisesRegex(ValueError, "outside the feature width"):
            select_cka_feature_channels(features, drop_indices="6")
        with self.assertRaisesRegex(ValueError, "remove all"):
            select_cka_feature_channels(features, drop_indices="0,1,2,3,4,5")

    def test_channel_selection_is_fixed_and_does_not_advance_global_rng(self):
        features = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16)
        rng_state = torch.random.get_rng_state().clone()
        selected = select_cka_feature_channels(
            features,
            keep_ratio=0.5,
            seed=123,
            salt=7,
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        repeated = select_cka_feature_channels(
            features,
            keep_ratio=0.5,
            seed=123,
            salt=7,
        )
        other_salt = select_cka_feature_channels(
            features,
            keep_ratio=0.5,
            seed=123,
            salt=8,
        )

        self.assertEqual(selected.shape, (2, 3, 8))
        self.assertTrue(torch.equal(selected, repeated))
        self.assertFalse(torch.equal(selected, other_salt))
        self.assertIs(
            select_cka_feature_channels(features, keep_ratio=1.0, seed=999),
            features,
        )

    def test_channel_selection_zeroes_dropped_channel_gradients(self):
        torch.manual_seed(31)
        features = torch.randn(2, 5, 12, requires_grad=True)
        selected = select_cka_feature_channels(
            features,
            keep_ratio=0.5,
            seed=17,
            salt=3,
        )
        marker = torch.arange(12).reshape(1, 1, 12)
        kept = select_cka_feature_channels(
            marker,
            keep_ratio=0.5,
            seed=17,
            salt=3,
        ).flatten()
        dropped_mask = torch.ones(12, dtype=torch.bool)
        dropped_mask[kept] = False

        selected.square().sum().backward()
        self.assertEqual(features.grad[..., dropped_mask].abs().sum().item(), 0.0)
        self.assertGreater(features.grad[..., kept].abs().sum().item(), 0.0)

    def test_channel_filtered_cka_supports_unequal_widths_and_backward(self):
        torch.manual_seed(37)
        x = torch.randn(2, 8, 12, requires_grad=True)
        y = torch.randn(2, 8, 20, requires_grad=True)
        loss = compute_linear_cka_loss(
            x,
            y,
            channel_keep_ratio=0.5,
            channel_seed=23,
            x_channel_drop_indices="1,11",
            y_channel_drop_indices="1,11,19",
            share_channel_indices=False,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(y.grad).all())
        x_kept = select_cka_feature_channels(
            torch.arange(12).reshape(1, 1, 12),
            keep_ratio=0.5,
            seed=23,
            salt=0,
            drop_indices="1,11",
        ).flatten()
        y_kept = select_cka_feature_channels(
            torch.arange(20).reshape(1, 1, 20),
            keep_ratio=0.5,
            seed=23,
            salt=1,
            drop_indices="1,11,19",
        ).flatten()
        x_dropped = torch.ones(12, dtype=torch.bool)
        y_dropped = torch.ones(20, dtype=torch.bool)
        x_dropped[x_kept] = False
        y_dropped[y_kept] = False
        self.assertEqual(x.grad[..., x_dropped].abs().sum().item(), 0.0)
        self.assertEqual(y.grad[..., y_dropped].abs().sum().item(), 0.0)
        self.assertGreater(x.grad[..., x_kept].abs().sum().item(), 0.0)
        self.assertGreater(y.grad[..., y_kept].abs().sum().item(), 0.0)

    def test_channel_filter_default_and_shared_identity_are_backward_compatible(self):
        torch.manual_seed(41)
        x = torch.randn(2, 9, 10)
        y = torch.randn(2, 9, 14)
        default = compute_linear_cka_loss(x, y)
        explicit_default = compute_linear_cka_loss(
            x,
            y,
            channel_keep_ratio=1.0,
            channel_seed=999,
            share_channel_indices=False,
        )
        self.assertTrue(torch.equal(default, explicit_default))

        identity = compute_linear_cka_loss(
            x,
            x.clone(),
            channel_keep_ratio=0.4,
            channel_seed=29,
            x_channel_drop_indices="1,3",
            y_channel_drop_indices="1,3",
            share_channel_indices=True,
        )
        self.assertEqual(identity.item(), 0.0)

        x_2d = torch.randn(3, 12)
        y_2d = torch.randn(3, 12)
        self.assertTrue(torch.equal(
            compute_linear_cka_loss(x_2d, y_2d),
            compute_linear_cka_loss(
                x_2d,
                y_2d,
                channel_keep_ratio=0.5,
                channel_seed=29,
            ),
        ))

    def test_masked_channel_filter_drops_only_hidden_target(self):
        torch.manual_seed(43)
        mask = torch.tensor([
            [True, False, True, True, False, True],
            [False, True, False, False, True, False],
            [True, False, False, False, False, False],
        ])
        actual_x = torch.randn(3, 6, 8, requires_grad=True)
        actual_y = torch.randn(3, 6, 12, requires_grad=True)
        ref_x = actual_x.detach().clone().requires_grad_(True)
        ref_y = actual_y.detach().clone().requires_grad_(True)
        dummy = _DummyCkaModel()
        dummy.model.config.cka_loss_channel_keep_ratio = 0.5
        dummy.model.config.cka_loss_channel_seed = 31
        dummy.model.config.cka_loss_hidden_channel_drop_indices = [1, 7]

        actual = LlavaLlamaForCausalLM._compute_masked_linear_cka_loss(
            dummy,
            actual_x,
            actual_y,
            mask,
        )
        selected_ref_x = select_cka_feature_channels(
            ref_x,
            keep_ratio=0.5,
            seed=31,
            salt=2,
            drop_indices=[1, 7],
        )
        selected_ref_y = select_cka_feature_channels(
            ref_y,
            keep_ratio=0.5,
            seed=31,
            salt=3,
            drop_indices=None,
        )
        reference = _reference_masked_cka_loss(
            selected_ref_x,
            selected_ref_y,
            mask,
        )
        actual.backward()
        reference.backward()

        self.assertTrue(torch.allclose(actual, reference, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(actual_x.grad, ref_x.grad, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(actual_y.grad, ref_y.grad, atol=2e-6, rtol=2e-6))

    def test_loss_is_applied_per_sample_before_mean(self):
        cka = torch.tensor([1.0, 0.0], requires_grad=True)
        loss = cka_similarity_to_loss(cka).mean()
        self.assertAlmostEqual(loss.item(), 0.5, places=7)

    def test_loss_gradients_are_linear(self):
        cka = torch.tensor([0.70, 0.75, 0.80], requires_grad=True)
        loss = cka_similarity_to_loss(cka).sum()
        loss.backward()
        self.assertTrue(torch.allclose(loss.detach(), torch.tensor(0.75), atol=1e-7))
        self.assertTrue(torch.equal(cka.grad, torch.tensor([-1.0, -1.0, -1.0])))

    def test_eps_validation(self):
        self.assertEqual(validate_cka_eps(1e-8), 1e-8)
        for invalid in (0.0, -1e-8, math.nan, math.inf, -math.inf, "bad", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_cka_eps(invalid)
                with self.assertRaises(ValueError):
                    compute_linear_cka_loss(
                        torch.randn(1, 3, 2),
                        torch.randn(1, 3, 4),
                        eps=invalid,
                    )

    def test_known_orthogonal_and_identical_samples(self):
        a = torch.tensor([1.0, -1.0, 0.0, 0.0]).view(1, 4, 1)
        b = torch.tensor([0.0, 0.0, 1.0, -1.0]).view(1, 4, 1)
        x = torch.cat((a, a), dim=0)
        y = torch.cat((a, b), dim=0)
        self.assertAlmostEqual(
            compute_linear_cka_loss(x, y).item(),
            0.5,
            places=6,
        )

    def test_scale_invariance_for_small_nonzero_features(self):
        torch.manual_seed(9)
        base = torch.randn(2, 8, 5)
        for scale in (1.0, 1e-2, 1e-6, 1e-10):
            with self.subTest(scale=scale):
                x = base * scale
                loss = compute_linear_cka_loss(x, x.clone())
                self.assertEqual(loss.item(), 0.0)

    def test_active_and_inactive_projector_gradients(self):
        torch.manual_seed(10)
        x = torch.randn(2, 8, 5)
        active_y = torch.randn(2, 8, 7, requires_grad=True)
        active_loss = compute_linear_cka_loss(x, active_y)
        active_loss.backward()
        self.assertGreater(active_y.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(active_y.grad).all())

        identical_y = x.detach().clone().requires_grad_(True)
        identical_loss = compute_linear_cka_loss(x, identical_y)
        identical_loss.backward()
        self.assertEqual(identical_loss.item(), 0.0)

    def test_masked_vectorized_value_and_grad_match_reference(self):
        torch.manual_seed(11)
        mask = torch.tensor(
            [
                [True, False, True, True, False, True],
                [False, True, False, False, True, False],
                [True, False, False, False, False, False],
            ]
        )
        actual_x = torch.randn(3, 6, 4, requires_grad=True)
        actual_y = torch.randn(3, 6, 7, requires_grad=True)
        ref_x = actual_x.detach().clone().requires_grad_(True)
        ref_y = actual_y.detach().clone().requires_grad_(True)
        dummy = _DummyCkaModel()

        actual = LlavaLlamaForCausalLM._compute_masked_linear_cka_loss(
            dummy, actual_x, actual_y, mask
        )
        reference = _reference_masked_cka_loss(ref_x, ref_y, mask)
        actual.backward()
        reference.backward()

        self.assertTrue(torch.allclose(actual, reference, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(actual_x.grad, ref_x.grad, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(actual_y.grad, ref_y.grad, atol=2e-6, rtol=2e-6))

    def test_masked_no_valid_sample_returns_connected_zero(self):
        x = torch.randn(2, 4, 3, requires_grad=True)
        y = torch.randn(2, 4, 5)
        mask = torch.tensor(
            [[True, False, False, False], [False, False, False, False]]
        )
        loss = LlavaLlamaForCausalLM._compute_masked_linear_cka_loss(
            _DummyCkaModel(), x, y, mask
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.abs().sum().item(), 0.0)

    def test_masked_cpu_autocast_matches_fp32(self):
        torch.manual_seed(12)
        x = torch.randn(2, 24, 12).to(torch.bfloat16)
        y = torch.randn(2, 24, 18).to(torch.bfloat16)
        mask = torch.ones(2, 24, dtype=torch.bool)
        dummy = _DummyCkaModel()
        reference = LlavaLlamaForCausalLM._compute_masked_linear_cka_loss(
            dummy, x.float(), y.float(), mask
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = LlavaLlamaForCausalLM._compute_masked_linear_cka_loss(
                dummy, x, y, mask
            )
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual, reference, atol=1e-6, rtol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_masked_cuda_fp16_autocast_value_and_grad_match_fp32(self):
        torch.manual_seed(14)
        mask = torch.tensor(
            [
                [True, False, True, True, True, False] * 8,
                [False, True, True, False, True, True] * 8,
            ],
            device="cuda",
        )
        actual_x = torch.randn(2, 48, 20, device="cuda").half().requires_grad_(True)
        actual_y = torch.randn(2, 48, 28, device="cuda").half().requires_grad_(True)
        ref_x = actual_x.detach().float().requires_grad_(True)
        ref_y = actual_y.detach().float().requires_grad_(True)
        dummy = _DummyCkaModel()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            actual = LlavaLlamaForCausalLM._compute_masked_linear_cka_loss(
                dummy, actual_x, actual_y, mask
            )
        reference = _reference_masked_cka_loss(ref_x, ref_y, mask)
        actual.backward()
        reference.backward()

        self.assertTrue(torch.allclose(actual, reference, atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(actual_x.grad.float(), ref_x.grad, atol=2e-3, rtol=2e-3)
        )
        self.assertTrue(
            torch.allclose(actual_y.grad.float(), ref_y.grad, atol=2e-3, rtol=2e-3)
        )

    def test_autocast_does_not_downcast_cka_reductions(self):
        torch.manual_seed(13)
        x = torch.randn(2, 32, 24).to(torch.bfloat16)
        y = torch.randn(2, 32, 40).to(torch.bfloat16)
        reference = compute_linear_cka_loss(x.float(), y.float())
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = compute_linear_cka_loss(x, y)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual, reference, atol=1e-6, rtol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_fp16_autocast_matches_fp32(self):
        torch.manual_seed(17)
        x = (100.0 * torch.randn(2, 128, 64, device="cuda")).half()
        y = (100.0 * torch.randn(2, 128, 96, device="cuda")).half()
        reference = compute_linear_cka_loss(x.float(), y.float())
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            actual = compute_linear_cka_loss(x, y)
        self.assertTrue(torch.isfinite(actual))
        self.assertTrue(torch.allclose(actual, reference, atol=1e-6, rtol=1e-6))

    def test_shape_validation_and_empty_batch(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            compute_linear_cka_loss(torch.randn(2, 3, 4), torch.randn(2, 12))
        with self.assertRaisesRegex(ValueError, "Batch size"):
            compute_linear_cka_loss(torch.randn(2, 3, 4), torch.randn(3, 3, 5))
        with self.assertRaisesRegex(ValueError, "Token length"):
            compute_linear_cka_loss(torch.randn(2, 3, 4), torch.randn(2, 4, 5))
        with self.assertRaisesRegex(ValueError, "Feature length"):
            compute_linear_cka_loss(torch.randn(2, 7), torch.randn(2, 8))

        x = torch.empty(0, 3, 4, requires_grad=True)
        y = torch.empty(0, 3, 5, requires_grad=True)
        loss = compute_linear_cka_loss(x, y)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)


if __name__ == "__main__":
    unittest.main()
