"""Loss-kernel tests; calibration selection and training integration are separate."""

import unittest

import torch

from llava.model.selected_state_cka import compute_selected_cka_loss


def direct_cka_loss(x, y):
    """The requested FP32, token-centered Gram formula for one sample."""
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    gx, gy = x @ x.T, y @ y.T
    return 1 - (gx * gy).sum() / (
        torch.linalg.matrix_norm(gx) * torch.linalg.matrix_norm(gy)
    )


class SelectedStateCKALossTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(19)

    def features(self, *shape, requires_grad=False):
        return torch.randn(
            *shape, generator=self.generator, requires_grad=requires_grad,
        )

    def test_gather_then_center_and_average_per_sample(self):
        vision = self.features(2, 8, 3)
        target = self.features(2, 8, 5)
        indices = [torch.tensor([6, 1, 4, 0]), torch.tensor([2, 7, 1, 5])]
        # Large differences between sample means expose a flattened-batch loss.
        vision[1] += 50
        target[1] -= 30
        result = compute_selected_cka_loss(vision, target, indices)
        expected = torch.stack([
            direct_cka_loss(vision[i, subset], target[i, subset])
            for i, subset in enumerate(indices)
        ]).mean()
        torch.testing.assert_close(result.loss, expected)
        self.assertEqual(result.valid_count, 2)
        self.assertEqual(result.invalid_count, 0)
        self.assertEqual(result.invalid_reasons, {})
        self.assertEqual(result.subset_sizes, (4, 4))

        gathered_x = torch.cat([vision[i, subset] for i, subset in enumerate(indices)])
        gathered_y = torch.cat([target[i, subset] for i, subset in enumerate(indices)])
        flattened_loss = direct_cka_loss(gathered_x, gathered_y)
        self.assertGreater(abs(float(result.loss - flattened_loss)), 0.05)

        # Changing excluded tokens must not change the centering or loss.
        changed_vision, changed_target = vision.clone(), target.clone()
        for i, subset in enumerate(indices):
            excluded = torch.ones(8, dtype=torch.bool)
            excluded[subset] = False
            changed_vision[i, excluded] = 1e6
            changed_target[i, excluded] = -1e6
        changed = compute_selected_cka_loss(changed_vision, changed_target, indices)
        torch.testing.assert_close(changed.loss, result.loss, rtol=0, atol=0)

    def test_attention_and_coverage_are_one_ordered_union(self):
        vision = self.features(1, 9, 4)
        target = self.features(1, 9, 6)
        attention = torch.tensor([7, 2, 5, 0])
        coverage = torch.tensor([8, 3])
        union = torch.cat((attention, coverage))
        result = compute_selected_cka_loss(vision, target, [union])
        torch.testing.assert_close(
            result.loss, direct_cka_loss(vision[0, union], target[0, union]),
        )
        separate_loss = (
            direct_cka_loss(vision[0, attention], target[0, attention])
            + direct_cka_loss(vision[0, coverage], target[0, coverage])
        ) / 2
        self.assertGreater(abs(float(result.loss - separate_loss)), 0.01)
        self.assertEqual(result.subset_sizes, (6,))

        # A shared permutation preserves pairings; changing only one side does not.
        shuffled = compute_selected_cka_loss(vision, target, [union.flip(0)])
        torch.testing.assert_close(shuffled.loss, result.loss)
        mismatched = direct_cka_loss(vision[0, union], target[0, union.flip(0)])
        self.assertGreater(abs(float(result.loss - mismatched)), 0.001)

    def test_only_selected_target_tokens_and_input_channels_receive_gradients(self):
        vision = self.features(2, 7, 3, requires_grad=True)
        target_storage = self.features(2, 7, 10, requires_grad=True)
        target = target_storage[:, :, ::2]
        indices = [torch.tensor([5, 1, 3, 0]), None]
        result = compute_selected_cka_loss(vision, target, indices)
        result.loss.backward()

        self.assertIsNone(vision.grad)
        self.assertTrue(torch.isfinite(target_storage.grad).all())
        self.assertGreater(float(target_storage.grad[0, indices[0], ::2].norm()), 0)
        self.assertEqual(int(torch.count_nonzero(target_storage.grad[:, :, 1::2])), 0)
        self.assertEqual(int(torch.count_nonzero(target_storage.grad[0, [2, 4, 6]])), 0)
        self.assertEqual(int(torch.count_nonzero(target_storage.grad[1])), 0)

        reference = target.detach().clone().requires_grad_(True)
        direct_cka_loss(vision[0, indices[0]].detach(), reference[0, indices[0]]).backward()
        torch.testing.assert_close(target_storage.grad[:, :, ::2], reference.grad)

    def test_invalid_samples_are_counted_and_excluded_from_the_mean(self):
        vision = self.features(9, 6, 3)
        target = self.features(9, 6, 5)
        vision[2, 1, 0] = float("nan")
        target[3, 1, 0] = float("inf")
        vision[4] = 3
        target[5] = -2
        vision[8] *= 1e20
        indices = [torch.tensor([0, 1, 3, 5]) for _ in range(9)]
        indices[1] = None
        indices[6] = torch.tensor([2])
        indices[7] = torch.empty(0, dtype=torch.long)

        result = compute_selected_cka_loss(vision, target, indices)
        torch.testing.assert_close(
            result.loss, direct_cka_loss(vision[0, indices[0]], target[0, indices[0]]),
        )
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(result.invalid_count, 8)
        self.assertEqual(result.subset_sizes, (4, 0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(result.invalid_reasons, {
            "missing_selection": 1,
            "nonfinite_features": 2,
            "degenerate_denominator": 2,
            "too_few_tokens": 2,
            "nonfinite_denominator": 1,
        })

    def test_nonfinite_values_outside_the_subset_do_not_invalidate_sample(self):
        vision = self.features(1, 6, 3)
        target = self.features(1, 6, 5, requires_grad=True)
        with torch.no_grad():
            vision[0, 4] = float("nan")
            target[0, 5] = float("nan")
        result = compute_selected_cka_loss(vision, target, [torch.tensor([0, 2, 3])])
        self.assertEqual(result.valid_count, 1)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        self.assertTrue(torch.isfinite(target.grad).all())
        self.assertEqual(int(torch.count_nonzero(target.grad[0, [1, 4, 5]])), 0)

    def test_all_invalid_returns_finite_graph_connected_zero(self):
        vision = self.features(3, 5, 3, requires_grad=True)
        target = self.features(3, 5, 4, requires_grad=True)
        with torch.no_grad():
            target[0] = float("nan")
            target[1] = 1
            target[2] = float("inf")
        result = compute_selected_cka_loss(
            vision, target, [torch.tensor([0, 1, 2]), torch.tensor([0, 2, 4]), None],
        )
        self.assertEqual(result.valid_count, 0)
        self.assertEqual(result.invalid_count, 3)
        self.assertEqual(result.subset_sizes, (0, 0, 0))
        self.assertEqual(float(result.loss), 0)
        self.assertTrue(result.loss.requires_grad)
        result.loss.backward()
        self.assertIsNone(vision.grad)
        torch.testing.assert_close(target.grad, torch.zeros_like(target), rtol=0, atol=0)

    def test_cpu_bfloat16_autocast_keeps_gram_loss_and_gradients_in_fp32(self):
        vision = self.features(1, 24, 13)
        target = self.features(1, 24, 17, requires_grad=True)
        indices = [torch.tensor([0, 3, 5, 6, 9, 12, 14, 18, 21, 23])]
        expected = direct_cka_loss(vision[0, indices[0]], target[0, indices[0]])
        expected_grad, = torch.autograd.grad(expected, target)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = compute_selected_cka_loss(vision, target, indices)
        self.assertEqual(result.loss.dtype, torch.float32)
        torch.testing.assert_close(result.loss, expected, rtol=0, atol=0)
        result.loss.backward()
        torch.testing.assert_close(target.grad, expected_grad, rtol=0, atol=0)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            low_precision = compute_selected_cka_loss(
                vision.bfloat16(), target.detach().bfloat16(), indices,
            )
        expected_rounded = direct_cka_loss(
            vision[0, indices[0]].bfloat16(), target[0, indices[0]].detach().bfloat16(),
        )
        self.assertEqual(low_precision.loss.dtype, torch.float32)
        torch.testing.assert_close(low_precision.loss, expected_rounded, rtol=0, atol=0)

    def test_duplicate_and_out_of_bounds_indices_raise(self):
        vision, target = self.features(1, 5, 3), self.features(1, 5, 4)
        for invalid in ([1, 1], [0, 5], [-1, 2], [-1], [5]):
            with self.subTest(indices=invalid), self.assertRaises(ValueError):
                compute_selected_cka_loss(vision, target, [torch.tensor(invalid)])

    def test_noninteger_or_nonvector_indices_raise(self):
        vision, target = self.features(1, 5, 3), self.features(1, 5, 4)
        for invalid in (
            [0, 1], torch.tensor([0., 1.]), torch.tensor([False, True]),
            torch.tensor([[0, 1]]), torch.tensor(1),
        ):
            with self.subTest(indices=invalid), self.assertRaises(ValueError):
                compute_selected_cka_loss(vision, target, [invalid])
        int32 = compute_selected_cka_loss(vision, target, [torch.tensor([0, 1, 2], dtype=torch.int32)])
        self.assertEqual(int32.valid_count, 1)

    def test_invalid_feature_shapes_selection_count_and_eps_raise(self):
        vision, target = self.features(1, 5, 3), self.features(1, 5, 4)
        indices = [torch.tensor([0, 1, 2])]
        for malformed_vision, malformed_target in (
            (vision[0], target), (vision, target[0]),
            (vision, target[:, :4]), (vision.expand(2, -1, -1), target),
        ):
            with self.subTest(shape=(malformed_vision.shape, malformed_target.shape)):
                with self.assertRaises(ValueError):
                    compute_selected_cka_loss(malformed_vision, malformed_target, indices)
        with self.assertRaises(ValueError):
            compute_selected_cka_loss(vision, target, [])
        for eps in (0, -1, float("nan"), float("inf")):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                compute_selected_cka_loss(vision, target, indices, eps=eps)


if __name__ == "__main__":
    unittest.main()
