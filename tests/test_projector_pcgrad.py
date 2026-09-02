import copy
import unittest

import torch
import torch.nn as nn

from llava.model.llava_arch import project_features_with_pcgrad


class ProjectorPcGradTests(unittest.TestCase):
    def _exact_parameter_grads(
        self,
        projector,
        inputs,
        main_gradient,
        cka_gradient,
        *,
        cka_scale=1.0,
        common_scale=1.0,
    ):
        parameters = tuple(
            parameter
            for parameter in projector.parameters()
            if parameter.requires_grad
        )
        projected_features = projector(inputs)
        task_loss = common_scale * (projected_features * main_gradient).sum()
        cka_loss = (
            common_scale
            * cka_scale
            * (projected_features * cka_gradient).sum()
        )
        task_grads = torch.autograd.grad(
            task_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        cka_grads = torch.autograd.grad(
            cka_loss,
            parameters,
            allow_unused=True,
        )

        reduction_dtype = (
            torch.float64
            if projected_features.dtype == torch.float64
            else torch.float32
        )
        dot_product = projected_features.new_zeros((), dtype=reduction_dtype)
        task_norm_sq = projected_features.new_zeros((), dtype=reduction_dtype)
        for task_grad, cka_grad in zip(task_grads, cka_grads):
            if task_grad is not None:
                task_float = task_grad.to(reduction_dtype)
                task_norm_sq = task_norm_sq + task_float.square().sum()
                if cka_grad is not None:
                    dot_product = dot_product + (
                        task_float * cka_grad.to(reduction_dtype)
                    ).sum()

        coefficient = torch.where(
            torch.isfinite(dot_product)
            & torch.isfinite(task_norm_sq)
            & (dot_product < 0)
            & (task_norm_sq > 0),
            dot_product
            / task_norm_sq.clamp_min(torch.finfo(reduction_dtype).tiny),
            dot_product.new_zeros(()),
        )
        expected = []
        for parameter, task_grad, cka_grad in zip(
            parameters,
            task_grads,
            cka_grads,
        ):
            if task_grad is None and cka_grad is None:
                merged = torch.zeros_like(parameter)
            elif task_grad is None:
                merged = cka_grad
            elif cka_grad is None:
                merged = task_grad
            else:
                task_float = task_grad.to(reduction_dtype)
                merged = (
                    task_float
                    + cka_grad.to(reduction_dtype)
                    - coefficient * task_float
                ).to(parameter.dtype)
            expected.append(merged)
        return tuple(expected), task_grads, cka_grads

    def _actual_parameter_grads(
        self,
        projector,
        inputs,
        main_gradient,
        cka_gradient,
        *,
        cka_scale=1.0,
        common_scale=1.0,
    ):
        projector.zero_grad(set_to_none=True)
        main_branch, cka_branch = project_features_with_pcgrad(
            projector,
            inputs,
        )
        loss = common_scale * (
            (main_branch * main_gradient).sum()
            + cka_scale * (cka_branch * cka_gradient).sum()
        )
        loss.backward()
        return tuple(
            parameter.grad.detach().clone()
            for parameter in projector.parameters()
            if parameter.requires_grad
        )

    def test_parameter_conflict_is_projected_even_when_output_dot_is_positive(self):
        projector = nn.Linear(2, 1, bias=False)
        reference_projector = copy.deepcopy(projector)
        inputs = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        main_gradient = torch.tensor([[1.0], [1.0]])
        cka_gradient = torch.tensor([[2.0], [-1.0]])

        self.assertGreater(
            float((main_gradient * cka_gradient).sum()),
            0.0,
        )
        expected, task_grads, cka_grads = self._exact_parameter_grads(
            reference_projector,
            inputs,
            main_gradient,
            cka_gradient,
        )
        parameter_dot = sum(
            (task_grad * cka_grad).sum()
            for task_grad, cka_grad in zip(task_grads, cka_grads)
        )
        self.assertLess(float(parameter_dot), 0.0)

        actual = self._actual_parameter_grads(
            projector,
            inputs,
            main_gradient,
            cka_gradient,
        )

        for actual_grad, expected_grad in zip(actual, expected):
            torch.testing.assert_close(actual_grad, expected_grad)
        torch.testing.assert_close(
            actual[0],
            torch.tensor([[3.4, 0.8]]),
        )

    def test_non_conflicting_parameter_gradients_are_summed(self):
        projector = nn.Linear(2, 2)
        reference_projector = copy.deepcopy(projector)
        inputs = torch.tensor([[0.25, -0.75], [1.0, 0.5]])
        main_gradient = torch.tensor([[1.0, 2.0], [0.5, 1.0]])
        cka_gradient = torch.tensor([[2.0, 1.0], [1.0, 0.5]])

        expected, task_grads, cka_grads = self._exact_parameter_grads(
            reference_projector,
            inputs,
            main_gradient,
            cka_gradient,
        )
        self.assertGreaterEqual(
            float(
                sum(
                    (task_grad * cka_grad).sum()
                    for task_grad, cka_grad in zip(task_grads, cka_grads)
                )
            ),
            0.0,
        )
        actual = self._actual_parameter_grads(
            projector,
            inputs,
            main_gradient,
            cka_gradient,
        )

        for actual_grad, expected_grad in zip(actual, expected):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_static_cka_weight_and_common_loss_scale_reach_exact_pcgrad(self):
        torch.manual_seed(17)
        projector = nn.Sequential(
            nn.Linear(2, 3, bias=False),
            nn.GELU(),
            nn.Linear(3, 2, bias=False),
        )
        reference_projector = copy.deepcopy(projector)
        inputs = torch.tensor([[0.25, -0.75], [1.0, 0.5]])
        main_gradient = torch.tensor([[1.0, -2.0], [0.5, 1.0]])
        cka_gradient = torch.tensor([[-3.0, 1.0], [2.0, -1.0]])
        cka_scale = 0.25 * 0.4
        common_scale = 128.0

        expected, _, _ = self._exact_parameter_grads(
            reference_projector,
            inputs,
            main_gradient,
            cka_gradient,
            cka_scale=cka_scale,
            common_scale=common_scale,
        )
        actual = self._actual_parameter_grads(
            projector,
            inputs,
            main_gradient,
            cka_gradient,
            cka_scale=cka_scale,
            common_scale=common_scale,
        )

        for actual_grad, expected_grad in zip(actual, expected):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_unused_branch_matches_ordinary_projector_backward(self):
        inputs = torch.tensor([[0.25, -0.75], [1.0, 0.5]])
        branch_gradient = torch.tensor([[2.0, -3.0], [1.0, 4.0]])

        for active_branch in ("main", "cka"):
            with self.subTest(active_branch=active_branch):
                projector = nn.Sequential(
                    nn.Linear(2, 3),
                    nn.GELU(),
                    nn.Linear(3, 2),
                )
                reference_projector = copy.deepcopy(projector)
                main_branch, cka_branch = project_features_with_pcgrad(
                    projector,
                    inputs,
                )
                ordinary_output = reference_projector(inputs)
                torch.testing.assert_close(main_branch, ordinary_output)
                torch.testing.assert_close(cka_branch, ordinary_output)

                active = (
                    main_branch
                    if active_branch == "main"
                    else cka_branch
                )
                (active * branch_gradient).sum().backward()
                (ordinary_output * branch_gradient).sum().backward()

                for actual_parameter, expected_parameter in zip(
                    projector.parameters(),
                    reference_projector.parameters(),
                ):
                    torch.testing.assert_close(
                        actual_parameter.grad,
                        expected_parameter.grad,
                    )

    def test_zero_task_gradient_leaves_cka_gradient_finite_and_unchanged(self):
        projector = nn.Linear(2, 2)
        reference_projector = copy.deepcopy(projector)
        inputs = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
        main_gradient = torch.zeros(2, 2)
        cka_gradient = torch.tensor([[2.0, -3.0], [0.5, 7.0]])

        expected, _, _ = self._exact_parameter_grads(
            reference_projector,
            inputs,
            main_gradient,
            cka_gradient,
        )
        actual = self._actual_parameter_grads(
            projector,
            inputs,
            main_gradient,
            cka_gradient,
        )

        for actual_grad, expected_grad in zip(actual, expected):
            self.assertTrue(torch.isfinite(actual_grad).all())
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_projector_input_keeps_the_ordinary_summed_gradient(self):
        torch.manual_seed(23)
        projector = nn.Sequential(
            nn.Linear(2, 3),
            nn.GELU(),
            nn.Linear(3, 2),
        )
        reference_projector = copy.deepcopy(projector)
        inputs = torch.tensor(
            [[0.25, -0.75], [1.0, 0.5]],
            requires_grad=True,
        )
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        main_gradient = torch.tensor([[1.0, 0.0], [0.5, -1.0]])
        cka_gradient = torch.tensor([[-2.0, 1.0], [1.0, 2.0]])

        main_branch, cka_branch = project_features_with_pcgrad(
            projector,
            inputs,
        )
        (
            (main_branch * main_gradient).sum()
            + (cka_branch * cka_gradient).sum()
        ).backward()

        reference_output = reference_projector(reference_inputs)
        (reference_output * (main_gradient + cka_gradient)).sum().backward()

        torch.testing.assert_close(inputs.grad, reference_inputs.grad)

    def test_fp16_large_projection_coefficient_stays_finite(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        projector = nn.Linear(1, 2, bias=False).to(
            device=device,
            dtype=torch.float16,
        )
        reference_projector = copy.deepcopy(projector)
        inputs = torch.ones(1, 1, device=device, dtype=torch.float16)
        main_gradient = torch.tensor(
            [[1e-5, 0.0]],
            device=device,
            dtype=torch.float16,
        )
        cka_gradient = torch.tensor(
            [[-1.0, 1.0]],
            device=device,
            dtype=torch.float16,
        )

        expected, _, _ = self._exact_parameter_grads(
            reference_projector,
            inputs,
            main_gradient,
            cka_gradient,
        )
        actual = self._actual_parameter_grads(
            projector,
            inputs,
            main_gradient,
            cka_gradient,
        )

        for actual_grad, expected_grad in zip(actual, expected):
            self.assertTrue(torch.isfinite(actual_grad).all())
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                rtol=1e-3,
                atol=1e-3,
            )

    def test_gradient_logging_probes_do_not_consume_private_graph(self):
        torch.manual_seed(31)
        projector = nn.Sequential(
            nn.Linear(2, 3),
            nn.GELU(),
            nn.Linear(3, 2),
        )
        reference_projector = copy.deepcopy(projector)
        inputs = torch.tensor([[0.25, -0.75], [1.0, 0.5]])
        main_gradient = torch.tensor([[1.0, -2.0], [0.5, 1.0]])
        cka_gradient = torch.tensor([[-3.0, 1.0], [2.0, -1.0]])

        expected, expected_task, expected_cka = self._exact_parameter_grads(
            reference_projector,
            inputs,
            main_gradient,
            cka_gradient,
        )
        main_branch, cka_branch = project_features_with_pcgrad(
            projector,
            inputs,
        )
        task_loss = (main_branch * main_gradient).sum()
        cka_loss = (cka_branch * cka_gradient).sum()
        parameters = tuple(projector.parameters())

        task_probe = torch.autograd.grad(
            task_loss,
            parameters,
            retain_graph=True,
        )
        cka_probe = torch.autograd.grad(
            cka_loss,
            parameters,
            retain_graph=True,
        )
        for actual_grad, expected_grad in zip(task_probe, expected_task):
            torch.testing.assert_close(actual_grad, expected_grad)
        for actual_grad, expected_grad in zip(cka_probe, expected_cka):
            torch.testing.assert_close(actual_grad, expected_grad)

        (task_loss + cka_loss).backward()
        for parameter, expected_grad in zip(parameters, expected):
            torch.testing.assert_close(parameter.grad, expected_grad)

    def test_zero3_partitioned_projector_fails_fast(self):
        projector = nn.Linear(2, 2)
        parameter = next(projector.parameters())
        parameter.ds_id = 0
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "does not support DeepSpeed ZeRO-3",
            ):
                project_features_with_pcgrad(
                    projector,
                    torch.ones(1, 2),
                )
        finally:
            del parameter.ds_id

    def test_frozen_projector_bypasses_pcgrad_and_preserves_input_gradient(self):
        projector = nn.Linear(2, 2)
        for parameter in projector.parameters():
            parameter.requires_grad_(False)
        reference_projector = copy.deepcopy(projector)
        inputs = torch.tensor([[1.0, -1.0]], requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        main_gradient = torch.tensor([[1.0, 2.0]])
        cka_gradient = torch.tensor([[-3.0, 4.0]])

        main_branch, cka_branch = project_features_with_pcgrad(
            projector,
            inputs,
        )
        (
            (main_branch * main_gradient).sum()
            + (cka_branch * cka_gradient).sum()
        ).backward()
        reference_output = reference_projector(reference_inputs)
        (reference_output * (main_gradient + cka_gradient)).sum().backward()

        torch.testing.assert_close(inputs.grad, reference_inputs.grad)


if __name__ == "__main__":
    unittest.main()
