import unittest
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from llava.train.vsp_gradient_controller import (
    VSPGradientController, _sequence_norm, combine_partitioned_vsp_gradients,
    vsp_controller_requested, vsp_rewrites_gradients,
)


def make_config(**overrides):
    values = {
        "use_pcgrad": False,
        "vsp_asymmetric_pcgrad": True,
        "vsp_apply_to_projector_only": False,
        "vsp_norm_cap": False,
        "vsp_pcgrad_threshold": 0.0,
        "vsp_proj_max_grad_ratio": 10.0,
        "vsp_llm_max_grad_ratio": 10.0,
        "vsp_grad_ema_beta": 0.95,
        "vsp_grad_log_interval": 10,
        "vsp_grad_eps": 1e-12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _distributed_projection_worker(rank, folder):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{folder}/rendezvous', rank=rank, world_size=2)
    try:
        controller = VSPGradientController(torch.nn.Linear(1, 1), make_config())
        result, _ = combine_partitioned_vsp_gradients(
            controller, {0:[torch.tensor([1. if rank == 0 else 10.])]},
            {0:[torch.tensor([-1.]) if rank == 0 else None]}, {}, {0:'projector'},
            torch.tensor(0.), log_stats=False,
        )
        torch.save(result[0][0], Path(folder)/f'rank-{rank}.pt')
    finally:
        dist.destroy_process_group()


class VSPProjectorOnlyPCGradTest(unittest.TestCase):
    def _run_controller(self, projector_only):
        controller = VSPGradientController(
            model=torch.nn.Linear(1, 1),
            config=make_config(vsp_apply_to_projector_only=projector_only),
        )
        return controller._process_group_gradients(
            "llm",
            main_gradients=[torch.tensor([1.0, 0.0])],
            proj_gradients=[torch.tensor([-1.0, 1.0])],
            final_gradients=[torch.tensor([-1.0, 2.0])],
            reference_tensor=torch.tensor(0.0),
        )

    def test_default_projects_both_auxiliary_losses(self):
        gradients, logs = self._run_controller(projector_only=False)

        torch.testing.assert_close(gradients[0], torch.tensor([1.0, 3.0]))
        self.assertGreater(logs["proj_projection_removed_fraction"], 0.0)
        self.assertGreater(logs["final_projection_removed_fraction"], 0.0)

    def test_projector_only_leaves_final_auxiliary_unprojected(self):
        gradients, logs = self._run_controller(projector_only=True)

        # main + projected projector auxiliary + untouched final auxiliary
        torch.testing.assert_close(gradients[0], torch.tensor([0.0, 3.0]))
        self.assertGreater(logs["proj_projection_removed_fraction"], 0.0)
        self.assertEqual(logs["final_projection_removed_fraction"], 0.0)
        self.assertEqual(logs["final_conflict"], 1.0)

    def test_all_disabled_bypasses_controller(self):
        config = make_config(vsp_asymmetric_pcgrad=False)
        self.assertFalse(vsp_controller_requested(config))
        self.assertFalse(vsp_rewrites_gradients(config))

    def test_none_and_zero_auxiliary_are_equivalent(self):
        for pcgrad, cap, value in ((True, False, -1.0), (False, True, 2.0)):
            outputs = []
            for missing in (None, torch.zeros(1)):
                controller = VSPGradientController(torch.nn.Linear(1, 1), make_config(
                    vsp_asymmetric_pcgrad=pcgrad, vsp_norm_cap=cap, vsp_proj_max_grad_ratio=0.5,
                ))
                result, logs = controller._process_group_gradients(
                    "projector", [torch.tensor([1.]), torch.tensor([10.])],
                    [torch.tensor([value]), missing], [None, None], torch.tensor(0.),
                )
                self.assertAlmostEqual(logs['main_grad_norm'], 101 ** 0.5, places=5)
                outputs.append(torch.cat(result))
            torch.testing.assert_close(*outputs)
            expected = torch.tensor([1/101, 10+10/101]) if pcgrad else torch.tensor([3., 10.])
            torch.testing.assert_close(outputs[0], expected)

    def test_missing_auxiliary_still_records_ce_norm(self):
        controller = VSPGradientController(torch.nn.Linear(1, 1), make_config())
        result, logs = controller._process_group_gradients(
            'llm', [torch.tensor([3., 4.])], [None], [None], torch.tensor(0.),
        )
        self.assertEqual(logs['main_grad_norm'], 5.)
        torch.testing.assert_close(result[0], torch.tensor([3., 4.]))

    def test_quiet_mode_preserves_gradients_and_norm_cap_ema(self):
        for pcgrad, cap in ((True, False), (False, True), (True, True)):
            controllers = [VSPGradientController(torch.nn.Linear(1, 1), make_config(
                vsp_asymmetric_pcgrad=pcgrad, vsp_norm_cap=cap, vsp_proj_max_grad_ratio=0.5,
            )) for _ in range(2)]
            for magnitude in (10., 1., 0., 3.):
                results = []
                for controller, verbose in zip(controllers, (False, True)):
                    result, stats = controller._process_group_gradients(
                        'projector', [torch.tensor([magnitude, 1.])],
                        [torch.tensor([-2., 3.])], [torch.tensor([1., -4.])],
                        torch.tensor(0.), log_stats=verbose,
                    )
                    logs = {}
                    controller._record_group_logs(logs, 'projector', stats, log_stats=verbose)
                    self.assertEqual(bool(logs), verbose)
                    results.append(result[0])
                torch.testing.assert_close(*results)
                self.assertEqual(controllers[0]._main_norm_reference('projector'),
                                 controllers[1]._main_norm_reference('projector'))

    def test_quiet_pcgrad_skips_logging_norm_passes(self):
        controller = VSPGradientController(torch.nn.Linear(1, 1), make_config())
        with mock.patch('llava.train.vsp_gradient_controller._sequence_norm', side_effect=AssertionError), \
             mock.patch('llava.train.vsp_gradient_controller._summed_sequence_norm', side_effect=AssertionError):
            result, _ = controller._process_group_gradients(
                'projector', [torch.tensor([1., 0.])], [torch.tensor([-1., 2.])],
                [None], torch.tensor(0.), log_stats=False,
            )
        torch.testing.assert_close(result[0], torch.tensor([1., 2.]))

    def test_chunked_norm_bounds_temporary_casts(self):
        gradient = torch.arange(17, dtype=torch.bfloat16)
        with mock.patch('torch.dot', wraps=torch.dot) as dot:
            result = _sequence_norm([gradient], torch.tensor(0.), chunk_size=4)
        self.assertEqual(dot.call_count, 5)
        self.assertTrue(all(call.args[0].numel() <= 4 for call in dot.call_args_list))
        torch.testing.assert_close(result.float(), gradient.float().norm())

    def test_partitioned_inplace_path_matches_dense_and_keeps_inputs_separate(self):
        for quiet in (False, True):
            config = make_config(vsp_norm_cap=True, vsp_proj_max_grad_ratio=0.5)
            dense = VSPGradientController(torch.nn.Linear(1, 1), config)
            partitioned = VSPGradientController(torch.nn.Linear(1, 1), config)
            main = [torch.tensor([1.]), torch.tensor([10.])]
            proj = [torch.tensor([-1.]), None]
            final = [None, torch.tensor([2.])]
            expected, _ = dense._process_group_gradients('projector', main, proj, final, torch.tensor(0.))
            parts, logs = combine_partitioned_vsp_gradients(
                partitioned, {0: copy.deepcopy(main)}, {0: copy.deepcopy(proj)},
                {0: copy.deepcopy(final)}, {0:'projector'}, torch.tensor(0.), log_stats=not quiet,
            )
            torch.testing.assert_close(torch.cat(parts[0]), torch.cat(expected))
            self.assertEqual(bool(logs), not quiet)

    def test_threshold_leaves_small_negative_cosine_unchanged(self):
        controller = VSPGradientController(torch.nn.Linear(1, 1), make_config(vsp_pcgrad_threshold=0.05))
        result, logs = controller._process_group_gradients(
            'projector', [torch.tensor([1., 0.])], [torch.tensor([-0.01, 1.])],
            [None], torch.tensor(0.),
        )
        torch.testing.assert_close(result[0], torch.tensor([0.99, 1.]))
        self.assertEqual(logs['projection_removed_fraction'], 0.)

    def test_nonfinite_gradient_fails_before_update(self):
        controller = VSPGradientController(torch.nn.Linear(1, 1), make_config())
        with self.assertRaisesRegex(RuntimeError, 'NaN or Inf'):
            controller._process_group_gradients(
                'projector', [torch.tensor([float('nan')])], [None], [None], torch.tensor(0.), log_stats=False,
            )

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), 'Gloo required')
    def test_two_rank_partitioned_projection_uses_global_norm(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(_distributed_projection_worker, args=(tmp,), nprocs=2, join=True)
            actual = torch.cat([torch.load(Path(tmp)/f'rank-{r}.pt', weights_only=True) for r in range(2)])
            torch.testing.assert_close(actual, torch.tensor([1/101, 10+10/101]))


if __name__ == "__main__":
    unittest.main()
