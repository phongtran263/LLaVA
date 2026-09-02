import unittest
from types import SimpleNamespace

import torch

from llava.train.vsp_gradient_controller import VSPGradientController


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


if __name__ == "__main__":
    unittest.main()
