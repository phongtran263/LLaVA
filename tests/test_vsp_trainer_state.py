"""VSP logging gates and checkpoint continuity, using tiny CPU models."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from transformers import Trainer, default_data_collator

from llava.train.train import TrainingArguments
from llava.train.llava_trainer import LLaVATrainer, VSP_STATE_NAME
from llava.train.vsp_gradient_controller import VSPGradientController
from tests.test_training_compatibility import make_model, sample


def trainer_for(model, output_dir, *, steps=3, diagnostics=False):
    model.config.vsp_gradient_diagnostics = diagnostics
    model.config.vsp_grad_log_interval = 2
    args = TrainingArguments(
        output_dir=str(output_dir), use_cpu=True, report_to=[], max_steps=steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=0.01, lr_scheduler_type='constant', optim='sgd',
        max_grad_norm=0., save_strategy='steps', save_steps=2,
        logging_steps=1, disable_tqdm=True, dataloader_pin_memory=False, seed=29,
    )
    args.tune_mm_mlp_adapter = model.config.tune_mm_mlp_adapter
    return LLaVATrainer(model=model, args=args, train_dataset=[sample() for _ in range(6)],
                       data_collator=default_data_collator)


class VSPTrainerStateTests(unittest.TestCase):
    def test_disabled_path_never_extracts_task_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = trainer_for(make_model(), tmp)
            with mock.patch('torch.autograd.grad', side_effect=AssertionError('unexpected task-gradient pass')):
                trainer.train()
            self.assertFalse(hasattr(trainer, '_vsp_gradient_controller'))

    def test_diagnostics_only_runs_at_interval_and_preserves_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(12)
            model = make_model()
            baseline = trainer_for(copy.deepcopy(model), Path(tmp)/'baseline')
            measured = trainer_for(copy.deepcopy(model), Path(tmp)/'measured', diagnostics=True)
            baseline.train()
            calls = []
            original = VSPGradientController.compute_diagnostics
            def record(controller, *args, **kwargs):
                calls.append(measured.state.global_step)
                return original(controller, *args, **kwargs)
            with mock.patch.object(VSPGradientController, 'compute_diagnostics', record):
                measured.train()
            self.assertEqual(calls, [0, 2])
            for p, q in zip(baseline.model.parameters(), measured.model.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_zero2_ordinary_backward_refreshes_accumulation_boundary(self):
        for cka in (True, False):
            with self.subTest(cka=cka), tempfile.TemporaryDirectory() as tmp:
                trainer = trainer_for(make_model(cka=cka), tmp, diagnostics=True)
                trainer.state.global_step = 1  # No diagnostics due at this step.
                trainer.is_deepspeed_enabled = True
                engine = mock.Mock()
                events = []
                engine.set_gradient_accumulation_boundary.side_effect = lambda sync: events.append(sync)
                with mock.patch.object(trainer, '_get_deepspeed_engine', return_value=engine), \
                     mock.patch.object(trainer.accelerator, 'backward', side_effect=lambda loss: events.append('backward')):
                    for sync in (False, True):
                        with mock.patch.object(trainer.accelerator.gradient_state, 'sync_gradients', sync):
                            trainer._training_step_with_cka_runtime_state(
                                trainer.model, default_data_collator([sample()]),
                            )
                self.assertEqual(events, [False, 'backward', True, 'backward'])

    def test_quiet_controller_checkpoint_resume_preserves_ema_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(21)
            initial = make_model()
            initial.config.vsp_asymmetric_pcgrad = True
            initial.config.vsp_norm_cap = True
            initial.config.vsp_proj_max_grad_ratio = 0.01
            full = trainer_for(copy.deepcopy(initial), Path(tmp)/'full')
            full.train()
            checkpoint = Path(tmp)/'full/checkpoint-2'
            self.assertTrue((checkpoint/VSP_STATE_NAME).is_file())
            saved = json.loads((checkpoint/VSP_STATE_NAME).read_text())
            self.assertTrue(saved['ema'])
            resumed = trainer_for(copy.deepcopy(initial), Path(tmp)/'resumed')
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(full._vsp_gradient_controller.state_dict(), resumed._vsp_gradient_controller.state_dict())
            for p, q in zip(full.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
            self.assertFalse(any(k.startswith('grad/') for entry in full.state.log_history for k in entry))

    def test_pretrain_adapter_checkpoint_includes_controller_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model(pretrain=True)
            model.config.vsp_norm_cap = True
            trainer = trainer_for(model, tmp, steps=2)
            trainer.train()
            checkpoint = Path(tmp)/'checkpoint-2'
            self.assertTrue((checkpoint/'mm_projector.bin').is_file())
            self.assertEqual(json.loads((checkpoint/VSP_STATE_NAME).read_text()),
                             trainer._vsp_gradient_controller.state_dict())

    def test_old_checkpoint_without_ema_warns_and_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model()
            model.config.vsp_norm_cap = True
            trainer = trainer_for(model, tmp)
            controller = trainer._get_vsp_gradient_controller(model)
            controller.ema['grad/projector/main_grad_norm_ema'] = 99.
            with mock.patch.object(Trainer, '_load_optimizer_and_scheduler'), \
                 mock.patch('llava.train.llava_trainer.logger.warning') as warning:
                trainer._load_optimizer_and_scheduler(tmp)
            self.assertEqual(controller.ema, {})
            warning.assert_called_once()


if __name__ == '__main__':
    unittest.main()
