"""Real tiny-model integration coverage for delayed CKA activation."""

import tempfile
import unittest

import torch
from transformers import default_data_collator

from llava.train.llava_trainer import LLaVATrainer
from llava.train.train import TrainingArguments
from tests.test_training_compatibility import make_model, sample


class CkaLossStartRatioIntegrationTests(unittest.TestCase):
    def test_forward_skips_cka_before_boundary_and_computes_both_terms_at_boundary(self):
        torch.manual_seed(61)
        model = make_model(cka=True, pretrain=False)
        model.config.cka_loss_start_ratio = 0.8
        model.config.cka_loss_layers = "1"

        hidden_cka_calls = []
        original_hidden_cka = model._compute_masked_linear_cka_loss

        def record_hidden_cka(*args, **kwargs):
            hidden_cka_calls.append(1)
            return original_hidden_cka(*args, **kwargs)

        model._compute_masked_linear_cka_loss = record_hidden_cka

        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                use_cpu=True,
                report_to=[],
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )
            trainer = LLaVATrainer(model=model, args=args)
            trainer.state.max_steps = 10
            batch = default_data_collator([sample()])

            trainer.state.global_step = 7
            inactive_loss = trainer.training_step(model, batch)

            self.assertTrue(torch.isfinite(inactive_loss))
            self.assertEqual(hidden_cka_calls, [])
            self.assertIsNone(model.last_cka_loss)
            self.assertIsNone(model.last_cka_projector_loss)
            self.assertIsNone(model.last_cka_layers_loss)
            self.assertTrue(model.config.cka_loss)

            model.zero_grad(set_to_none=True)
            trainer.state.global_step = 8
            active_loss = trainer.training_step(model, batch)

        self.assertEqual(len(hidden_cka_calls), 1)
        self.assertTrue(trainer._last_cka_schedule_active)
        self.assertTrue(model.config.cka_loss)
        self.assertTrue(torch.isfinite(model.last_cka_projector_loss))
        self.assertTrue(torch.isfinite(model.last_cka_layers_loss))
        self.assertGreater(model.last_cka_projector_loss.item(), 0.0)
        self.assertGreater(model.last_cka_layers_loss.item(), 0.0)

        expected_active_loss = (
            model.last_text_loss
            + model.last_cka_projector_loss * model.config.cka_loss_projector_weight
            + model.last_cka_layers_loss * model.config.cka_loss_final_hidden_weight
        )
        torch.testing.assert_close(active_loss, expected_active_loss)
        # Backward alone does not update parameters, so these deterministic
        # forwards have the same text objective. The inactive step therefore
        # contains neither projector nor hidden CKA.
        torch.testing.assert_close(inactive_loss, model.last_text_loss)


if __name__ == "__main__":
    unittest.main()
