"""CPU integration checks for the LLaVA Trainer across Transformers versions."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import default_data_collator

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from llava.train.train import TrainingArguments
from llava.train.llava_trainer import LLaVATrainer
from tests.test_cka_backbones import (
    _FakeVisionTower,
    _backbone_factories,
    _configure_multimodal_cka,
    _projector,
)


def make_model(cka=True, pretrain=False):
    config = _configure_multimodal_cka(LlavaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    ))
    config.cka_loss = cka
    config.use_pcgrad = False
    config.cka_loss_projector_weight = 0.1
    config.cka_loss_final_hidden_weight = 0.1
    config.tune_mm_mlp_adapter = pretrain
    config._attn_implementation = "eager"
    model = LlavaLlamaForCausalLM(config)
    model.get_model().vision_tower = _FakeVisionTower()
    model.get_model().mm_projector = _projector()
    if pretrain:
        model.requires_grad_(False)
        model.get_model().mm_projector.requires_grad_(True)
    return model


def sample():
    return {
        "input_ids": torch.tensor([1, IMAGE_TOKEN_INDEX, 2, 3]),
        "labels": torch.tensor([1, IMAGE_TOKEN_INDEX, 2, 3]),
        "images": torch.ones(3, 2, 2),
    }


class TrainingCompatibilityTests(unittest.TestCase):
    def _train(self, model, output_dir, batch_size, accumulation):
        args = TrainingArguments(
            output_dir=output_dir, use_cpu=True, report_to=[],
            max_steps=1, per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            learning_rate=0.01, lr_scheduler_type="constant",
            optim="sgd", max_grad_norm=0.0,
            save_strategy="steps", save_steps=1, logging_steps=1,
            disable_tqdm=True, dataloader_pin_memory=False,
        )
        args.tune_mm_mlp_adapter = model.config.tune_mm_mlp_adapter
        trainer = LLaVATrainer(
            model=model, args=args, train_dataset=[sample(), sample()],
            data_collator=default_data_collator,
        )
        result = trainer.train()
        self.assertEqual(result.global_step, 1)
        self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
        self.assertTrue(any("loss" in entry for entry in trainer.state.log_history))
        checkpoint = Path(output_dir) / "checkpoint-1"
        if args.tune_mm_mlp_adapter:
            self.assertTrue((checkpoint / "mm_projector.bin").is_file())
        else:
            self.assertTrue((checkpoint / "model.safetensors").is_file())
            self.assertTrue((checkpoint / "trainer_state.json").is_file())
        return trainer

    def test_pretrain_and_finetune_accumulation_match_full_batch(self):
        for pretrain, cka in ((True, True), (False, False), (False, True)):
            with self.subTest(pretrain=pretrain, cka=cka), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(10)
                initial = make_model(cka=cka, pretrain=pretrain)
                full = copy.deepcopy(initial)
                accumulated = copy.deepcopy(initial)
                self._train(full, str(Path(tmp) / "full"), 2, 1)
                self._train(accumulated, str(Path(tmp) / "accumulated"), 1, 2)
                for name, parameter in full.named_parameters():
                    torch.testing.assert_close(
                        parameter, dict(accumulated.named_parameters())[name],
                        atol=2e-7, rtol=1e-5,
                    )
                self.assertFalse(torch.equal(
                    initial.model.mm_projector.weight, full.model.mm_projector.weight,
                ))
                self.assertEqual(
                    torch.equal(initial.lm_head.weight, full.lm_head.weight), pretrain,
                )

    def test_projector_pcgrad_training_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model(pretrain=True)
            model.config.vsp_asymmetric_pcgrad = True
            model.config.vsp_gradient_diagnostics = True
            model.config.vsp_apply_to_projector_only = True
            initial_projector = model.model.mm_projector.weight.detach().clone()
            trainer = self._train(model, tmp, 2, 1)
            self.assertFalse(torch.equal(initial_projector, model.model.mm_projector.weight))
            self.assertTrue(any("grad/projector/main_grad_norm" in entry
                                for entry in trainer.state.log_history))

    def test_attention_subset_with_selected_hidden_cka(self):
        for name, model_cls, config_factory in _backbone_factories():
            if name not in {"llama", "qwen2", "qwen3", "gemma3", "phi3"}:
                continue
            with self.subTest(backbone=name):
                config = _configure_multimodal_cka(config_factory())
                config.cka_loss_layers = "1,final"
                config.cka_loss_subset_select_layer = 1
                config.cka_loss_subset_max_ratio = 0.5
                config._attn_implementation = "eager"
                model = model_cls(config).train()
                model.model.vision_tower = _FakeVisionTower()
                model.model.mm_projector = _projector()
                batch = default_data_collator([sample()])
                result = model(**batch)
                mask = model.last_cka_subset_vision_feature_mask
                self.assertIsNotNone(mask)
                self.assertGreater(int(mask.sum()), 0)
                self.assertLess(int(mask.sum()), 5)
                self.assertEqual(len(model.last_cka_per_layer_losses), 2)
                (result.loss + result.projector_cka_loss + sum(result.aux_losses)).backward()
                self.assertTrue(torch.isfinite(model.model.mm_projector.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
