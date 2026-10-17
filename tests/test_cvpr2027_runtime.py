import os
from pathlib import Path
import tempfile
import unittest
from contextlib import nullcontext
import json
from types import SimpleNamespace
import copy
import numpy as np
from unittest import mock

import torch

from tests.test_cvpr2027_cka import _tiny_model, _batch


class ExperimentValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_validation_uses_ce_and_restores_cka_configuration(self):
        import llava.train.train as training
        from scripts.cvpr2027.entrypoint import main
        from transformers import TrainingArguments

        with tempfile.TemporaryDirectory() as directory:
            original_trainer = training.LLaVATrainer
            try:
                with mock.patch.dict(os.environ, {"CVPR_LOG_DIR": directory, "CVPR_MODE": "train"}):
                    with mock.patch.object(training, "train"):
                        main()
                cls = training.LLaVATrainer
                model = _tiny_model("llama", projector_weight=0, final_weight=0.1)
                args = TrainingArguments(output_dir=directory, report_to=[], use_cpu=True)
                trainer = cls(model=model, args=args)
                model.eval()
                inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, 3]])}
                expected = model(**inputs).loss
                result, outputs = trainer.compute_loss(model, dict(inputs), return_outputs=True)
                self.assertTrue(torch.equal(result, expected))
                self.assertIsNotNone(outputs.logits)
                self.assertTrue(model.config.cka_loss)
                with mock.patch.object(model, "forward", side_effect=RuntimeError("forward failed")):
                    with self.assertRaisesRegex(RuntimeError, "forward failed"):
                        trainer.compute_loss(model, dict(inputs))
                self.assertTrue(model.config.cka_loss)
            finally:
                training.LLaVATrainer = original_trainer

    def test_standalone_bf16_calibration_handles_fp32_clip_roundtrip(self):
        from scripts.cvpr2027.calibration import calibrate
        from transformers import default_data_collator

        model = _tiny_model().bfloat16()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        vision = model.get_vision_tower()
        batch = _batch()

        class Dataset:
            list_data_dict = [{"id": str(i), "image": f"image-{i}"} for i in range(2)]

            def __getitem__(self, index):
                return {name: value[index] for name, value in batch.items()}

        trainer = SimpleNamespace(
            model=model, args=SimpleNamespace(device=torch.device("cpu"), bf16=True, fp16=False),
            train_dataset=Dataset(), data_collator=default_data_collator,
            _prepare_inputs=lambda inputs: inputs, compute_loss_context_manager=nullcontext,
            _clear_gradient_log_tensors=lambda _model: None)
        env = {"CVPR_CALIBRATION_BATCHES": "2", "CVPR_CALIBRATION_BATCH_SIZE": "1",
               "CVPR_CALIBRATION_SEED": "42", "CVPR_CALIBRATION_ID": "test"}
        with tempfile.TemporaryDirectory() as directory:
            # Native CLIP casts encoded features back to the pixels' dtype.
            def forward(images):
                return vision.features.unsqueeze(0).expand(images.shape[0], -1, -1).to(images.dtype)
            with mock.patch.object(vision, "forward", side_effect=forward):
                with mock.patch.dict(os.environ, env), mock.patch("torch.cuda.is_available", return_value=False):
                    result = calibrate(trainer, directory)
            self.assertGreater(result["recommended_lambda_proj"], 0)
            self.assertTrue((Path(directory) / "gradient_norms.csv").is_file())
            self.assertEqual(json.loads((Path(directory) / "calibration_samples.json").read_text())["seed"], 42)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_stage1_runtime_saves_and_resumes_full_training_state(self):
        import llava.train.train as training
        from scripts.cvpr2027.entrypoint import main
        from llava.train.train import TrainingArguments
        from transformers import default_data_collator
        from transformers.trainer_callback import PrinterCallback

        batch = _batch()
        dataset = [{name: value[i % 2] for name, value in batch.items()} for i in range(8)]
        original_trainer = training.LLaVATrainer
        with tempfile.TemporaryDirectory() as directory:
            try:
                with mock.patch.dict(os.environ, {"CVPR_LOG_DIR": str(Path(directory) / "logs"), "CVPR_MODE": "stage1"}):
                    with mock.patch.object(training, "train"):
                        main()
                trainer_cls = training.LLaVATrainer
                initial = _tiny_model(projector_weight=0.1, final_weight=0)
                initial.requires_grad_(False)
                initial.get_model().mm_projector.requires_grad_(True)
                initial.config.tune_mm_mlp_adapter = True

                def trainer_for(output):
                    args = TrainingArguments(
                        output_dir=str(output), use_cpu=True, report_to=[], max_steps=3,
                        per_device_train_batch_size=2,
                        learning_rate=0.003, lr_scheduler_type="constant", max_grad_norm=0,
                        save_strategy="steps", save_steps=2, logging_steps=1, disable_tqdm=True,
                        seed=291, data_seed=291, remove_unused_columns=False, dataloader_pin_memory=False)
                    args.tune_mm_mlp_adapter = True
                    trainer = trainer_cls(model=copy.deepcopy(initial), args=args, train_dataset=dataset,
                                          data_collator=default_data_collator)
                    trainer.remove_callback(PrinterCallback)
                    return trainer

                full = trainer_for(Path(directory) / "full")
                full.train()
                checkpoint = Path(directory) / "full/checkpoint-2"
                for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
                    self.assertTrue((checkpoint / name).is_file(), name)
                resumed = trainer_for(Path(directory) / "resumed")
                rng_types = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
                context = torch.serialization.safe_globals(rng_types) if hasattr(torch.serialization, "safe_globals") else nullcontext()
                with context:
                    resumed.train(resume_from_checkpoint=str(checkpoint))
                self.assertEqual(resumed.state.global_step, 3)
                for name, value in full.model.state_dict().items():
                    torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
                for key, values in full.optimizer.state_dict()["state"].items():
                    for field, value in values.items():
                        torch.testing.assert_close(resumed.optimizer.state_dict()["state"][key][field], value, rtol=0, atol=0)
            finally:
                training.LLaVATrainer = original_trainer


if __name__ == "__main__":
    unittest.main()
