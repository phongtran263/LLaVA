#!/usr/bin/env python3
"""Reuse the native training loader; attach telemetry or a save-free CKA probe."""
import json
import os
from pathlib import Path
import sys
import time

# DeepSpeed launches this file by path, while calibration launches it as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import torch
    from transformers import Trainer, TrainerCallback
    import llava.train.train as training
    from scripts.cvpr2027.calibration import calibrate

    log_dir = Path(os.environ["CVPR_LOG_DIR"])
    log_dir.mkdir(parents=True, exist_ok=True)

    class Telemetry(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.start = self.last = time.monotonic()
            self.last_step = state.global_step
            if args.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            if state.is_world_process_zero:
                groups = {}
                for name, param in model.named_parameters():
                    group = "vision" if "vision_tower" in name else ("projector" if "mm_projector" in name else "language")
                    counts = groups.setdefault(group, {"total": 0, "trainable": 0})
                    size = getattr(param, "ds_numel", param.numel())
                    counts["total"] += int(size)
                    if param.requires_grad:
                        counts["trainable"] += int(size)
                if groups.get("vision", {}).get("trainable", 0):
                    raise RuntimeError("Controlled ablations require a frozen vision encoder.")
                (log_dir / "parameter_groups.json").write_text(json.dumps(groups, indent=2) + "\n")

        def on_log(self, args, state, control, logs=None, **kwargs):
            now = time.monotonic()
            elapsed = now - self.last
            steps = state.global_step - self.last_step
            values = dict(logs or {})
            values.update(step=state.global_step, elapsed_seconds=now - self.start,
                          world_size=args.world_size,
                          throughput_examples_per_second=(steps * args.per_device_train_batch_size *
                                                         args.gradient_accumulation_steps * args.world_size /
                                                         elapsed if elapsed > 0 else 0))
            if args.device.type == "cuda":
                values.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            # Every rank records its own VRAM; process zero also supplies the report stream.
            filename = "telemetry.jsonl" if state.is_world_process_zero else f"telemetry-rank{args.process_index}.jsonl"
            with (log_dir / filename).open("a") as stream:
                stream.write(json.dumps(values, default=str, allow_nan=False) + "\n")
            self.last, self.last_step = now, state.global_step

        def on_train_end(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                state.save_to_json(str(log_dir / "trainer_state.json"))

    native_trainer = training.LLaVATrainer

    class ExperimentTrainer(native_trainer):
        def __init__(self, *args, **kwargs):
            callbacks = list(kwargs.pop("callbacks", []) or [])
            callbacks.append(Telemetry())
            super().__init__(*args, callbacks=callbacks, **kwargs)

        def _save_checkpoint(self, model, trial, *args, **kwargs):
            if getattr(self.args, "tune_mm_mlp_adapter", False):
                # Native stage-1 snapshots save only the adapter and cannot
                # resume optimizer/scheduler/RNG. This package promises resume.
                return Trainer._save_checkpoint(self, model, trial, *args, **kwargs)
            return super()._save_checkpoint(model, trial, *args, **kwargs)

        def _save(self, output_dir=None, state_dict=None):
            if getattr(self.args, "tune_mm_mlp_adapter", False):
                training.sanitize_generation_config_for_save(self.model)
                return Trainer._save(self, output_dir, state_dict)
            return super()._save(output_dir, state_dict)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Native CKA forward returns split objectives only in training mode.
            # Validation uses the ordinary CE scalar/2-tuple Trainer contract.
            if not model.training:
                enabled = self.model.config.cka_loss
                self.model.config.cka_loss = False
                try:
                    return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
                finally:
                    self.model.config.cka_loss = enabled
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            if os.environ.get("CVPR_MODE") == "smoke":
                losses = [result] if torch.is_tensor(result) else [result[0]]
                if self.model.config.cka_loss:
                    losses.extend([result[1], *(result[2] or [])])
                if any(not torch.isfinite(loss).all().item() for loss in losses if torch.is_tensor(loss)):
                    raise RuntimeError("Smoke detected NaN/Inf CE or structural loss.")
            return result

    training.LLaVATrainer = ExperimentTrainer
    if os.environ.get("CVPR_VALIDATION_DATA") or os.environ.get("CVPR_MODE") == "smoke":
        original_data_module = training.make_supervised_data_module

        def data_module(tokenizer, data_args, training_args=None):
            import copy
            result = original_data_module(tokenizer, data_args, training_args)
            if os.environ.get("CVPR_MODE") == "smoke":
                import random
                samples = result["train_dataset"].list_data_dict
                candidates = [i for i, sample in enumerate(samples) if sample.get("image")]
                if len(candidates) < 2:
                    raise ValueError("Real-data smoke needs at least two image-containing training examples.")
                indices = random.Random(training_args.seed).sample(candidates, min(8, len(candidates)))
                result["train_dataset"].list_data_dict = [samples[i] for i in indices]
                (log_dir / "smoke_samples.json").write_text(json.dumps({"indices": indices, "seed": training_args.seed}) + "\n")
            if not os.environ.get("CVPR_VALIDATION_DATA"):
                return result
            validation_args = copy.copy(data_args)
            validation_args.image_folder = os.environ["CVPR_VALIDATION_IMAGES"]
            validation_args.train_data_fraction = 1.0
            result["eval_dataset"] = training.LazySupervisedDataset(
                tokenizer=tokenizer, data_path=os.environ["CVPR_VALIDATION_DATA"], data_args=validation_args)
            def sample_key(sample):
                # Exact conversation duplicates must not enter validation.
                import hashlib
                value = json.dumps({"image": sample.get("image"), "conversations": sample.get("conversations")},
                                   sort_keys=True, ensure_ascii=False)
                return hashlib.sha256(value.encode()).hexdigest()
            training_keys = {sample_key(sample) for sample in result["train_dataset"].list_data_dict}
            if any(sample_key(sample) in training_keys for sample in result["eval_dataset"].list_data_dict):
                raise ValueError("Validation contains training conversations. Provide a fixed held-out split for every variant.")
            return result

        training.make_supervised_data_module = data_module
    if os.environ.get("CVPR_MODE") == "calibration":
        # The native debug branch loads exactly the ordinary initialization and
        # exits before trainer.train/save_state/save_model. Substitute our probe.
        training.run_cka_debug_compare = lambda trainer, args: calibrate(trainer, log_dir)
    training.train(attn_implementation="flash_attention_2")


if __name__ == "__main__":
    main()
