"""Tiny real-Trainer Stage 2 tests; CPU/offline only, never model downloads."""

import copy
import json
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from accelerate.state import AcceleratorState
from transformers import TrainerCallback, default_data_collator

from llava.train.llava_trainer import LLaVATrainer
from llava.train.protected_cka_refinement import ProtectedCKARefinement
from llava.train.protected_cka_stage2 import (
    CHECKPOINT_MARKER, ProtectedCKAStage2, ProtectedModalitySampler, REFERENCE_NAME, STATE_NAME, STUDENT_NAME,
)
from llava.train.protected_cka_stage2_math import protected_group_gradient
from llava.train.train import (
    TrainingArguments, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3,
)
from tests.test_protected_cka_runtime import (
    FixedCaptionDataset, assert_nested_equal, load_payload, runtime_trainer as stage1_trainer,
)
from tests.test_protected_cka_stage2_forward import make_stage2


class FixedInstructionDataset(FixedCaptionDataset):
    def __init__(self, count, seed, modality="mixed"):
        super().__init__(count, seed)
        self.modality_lengths = []
        for index, row in enumerate(self.rows):
            has_image = modality == "images" or (modality == "mixed" and index % 2 == 0)
            self.list_data_dict[index]["modality"] = "image" if has_image else "text"
            self.modality_lengths.append(5 + index % 3 if has_image else -(5 + index % 3))
            if not has_image:
                row["input_ids"][1] = 8  # Retain a dummy image; it must never create CKA.


def stage2_trainer(model, output_dir, *, count=8, modality="mixed", **overrides):
    values = dict(
        output_dir=str(output_dir), use_cpu=True, report_to=[], max_steps=4,
        per_device_train_batch_size=1, per_device_eval_batch_size=2,
        gradient_accumulation_steps=2, learning_rate=0.003, mm_projector_lr=0.0015,
        lr_scheduler_type="linear", optim="adamw_torch", max_grad_norm=0.4,
        save_strategy="no", eval_strategy="no", logging_steps=1, disable_tqdm=True,
        dataloader_pin_memory=False, dataloader_num_workers=0, remove_unused_columns=False,
        seed=29, data_seed=61, protected_stage2=True, cka_grad_scope="existing",
        protected_stage2_start_phase_b=True, protected_stage2_refine_lr=0.002,
        protected_stage2_warmup_steps=0, protected_stage2_ce_gradient_calibration_batches=2,
        protected_stage2_eval_interval=1, protected_stage2_max_retries=1,
        protected_stage2_ce_tolerance=10., protected_stage2_text_kl_budget=10.,
        protected_stage2_min_phase_a_steps=2, protected_stage2_max_phase_a_steps=4,
        protected_stage2_min_phase_b_steps=1,
        protected_stage2_max_phase_b_steps=4, protected_stage2_plateau_window=1,
        protected_stage2_plateau_patience=1, protected_stage2_plateau_min_delta=100.,
        protected_stage2_kl_chunk_size=2,
    )
    values.update(overrides)
    args = TrainingArguments(**values)
    args.tune_mm_mlp_adapter = False
    model.config.tune_mm_mlp_adapter = False
    trainer = LLaVATrainer(
        model=model, args=args, train_dataset=FixedInstructionDataset(count, 111, modality),
        eval_dataset=FixedInstructionDataset(3, 999, modality), data_collator=default_data_collator,
    )
    trainer.log = lambda logs, *args, **kwargs: None
    return trainer


class ProtectedStage2RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        self.environment = mock.patch.dict(os.environ, {"ACCELERATE_MIXED_PRECISION": "no"})
        self.environment.start()
        torch.manual_seed(73)
        self.initial = make_stage2()
        # Real backbones have a nonempty identity; HF save_pretrained strips
        # this key from config.json, unlike the protected runtime signature.
        self.initial.config._name_or_path = "test/qwen2-tiny"
        self.initial.name_or_path = self.initial.config._name_or_path

    def tearDown(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        self.environment.stop()

    def assert_trainers_equal(self, left, right):
        assert_nested_equal(self, left.model.state_dict(), right.model.state_dict())
        assert_nested_equal(self, left.optimizer.state_dict(), right.optimizer.state_dict())
        assert_nested_equal(self, left.lr_scheduler.state_dict(), right.lr_scheduler.state_dict())
        self.assertEqual(left.state.global_step, right.state.global_step)
        self.assertEqual(left.state.epoch, right.state.epoch)

    def test_new_stage2_arguments_do_not_change_existing_stage1_checkpoint_signature(self):
        model = copy.deepcopy(self.initial).requires_grad_(False)
        model.get_model().mm_projector.requires_grad_(True)
        model.config.cka_loss = False
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage1_trainer(model, tmp)
            runtime = ProtectedCKARefinement(trainer)
            actual = runtime._signature()
            self.assertFalse(any(key.startswith("protected_stage2") for key in actual["config"]))
            # Emulate the pre-Stage-2 TrainingArguments instance without editing
            # Stage 1's implementation or excluding any of its original fields.
            original_args = runtime.args
            runtime.args = SimpleNamespace(**{
                key: value for key, value in vars(original_args).items()
                if not key.startswith("protected_stage2")
            })
            self.assertEqual(runtime._signature(), actual)
            runtime.args = original_args
            runtime.args.protected_stage2_kl_weight = 123.
            runtime.args.protected_stage2_reference_device = "cpu"
            self.assertEqual(runtime._signature(), actual)

    def test_phase_a_matches_native_instruction_ce_with_training_dropout(self):
        with tempfile.TemporaryDirectory() as tmp:
            ordinary_model = copy.deepcopy(self.initial)
            ordinary_model.config.cka_loss = False
            ordinary = stage2_trainer(ordinary_model, Path(tmp) / "ordinary", max_steps=2,
                                      protected_stage2=False)
            ordinary.train()
            protected = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "protected", max_steps=2,
                                       protected_stage2_start_phase_b=False,
                                       protected_stage2_plateau_patience=100)
            protected.train()
            self.assert_trainers_equal(ordinary, protected)
            self.assertIsNone(protected._protected_cka_runtime.reference)
            self.assertFalse(torch.equal(ordinary.model.lm_head.weight, self.initial.lm_head.weight))

    def test_phase_a_plateau_transitions_with_separate_refinement_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "full", max_steps=7,
                                  protected_stage2_start_phase_b=False,
                                  protected_stage2_max_phase_b_steps=2)
            result = full.train()
            runtime = full._protected_cka_runtime
            self.assertEqual(runtime.phase_b_start, 2)
            self.assertEqual(runtime.transition_reason, "plateau")
            self.assertEqual(result.global_step, 4)
            self.assertEqual(runtime.finished_reason, "refinement_budget_complete")
            transition = Path(tmp) / "full/checkpoint-2"
            self.assertEqual(load_payload(transition / "optimizer.pt")["state"], {})
            transition_state = load_payload(transition / STUDENT_NAME)
            assert_nested_equal(self, runtime.reference.state_dict(), transition_state)
            self.assertFalse(torch.equal(runtime.reference.lm_head.weight, self.initial.lm_head.weight))
            resumed = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "resumed", max_steps=7,
                                     protected_stage2_start_phase_b=False,
                                     protected_stage2_max_phase_b_steps=2)
            resumed.train(resume_from_checkpoint=str(Path(tmp) / "full/checkpoint-1"))
            self.assert_trainers_equal(full, resumed)
            assert_nested_equal(self, runtime.state_dict(), resumed._protected_cka_runtime.state_dict())

    def test_no_plateau_preserves_ce_only_and_reports_phase_a_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp,
                                     protected_stage2_start_phase_b=False,
                                     protected_stage2_max_phase_a_steps=2,
                                     protected_stage2_plateau_patience=100)
            result = trainer.train()
            self.assertEqual(result.global_step, 2)
            self.assertEqual(trainer._protected_cka_runtime.phase, "A")
            self.assertIsNone(trainer._protected_cka_runtime.reference)
            self.assertEqual(trainer._protected_cka_runtime.finished_reason, "phase_a_limit_without_plateau")

    def test_budget_reservation_transitions_exactly_off_validation_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=9,
                protected_stage2_start_phase_b=False, protected_stage2_max_phase_a_steps=9,
                protected_stage2_min_phase_b_steps=3, protected_stage2_max_phase_b_steps=3,
                protected_stage2_eval_interval=4, protected_stage2_plateau_patience=100)
            logs = []
            trainer.log = lambda item, *a, **kw: logs.append(item)
            result = trainer.train()
            runtime = trainer._protected_cka_runtime
            self.assertEqual(runtime.phase_b_start, 6)
            self.assertEqual(runtime.transition_reason, "budget_reservation")
            self.assertEqual(result.global_step, 9)
            self.assertEqual(runtime.phase_b_horizon, 3)
            transition = Path(tmp) / "checkpoint-6"
            assert_nested_equal(self, runtime.reference.state_dict(), load_payload(transition / STUDENT_NAME))
            self.assertFalse(load_payload(transition / STATE_NAME)["refinement_accepted"])
            self.assertTrue(load_payload(Path(tmp) / "checkpoint-9" / STATE_NAME)["refinement_accepted"])
            self.assertTrue(any(item.get("protected2/transition_reason") == "budget_reservation" for item in logs))

    def test_reservation_and_plateau_both_wait_for_minimum_ce_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=7,
                protected_stage2_start_phase_b=False, protected_stage2_max_phase_a_steps=7,
                protected_stage2_min_phase_a_steps=5, protected_stage2_min_phase_b_steps=3,
                protected_stage2_max_phase_b_steps=3)
            trainer.train()
            self.assertEqual(trainer._protected_cka_runtime.phase_b_start, 5)

    def test_default_reservation_for_2173_steps_fires_at_1823_not_next_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=2173,
                protected_stage2_start_phase_b=False, protected_stage2_max_phase_a_steps=2173,
                protected_stage2_min_phase_a_steps=1500, protected_stage2_min_phase_b_steps=350,
                protected_stage2_max_phase_b_steps=350, protected_stage2_eval_interval=100)
            runtime = ProtectedCKAStage2(trainer)
            control = SimpleNamespace(should_training_stop=False)
            with mock.patch.object(runtime, "validate", return_value={"ce": 2.6}) as validate, \
                 mock.patch.object(runtime, "enter_phase_b") as enter:
                trainer.state.global_step = 1822
                runtime.step_end(control)
                validate.assert_not_called()
                trainer.state.global_step = 1823
                runtime.step_end(control)
                validate.assert_called_once()
                enter.assert_called_once_with(reason="budget_reservation")

    def test_zero_reservation_keeps_plateau_only_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=3,
                protected_stage2_start_phase_b=False, protected_stage2_max_phase_a_steps=3,
                protected_stage2_min_phase_b_steps=0, protected_stage2_plateau_patience=100)
            trainer.train()
            self.assertEqual(trainer._protected_cka_runtime.phase, "A")
            self.assertIsNone(trainer._protected_cka_runtime.reference)

    def test_direct_phase_b_teacher_comes_from_stage2_ce_checkpoint_not_stage1(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "source", max_steps=1,
                                    protected_stage2_start_phase_b=False,
                                    protected_stage2_plateau_patience=100)
            source.train()
            checkpoint = Path(tmp) / "source/checkpoint-1"
            self.assertNotIn("_name_or_path", json.loads((checkpoint / "config.json").read_text()))
            self.assertEqual(load_payload(checkpoint / STATE_NAME)["signature"]["model"]["_name_or_path"],
                             "test/qwen2-tiny")
            initialized = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "continued", max_steps=2,
                                         protected_stage2_source_stage2_checkpoint=str(checkpoint))
            initialized.train()
            assert_nested_equal(self, initialized._protected_cka_runtime.reference.state_dict(), source.model.state_dict())
            self.assertEqual(initialized._protected_cka_runtime.transition_reason, "explicit_continuation")
            self.assertEqual(initialized._protected_cka_runtime.source_provenance["kind"], "protected_stage2_ce_only")
            self.assertFalse(torch.equal(source.model.lm_head.weight, self.initial.lm_head.weight))
            # A Phase-B protected source is not a CE-only initialization source.
            bad = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "bad", max_steps=2,
                                  protected_stage2_source_stage2_checkpoint=str(Path(tmp) / "continued/checkpoint-1"))
            with self.assertRaisesRegex(ValueError, "CE-only"):
                bad.train()
            # Explicit contradictory config identity may not be overwritten
            # using the otherwise-valid protected signature fallback.
            config = json.loads((checkpoint / "config.json").read_text())
            config["_name_or_path"] = "test/different-backbone"
            (checkpoint / "config.json").write_text(json.dumps(config))
            wrong_identity = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "wrong-identity", max_steps=2,
                protected_stage2_source_stage2_checkpoint=str(checkpoint))
            with self.assertRaisesRegex(ValueError, "mismatch.*_name_or_path"):
                wrong_identity.train()
            for key, value in (("model_type", "a-different-family"), ("pad_token_id", 19)):
                changed = dict(config, _name_or_path=source.model.config._name_or_path)
                changed[key] = value
                (checkpoint / "config.json").write_text(json.dumps(changed))
                incompatible = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / f"wrong-{key}", max_steps=2,
                    protected_stage2_source_stage2_checkpoint=str(checkpoint))
                with self.assertRaisesRegex(ValueError, f"mismatch.*{key}"):
                    incompatible.train()

    def test_completed_ce_source_allows_new_run_settings_but_not_different_data_or_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "source", max_steps=1,
                protected_stage2_start_phase_b=False, protected_stage2_plateau_patience=100)
            source.train()
            checkpoint = Path(tmp) / "source/checkpoint-1"
            source_state = load_payload(checkpoint / STATE_NAME)
            self.assertIsNotNone(source_state["finished_reason"])
            good = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "good", max_steps=2,
                seed=55, per_device_train_batch_size=2, gradient_accumulation_steps=1,
                learning_rate=0.006, protected_stage2_refine_lr=0.001,
                protected_stage2_margin=0.07, protected_stage2_warmup_steps=1,
                protected_stage2_source_stage2_checkpoint=str(checkpoint))
            good.train()
            assert_nested_equal(self, good._protected_cka_runtime.reference.state_dict(), source.model.state_dict())
            self.assertEqual(load_payload(Path(tmp) / "good/checkpoint-0/optimizer.pt")["state"], {})
            inconsistent = copy.deepcopy(source_state)
            inconsistent["phase_b_start"] = 0
            with self.assertRaisesRegex(ValueError, "CE-only"):
                good._protected_cka_runtime._validate_continuation_source(checkpoint, inconsistent)
            inconsistent = copy.deepcopy(source_state)
            inconsistent.pop("has_reference")
            with self.assertRaisesRegex(ValueError, "CE-only"):
                good._protected_cka_runtime._validate_continuation_source(checkpoint, inconsistent)
            # Exact resume is deliberately stricter than a new continuation.
            strict = ProtectedCKAStage2(stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "strict", max_steps=2))
            with self.assertRaisesRegex(ValueError, "checkpoint/config/data mismatch"):
                strict.load_checkpoint(checkpoint)
            for difference in ("train_dataset", "eval_dataset", "model"):
                with self.subTest(difference=difference):
                    bad_model = copy.deepcopy(self.initial)
                    if difference == "model":
                        bad_model.config.rope_theta = 1234.
                    bad = stage2_trainer(bad_model, Path(tmp) / difference, max_steps=2,
                        count=4 if difference == "train_dataset" else 8,
                        protected_stage2_source_stage2_checkpoint=str(checkpoint))
                    if difference == "eval_dataset":
                        bad.eval_dataset.list_data_dict[0]["id"] = "changed-validation"
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        bad.train()

    def test_legacy_exact_resume_requires_explicit_zero_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "full", max_steps=2,
                protected_stage2_min_phase_b_steps=0)
            full.train()
            checkpoint = Path(tmp) / "full/checkpoint-1"
            legacy = load_payload(checkpoint / STATE_NAME)
            legacy["signature"]["config"].pop("protected_stage2_min_phase_b_steps")
            torch.save(legacy, checkpoint / STATE_NAME)
            same = ProtectedCKAStage2(stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "same", max_steps=2,
                protected_stage2_min_phase_b_steps=0))
            same.load_checkpoint(checkpoint)
            self.assertEqual(same.phase, "B")
            changed = ProtectedCKAStage2(stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "changed", max_steps=2,
                protected_stage2_min_phase_b_steps=350))
            with self.assertRaisesRegex(ValueError, "checkpoint/config/data mismatch"):
                changed.load_checkpoint(checkpoint)
            legacy_args = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "old-args", max_steps=2)
            del legacy_args.args.__dict__["protected_stage2_min_phase_b_steps"]
            self.assertEqual(legacy_args.args.protected_stage2_min_phase_b_steps, 350)
            implicit_default = ProtectedCKAStage2(legacy_args)
            self.assertEqual(implicit_default.signature["config"]["protected_stage2_min_phase_b_steps"], 350)
            with self.assertRaisesRegex(ValueError, "checkpoint/config/data mismatch"):
                implicit_default.load_checkpoint(checkpoint)

    def test_legacy_hf_full_source_restores_every_weight_and_rejects_missing_projector(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_model = copy.deepcopy(self.initial)
            source_model.config.cka_loss = False
            source = stage2_trainer(source_model, Path(tmp) / "source", max_steps=1, protected_stage2=False)
            source.train()
            export = Path(tmp) / "legacy-full"
            source.model.save_pretrained(export, safe_serialization=False)
            self.assertNotIn("_name_or_path", json.loads((export / "config.json").read_text()))
            missing_identity = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "missing-identity", max_steps=1,
                protected_stage2_source_stage2_checkpoint=str(export))
            with self.assertRaisesRegex(ValueError, "lacks model identity"):
                missing_identity.train()
            # Legacy exports are accepted only with independently preserved
            # model identity. This fixture supplies the known source metadata.
            config = json.loads((export / "config.json").read_text())
            config["_name_or_path"] = source.model.config._name_or_path
            (export / "config.json").write_text(json.dumps(config))
            continued = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "continued", max_steps=1,
                protected_stage2_source_stage2_checkpoint=str(export))
            continued.train()
            assert_nested_equal(self, continued._protected_cka_runtime.reference.state_dict(), source.model.state_dict())
            self.assertFalse(torch.equal(source.model.lm_head.weight, self.initial.lm_head.weight))
            broken = Path(tmp) / "missing-projector"
            shutil.copytree(export, broken)
            weights = load_payload(broken / "pytorch_model.bin")
            torch.save({key: value for key, value in weights.items() if "mm_projector" not in key},
                       broken / "pytorch_model.bin")
            bad = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "bad", max_steps=1,
                protected_stage2_source_stage2_checkpoint=str(broken))
            with self.assertRaises((ValueError, RuntimeError)):
                bad.train()

    def test_legacy_lora_export_restores_adapter_projector_other_trainables_and_flags(self):
        from peft import LoraConfig, get_peft_model

        def adapted():
            model = get_peft_model(copy.deepcopy(self.initial), LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.3, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
            model.get_model().mm_projector.requires_grad_(True)
            model.get_base_model().lm_head.requires_grad_(True)
            return model

        with tempfile.TemporaryDirectory() as tmp:
            source_model = adapted()
            source_model.config.cka_loss = False
            source = stage2_trainer(source_model, Path(tmp) / "source", max_steps=1,
                lora_enable=True, lora_r=2, lora_alpha=4, lora_dropout=0.3,
                protected_stage2_start_phase_b=False, protected_stage2_plateau_patience=100)
            source.train()
            export = Path(tmp) / "legacy-lora"
            source.model.save_pretrained(export, state_dict=get_peft_state_maybe_zero_3(source.model.named_parameters(), "none"))
            source.model.config.save_pretrained(export)
            non_lora = get_peft_state_non_lora_maybe_zero_3(source.model.named_parameters())
            self.assertTrue(any("mm_projector" in key for key in non_lora))
            self.assertTrue(any("lm_head" in key for key in non_lora))
            torch.save(non_lora, export / "non_lora_trainables.bin")
            model = adapted()
            expected_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
            continued = stage2_trainer(model, Path(tmp) / "continued", max_steps=1,
                lora_enable=True, lora_r=2, lora_alpha=4, lora_dropout=0.3,
                protected_stage2_source_stage2_checkpoint=str(export))
            continued.train()
            assert_nested_equal(self, continued._protected_cka_runtime.reference.state_dict(), source.model.state_dict())
            self.assertEqual({name: parameter.requires_grad for name, parameter in model.named_parameters()}, expected_flags)
            broken = Path(tmp) / "incomplete-lora"
            shutil.copytree(export, broken)
            torch.save({key: value for key, value in non_lora.items() if "mm_projector" not in key},
                       broken / "non_lora_trainables.bin")
            bad = stage2_trainer(adapted(), Path(tmp) / "bad", max_steps=1,
                lora_enable=True, lora_r=2, lora_alpha=4, lora_dropout=0.3,
                protected_stage2_source_stage2_checkpoint=str(broken))
            with self.assertRaises((ValueError, RuntimeError)):
                bad.train()

    def test_complete_phase_b_resume_restores_all_state_and_rng_across_epochs(self):
        for grouped in (False, True):
            with self.subTest(grouped=grouped), tempfile.TemporaryDirectory() as tmp:
                count, steps = (8, 6) if grouped else (4, 5)
                full = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "full", count=count, max_steps=steps,
                                      protected_stage2_max_phase_b_steps=steps, group_by_modality_length=grouped)
                full.train()
                final_rng, python_rng = torch.get_rng_state(), random.getstate()
                checkpoint = Path(tmp) / "full/checkpoint-3"
                for name in (STUDENT_NAME, REFERENCE_NAME, STATE_NAME, "optimizer.pt", "scheduler.pt",
                             "rng_state.pth", "trainer_state.json"):
                    self.assertTrue((checkpoint / name).is_file(), name)
                resumed = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "resumed", count=count, max_steps=steps,
                                         protected_stage2_max_phase_b_steps=steps, group_by_modality_length=grouped)
                resumed.train(resume_from_checkpoint=str(checkpoint))
                self.assert_trainers_equal(full, resumed)
                assert_nested_equal(self, full._protected_cka_runtime.state_dict(), resumed._protected_cka_runtime.state_dict())
                assert_nested_equal(self, full._protected_cka_runtime.reference.state_dict(),
                                    resumed._protected_cka_runtime.reference.state_dict())
                torch.testing.assert_close(torch.get_rng_state(), final_rng, rtol=0, atol=0)
                self.assertEqual(random.getstate(), python_rng)
                self.assertEqual(full.state.epoch, steps / (count / 2))

    def test_mixed_modality_sampler_is_epoch_addressable_without_global_rng(self):
        sampler = ProtectedModalitySampler(batch_size=1, world_size=2, lengths=[5, -5, 7, -7, 6, -6, 8, -8],
                                           group_by_modality=True, seed=45)
        sampler.set_epoch(3)
        initial_rng = torch.get_rng_state()
        first = list(sampler)
        torch.testing.assert_close(torch.get_rng_state(), initial_rng, rtol=0, atol=0)
        torch.manual_seed(999)
        sampler.set_epoch(3)
        self.assertEqual(first, list(sampler))

    def test_bf16_failed_gates_restore_full_student_teacher_optimizer_and_retry_state(self):
        class Depths(TrainerCallback):
            def __init__(self):
                self.values = []

            def on_train_begin(self, args, state, control, model=None, **kwargs):
                forward, depth = model.forward, 0
                while hasattr(forward, "__wrapped__"):
                    depth += 1
                    forward = forward.__wrapped__
                self.values.append(depth)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=2, bf16=True)
            depths = Depths()
            trainer.add_callback(depths)
            with mock.patch.object(ProtectedCKAStage2, "gate", return_value=False) as gate:
                result = trainer.train()
            runtime = trainer._protected_cka_runtime
            self.assertEqual(gate.call_count, 2)
            self.assertEqual(result.global_step, 0)
            self.assertEqual(depths.values, [depths.values[0]] * 2)
            assert_nested_equal(self, trainer.model.state_dict(), self.initial.state_dict())
            assert_nested_equal(self, runtime.reference.state_dict(), self.initial.state_dict())
            self.assertEqual(runtime.retries, 2)
            self.assertEqual(runtime.aux_ratios, {"projector": 0.025, "llm": 0.025})
            self.assertEqual(runtime.finished_reason, "gate_retry_exhausted")
            self.assertFalse(runtime.refinement_accepted)
            self.assertEqual(result.metrics["protected2_refinement_accepted"], 0)
            checkpoint = Path(tmp) / "checkpoint-0"
            assert_nested_equal(self, trainer.optimizer.state_dict(), load_payload(checkpoint / "optimizer.pt"))
            assert_nested_equal(self, trainer.lr_scheduler.state_dict(), load_payload(checkpoint / "scheduler.pt"))
            torch.testing.assert_close(torch.get_rng_state(), load_payload(checkpoint / "rng_state.pth")["cpu"])
            self.assertFalse((Path(tmp) / "checkpoint-1").exists())
            self.assertFalse(load_payload(checkpoint / STATE_NAME)["refinement_accepted"])

    def test_projector_only_auxiliary_leaves_llm_main_gradients_exactly_intact(self):
        observed = []

        def record(main, cka, ema=None, **kwargs):
            final, new_ema, logs = protected_group_gradient(main, cka, ema, **kwargs)
            for group, before, auxiliary, after in zip(kwargs["group_names"], main, cka, final):
                if group == "llm":
                    self.assertIsNone(auxiliary)
                    self.assertIs(after, before)
            observed.append(logs)
            return final, new_ema, logs

        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=2, cka_grad_scope="projector_only")
            with mock.patch("llava.train.protected_cka_stage2.protected_group_gradient", side_effect=record):
                trainer.train()
            self.assertEqual(len(observed), 2)
            self.assertTrue(all(entry["llm"]["scaled_aux_norm"] == 0 for entry in observed))
            self.assertTrue(any(entry["projector"]["scaled_aux_norm"] > 0 for entry in observed))
            self.assertFalse(torch.equal(trainer.model.lm_head.weight, self.initial.lm_head.weight))
            self.assertTrue(all("mm_projector" in name for name in trainer._protected_cka_runtime.seen_cka_support))

    def test_all_text_only_effective_batch_has_no_auxiliary_but_trains_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=2, modality="text")
            trainer.train()
            runtime = trainer._protected_cka_runtime
            self.assertEqual(runtime.seen_cka_support, set())
            for group in ("projector", "llm"):
                self.assertEqual(runtime.latest_logs[f"protected2/grad/{group}/scaled_aux_norm"], 0.)
                self.assertEqual(runtime.latest_logs[f"protected2/grad/{group}/active_tensors"], 0)
            self.assertFalse(torch.equal(trainer.model.lm_head.weight, self.initial.lm_head.weight))
            metrics = runtime.validate()
            self.assertEqual(metrics["multimodal_tokens"], 0)
            self.assertGreater(metrics["text_only_tokens"], 0)
            self.assertIn("text_only_ce", metrics)

    def test_full_ft_nonreentrant_checkpointing_and_bf16_keep_llm_trainable(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=2, bf16=True,
                                     gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
            trainer.train()
            self.assertFalse(torch.equal(trainer.model.lm_head.weight, self.initial.lm_head.weight))
            self.assertGreater(trainer._protected_cka_runtime.ce_grad_refs["llm"], 0.)
            assert_nested_equal(self, trainer._protected_cka_runtime.reference.state_dict(), self.initial.state_dict())
            self.assertTrue(all(not parameter.requires_grad for parameter in trainer._protected_cka_runtime.reference.parameters()))

    def test_lora_adapters_and_non_lora_projector_train_with_immutable_teacher(self):
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(copy.deepcopy(self.initial), LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.3, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        model.get_model().mm_projector.requires_grad_(True)
        initial = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(model, tmp, max_steps=2, lora_enable=True, lora_r=2, lora_alpha=4, lora_dropout=0.3)
            trainer.train()
            changed = {name for name, value in model.state_dict().items() if not torch.equal(value, initial[name])}
            self.assertTrue(any("lora_B" in name for name in changed))
            self.assertTrue(any("mm_projector" in name for name in changed))
            self.assertTrue(all("lora_" in name or "mm_projector" in name for name in changed))
            assert_nested_equal(self, trainer._protected_cka_runtime.reference.state_dict(), initial)
            resumed_model = get_peft_model(copy.deepcopy(self.initial), LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.3, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
            resumed_model.get_model().mm_projector.requires_grad_(True)
            resumed_model.load_state_dict(initial)
            resumed = stage2_trainer(resumed_model, Path(tmp) / "resumed", max_steps=2,
                                     lora_enable=True, lora_r=2, lora_alpha=4, lora_dropout=0.3)
            resumed.train(resume_from_checkpoint=str(Path(tmp) / "checkpoint-1"))
            self.assert_trainers_equal(trainer, resumed)
            assert_nested_equal(self, resumed._protected_cka_runtime.reference.state_dict(), initial)

    def test_continuation_controls_use_same_source_and_zero_cka_matches_ce_kl(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / "source", max_steps=1,
                                    protected_stage2_start_phase_b=False,
                                    protected_stage2_plateau_patience=100)
            source.train()
            runs = {}
            for mode in ("ce_only", "ce_kl", "protected"):
                runs[mode] = stage2_trainer(copy.deepcopy(self.initial), Path(tmp) / mode, max_steps=2,
                    protected_stage2_continuation=mode, protected_stage2_lambda_max=0.,
                    protected_stage2_source_stage2_checkpoint=str(Path(tmp) / "source/checkpoint-1"))
                runs[mode].train()
                runtime = runs[mode]._protected_cka_runtime
                assert_nested_equal(self, runtime.reference.state_dict(), source.model.state_dict())
                self.assertGreater(runtime.latest_logs["protected2/time/effective_batch_compute_seconds"], 0.)
                self.assertEqual(runs[mode].state.global_step, 2)
            self.assertEqual(runs["ce_only"]._protected_cka_runtime.latest_logs["protected2/text_kl"], 0.)
            self.assert_trainers_equal(runs["ce_kl"], runs["protected"])

    def test_validation_checks_answer_kl_even_when_training_kl_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=1,
                                     protected_stage2_continuation="ce_only",
                                     protected_stage2_kl_weight=0.)
            trainer.train()
            runtime = trainer._protected_cka_runtime
            self.assertFalse(runtime.use_kl)
            teacher = copy.deepcopy(runtime.reference.state_dict())
            with torch.no_grad():
                trainer.model.lm_head.weight.add_(torch.randn_like(trainer.model.lm_head.weight) * 0.2)
            metrics = runtime.validate()
            self.assertGreater(metrics["text_kl"], 0.)
            self.assertGreater(metrics["multimodal_tokens"], 0)
            self.assertGreater(metrics["text_only_tokens"], 0)
            trainer.args.protected_stage2_text_kl_budget = 0.
            self.assertFalse(runtime.gate(metrics))
            assert_nested_equal(self, runtime.reference.state_dict(), teacher)

    def test_ce_only_control_logs_failed_gate_but_completes_fixed_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=3,
                                     protected_stage2_continuation="ce_only")
            logs = []
            trainer.log = lambda item, *a, **kw: logs.append(item)
            with mock.patch.object(ProtectedCKAStage2, "gate", return_value=False):
                result = trainer.train()
            runtime = trainer._protected_cka_runtime
            self.assertEqual(result.global_step, 3)
            self.assertEqual(runtime.retries, 0)
            self.assertFalse(runtime.refinement_accepted)
            self.assertEqual(result.metrics["protected2_accepted_refinement_steps"], 0)
            self.assertEqual(result.metrics["protected2_phase_b_steps"], 3)
            self.assertEqual(result.metrics["protected2_requested_phase_b_steps"], 3)
            verdicts = [item for item in logs if "protected2/gate_pass" in item]
            self.assertEqual(len(verdicts), 3)
            self.assertTrue(all(item["protected2/gate_pass"] == 0 and item["protected2/gate_diagnostic_only"] == 1
                                for item in verdicts))
            self.assertFalse(torch.equal(trainer.model.lm_head.weight, self.initial.lm_head.weight))

    def test_checkpoint_retention_keeps_two_latest_and_ce_snapshot_with_reference_hardlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=4, save_total_limit=1)
            trainer.train()
            root = Path(tmp)
            self.assertEqual(sorted(path.name for path in root.glob("checkpoint-*")), ["checkpoint-3", "checkpoint-4"])
            self.assertTrue((root / "ce-only-checkpoint-0" / STUDENT_NAME).is_file())
            self.assertEqual((root / "checkpoint-3" / REFERENCE_NAME).stat().st_ino,
                             (root / "checkpoint-4" / REFERENCE_NAME).stat().st_ino)
            assert_nested_equal(self, load_payload(root / "checkpoint-4" / REFERENCE_NAME), self.initial.state_dict())
            self.assertEqual(Path(trainer._protected_cka_runtime.accepted_checkpoint), root / "checkpoint-4")

    def test_retention_ignores_unknown_incomplete_foreign_symlink_and_source_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            trainer = stage2_trainer(copy.deepcopy(self.initial), root, max_steps=2)
            trainer.train()
            runtime = trainer._protected_cka_runtime
            self.assertTrue((root / "checkpoint-0").is_dir())  # None preserves old behavior.
            original = root / "checkpoint-0"
            foreign = Path(tmp) / "external-source"
            shutil.copytree(original, foreign)
            (root / "checkpoint-90").symlink_to(foreign, target_is_directory=True)
            (root / "checkpoint-91").mkdir()  # Unknown/unmarked user directory.
            for step in (92, 93, 94):
                target = root / f"checkpoint-{step}"
                shutil.copytree(original, target)
                marker = runtime._checkpoint_marker(step)
                if step == 92:
                    marker["output_dir"] = str(foreign)
                elif step == 93:
                    (target / "scheduler.pt").unlink()
                else:
                    marker["signature_sha256"] = "a different training recipe"
                (target / CHECKPOINT_MARKER).write_text(json.dumps(marker))
            trainer.args.save_total_limit = 1
            trainer.args.protected_stage2_source_stage2_checkpoint = str(original)
            runtime._rotate_checkpoints(root / "checkpoint-2")
            self.assertTrue(original.is_dir())
            trainer.args.protected_stage2_source_stage2_checkpoint = None
            runtime._rotate_checkpoints(root / "checkpoint-2")
            self.assertFalse(original.exists())
            self.assertTrue(foreign.is_dir())
            self.assertTrue((root / "ce-only-checkpoint-0").is_dir())
            for step in range(90, 95):
                self.assertTrue((root / f"checkpoint-{step}").exists())
            self.assertTrue((root / "checkpoint-1").is_dir())
            self.assertTrue((root / "checkpoint-2").is_dir())

    def test_retention_never_runs_on_existing_destination_or_failed_atomic_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = stage2_trainer(copy.deepcopy(self.initial), tmp, max_steps=2)
            trainer.train()
            runtime = trainer._protected_cka_runtime
            trainer.args.save_total_limit = 1
            with mock.patch.object(runtime, "_rotate_checkpoints") as rotate:
                with self.assertRaises(FileExistsError):
                    runtime.save_checkpoint()
                rotate.assert_not_called()
            trainer.state.global_step = 3
            failures = (
                mock.patch.object(trainer, "_save_optimizer_and_scheduler", side_effect=OSError("disk full")),
                mock.patch("llava.train.protected_cka_stage2.os.replace", side_effect=OSError("rename failed")),
            )
            for failure in failures:
                with failure, mock.patch.object(runtime, "_rotate_checkpoints") as rotate:
                    with self.assertRaises(OSError):
                        runtime.save_checkpoint()
                    rotate.assert_not_called()
                self.assertFalse((Path(tmp) / "checkpoint-3").exists())
                for step in (0, 1, 2):
                    self.assertTrue((Path(tmp) / f"checkpoint-{step}").is_dir())


if __name__ == "__main__":
    unittest.main()
