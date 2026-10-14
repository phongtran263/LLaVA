"""Real Trainer CPU/offline lifecycle checks for protected Stage 1 refinement."""

import copy
import json
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from accelerate.state import AcceleratorState
from transformers import TrainerCallback, default_data_collator

from llava.train.llava_trainer import LLaVATrainer
from llava.train.protected_cka_refinement import PlateauMonitor, ProtectedCKARefinement, STATE_NAME
from llava.train.train import TrainingArguments
from tests.test_protected_cka_forward import make_protected_model, multimodal_batch


class FixedCaptionDataset(torch.utils.data.Dataset):
    """Precomputed images, with a JSON identity independent of the output path."""

    def __init__(self, count, seed):
        self.list_data_dict = [{"id": f"{seed}-{index}"} for index in range(count)]
        self.rows = []
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for index in range(count):
                batch = multimodal_batch()
                row = {key: value[0].clone() for key, value in batch.items()}
                row["input_ids"][-1] = row["labels"][-1] = 4 + index % 10
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {key: value.clone() for key, value in self.rows[index].items()}


def runtime_trainer(model, output_dir, *, train_count=12, **overrides):
    values = dict(
        output_dir=str(output_dir), use_cpu=True, report_to=[], max_steps=4,
        per_device_train_batch_size=1, per_device_eval_batch_size=2,
        gradient_accumulation_steps=2, learning_rate=0.003,
        lr_scheduler_type="linear", optim="adamw_torch", max_grad_norm=0.4,
        save_strategy="no", eval_strategy="no", logging_steps=1,
        disable_tqdm=True, dataloader_pin_memory=False, dataloader_num_workers=0,
        remove_unused_columns=False, seed=29, data_seed=61,
        protected_cka_refinement=True, protected_cka_start_phase_b=True,
        protected_cka_refine_lr=0.002, protected_cka_warmup_steps=0,
        protected_cka_ce_gradient_calibration_batches=2, protected_cka_eval_interval=1,
        protected_cka_min_phase_a_steps=2, protected_cka_max_phase_a_steps=4,
        protected_cka_min_phase_b_steps=0,
        protected_cka_plateau_window=1, protected_cka_plateau_patience=1,
        protected_cka_plateau_min_delta=100., protected_cka_max_retries=1,
        protected_cka_ce_tolerance=10., protected_cka_text_kl_budget=10.,
        protected_cka_drift_budget=10., protected_cka_kl_chunk_size=2,
    )
    values.update(overrides)
    args = TrainingArguments(**values)
    args.tune_mm_mlp_adapter = True
    model.config.tune_mm_mlp_adapter = True
    trainer = LLaVATrainer(
        model=model, args=args, train_dataset=FixedCaptionDataset(train_count, 111),
        eval_dataset=FixedCaptionDataset(3, 999), data_collator=default_data_collator,
    )
    # Keep test output concise without changing Trainer callback/state behavior.
    trainer.log = lambda logs, *args, **kwargs: None
    return trainer


def load_payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_nested_equal(test, actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            assert_nested_equal(test, actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        test.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            assert_nested_equal(test, left, right)
    else:
        test.assertEqual(actual, expected)


class ProtectedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        # Accelerate's global state otherwise lets this module's BF16 test
        # silently enable AMP for later FP32 trainers in the same test process.
        AcceleratorState._reset_state(reset_partial_state=True)
        self.precision_environment = mock.patch.dict(os.environ, {"ACCELERATE_MIXED_PRECISION": "no"})
        self.precision_environment.start()
        torch.manual_seed(73)
        self.initial, _ = make_protected_model()

    def tearDown(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        self.precision_environment.stop()

    def assert_trainers_equal(self, left, right):
        assert_nested_equal(self, left.model.state_dict(), right.model.state_dict())
        assert_nested_equal(self, left.optimizer.state_dict(), right.optimizer.state_dict())
        assert_nested_equal(self, left.lr_scheduler.state_dict(), right.lr_scheduler.state_dict())
        self.assertEqual(left.state.global_step, right.state.global_step)
        self.assertEqual(left.state.epoch, right.state.epoch)

    def test_plateau_monitor_uses_window_patience_and_eligibility(self):
        monitor = PlateauMonitor(window=2, patience=2, min_delta=0.01)
        self.assertFalse(monitor.update(3.))
        self.assertFalse(monitor.update(2.))
        self.assertFalse(monitor.update(2., eligible=False))
        self.assertFalse(monitor.update(2.))
        self.assertTrue(monitor.update(2.))
        self.assertFalse(monitor.update(1.))
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            monitor.update(float("nan"))

    def test_full_resume_restores_projector_optimizer_scheduler_ema_and_fixed_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "full")
            full.train()
            checkpoint = Path(tmp) / "full/checkpoint-2"
            for name in ("mm_projector.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                         "trainer_state.json", STATE_NAME):
                self.assertTrue((checkpoint / name).is_file(), name)
            resumed = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "resumed")
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assert_trainers_equal(full, resumed)
            assert_nested_equal(self, full._protected_cka_runtime.state_dict(),
                                resumed._protected_cka_runtime.state_dict())
            for name, value in self.initial.state_dict().items():
                if "mm_projector" not in name:
                    torch.testing.assert_close(full.model.state_dict()[name], value, rtol=0, atol=0)
            assert_nested_equal(self, full._protected_cka_runtime.reference.state_dict(),
                                self.initial.get_model().mm_projector.state_dict())
            self.assertGreater(full._protected_cka_runtime.ce_grad_ref, 0.)
            self.assertEqual(full._protected_cka_runtime.accum_count, 0)

    def test_phase_a_plateau_resets_optimizer_lr_and_preserves_resume_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "full",
                                   protected_cka_start_phase_b=False)
            full.train()
            runtime = full._protected_cka_runtime
            self.assertEqual(runtime.phase_b_start, 2)
            self.assertEqual(runtime.phase, "B")
            transition = Path(tmp) / "full/checkpoint-2"
            self.assertTrue((Path(tmp) / "full/ce-only-checkpoint-2/mm_projector.bin").is_file())
            optimizer = load_payload(transition / "optimizer.pt")
            self.assertEqual(optimizer["state"], {})
            for group in optimizer["param_groups"]:
                self.assertEqual(group["lr"], full.args.protected_cka_refine_lr)
                self.assertEqual(group["initial_lr"], full.args.protected_cka_refine_lr)
            reference = load_payload(transition / STATE_NAME)["reference"]
            assert_nested_equal(self, runtime.reference.state_dict(), reference)
            resumed = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "resumed",
                                      protected_cka_start_phase_b=False)
            resumed.train(resume_from_checkpoint=str(Path(tmp) / "full/checkpoint-3"))
            self.assert_trainers_equal(full, resumed)
            self.assertEqual(resumed._protected_cka_runtime.phase_b_start, 2)
            phase_a_resumed = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "phase-a-resumed",
                                              protected_cka_start_phase_b=False)
            phase_a_resumed.train(resume_from_checkpoint=str(Path(tmp) / "full/checkpoint-1"))
            self.assert_trainers_equal(full, phase_a_resumed)
            assert_nested_equal(self, runtime.state_dict(), phase_a_resumed._protected_cka_runtime.state_dict())

    def test_phase_a_limit_stops_without_forcing_refinement(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = runtime_trainer(copy.deepcopy(self.initial), tmp,
                                      protected_cka_start_phase_b=False,
                                      protected_cka_max_phase_a_steps=2,
                                      protected_cka_plateau_patience=100)
            result = trainer.train()
            runtime = trainer._protected_cka_runtime
            self.assertEqual(result.global_step, 2)
            self.assertEqual(runtime.phase, "A")
            self.assertIsNone(runtime.reference)
            self.assertEqual(runtime.finished_reason, "phase_a_limit_without_plateau")

    def test_budget_reservation_validates_at_exact_off_cadence_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = runtime_trainer(copy.deepcopy(self.initial), tmp, max_steps=2173,
                                      protected_cka_start_phase_b=False,
                                      protected_cka_max_phase_a_steps=2173,
                                      protected_cka_min_phase_a_steps=1500,
                                      protected_cka_min_phase_b_steps=350,
                                      protected_cka_eval_interval=100,
                                      protected_cka_plateau_window=3,
                                      protected_cka_plateau_patience=3)
            runtime = ProtectedCKARefinement(trainer)
            with mock.patch.object(runtime, "validate", return_value={"ce": 3.}) as validate, \
                    mock.patch.object(runtime, "enter_phase_b") as enter, \
                    mock.patch.object(runtime, "save_checkpoint"):
                trainer.state.global_step = 1822
                runtime.step_end(trainer.control)
                validate.assert_not_called()
                trainer.state.global_step = 1823
                runtime.step_end(trainer.control)
                validate.assert_called_once()
                enter.assert_called_once_with(reason="budget_reservation")

    def test_reservation_and_plateau_both_obey_minimum_phase_a(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = runtime_trainer(copy.deepcopy(self.initial), tmp, max_steps=10,
                                      protected_cka_start_phase_b=False,
                                      protected_cka_max_phase_a_steps=10,
                                      protected_cka_min_phase_a_steps=7,
                                      protected_cka_min_phase_b_steps=3)
            runtime = ProtectedCKARefinement(trainer)
            with mock.patch.object(runtime, "validate", return_value={"ce": 3.}), \
                    mock.patch.object(runtime, "enter_phase_b") as enter, \
                    mock.patch.object(runtime, "save_checkpoint"):
                for step in range(1, 7):
                    trainer.state.global_step = step
                    runtime.step_end(trainer.control)
                enter.assert_not_called()
                trainer.state.global_step = 7
                runtime.step_end(trainer.control)
                enter.assert_called_once()

    def test_budget_transition_snapshot_and_resume_preserve_reference_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = dict(max_steps=5, protected_cka_start_phase_b=False,
                           protected_cka_max_phase_a_steps=5,
                           protected_cka_min_phase_a_steps=2,
                           protected_cka_min_phase_b_steps=2,
                           protected_cka_eval_interval=2,
                           protected_cka_plateau_patience=100)
            full = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "full", **options)
            full.train()
            runtime = full._protected_cka_runtime
            self.assertEqual(runtime.phase_b_start, 3)
            self.assertEqual(runtime.transition_reason, "budget_reservation")
            reference = load_payload(Path(tmp) / "full/ce-only-checkpoint-3/mm_projector.bin")
            assert_nested_equal(self, runtime.reference.state_dict(),
                                {key.split("mm_projector.", 1)[-1]: value for key, value in reference.items()})
            self.assertTrue(runtime.refinement_accepted)
            resumed = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "resume", **options)
            resumed.train(resume_from_checkpoint=str(Path(tmp) / "full/checkpoint-3"))
            self.assert_trainers_equal(full, resumed)
            assert_nested_equal(self, runtime.state_dict(), resumed._protected_cka_runtime.state_dict())

    def _completed_legacy_source(self, directory):
        trainer = runtime_trainer(copy.deepcopy(self.initial), directory, max_steps=2,
                                  protected_cka_start_phase_b=False,
                                  protected_cka_max_phase_a_steps=2,
                                  protected_cka_plateau_patience=100)
        trainer.train()
        source = Path(directory) / "checkpoint-2"
        payload = load_payload(source / STATE_NAME)
        # Emulate the old completed CE-only run: old signatures predate the new
        # policy fields and the source horizon is not the continuation horizon.
        payload["signature"]["config"].pop("protected_cka_min_phase_b_steps")
        payload["signature"]["config"].pop("protected_cka_continuation")
        payload["signature"]["config"]["max_steps"] = 2173
        payload["signature"]["config"]["protected_cka_min_phase_a_steps"] = 200
        torch.save(payload, source / STATE_NAME)
        trainer.state.global_step = 2173
        trainer.state.save_to_json(str(source / "trainer_state.json"))
        return source, trainer

    def test_completed_legacy_phase_a_source_starts_fresh_b_and_fair_ce_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, ce_source = self._completed_legacy_source(Path(tmp) / "source")
            common = dict(max_steps=3, protected_cka_source_stage1_checkpoint=str(source),
                          protected_cka_max_phase_a_steps=3,
                          protected_cka_min_phase_b_steps=350,
                          protected_cka_min_phase_a_steps=1500)
            ce = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "ce",
                                 protected_cka_continuation="ce_only", **common)
            protected = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "protected",
                                        protected_cka_lambda_max=0., protected_cka_kl_weight=0., **common)
            ce.train()
            protected.train()
            self.assert_trainers_equal(ce, protected)
            for trainer in (ce, protected):
                runtime = trainer._protected_cka_runtime
                self.assertEqual(runtime.phase_b_start, 0)
                self.assertEqual(runtime.transition_reason, "explicit_continuation")
                self.assertEqual(runtime.source_provenance["global_step"], 2173)
                assert_nested_equal(self, runtime.reference.state_dict(),
                                    ce_source.model.get_model().mm_projector.state_dict())
                initial_optimizer = load_payload(Path(trainer.args.output_dir) / "checkpoint-0/optimizer.pt")
                self.assertEqual(initial_optimizer["state"], {})
                for group in initial_optimizer["param_groups"]:
                    self.assertEqual(group["initial_lr"], trainer.args.protected_cka_refine_lr)

    def test_source_rejects_changed_model_data_split_or_refined_weights_before_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = self._completed_legacy_source(Path(tmp) / "source")
            original = load_payload(source / STATE_NAME)
            for mismatch in ("model", "train_dataset", "eval_dataset", "split", "phase"):
                with self.subTest(mismatch=mismatch):
                    payload = copy.deepcopy(original)
                    if mismatch == "phase":
                        payload["phase"] = "B"
                    elif mismatch == "split":
                        payload["signature"]["config"]["protected_cka_split_fingerprint"] = "other"
                    else:
                        payload["signature"][mismatch] = "other"
                    torch.save(payload, source / STATE_NAME)
                    trainer = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / mismatch,
                                              protected_cka_source_stage1_checkpoint=str(source))
                    with self.assertRaisesRegex(ValueError, "provenance|CE-only"):
                        trainer.train()
                    self.assertEqual(trainer.state.global_step, 0)
                    self.assertIsNone(trainer.optimizer)
                    assert_nested_equal(self, trainer.model.state_dict(), self.initial.state_dict())

    def test_source_checks_full_frozen_config_beyond_short_model_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = self._completed_legacy_source(Path(tmp) / "source")
            config_file = source / "config.json"
            with config_file.open() as handle:
                original = json.load(handle)
            changed = {"model_type": "different_backbone_family",
                       "num_attention_heads": self.initial.config.num_attention_heads + 1,
                       "mm_vision_select_feature": "cls_patch",
                       "llava_conversation_version": "different-prompt-template",
                       "eos_token_id": 99, "tokenizer_padding_side": "left"}
            for key, value in changed.items():
                with self.subTest(key=key):
                    config = dict(original, **{key: value})
                    with config_file.open("w") as handle:
                        json.dump(config, handle)
                    trainer = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / key,
                                              protected_cka_source_stage1_checkpoint=str(source))
                    with self.assertRaisesRegex(ValueError, key):
                        trainer.train()
                    self.assertEqual(trainer.state.global_step, 0)
                    self.assertIsNone(trainer.optimizer)
                    assert_nested_equal(self, trainer.model.state_dict(), self.initial.state_dict())

    def test_named_source_save_pretrained_roundtrip_uses_verified_identity_only_if_omitted(self):
        self.initial.config._name_or_path = "test/nonempty-frozen-backbone"
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = self._completed_legacy_source(Path(tmp) / "source")
            config_file = source / "config.json"
            with config_file.open() as handle:
                serialized = json.load(handle)
            # Real save_pretrained in Transformers 4.51.3 removes this field;
            # empty-name toy models did not exercise the production failure.
            self.assertNotIn("_name_or_path", serialized)
            self.assertNotIn("name_or_path", serialized)
            trainer = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "continuation",
                                      protected_cka_source_stage1_checkpoint=str(source))
            trainer.train()
            self.assertEqual(trainer.state.global_step, trainer.args.max_steps)
            self.assertEqual(trainer._protected_cka_runtime.source_provenance["signature"]["model"]["_name_or_path"],
                             self.initial.config._name_or_path)
            for field in ("_name_or_path", "name_or_path"):
                with self.subTest(field=field):
                    with config_file.open("w") as handle:
                        json.dump(dict(serialized, **{field: "different/frozen-backbone"}), handle)
                    rejected = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / field,
                                               protected_cka_source_stage1_checkpoint=str(source))
                    with self.assertRaisesRegex(ValueError, "name_or_path"):
                        rejected.train()
                    self.assertEqual(rejected.state.global_step, 0)
                    self.assertIsNone(rejected.optimizer)
                    assert_nested_equal(self, rejected.model.state_dict(), self.initial.state_dict())

    def test_signature_records_actual_policy_inherited_by_unpickled_old_dataclass(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "full")
            full.train()
            source = Path(tmp) / "full/checkpoint-2"
            payload = load_payload(source / STATE_NAME)
            payload["signature"]["config"].pop("protected_cka_min_phase_b_steps")
            torch.save(payload, source / STATE_NAME)
            trainer = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "changed")
            # torch-loaded older dataclasses inherit newly-added class defaults
            # even though vars(instance) does not contain those fields.
            del trainer.args.__dict__["protected_cka_min_phase_b_steps"]
            del trainer.args.__dict__["protected_cka_continuation"]
            self.assertEqual(trainer.args.protected_cka_min_phase_b_steps, 350)
            runtime = ProtectedCKARefinement(trainer)
            self.assertEqual(runtime.signature["config"]["protected_cka_min_phase_b_steps"], 350)
            self.assertEqual(runtime.signature["config"]["protected_cka_continuation"], "protected")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                runtime.load_checkpoint(source)

    def test_legacy_full_resume_only_normalizes_absent_fields_to_old_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "full")
            full.train()
            source = Path(tmp) / "full/checkpoint-2"
            payload = load_payload(source / STATE_NAME)
            payload["signature"]["config"].pop("protected_cka_min_phase_b_steps")
            payload["signature"]["config"].pop("protected_cka_continuation")
            torch.save(payload, source / STATE_NAME)
            resumed = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "resume")
            resumed.train(resume_from_checkpoint=str(source))
            self.assert_trainers_equal(full, resumed)
            changed = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "changed",
                                      protected_cka_min_phase_b_steps=1)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                changed.train(resume_from_checkpoint=str(source))

    def test_ce_only_gate_is_diagnostic_but_protected_gate_still_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            common = dict(max_steps=3, protected_cka_max_retries=0)
            ce = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "ce",
                                 protected_cka_continuation="ce_only", **common)
            logs = []
            ce.log = lambda record, *args, **kwargs: logs.append(record)
            protected = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "protected", **common)
            with mock.patch.object(ProtectedCKARefinement, "gate", return_value=False):
                ce_result = ce.train()
                protected_result = protected.train()
            self.assertEqual(ce_result.global_step, 3)
            self.assertEqual(ce._protected_cka_runtime.retries, 0)
            self.assertFalse(ce._protected_cka_runtime.refinement_accepted)
            self.assertTrue((Path(tmp) / "ce/checkpoint-3").is_dir())
            self.assertTrue(any(record.get("protected/gate_pass") == 0 and
                                record.get("protected/gate_diagnostic_only") == 1 for record in logs))
            self.assertEqual(protected_result.global_step, 0)
            self.assertEqual(protected_result.metrics["protected_requested_phase_b_steps"], 3)
            self.assertEqual(protected_result.metrics["protected_completed_phase_b_steps"], 0)
            self.assertEqual(protected_result.metrics["protected_accepted_refinement_steps"], 0)
            assert_nested_equal(self, protected.model.state_dict(), self.initial.state_dict())

    def test_ce_only_control_still_refuses_nonfinite_validation(self):
        original_validate = ProtectedCKARefinement.validate

        def nonfinite_after_update(runtime):
            metrics = original_validate(runtime)
            if runtime.trainer.state.global_step:
                metrics["text_kl"] = float("nan")
            return metrics

        with tempfile.TemporaryDirectory() as tmp:
            ce = runtime_trainer(copy.deepcopy(self.initial), tmp, max_steps=2,
                                 protected_cka_max_retries=0, protected_cka_continuation="ce_only")
            with mock.patch.object(ProtectedCKARefinement, "validate", nonfinite_after_update):
                result = ce.train()
            self.assertEqual(result.global_step, 0)
            self.assertFalse(ce._protected_cka_runtime.refinement_accepted)
            self.assertFalse((Path(tmp) / "checkpoint-1").exists())
            assert_nested_equal(self, ce.model.state_dict(), self.initial.state_dict())

    def test_ce_kl_control_is_not_reported_as_accepted_cka_refinement(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = runtime_trainer(copy.deepcopy(self.initial), tmp, max_steps=2,
                                      protected_cka_continuation="ce_kl")
            trainer.train()
            runtime = trainer._protected_cka_runtime
            self.assertEqual(trainer.state.global_step, 2)
            self.assertFalse(runtime.refinement_accepted)
            self.assertEqual(runtime.budget_metrics()["accepted_refinement_steps"], 0)
            self.assertFalse(load_payload(Path(tmp) / "checkpoint-2" / STATE_NAME)["refinement_accepted"])

    def test_divisible_accumulation_resume_across_epochs_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "full", train_count=4,
                                   max_steps=5, protected_cka_max_phase_a_steps=5)
            full.train()
            resumed = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "resumed", train_count=4,
                                      max_steps=5, protected_cka_max_phase_a_steps=5)
            resumed.train(resume_from_checkpoint=str(Path(tmp) / "full/checkpoint-3"))
            self.assert_trainers_equal(full, resumed)
            assert_nested_equal(self, full._protected_cka_runtime.state_dict(),
                                resumed._protected_cka_runtime.state_dict())
            self.assertEqual(full.state.epoch, 2.5)

    def test_partial_accumulation_epoch_is_rejected_before_any_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = runtime_trainer(copy.deepcopy(self.initial), tmp, train_count=5)
            with self.assertRaisesRegex(ValueError, "accum|divis"):
                trainer.train()
            self.assertEqual(trainer.state.global_step, 0)
            assert_nested_equal(self, trainer.model.state_dict(), self.initial.state_dict())

    def test_incomplete_checkpoint_cannot_silently_resume_fresh_optimizer_or_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            trained = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "full", max_steps=2,
                                      protected_cka_max_phase_a_steps=2)
            trained.train()
            source = Path(tmp) / "full/checkpoint-1"
            for missing in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                            "mm_projector.bin", STATE_NAME):
                with self.subTest(missing=missing):
                    broken = Path(tmp) / ("missing-" + missing.replace(".", "-"))
                    shutil.copytree(source, broken)
                    (broken / missing).unlink()
                    resumed = runtime_trainer(copy.deepcopy(self.initial), broken / "new-output", max_steps=2,
                                              protected_cka_max_phase_a_steps=2)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        resumed.train(resume_from_checkpoint=str(broken))
                    self.assertEqual(resumed.state.global_step, 0)

    def test_bf16_retry_exhaustion_restores_all_safe_state_and_reduces_cap(self):
        class ForwardDepth(TrainerCallback):
            def __init__(self):
                self.depths = []

            def on_train_begin(self, args, state, control, model=None, **kwargs):
                forward, depth = model.forward, 0
                while hasattr(forward, "__wrapped__"):
                    depth += 1
                    forward = forward.__wrapped__
                self.depths.append(depth)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = runtime_trainer(copy.deepcopy(self.initial), tmp, bf16=True,
                                      max_steps=2, protected_cka_max_phase_a_steps=2)
            depths = ForwardDepth()
            trainer.add_callback(depths)
            with mock.patch.object(ProtectedCKARefinement, "gate", return_value=False) as gate:
                result = trainer.train()
            self.assertEqual(gate.call_count, 2)
            self.assertEqual(result.global_step, 0)
            self.assertEqual(depths.depths, [depths.depths[0]] * 2)
            runtime = trainer._protected_cka_runtime
            self.assertEqual(runtime.retries, 2)
            self.assertEqual(runtime.aux_ratio, trainer.args.protected_cka_aux_ratio * 0.25)
            self.assertEqual(runtime.finished_reason, "gate_retry_exhausted")
            assert_nested_equal(self, trainer.model.state_dict(), self.initial.state_dict())
            self.assertIsNone(runtime.ema)
            checkpoint = Path(tmp) / "checkpoint-0"
            assert_nested_equal(self, trainer.optimizer.state_dict(), load_payload(checkpoint / "optimizer.pt"))
            assert_nested_equal(self, trainer.lr_scheduler.state_dict(), load_payload(checkpoint / "scheduler.pt"))
            torch.testing.assert_close(torch.get_rng_state(), load_payload(checkpoint / "rng_state.pth")["cpu"])
            self.assertFalse((Path(tmp) / "checkpoint-1").exists())
            saved = load_payload(checkpoint / STATE_NAME)
            self.assertEqual(saved["retries"], 2)
            self.assertEqual(saved["aux_ratio"], runtime.aux_ratio)

    def test_retry_replays_same_batches_and_matches_clean_run_with_reduced_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            retried = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "retry",
                                      max_steps=3, protected_cka_max_phase_a_steps=3)
            calls = []

            def fail_once(runtime, metrics):
                calls.append(runtime.trainer.state.global_step)
                return len(calls) > 1

            with mock.patch.object(ProtectedCKARefinement, "gate", fail_once):
                retried.train()
            clean = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "clean",
                                    max_steps=3, protected_cka_max_phase_a_steps=3,
                                    protected_cka_aux_ratio=0.05)
            clean.train()
            self.assertEqual(calls, [1, 1, 2, 3])
            self.assert_trainers_equal(retried, clean)
            torch.testing.assert_close(retried._protected_cka_runtime.ema,
                                       clean._protected_cka_runtime.ema, rtol=0, atol=0)

    def test_zero_cka_and_kl_matches_ce_only_continuation(self):
        # An ordinary deterministic CE run is the baseline; disabled auxiliary
        # losses must not perturb its data order, optimizer, LR, or update count.
        for module in self.initial.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = 0.
        self.initial.config.attention_dropout = 0.
        with tempfile.TemporaryDirectory() as tmp:
            ce = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "ce",
                                 protected_cka_refinement=False, learning_rate=0.002)
            protected = runtime_trainer(copy.deepcopy(self.initial), Path(tmp) / "protected",
                                        learning_rate=0.002, protected_cka_lambda_max=0.,
                                        protected_cka_kl_weight=0.)
            ce.train()
            protected.train()
            self.assert_trainers_equal(ce, protected)

    def test_early_epoch_stop_still_runs_gate_and_returns_safe_projector(self):
        class StopAfterEpoch(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                control.should_training_stop = True

        with tempfile.TemporaryDirectory() as tmp:
            trainer = runtime_trainer(copy.deepcopy(self.initial), tmp, train_count=4,
                                      max_steps=5, protected_cka_eval_interval=99,
                                      protected_cka_max_retries=0)
            trainer.add_callback(StopAfterEpoch())
            with mock.patch.object(ProtectedCKARefinement, "gate", return_value=False) as gate:
                result = trainer.train()
            self.assertEqual(gate.call_count, 1)
            self.assertEqual(result.global_step, 0)
            assert_nested_equal(self, trainer.model.state_dict(), self.initial.state_dict())
            self.assertFalse((Path(tmp) / "checkpoint-2").exists())

    def test_calibration_preserves_parameters_gradients_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = runtime_trainer(copy.deepcopy(self.initial), tmp)
            runtime = ProtectedCKARefinement(trainer)
            model_before = copy.deepcopy(trainer.model.state_dict())
            torch_state, python_state, numpy_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
            runtime.calibrate()
            assert_nested_equal(self, trainer.model.state_dict(), model_before)
            self.assertTrue(all(parameter.grad is None for parameter in trainer.model.parameters()))
            torch.testing.assert_close(torch.get_rng_state(), torch_state)
            self.assertEqual(random.getstate(), python_state)
            after_numpy = np.random.get_state()
            self.assertEqual(after_numpy[0], numpy_state[0])
            np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
            self.assertEqual(after_numpy[2:], numpy_state[2:])


if __name__ == "__main__":
    unittest.main()
