"""Tiny CPU fixtures for the read-only metadata/header completion audit."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/audit_protected_pipeline.py"
SPEC = importlib.util.spec_from_file_location("audit_protected_pipeline", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class PipelineAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stage1 = self.root / "stage1"
        self.stage2 = self.root / "stage2"
        self.source = self.root / "ce-only-source" / "checkpoint-2173"
        self.source.mkdir(parents=True)
        self.weights = {"weight": torch.ones(1)}
        torch.save(self.weights, self.source / "mm_projector.bin")
        self.states = {}
        for stage, root in ((1, self.stage1), (2, self.stage2)):
            for mode in ("ce_only", "protected"):
                self._fixture(root, stage, mode)

    def _fixture(self, root, stage, mode):
        directory = root / mode
        step, start = (350, 0) if stage == 1 else (5182, 4832)
        checkpoint = directory / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        prefix = "protected_cka_" if stage == 1 else "protected_stage2_"
        config = {"max_steps": step, prefix + "continuation": mode, prefix + "split_fingerprint": f"stage{stage}-holdout",
                  "seed": 42, "per_device_train_batch_size": 16 if stage == 1 else 8,
                  "gradient_accumulation_steps": 16, "lr_scheduler_type": "cosine",
                  "learning_rate": 1e-5 if stage == 1 else 2e-5, prefix + "refine_lr": 1e-5,
                  prefix + "validation_max_samples": 1840 if stage == 1 else 2002}
        if stage == 2:
            config.update(protected_stage2_min_phase_a_steps=4832, protected_stage2_max_phase_a_steps=4832,
                          protected_stage2_max_phase_b_steps=350)
        state = dict(version=1, phase="B", phase_b_start=start, refinement_accepted=(mode == "protected"),
                     finished_reason=None if stage == 1 else "refinement_budget_complete", retries=0,
                     signature={"config": config, "model": {"_name_or_path": "Qwen/Qwen2.5-3B-Instruct"},
                                "train_dataset": f"stage{stage}-train", "eval_dataset": f"stage{stage}-validation"})
        if stage == 1:
            state.update(reference=self.weights, source_provenance={"checkpoint": str(self.source), "global_step": 2173,
                         "signature": {"config": {"max_steps": 2173}, "model": "source-identity"}})
        else:
            state.update(has_reference=True, source_provenance=None)
        state_path = checkpoint / ("protected_cka_state.pt" if stage == 1 else "protected_cka_stage2_state.pt")
        self.states[stage, mode] = state_path, state
        torch.save(state, state_path)
        for folder in (directory, checkpoint):
            (folder / "trainer_state.json").write_text(json.dumps({"global_step": step}))
            (folder / "config.json").write_text(json.dumps({"model_type": "llava_qwen"}))
        for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            torch.save({}, checkpoint / filename)
        if stage == 1:
            torch.save(self.weights, checkpoint / "mm_projector.bin")
            torch.save(self.weights, directory / "mm_projector.bin")
        else:
            torch.save(self.weights, checkpoint / "student_model.pt")
            torch.save(self.weights, checkpoint / "reference_model.pt")
            save_file(self.weights, str(directory / "model.safetensors"))

    def audit(self):
        return AUDIT.audit_pipeline(self.stage1, self.stage2)

    def update(self, stage, mode, fn):
        path, original = self.states[stage, mode]
        state = copy.deepcopy(original)
        fn(state)
        torch.save(state, path)

    def test_complete_matched_pipeline_reports_verified_scope_without_writes(self):
        before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in self.root.rglob("*") if path.is_file()}
        report = self.audit()
        self.assertTrue(report["complete"], report["errors"])
        self.assertIn("not verified", report["verification_scope"])
        for stage in ("stage1", "stage2"):
            self.assertTrue(report["stages"][stage]["matched_recorded_settings"])
            self.assertEqual(report["stages"][stage]["arms"]["protected"]["phase_b_steps"], 350)
        self.assertEqual(len(report["stages"]["stage1"]["arms"]["protected"]["source_projector_sha256"]), 64)
        after = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_any_additional_recorded_setting_difference_is_rejected(self):
        self.update(2, "protected", lambda state: state["signature"]["config"].update(optim="different"))
        report = self.audit()
        self.assertFalse(report["complete"])
        self.assertTrue(any("differ beyond continuation" in error for error in report["errors"]))

    def test_incomplete_wrong_boundary_and_failed_gate_are_rejected(self):
        for key, value in (("phase_b_start", 4800), ("refinement_accepted", False),
                           ("finished_reason", "gate_retry_exhausted")):
            with self.subTest(key=key):
                self.update(2, "protected", lambda state: state.update({key: value}))
                self.assertFalse(self.audit()["complete"])
        self.update(2, "protected", lambda state: None)
        (self.stage2 / "protected" / "trainer_state.json").write_text(json.dumps({"global_step": 5000}))
        self.assertFalse(self.audit()["complete"])

    def test_control_cannot_be_claimed_as_accepted_refinement(self):
        self.update(1, "ce_only", lambda state: state.update(refinement_accepted=True))
        self.assertFalse(self.audit()["complete"])

    def test_original_source_and_split_must_match(self):
        self.update(1, "protected", lambda state: state["source_provenance"]["signature"].update(model="different"))
        report = self.audit()
        self.assertFalse(report["complete"])
        self.assertTrue(any("source provenance differs" in error for error in report["errors"]))
        self.update(1, "protected", lambda state: None)
        self.update(2, "protected", lambda state: state["signature"].update(eval_dataset="other-holdout"))
        self.assertFalse(self.audit()["complete"])

    def test_sharded_safetensors_index_is_checked_without_tensor_loading(self):
        directory = self.stage2 / "protected"
        (directory / "model.safetensors").unlink()
        save_file({"a": torch.ones(1)}, str(directory / "part-1.safetensors"))
        save_file({"b": torch.ones(1)}, str(directory / "part-2.safetensors"))
        index = directory / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}))
        self.assertTrue(self.audit()["complete"])
        index.write_text(json.dumps({"weight_map": {"missing-name": "part-1.safetensors"}}))
        self.assertFalse(self.audit()["complete"])

    def test_truncated_export_cannot_pass_on_valid_checkpoint_metadata(self):
        path = self.stage2 / "protected" / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        self.assertFalse(self.audit()["complete"])

    def test_cli_emits_json_and_nonzero_for_incomplete_runs(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        for complete in (True, False):
            if not complete:
                (self.stage1 / "ce_only" / "mm_projector.bin").unlink()
            result = subprocess.run([sys.executable, str(SCRIPT), "--stage1-root", str(self.stage1),
                                     "--stage2-root", str(self.stage2)], env=env, capture_output=True, text=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["complete"], complete)
            self.assertEqual(result.returncode, 0 if complete else 1, result.stderr)


if __name__ == "__main__":
    unittest.main()
