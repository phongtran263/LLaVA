"""CPU-only launch contract checks; never invoke model training."""

import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/7b/finetune_qwen2_5_3b_protected_pipeline_pair.sh"


class ProtectedPipelinePairScriptTests(unittest.TestCase):
    def run_script(self, *, arguments=(), **settings):
        env = dict(os.environ, DRY_RUN="1", WORLD_SIZE="1")
        env.update({key: str(value) for key, value in settings.items()})
        return subprocess.run(["bash", str(SCRIPT), *arguments], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)

    @staticmethod
    def commands(output):
        commands = []
        for line in output.splitlines():
            if "llava/train/train_mem.py" not in line:
                continue
            command = shlex.split(line)
            cli = command[command.index("llava/train/train_mem.py") + 1:]
            commands.append(dict(zip(cli[::2], cli[1::2])))
        return commands

    def test_dry_run_matches_recipe_without_existing_source_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "missing-stage1", Path(directory) / "stage2"
            result = self.run_script(STAGE1_PAIR_ROOT=source, STAGE2_PAIR_ROOT=output)
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = self.commands(result.stdout)
            self.assertEqual(len(commands), 2)
            ignored = {"--pretrain_mm_mlp_adapter", "--protected_stage2_continuation", "--output_dir", "--run_name"}
            self.assertEqual({k: v for k, v in commands[0].items() if k not in ignored},
                             {k: v for k, v in commands[1].items() if k not in ignored})
            expected = {
                "--model_name_or_path": "Qwen/Qwen2.5-3B-Instruct", "--max_steps": "5182",
                "--protected_stage2_min_phase_a_steps": "4832", "--protected_stage2_max_phase_a_steps": "4832",
                "--protected_stage2_min_phase_b_steps": "350", "--protected_stage2_max_phase_b_steps": "350",
                "--per_device_train_batch_size": "8", "--gradient_accumulation_steps": "16",
                "--protected_stage2_validation_max_samples": "2002", "--seed": "42",
                "--gradient_checkpointing": "True", "--dataloader_num_workers": "0",
                "--learning_rate": "2e-5", "--protected_stage2_refine_lr": "1e-5",
                "--protected_stage2_eval_interval": "100", "--save_total_limit": "2",
                "--cka_loss_projector_weight": "0.1", "--cka_loss_final_hidden_weight": "0.1",
                "--cka_loss_layers": "final", "--protected_stage2_margin": "0.02",
                "--protected_stage2_kl_weight": "0.1", "--protected_stage2_warmup_steps": "100",
                "--protected_stage2_ce_tolerance": "0.02", "--protected_stage2_text_kl_budget": "0.05",
                "--protected_stage2_start_phase_b": "False", "--protected_stage2": "True",
            }
            for mode, command in zip(("ce_only", "protected"), commands):
                for key, value in expected.items():
                    self.assertEqual(command[key], value, key)
                self.assertEqual(command["--protected_stage2_continuation"], mode)
                self.assertEqual(command["--pretrain_mm_mlp_adapter"], str(source / mode / "checkpoint-350/mm_projector.bin"))
                self.assertEqual(command["--output_dir"], str(output / mode))
            self.assertFalse(output.exists())

    def test_stale_source_resume_anchor_backend_and_recipe_environment_is_cleared(self):
        result = self.run_script(
            SOURCE_STAGE1_CHECKPOINT="stale-stage1", SOURCE_STAGE2_CHECKPOINT="stale-stage2",
            RESUME_FROM_CHECKPOINT="stale-resume", PRETRAIN_ADAPTER="stale-adapter",
            CKA_ANCHOR_LAYER="12", CKA_VISION_ANCHOR_LAYER="20", CKA_PROJECTOR_VISION_ANCHOR_LAYER="24",
            CKA_FINAL_VISION_ANCHOR_LAYER="24", DEEPSPEED="old", DEEPSPEED_CONFIG="old", GPU_INCLUDE="localhost:0,1",
            STAGE2_START_PHASE_B="True", STAGE2_CONTINUATION="ce_kl", STAGE2_DRIFT_BUDGET="0.0001",
            STAGE2_VALIDATION_DATA_PATH="old.json", STAGE2_VALIDATION_IMAGE_FOLDER="old-images",
            MODEL_NAME_OR_PATH="wrong-model", MAX_STEPS="1", PER_DEVICE_TRAIN_BATCH_SIZE="99",
            GRADIENT_ACCUMULATION_STEPS="1", SEED="17", CKA_LOSS_ENABLED="False", CKA_LAYERS="all",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in self.commands(result.stdout):
            for key in ("--protected_stage2_source_stage2_checkpoint", "--resume_from_checkpoint",
                        "--cka_loss_anchor_layer", "--cka_loss_vision_anchor_layer",
                        "--cka_loss_projector_vision_anchor_layer", "--cka_loss_final_vision_anchor_layer",
                        "--protected_stage2_drift_budget", "--protected_stage2_validation_data_path"):
                self.assertNotIn(key, command)
            self.assertEqual(command["--max_steps"], "5182")
            self.assertEqual(command["--seed"], "42")
            self.assertEqual(command["--cka_loss"], "True")
            self.assertEqual(command["--cka_loss_layers"], "final")

    def test_rejects_cli_overrides_invalid_dry_run_and_existing_output_before_launch(self):
        self.assertNotEqual(self.run_script(arguments=("--max_steps", "1")).returncode, 0)
        self.assertNotEqual(self.run_script(DRY_RUN="yes").returncode, 0)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "protected").mkdir()
            result = self.run_script(DRY_RUN="0", STAGE2_PAIR_ROOT=directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing existing", result.stderr)
            self.assertNotIn("llava/train/train_mem.py", result.stdout)

    def make_source_pair(self, root):
        import torch
        common = dict(version=1, phase="B", phase_b_start=0, finished_reason=None,
                      reference={"weight": torch.ones(1)},
                      source_provenance={"checkpoint": "/verified/source", "global_step": 2173,
                                         "signature": {"identity": "original-ce-only"}},
                      signature={"model": {"_name_or_path": "Qwen/Qwen2.5-3B-Instruct"},
                                 "train_dataset": "train-hash", "eval_dataset": "val-hash",
                                 "config": {"max_steps": 350, "protected_cka_split_fingerprint": "split-hash"}})
        for mode in ("ce_only", "protected"):
            checkpoint = root / mode / "checkpoint-350"
            checkpoint.mkdir(parents=True)
            state = copy.deepcopy(common)
            state["refinement_accepted"] = mode == "protected"
            state["signature"]["config"]["protected_cka_continuation"] = mode
            torch.save(state, checkpoint / "protected_cka_state.pt")
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 350}))
            (checkpoint / "mm_projector.bin").write_bytes(b"fixture-weights-not-loaded")

    def test_preflight_accepts_valid_pair_and_never_reads_projector_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source_pair(root / "stage1")
            # The shim executes the real metadata validator but replaces model
            # training with argument printing. No training library is imported
            # for either prospective GPU command.
            shim = root / "python-check-only"
            shim.write_text(f"#!{sys.executable}\nimport os, shlex, sys\n"
                            f"if sys.argv[1] == '-': os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
                            "print(shlex.join(sys.argv))\n")
            shim.chmod(0o755)
            result = self.run_script(DRY_RUN="0", STAGE1_PAIR_ROOT=root / "stage1",
                                     STAGE2_PAIR_ROOT=root / "stage2", PYTHON_BIN=shim)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Stage-1 pair verified", result.stdout)
            self.assertEqual(len(self.commands(result.stdout)), 2)
            self.assertFalse((root / "stage2").exists())

    def test_preflight_rejects_failed_gate_incomplete_step_and_different_source(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source_pair(root / "stage1")
            checkpoint = root / "stage1/protected/checkpoint-350"
            original = torch.load(checkpoint / "protected_cka_state.pt", weights_only=False)
            for mismatch in ("gate", "step", "source", "split", "learning_rate", "seed", "per_device_train_batch_size"):
                with self.subTest(mismatch=mismatch):
                    state = copy.deepcopy(original)
                    step = 350
                    if mismatch == "gate":
                        state["refinement_accepted"] = False
                    elif mismatch == "step":
                        step = 349
                    elif mismatch == "source":
                        state["source_provenance"]["checkpoint"] = "/different/source"
                    elif mismatch == "split":
                        state["signature"]["config"]["protected_cka_split_fingerprint"] = "different"
                    else:
                        state["signature"]["config"][mismatch] = 17
                    torch.save(state, checkpoint / "protected_cka_state.pt")
                    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
                    result = self.run_script(DRY_RUN="0", STAGE1_PAIR_ROOT=root / "stage1",
                                             STAGE2_PAIR_ROOT=root / "stage2", PYTHON_BIN=sys.executable)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("llava/train/train_mem.py", result.stdout)
                    self.assertFalse((root / "stage2").exists())


if __name__ == "__main__":
    unittest.main()
