import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/7b/pretrain_qwen2_5_3b_protected_cka.sh"


class Qwen3BProtectedLauncherTests(unittest.TestCase):
    def launch(self, *, ablation=False, extra_args=(), **updates):
        # Keep inherited run settings from silently selecting another model/run.
        environment = {key: value for key, value in os.environ.items() if key in (
            "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")}
        environment.update(DRY_RUN="1", MAX_STEPS="70000")
        if ablation:
            environment.update(SOURCE_STAGE1_CHECKPOINT="/source/ce-only/checkpoint-2173",
                               ABLATION_OUTPUT_ROOT="/output/stage1-comparison")
        environment.update(updates)
        script = SCRIPT.with_name("pretrain_qwen2_5_3b_protected_ablation.sh") if ablation else SCRIPT
        return subprocess.run(["bash", str(script), *extra_args], cwd="/tmp", env=environment,
                              text=True, capture_output=True)

    def test_default_stage1_plateau_and_safe_single_process(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        expected = {
            "--model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
            "--version": "plain", "--protected_cka_refinement": "True",
            "--protected_cka_start_phase_b": "False", "--learning_rate": "1e-3",
            "--protected_cka_refine_lr": "1e-5", "--max_steps": "70000",
            "--per_device_train_batch_size": "16", "--gradient_accumulation_steps": "16",
            "--gradient_checkpointing": "False", "--dataloader_num_workers": "0",
            "--protected_cka_eval_interval": "1000", "--report_to": "wandb",
            "--protected_cka_min_phase_a_steps": "1500", "--protected_cka_min_phase_b_steps": "350",
            "--cka_loss": "False", "--vsp_asymmetric_pcgrad": "False",
        }
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value, flag)
        self.assertIn("CUDA_VISIBLE_DEVICES=3", command)
        self.assertNotIn("--deepspeed", command)
        self.assertNotIn("--pretrain_mm_mlp_adapter", command)
        self.assertNotIn("--protected_cka_validation_data_path", command)

    def test_overrides_and_optional_resume_are_forwarded(self):
        result = self.launch(GPU_ID="1", MAX_STEPS="90000", EVAL_INTERVAL="500",
                             OUTPUT_DIR="/tmp/custom output", RUN_NAME="custom-run",
                             RESUME_FROM_CHECKPOINT="/tmp/checkpoint-2000")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        for flag, value in (("--max_steps", "90000"), ("--protected_cka_eval_interval", "500"),
                            ("--output_dir", "/tmp/custom output"), ("--run_name", "custom-run"),
                            ("--resume_from_checkpoint", "/tmp/checkpoint-2000")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", command)

    def test_budget_must_be_explicit_and_multiple_gpus_rejected(self):
        for changes in ({"MAX_STEPS": ""}, {"GPU_ID": "0,1"}):
            result = self.launch(**changes)
            self.assertNotEqual(result.returncode, 0)

    def test_continuations_are_a_matched_pair_from_verified_ce_only_source(self):
        result = self.launch(ablation=True, MAX_STEPS="350", REFINE_LR="3e-6", SEED="17",
                             PRETRAIN_ADAPTER="/ignored/old-initializer.bin", VALIDATION_MAX_SAMPLES="1840")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 2)
        self.assertEqual([cmd[cmd.index("--protected_cka_continuation") + 1] for cmd in commands],
                         ["ce_only", "protected"])
        for flag in ("--model_name_or_path", "--data_path", "--image_folder", "--learning_rate", "--max_steps",
                     "--lr_scheduler_type", "--warmup_ratio", "--seed", "--protected_cka_warmup_steps",
                     "--protected_cka_validation_max_samples", "--protected_cka_source_stage1_checkpoint",
                     "--per_device_train_batch_size", "--gradient_accumulation_steps"):
            self.assertEqual(commands[0][commands[0].index(flag) + 1], commands[1][commands[1].index(flag) + 1], flag)
        for cmd in commands:
            self.assertEqual(cmd[cmd.index("--learning_rate") + 1], "3e-6")
            self.assertEqual(cmd[cmd.index("--max_steps") + 1], "350")
            self.assertEqual(cmd[cmd.index("--protected_cka_start_phase_b") + 1], "True")
            self.assertEqual(cmd[cmd.index("--protected_cka_refinement") + 1], "True")
            self.assertNotIn("--resume_from_checkpoint", cmd)
            self.assertNotIn("--pretrain_mm_mlp_adapter", cmd)
        self.assertEqual([cmd[cmd.index("--output_dir") + 1] for cmd in commands],
                         ["/output/stage1-comparison/ce_only", "/output/stage1-comparison/protected"])

    def test_ablation_rejects_resume_cli_overrides_and_distributed_options(self):
        for settings in ({"RESUME_FROM_CHECKPOINT": "/old/checkpoint-100"}, {"GPU_ID": "0,1"},
                         {"WORLD_SIZE": "2"}, {"DEEPSPEED_CONFIG": "zero2.json"}):
            with self.subTest(settings=settings):
                result = self.launch(ablation=True, **settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(result.stdout.strip())
        result = self.launch(ablation=True, extra_args=("--max_steps", "999"))
        self.assertNotEqual(result.returncode, 0)

    def test_ablation_rejects_existing_output_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "protected").mkdir()
            result = self.launch(ablation=True, ABLATION_OUTPUT_ROOT=temp,
                                 DRY_RUN="0", PYTHON_BIN="/not-executable")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing", result.stderr)
            self.assertFalse(result.stdout.strip())

    def test_source_must_use_protected_runtime_new_run_route(self):
        for settings in ({"START_PHASE_B": "False"},
                         {"START_PHASE_B": "True", "PROTECTED_CKA_REFINEMENT": "False"},
                         {"START_PHASE_B": "True", "RESUME_FROM_CHECKPOINT": "/old/run"}):
            with self.subTest(settings=settings):
                result = self.launch(SOURCE_STAGE1_CHECKPOINT="/source/ce-only", **settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
