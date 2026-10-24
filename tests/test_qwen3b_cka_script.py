"""Verify the Qwen3B CKA-only suite using real child scripts and fake DeepSpeed."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from tests import test_qwen3b_random_projector_script as baseline_fixtures


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/v1_5/7b/train_qwen2_5_3b_cka_ablations.sh"


class Qwen3bCkaScriptTests(unittest.TestCase):
    def setUp(self):
        # Reuse only fixture setup and argv helpers; no baseline tests are
        # inherited or collected in this module.
        baseline_fixtures.Qwen3bRandomProjectorScriptTests.setUp(self)

    commands = staticmethod(baseline_fixtures.Qwen3bRandomProjectorScriptTests.commands)
    flags = staticmethod(baseline_fixtures.Qwen3bRandomProjectorScriptTests.flags)
    calls = baseline_fixtures.Qwen3bRandomProjectorScriptTests.calls

    def run_script(self, *arguments, **settings):
        env = dict(self.env)
        env.update({key: str(value) for key, value in settings.items()})
        return subprocess.run(["bash", str(SCRIPT), *arguments], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=30)

    def jobs(self):
        return [(repeat, setting, stage) for repeat in (1, 2, 3)
                for setting, stage in (("cka_proj", "pretrain"), ("cka_proj", "finetune"),
                                       ("cka_proj_final", "finetune"))]

    def output_for(self, repeat, setting, stage):
        return self.output / f"repeat-{repeat}" / setting / f"llava-{stage}"

    def assert_job(self, command, repeat, setting, stage):
        flags = self.flags(command)
        self.assertEqual(flags["--model_name_or_path"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(flags["--mm_projector_type"], "mlp2x_gelu")
        self.assertEqual(flags["--seed"], "42")
        self.assertNotIn("--mm_projector_init_seed", flags)
        self.assertEqual(flags["--cka_loss"], "True")
        self.assertEqual(float(flags["--cka_loss_projector_weight"]), 0.1)
        final = setting == "cka_proj_final"
        self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), 0.1 if final else 0)
        self.assertEqual(flags["--cka_loss_layers"], "final" if final else "-1")
        self.assertEqual(flags["--output_dir"], str(self.output_for(repeat, setting, stage)))
        self.assertEqual(command[command.index("--include") + 1], "localhost:2,3")
        pretrain = stage == "pretrain"
        self.assertEqual(flags["--per_device_train_batch_size"], "32" if pretrain else "4")
        self.assertEqual(flags["--gradient_accumulation_steps"], "4" if pretrain else "16")
        self.assertEqual(flags["--tune_mm_mlp_adapter"], "True" if pretrain else "False")
        if pretrain:
            self.assertEqual(setting, "cka_proj")
            self.assertNotIn("--pretrain_mm_mlp_adapter", flags)
        else:
            self.assertEqual(flags["--pretrain_mm_mlp_adapter"],
                             str(self.output_for(repeat, "cka_proj", "pretrain") / "mm_projector.bin"))

    def test_default_plan_has_three_projector_pretrains_and_six_cka_finetunes(self):
        self.env.pop("CONDA_PREFIX")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 9, result.stdout)
        for command, job in zip(commands, self.jobs()):
            self.assert_job(command, *job)
        flags = [self.flags(command) for command in commands]
        self.assertEqual(sum(entry["--tune_mm_mlp_adapter"] == "True" for entry in flags), 3)
        self.assertEqual(sum(entry["--tune_mm_mlp_adapter"] == "False" for entry in flags), 6)
        self.assertEqual(len({entry["--run_name"] for entry in flags}), 9)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_actual_children_share_one_projector_adapter_per_repeat_without_baseline(self):
        result = self.run_script(DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        self.assertEqual(len(commands), 9)
        for command, job in zip(commands, self.jobs()):
            self.assert_job(command, *job)
        for repeat in (1, 2, 3):
            adapter = self.output_for(repeat, "cka_proj", "pretrain") / "mm_projector.bin"
            self.assertTrue(adapter.is_file())
            self.assertFalse((self.output / f"repeat-{repeat}" / "baseline").exists())
            self.assertFalse(self.output_for(repeat, "cka_proj_final", "pretrain").exists())
        self.assertEqual(self.preflight_log.read_text().splitlines(), ["dependency-check"] * 6)

    def test_cka_settings_model_and_unseeded_projector_override_inherited_configuration(self):
        result = self.run_script(SETTINGS="baseline", MODEL_NAME_OR_PATH="Qwen/Qwen2.5-7B-Instruct",
                                 RUN_NAME="old-model", MM_PROJECTOR_TYPE="coupling1x_gelu",
                                 PROJECTOR_INIT_SEED_MODE="seeded", PROJECTOR_INIT_SEED_BASE="unused-invalid-base",
                                 MM_PROJECTOR_INIT_SEED="1234", CKA_WEIGHT="0.5", CKA_LOSS_ENABLED="False")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 9)
        for command, job in zip(commands, self.jobs()):
            self.assert_job(command, *job)

    def test_pretrain_stage_is_deduplicated_and_finetune_requires_only_projector_adapters(self):
        pretrain = self.run_script(STAGE="pretrain")
        self.assertEqual(pretrain.returncode, 0, pretrain.stderr)
        commands = self.commands(pretrain.stdout)
        self.assertEqual(len(commands), 3)
        for command, repeat in zip(commands, (1, 2, 3)):
            self.assert_job(command, repeat, "cka_proj", "pretrain")

        protected = {}
        for repeat in (1, 2):
            adapter = self.output_for(repeat, "cka_proj", "pretrain") / "mm_projector.bin"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(f"repeat-{repeat}-projector".encode())
            protected[adapter] = adapter.read_bytes()
        missing = self.run_script(STAGE="finetune", DRY_RUN=0)
        self.assertNotEqual(missing.returncode, 0)
        self.assertFalse(self.log.exists())
        adapter = self.output_for(3, "cka_proj", "pretrain") / "mm_projector.bin"
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b"repeat-3-projector")
        protected[adapter] = adapter.read_bytes()
        result = self.run_script(STAGE="finetune", DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        jobs = [job for job in self.jobs() if job[2] == "finetune"]
        self.assertEqual(len(commands), 6)
        for command, job in zip(commands, jobs):
            self.assert_job(command, *job)
        for path, content in protected.items():
            self.assertEqual(path.read_bytes(), content)

    def test_repeat_start_and_resource_overrides_apply_to_both_cka_settings(self):
        result = self.run_script(REPEAT_START=2, REPEATS=1, GPU_INCLUDE="localhost:1",
                                 PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE=2, PRETRAIN_GRADIENT_ACCUMULATION_STEPS=3,
                                 PER_DEVICE_TRAIN_BATCH_SIZE=4, GRADIENT_ACCUMULATION_STEPS=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 3)
        for command, (_, setting, stage) in zip(commands, self.jobs()[3:6]):
            flags = self.flags(command)
            self.assertEqual(flags["--output_dir"], str(self.output_for(2, setting, stage)))
            self.assertEqual(flags["--seed"], "42")
            self.assertNotIn("--mm_projector_init_seed", flags)
            self.assertEqual(command[command.index("--include") + 1], "localhost:1")
            self.assertEqual(flags["--per_device_train_batch_size"], "2" if stage == "pretrain" else "4")
            self.assertEqual(flags["--gradient_accumulation_steps"], "3" if stage == "pretrain" else "5")

    def test_late_existing_output_prevents_any_launch_and_preserves_checkpoint(self):
        output = self.output_for(3, "cka_proj_final", "finetune")
        output.parent.mkdir(parents=True)
        output.symlink_to(self.root / "missing-target")
        protected = {}
        for repeat in (1, 2, 3):
            adapter = self.output_for(repeat, "cka_proj", "pretrain") / "mm_projector.bin"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(f"preserve-repeat-{repeat}-projector".encode())
            protected[adapter] = adapter.read_bytes()
        for dry_run in (0, 1):
            result = self.run_script(STAGE="finetune", DRY_RUN=dry_run, RESUME_FROM_CHECKPOINT="old-checkpoint")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.log.exists())
            for adapter, contents in protected.items():
                self.assertEqual(adapter.read_bytes(), contents)

    def test_child_failure_prevents_later_jobs(self):
        for stage, count in (("pretrain", 1), ("finetune", 2)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "calls.jsonl"
                result = self.run_script(DRY_RUN=0, FAIL_STAGE=stage,
                                         EXPERIMENT_ROOT=Path(directory) / "outputs", CALL_LOG=log)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual(len(log.read_text().splitlines()), count)

    def test_default_output_root_is_separate_from_baseline_runs(self):
        isolated = self.root / "isolated repo"
        script_dir = isolated / "scripts/v1_5/7b"
        script_dir.mkdir(parents=True)
        for name in (SCRIPT.name, "train_cka_two_stage.sh", "pretrain_qwen2_5.sh", "finetune_qwen2_5.sh"):
            shutil.copyfile(SCRIPT.with_name(name), script_dir / name)
        env = dict(self.env)
        env.pop("EXPERIMENT_ROOT")
        result = subprocess.run(["bash", str(script_dir / SCRIPT.name)], cwd=isolated, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 9)
        default = Path("checkpoints/cka-two-stage/qwen2.5-3b-mlp2x_gelu-random-projector-cka-seed42")
        for command in commands:
            self.assertTrue(Path(self.flags(command)["--output_dir"]).is_relative_to(default))
        self.assertFalse((isolated / "checkpoints").exists())

    def test_all_nine_jobs_parse_with_cka_and_no_projector_seed_in_native_arguments(self):
        import transformers.utils
        import transformers.utils.import_utils

        with mock.patch.object(transformers.utils, "is_flash_attn_2_available", return_value=False), \
             mock.patch.object(transformers.utils.import_utils, "is_flash_attn_2_available", return_value=False):
            from transformers import HfArgumentParser
            from llava.train.train import DataArguments, ModelArguments, TrainingArguments

        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 9)
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        for command, (_, setting, _) in zip(commands, self.jobs()):
            argv = command[command.index("llava/train/train_mem.py") + 1:]
            with mock.patch.object(TrainingArguments, "__post_init__", return_value=None):
                model, _, training = parser.parse_args_into_dataclasses(argv, look_for_args_file=False)
            self.assertEqual(model.model_name_or_path, "Qwen/Qwen2.5-3B-Instruct")
            self.assertIsNone(model.mm_projector_init_seed)
            self.assertTrue(model.cka_loss)
            self.assertEqual(model.cka_loss_projector_weight, 0.1)
            self.assertEqual(model.cka_loss_final_hidden_weight, 0.1 if setting == "cka_proj_final" else 0)
            self.assertEqual(model.cka_loss_layers, "final" if setting == "cka_proj_final" else "-1")
            self.assertEqual(training.seed, 42)


if __name__ == "__main__":
    unittest.main()
