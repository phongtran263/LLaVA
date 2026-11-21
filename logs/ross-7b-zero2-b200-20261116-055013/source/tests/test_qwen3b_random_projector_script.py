"""Check three Qwen3B baseline repeats without a seeded projector or GPU jobs."""

from pathlib import Path
import shutil
import subprocess
import unittest
from unittest import mock

from tests import test_cka_location_ablation_script as launch_fixtures


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/v1_5/7b/train_qwen2_5_3b_baseline.sh"


class Qwen3bRandomProjectorScriptTests(unittest.TestCase):
    def setUp(self):
        launch_fixtures.CkaLocationAblationScriptTests.setUp(self)
        for key in ("INIT_MODE", "SETTINGS", "REPEATS", "REPEAT_START", "CKA_WEIGHT",
                    "PROJECTOR_INIT_SEED_BASE", "PROJECTOR_INIT_SEED_MODE", "MM_PROJECTOR_INIT_SEED"):
            self.env.pop(key, None)

    commands = staticmethod(launch_fixtures.CkaLocationAblationScriptTests.commands)
    flags = staticmethod(launch_fixtures.CkaLocationAblationScriptTests.flags)
    calls = launch_fixtures.CkaLocationAblationScriptTests.calls

    def run_script(self, *arguments, **settings):
        env = dict(self.env)
        env.update({key: str(value) for key, value in settings.items()})
        return subprocess.run(["bash", str(SCRIPT), *arguments], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=30)

    def jobs(self):
        return [(repeat, "baseline", stage) for repeat in (1, 2, 3) for stage in ("pretrain", "finetune")]

    def output_for(self, repeat, setting, stage):
        return self.output / f"repeat-{repeat}" / setting / f"llava-{stage}"

    def assert_job(self, command, repeat, setting, stage):
        flags = self.flags(command)
        self.assertEqual(flags["--model_name_or_path"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(flags["--mm_projector_type"], "mlp2x_gelu")
        self.assertEqual(flags["--seed"], "42")
        self.assertNotIn("--mm_projector_init_seed", flags)
        self.assertEqual(setting, "baseline")
        self.assertEqual(flags["--cka_loss"], "False")
        self.assertEqual(float(flags["--cka_loss_projector_weight"]), 0)
        self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), 0)
        self.assertEqual(flags["--cka_loss_layers"], "-1")
        self.assertEqual(flags["--output_dir"], str(self.output_for(repeat, setting, stage)))
        self.assertEqual(command[command.index("--include") + 1], "localhost:2,3")
        pretrain = stage == "pretrain"
        self.assertEqual(flags["--per_device_train_batch_size"], "32" if pretrain else "4")
        self.assertEqual(flags["--gradient_accumulation_steps"], "4" if pretrain else "16")
        self.assertEqual(flags["--tune_mm_mlp_adapter"], "True" if pretrain else "False")
        if pretrain:
            self.assertNotIn("--pretrain_mm_mlp_adapter", flags)
        else:
            self.assertEqual(flags["--pretrain_mm_mlp_adapter"],
                             str(self.output_for(repeat, "baseline", "pretrain") / "mm_projector.bin"))

    def test_default_three_baseline_repeats_have_six_jobs_and_no_projector_seed_flag(self):
        self.env.pop("CONDA_PREFIX")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 6, result.stdout)
        for command, job in zip(commands, self.jobs()):
            self.assert_job(command, *job)
        flags = [self.flags(command) for command in commands]
        self.assertEqual(len({entry["--run_name"] for entry in flags}), 6)
        self.assertEqual(sum(entry["--tune_mm_mlp_adapter"] == "True" for entry in flags), 3)
        self.assertEqual(sum(entry["--tune_mm_mlp_adapter"] == "False" for entry in flags), 3)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_actual_child_launches_route_each_repeat_to_its_own_baseline_adapter(self):
        result = self.run_script(DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        self.assertEqual(len(commands), 6)
        for command, job in zip(commands, self.jobs()):
            self.assert_job(command, *job)
        for repeat in (1, 2, 3):
            self.assertTrue((self.output_for(repeat, "baseline", "pretrain") / "mm_projector.bin").is_file())
            self.assertFalse((self.output / f"repeat-{repeat}" / "cka_proj").exists())
            self.assertFalse(self.output_for(repeat, "cka_proj_final", "pretrain").exists())
        self.assertEqual(self.preflight_log.read_text().splitlines(), ["dependency-check"] * 3)

    def test_model_and_random_mode_are_pinned_despite_inherited_seed_configuration(self):
        result = self.run_script(MODEL_NAME_OR_PATH="lmsys/vicuna-7b-v1.5", RUN_NAME="old-model",
                                 MM_PROJECTOR_TYPE="coupling1x_gelu", PROJECTOR_INIT_SEED_MODE="seeded",
                                 PROJECTOR_INIT_SEED_BASE="invalid-unused-base", MM_PROJECTOR_INIT_SEED="1234",
                                 SETTINGS="cka_proj cka_proj_final", CKA_LOSS_ENABLED="True",
                                 CKA_PROJECTOR_WEIGHT="0.9", CKA_FINAL_HIDDEN_WEIGHT="0.8")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 6)
        for command, job in zip(commands, self.jobs()):
            self.assert_job(command, *job)

    def test_repeat_stage_and_resource_overrides_still_work(self):
        result = self.run_script(REPEAT_START=2, REPEATS=1, SETTINGS="cka_proj_final", GPU_INCLUDE="localhost:1",
                                 PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE=2, PRETRAIN_GRADIENT_ACCUMULATION_STEPS=3,
                                 PER_DEVICE_TRAIN_BATCH_SIZE=4, GRADIENT_ACCUMULATION_STEPS=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 2)
        for command, stage in zip(commands, ("pretrain", "finetune")):
            flags = self.flags(command)
            self.assertEqual(flags["--output_dir"], str(self.output_for(2, "baseline", stage)))
            self.assertEqual(flags["--seed"], "42")
            self.assertNotIn("--mm_projector_init_seed", flags)
            self.assertEqual(command[command.index("--include") + 1], "localhost:1")
            self.assertEqual(flags["--per_device_train_batch_size"], "2" if stage == "pretrain" else "4")
            self.assertEqual(flags["--gradient_accumulation_steps"], "3" if stage == "pretrain" else "5")
        pretrain = self.run_script(STAGE="pretrain", SETTINGS="cka_proj cka_proj_final", REPEATS=1)
        self.assertEqual(pretrain.returncode, 0, pretrain.stderr)
        command, = self.commands(pretrain.stdout)
        self.assert_job(command, 1, "baseline", "pretrain")

    def test_existing_last_output_blocks_every_job_and_resume_settings_are_ignored(self):
        output = self.output_for(3, "baseline", "finetune")
        output.parent.mkdir(parents=True)
        output.symlink_to(self.root / "missing-target")
        for dry_run in (0, 1):
            result = self.run_script(DRY_RUN=dry_run, RESUME_FROM_CHECKPOINT="old-checkpoint")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.log.exists())
            self.assertFalse(self.output_for(1, "baseline", "pretrain").exists())

    def test_invalid_repeat_or_training_seed_rejects_before_any_launch(self):
        for values in ({"REPEATS": "0"}, {"REPEAT_START": "0"},
                       {"SEED": "43"}, {"SEEDS": "42 43"}):
            with self.subTest(values=values):
                result = self.run_script(**values)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.log.exists())
                self.assertFalse(self.output.exists())

    def test_default_output_root_marks_random_projector_runs(self):
        # Copy the real shell sources into a temporary repository so checking
        # default destinations never depends on existing user checkpoints.
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
        self.assertEqual(len(commands), 6)
        default_root = Path("checkpoints/baseline-two-stage/qwen2.5-3b-mlp2x_gelu-random-projector-seed42")
        for command in commands:
            self.assertTrue(Path(self.flags(command)["--output_dir"]).is_relative_to(default_root))
        self.assertFalse((isolated / "checkpoints").exists())

    def test_all_six_baseline_jobs_parse_with_unseeded_projector_in_native_arguments(self):
        import transformers.utils
        import transformers.utils.import_utils

        with mock.patch.object(transformers.utils, "is_flash_attn_2_available", return_value=False), \
             mock.patch.object(transformers.utils.import_utils, "is_flash_attn_2_available", return_value=False):
            from transformers import HfArgumentParser
            from llava.train.train import DataArguments, ModelArguments, TrainingArguments

        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 6)
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        for command in commands:
            argv = command[command.index("llava/train/train_mem.py") + 1:]
            with mock.patch.object(TrainingArguments, "__post_init__", return_value=None):
                model, _, training = parser.parse_args_into_dataclasses(argv, look_for_args_file=False)
            self.assertEqual(model.model_name_or_path, "Qwen/Qwen2.5-3B-Instruct")
            self.assertIsNone(model.mm_projector_init_seed)
            self.assertFalse(model.cka_loss)
            self.assertEqual(model.cka_loss_projector_weight, 0)
            self.assertEqual(model.cka_loss_final_hidden_weight, 0)
            self.assertEqual(model.cka_loss_layers, "-1")
            self.assertEqual(training.seed, 42)


if __name__ == "__main__":
    unittest.main()
