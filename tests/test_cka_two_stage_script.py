"""Verify repeated two-stage CKA jobs using real child scripts and fake DeepSpeed."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tests import test_cka_location_ablation_script as launch_fixtures


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/v1_5/7b/train_cka_two_stage.sh"
SETTINGS = ("baseline", "cka_proj", "cka_proj_final")


class CkaTwoStageScriptTests(unittest.TestCase):
    def setUp(self):
        # Composition reuses the narrowly scoped dependency-check shim and fake
        # DeepSpeed without inheriting or rerunning the older placement tests.
        launch_fixtures.CkaLocationAblationScriptTests.setUp(self)
        for key in ("INIT_MODE", "SETTINGS", "REPEATS", "CKA_WEIGHT"):
            self.env.pop(key, None)

    commands = staticmethod(launch_fixtures.CkaLocationAblationScriptTests.commands)
    flags = staticmethod(launch_fixtures.CkaLocationAblationScriptTests.flags)
    calls = launch_fixtures.CkaLocationAblationScriptTests.calls

    def run_script(self, *arguments, **settings):
        env = dict(self.env)
        env.update({key: str(value) for key, value in settings.items()})
        return subprocess.run(["bash", str(SCRIPT), *arguments], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=30)

    def output_for(self, repeat, setting, stage):
        return self.output / f"repeat-{repeat}" / setting / f"llava-{stage}"

    def jobs(self):
        return [(repeat, setting, stage) for repeat in (1, 2) for setting in SETTINGS
                for stage in ("pretrain", "finetune")]

    def assert_objective(self, flags, setting):
        projector, final = {"baseline": (0, 0), "cka_proj": (0.1, 0),
                            "cka_proj_final": (0.1, 0.1)}[setting]
        self.assertEqual(flags["--cka_loss"], "False" if setting == "baseline" else "True")
        self.assertEqual(float(flags["--cka_loss_projector_weight"]), projector)
        self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), final)
        self.assertEqual(flags["--cka_loss_layers"], "final" if final else "-1")

    def assert_default_job(self, command, repeat, setting, stage):
        flags = self.flags(command)
        self.assert_objective(flags, setting)
        self.assertEqual(flags["--seed"], "42")
        self.assertEqual(flags["--model_name_or_path"], "lmsys/vicuna-7b-v1.5")
        self.assertEqual(flags["--mm_projector_type"], "mlp2x_gelu")
        self.assertEqual(command[command.index("--include") + 1], "localhost:1,2")
        pretrain = stage == "pretrain"
        self.assertEqual(flags["--tune_mm_mlp_adapter"], "True" if pretrain else "False")
        self.assertEqual(flags["--per_device_train_batch_size"], "32" if pretrain else "8")
        self.assertEqual(flags["--gradient_accumulation_steps"], "4" if pretrain else "8")
        self.assertEqual(flags["--output_dir"], str(self.output_for(repeat, setting, stage)))
        if pretrain:
            self.assertNotIn("--pretrain_mm_mlp_adapter", flags)
        else:
            self.assertEqual(flags["--pretrain_mm_mlp_adapter"],
                             str(self.output_for(repeat, setting, "pretrain") / "mm_projector.bin"))

    def test_default_dry_run_has_twelve_jobs_and_matching_objectives_in_both_stages(self):
        self.env.pop("CONDA_PREFIX")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 12, result.stdout)
        for command, job in zip(commands, self.jobs()):
            with self.subTest(job=job):
                self.assert_default_job(command, *job)
        flags = [self.flags(command) for command in commands]
        self.assertEqual(len({entry["--output_dir"] for entry in flags}), 12)
        self.assertEqual(len({entry["--run_name"] for entry in flags}), 12)
        adapters = [entry["--pretrain_mm_mlp_adapter"] for entry in flags if "--pretrain_mm_mlp_adapter" in entry]
        self.assertEqual(len(set(adapters)), 6)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_actual_children_route_each_of_six_generated_adapters_to_its_own_finetune(self):
        result = self.run_script(DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        self.assertEqual(len(commands), 12)
        for command, job in zip(commands, self.jobs()):
            self.assert_default_job(command, *job)
        for repeat in (1, 2):
            for setting in SETTINGS:
                adapter = self.output_for(repeat, setting, "pretrain") / "mm_projector.bin"
                self.assertTrue(adapter.is_file())
                self.assertGreater(adapter.stat().st_size, 0)
        self.assertEqual(self.preflight_log.read_text().splitlines(), ["dependency-check"] * 6)

    def test_stage_and_setting_subsets_launch_only_the_selected_jobs(self):
        result = self.run_script(STAGE="pretrain", SETTINGS="cka_proj_final", REPEATS=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        command, = self.commands(result.stdout)
        self.assert_default_job(command, 1, "cka_proj_final", "pretrain")
        for setting in ("baseline", "cka_proj_final"):
            adapter = self.output_for(1, setting, "pretrain") / "mm_projector.bin"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(setting.encode())
        result = self.run_script(STAGE="finetune", SETTINGS="baseline cka_proj_final", REPEATS=1, DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        self.assertEqual(len(commands), 2)
        for command, setting in zip(commands, ("baseline", "cka_proj_final")):
            self.assert_default_job(command, 1, setting, "finetune")

    def test_finetune_checks_every_required_adapter_before_launching_any_job(self):
        missing = self.output_for(2, "cka_proj_final", "pretrain") / "mm_projector.bin"
        for repeat in (1, 2):
            for setting in SETTINGS:
                adapter = self.output_for(repeat, setting, "pretrain") / "mm_projector.bin"
                if adapter != missing:
                    adapter.parent.mkdir(parents=True)
                    adapter.write_bytes(b"existing-stage1")
        for kind in ("missing", "empty", "directory"):
            with self.subTest(kind=kind):
                if kind != "missing":
                    missing.parent.mkdir(parents=True, exist_ok=True)
                    missing.touch() if kind == "empty" else missing.mkdir()
                result = self.run_script(STAGE="finetune", DRY_RUN=0)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.log.exists())
                self.assertFalse(any(self.output.glob("repeat-*/*/llava-finetune")))
                if kind == "empty":
                    missing.unlink()
                elif kind == "directory":
                    missing.rmdir()

    def test_all_selected_outputs_are_checked_before_the_first_launch(self):
        for stage in ("pretrain", "finetune"):
            output = self.output_for(2, "cka_proj_final", stage)
            for kind in ("directory", "file", "dangling-symlink"):
                with self.subTest(stage=stage, kind=kind):
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if kind == "directory":
                        output.mkdir()
                    elif kind == "file":
                        output.write_text("preserve")
                    else:
                        output.symlink_to(self.root / "missing-target")
                    for dry_run in (0, 1):
                        result = self.run_script(DRY_RUN=dry_run)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertFalse(self.log.exists())
                    output.rmdir() if kind == "directory" else output.unlink()

    def test_child_failure_or_missing_adapter_prevents_later_jobs(self):
        scenarios = (({"FAIL_STAGE": "pretrain"}, 1, 7),
                     ({"FAIL_STAGE": "finetune"}, 2, 7),
                     ({"ADAPTER_MODE": "missing"}, 1, None),
                     ({"ADAPTER_MODE": "empty"}, 1, None),
                     ({"ADAPTER_MODE": "cka-missing"}, 3, None))
        for settings, expected_count, status in scenarios:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as directory:
                call_log = Path(directory) / "calls.jsonl"
                result = self.run_script(DRY_RUN=0, EXPERIMENT_ROOT=Path(directory) / "outputs",
                                         CALL_LOG=call_log, **settings)
                self.assertNotEqual(result.returncode, 0)
                if status is not None:
                    self.assertEqual(result.returncode, status)
                self.assertEqual(len(call_log.read_text().splitlines()), expected_count)

    def test_invalid_inputs_reject_before_launch(self):
        for settings in ({"STAGE": "invalid"}, {"SETTINGS": "baseline unknown"},
                         {"SETTINGS": "baseline baseline"}, {"SETTINGS": ""},
                         {"REPEATS": "0"}, {"REPEATS": "-1"},
                         {"REPEATS": "two"}, {"DRY_RUN": "yes"}, {"SEED": "43"},
                         {"SEEDS": "42 43"}, {"STAGE1_SEED": "43"},
                         *({"CKA_WEIGHT": value} for value in ("nan", "inf", "-1", "0", ""))):
            with self.subTest(settings=settings):
                result = self.run_script(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.log.exists())
                self.assertFalse(self.output.exists())
        self.assertNotEqual(self.run_script("--seed", "43").returncode, 0)

    def test_weight_override_changes_each_active_term_in_both_stages(self):
        result = self.run_script(CKA_WEIGHT="2e-1", REPEATS=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 6)
        for command, setting in zip(commands, (setting for setting in SETTINGS for _ in range(2))):
            flags = self.flags(command)
            self.assertEqual(float(flags["--cka_loss_projector_weight"]), 0 if setting == "baseline" else 0.2)
            self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), 0.2 if setting == "cka_proj_final" else 0)
            self.assertEqual(flags["--cka_loss_layers"], "final" if setting == "cka_proj_final" else "-1")
            self.assertEqual(flags["--cka_loss"], "False" if setting == "baseline" else "True")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_inherited_auxiliary_options_cannot_change_the_selected_objectives(self):
        result = self.run_script(
            CKA_LOSS_ENABLED="False", CKA_PROJECTOR_WEIGHT="0.9", CKA_FINAL_HIDDEN_WEIGHT="0.8",
            CKA_LAYERS="all", CKA_LOSS_TAU="0.5", CKA_LOSS_START_RATIO="0.8", CKA_FINAL_MLP_HEAD="True",
            CKA_ANCHOR_LAYER="12", CKA_VISION_ANCHOR_LAYER="20", CKA_PROJECTOR_VISION_ANCHOR_LAYER="21",
            CKA_FINAL_VISION_ANCHOR_LAYER="22", CKA_LOSS_SUBSET_SELECT_LAYER="8",
            VSP_PCGRAD="True", VSP_NORM_CAP="True", VSP_APPLY_TO_PROJECTOR_ONLY="True",
            VSP_DIAGNOSTICS="True", RESUME_FROM_CHECKPOINT="old-checkpoint", STOP_AFTER_STEP_RATIO="0.8",
            WANDB_RUN_ID="old-run", WANDB_RESUME="must", PRETRAIN_ADAPTER=self.root / "unrelated-adapter.bin",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 12)
        for command, job in zip(commands, self.jobs()):
            self.assert_default_job(command, *job)
            flags = self.flags(command)
            for key in ("--use_pcgrad", "--vsp_gradient_diagnostics", "--vsp_asymmetric_pcgrad",
                        "--vsp_apply_to_projector_only", "--vsp_norm_cap", "--cka_loss_final_mlp_head"):
                self.assertEqual(flags[key], "False")
            self.assertEqual(float(flags["--cka_loss_tau"]), 0)
            self.assertEqual(float(flags["--cka_loss_start_ratio"]), 0)
            self.assertNotIn("--resume_from_checkpoint", flags)
            self.assertNotIn("--stop_after_step_ratio", flags)
            for key in flags:
                self.assertNotIn("anchor", key)
                self.assertNotIn("subset", key)
            self.assertIn("WANDB_RUN_ID", command)
            self.assertIn("WANDB_RESUME", command)
            self.assertNotIn("WANDB_RUN_ID=old-run", command)
            self.assertNotIn("WANDB_RESUME=must", command)

    def test_native_argument_parser_accepts_all_twelve_jobs(self):
        import transformers.utils
        import transformers.utils.import_utils

        with mock.patch.object(transformers.utils, "is_flash_attn_2_available", return_value=False), \
             mock.patch.object(transformers.utils.import_utils, "is_flash_attn_2_available", return_value=False):
            from transformers import HfArgumentParser
            from llava.train.train import DataArguments, ModelArguments, TrainingArguments

        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 12)
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        for command, (_, setting, _) in zip(commands, self.jobs()):
            argv = command[command.index("llava/train/train_mem.py") + 1:]
            with mock.patch.object(TrainingArguments, "__post_init__", return_value=None):
                model, _, training = parser.parse_args_into_dataclasses(argv, look_for_args_file=False)
            self.assertEqual(model.cka_loss, setting != "baseline")
            self.assertEqual(model.cka_loss_projector_weight, 0 if setting == "baseline" else 0.1)
            self.assertEqual(model.cka_loss_final_hidden_weight, 0.1 if setting == "cka_proj_final" else 0)
            self.assertEqual(model.cka_loss_layers, "final" if setting == "cka_proj_final" else "-1")
            self.assertEqual(model.cka_loss_geometry, "linear")
            self.assertIsNone(model.cka_loss_anchor_layer)
            self.assertIsNone(model.cka_loss_subset_select_layer)
            self.assertFalse(model.vsp_asymmetric_pcgrad)
            self.assertFalse(training.protected_cka_refinement)
            self.assertFalse(training.protected_stage2)
            self.assertEqual(training.seed, 42)


if __name__ == "__main__":
    unittest.main()
