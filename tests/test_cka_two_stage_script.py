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
        for key in ("INIT_MODE", "SETTINGS", "REPEATS", "REPEAT_START", "CKA_WEIGHT",
                    "PROJECTOR_INIT_SEED_BASE", "MM_PROJECTOR_INIT_SEED", "PROJECTOR_INIT_SEED_MODE"):
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
                for stage in (("finetune",) if setting == "cka_proj_final" else ("pretrain", "finetune"))]

    def assert_objective(self, flags, setting):
        projector, final = {"baseline": (0, 0), "cka_proj": (0.1, 0),
                            "cka_proj_final": (0.1, 0.1)}[setting]
        self.assertEqual(flags["--cka_loss"], "False" if setting == "baseline" else "True")
        self.assertEqual(float(flags["--cka_loss_projector_weight"]), projector)
        self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), final)
        self.assertEqual(flags["--cka_loss_layers"], "final" if final else "-1")

    def assert_default_job(self, command, repeat, setting, stage, *, projector_seed_base=1000):
        flags = self.flags(command)
        self.assert_objective(flags, setting)
        self.assertEqual(flags["--seed"], "42")
        self.assertEqual(int(flags["--mm_projector_init_seed"]), projector_seed_base + repeat)
        self.assertEqual(flags["--model_name_or_path"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(flags["--mm_projector_type"], "mlp2x_gelu")
        self.assertEqual(command[command.index("--include") + 1], "localhost:2,3")
        pretrain = stage == "pretrain"
        self.assertEqual(flags["--tune_mm_mlp_adapter"], "True" if pretrain else "False")
        self.assertEqual(flags["--per_device_train_batch_size"], "32" if pretrain else "4")
        self.assertEqual(flags["--gradient_accumulation_steps"], "4" if pretrain else "16")
        self.assertEqual(flags["--output_dir"], str(self.output_for(repeat, setting, stage)))
        if pretrain:
            self.assertNotIn("--pretrain_mm_mlp_adapter", flags)
        else:
            pretrain_setting = "cka_proj" if setting == "cka_proj_final" else setting
            self.assertEqual(flags["--pretrain_mm_mlp_adapter"],
                             str(self.output_for(repeat, pretrain_setting, "pretrain") / "mm_projector.bin"))

    def test_default_dry_run_has_ten_jobs_and_shares_projector_initialization(self):
        self.env.pop("CONDA_PREFIX")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 10, result.stdout)
        for command, job in zip(commands, self.jobs()):
            with self.subTest(job=job):
                self.assert_default_job(command, *job)
        flags = [self.flags(command) for command in commands]
        self.assertEqual(len({entry["--output_dir"] for entry in flags}), 10)
        self.assertEqual(len({entry["--run_name"] for entry in flags}), 10)
        adapters = [entry["--pretrain_mm_mlp_adapter"] for entry in flags if "--pretrain_mm_mlp_adapter" in entry]
        self.assertEqual(len(set(adapters)), 4)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_projector_seed_base_override_is_stable_when_selecting_a_later_repeat(self):
        complete = self.run_script(PROJECTOR_INIT_SEED_BASE=9000)
        self.assertEqual(complete.returncode, 0, complete.stderr)
        full_commands = self.commands(complete.stdout)
        self.assertEqual(len(full_commands), 10)
        for command, job in zip(full_commands, self.jobs()):
            self.assert_default_job(command, *job, projector_seed_base=9000)
        for base in (0, 9000):
            with self.subTest(base=base):
                later = self.run_script(PROJECTOR_INIT_SEED_BASE=base, REPEAT_START=2, REPEATS=1)
                self.assertEqual(later.returncode, 0, later.stderr)
                commands = self.commands(later.stdout)
                self.assertEqual(len(commands), 5)
                for command, job in zip(commands, self.jobs()[5:]):
                    self.assert_default_job(command, *job, projector_seed_base=base)
                if base == 9000:
                    self.assertEqual([self.flags(command) for command in commands],
                                     [self.flags(command) for command in full_commands[5:]])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_child_launchers_omit_projector_seed_argument_when_no_seed_is_supplied(self):
        for name in ("pretrain_qwen2_5.sh", "finetune_qwen2_5.sh"):
            with self.subTest(child=name):
                child = SCRIPT.with_name(name)
                result = subprocess.run(["bash", str(child)], cwd=REPO, env=self.env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                command, = self.commands(result.stdout)
                self.assertNotIn("--mm_projector_init_seed", self.flags(command))
        self.assertFalse(self.log.exists())

    def test_random_projector_mode_ignores_seed_base_but_keeps_training_seed_fixed(self):
        for base in ("not-a-seed", "", "-1", str(2**80)):
            with self.subTest(base=base):
                result = self.run_script(PROJECTOR_INIT_SEED_MODE="random", PROJECTOR_INIT_SEED_BASE=base,
                                         MM_PROJECTOR_INIT_SEED="1234")
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.commands(result.stdout)
                self.assertEqual(len(commands), 10)
                for command, (repeat, setting, stage) in zip(commands, self.jobs()):
                    flags = self.flags(command)
                    self.assertEqual(flags["--seed"], "42")
                    self.assertNotIn("--mm_projector_init_seed", flags)
                    self.assert_objective(flags, setting)
                    self.assertEqual(flags["--output_dir"], str(self.output_for(repeat, setting, stage)))
                    if stage == "finetune":
                        producer = "baseline" if setting == "baseline" else "cka_proj"
                        self.assertEqual(flags["--pretrain_mm_mlp_adapter"],
                                         str(self.output_for(repeat, producer, "pretrain") / "mm_projector.bin"))
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_random_projector_mode_still_validates_repeat_ranges_and_mode(self):
        for values in ({"REPEAT_START": "0"}, {"REPEATS": "0"}, {"REPEAT_START": "01"},
                       {"REPEAT_START": str(2**63 - 1)}, {"REPEATS": str(2**63)}):
            with self.subTest(values=values):
                result = self.run_script(PROJECTOR_INIT_SEED_MODE="random", **values)
                self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(self.run_script(PROJECTOR_INIT_SEED_MODE="unknown").returncode, 0)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.output.exists())

    def test_actual_children_share_four_generated_adapters_within_their_own_repeat(self):
        result = self.run_script(DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        self.assertEqual(len(commands), 10)
        for command, job in zip(commands, self.jobs()):
            self.assert_default_job(command, *job)
        for repeat in (1, 2):
            for setting in ("baseline", "cka_proj"):
                adapter = self.output_for(repeat, setting, "pretrain") / "mm_projector.bin"
                self.assertTrue(adapter.is_file())
                self.assertGreater(adapter.stat().st_size, 0)
            self.assertFalse(self.output_for(repeat, "cka_proj_final", "pretrain").exists())
        self.assertEqual(self.preflight_log.read_text().splitlines(), ["dependency-check"] * 6)

    def test_stage_and_setting_subsets_launch_only_the_selected_jobs(self):
        result = self.run_script(STAGE="pretrain", SETTINGS="cka_proj_final", REPEATS=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        command, = self.commands(result.stdout)
        self.assert_default_job(command, 1, "cka_proj", "pretrain")
        for setting in ("baseline", "cka_proj"):
            adapter = self.output_for(1, setting, "pretrain") / "mm_projector.bin"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(setting.encode())
        result = self.run_script(STAGE="finetune", SETTINGS="baseline cka_proj_final", REPEATS=1, DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        self.assertEqual(len(commands), 2)
        for command, setting in zip(commands, ("baseline", "cka_proj_final")):
            self.assert_default_job(command, 1, setting, "finetune")

    def test_projector_pretraining_is_deduplicated_in_either_setting_order(self):
        scenarios = (
            ("pretrain", "cka_proj cka_proj_final", [("cka_proj", "pretrain")]),
            ("pretrain", "cka_proj_final cka_proj", [("cka_proj", "pretrain")]),
            ("pretrain", "baseline cka_proj cka_proj_final",
             [("baseline", "pretrain"), ("cka_proj", "pretrain")]),
            ("all", "cka_proj cka_proj_final",
             [("cka_proj", "pretrain"), ("cka_proj", "finetune"), ("cka_proj_final", "finetune")]),
            ("all", "cka_proj_final cka_proj",
             [("cka_proj", "pretrain"), ("cka_proj_final", "finetune"), ("cka_proj", "finetune")]),
        )
        for stage, settings, jobs in scenarios:
            with self.subTest(stage=stage, settings=settings):
                result = self.run_script(STAGE=stage, SETTINGS=settings, REPEATS=1)
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.commands(result.stdout)
                self.assertEqual(len(commands), len(jobs))
                for command, (setting, job_stage) in zip(commands, jobs):
                    self.assert_default_job(command, 1, setting, job_stage)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_final_only_all_builds_projector_adapter_and_preserves_unused_final_pretrain(self):
        unused = self.output_for(2, "cka_proj_final", "pretrain") / "mm_projector.bin"
        unused.parent.mkdir(parents=True)
        unused.write_bytes(b"older-final-pretraining-must-remain-intact")
        projector = self.output_for(2, "cka_proj", "pretrain") / "mm_projector.bin"
        projector.parent.mkdir(parents=True)
        projector.write_bytes(b"existing-projector-requires-finetune-only-stage")
        rejected = self.run_script(REPEAT_START=2, REPEATS=1, SETTINGS="cka_proj_final", DRY_RUN=0)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(self.log.exists())
        self.assertEqual(projector.read_bytes(), b"existing-projector-requires-finetune-only-stage")
        projector.unlink()
        projector.parent.rmdir()
        result = self.run_script(REPEAT_START=2, REPEATS=1, SETTINGS="cka_proj_final", DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        self.assertEqual(len(commands), 2)
        self.assert_default_job(commands[0], 2, "cka_proj", "pretrain")
        self.assert_default_job(commands[1], 2, "cka_proj_final", "finetune")
        self.assertEqual(unused.read_bytes(), b"older-final-pretraining-must-remain-intact")
        self.assertFalse(self.output_for(2, "cka_proj", "finetune").exists())

    def test_final_only_finetune_requires_projector_adapter_from_the_same_repeat(self):
        other_repeat = self.output_for(1, "cka_proj", "pretrain") / "mm_projector.bin"
        unused_final = self.output_for(2, "cka_proj_final", "pretrain") / "mm_projector.bin"
        for adapter in (other_repeat, unused_final):
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b"unrelated-checkpoint-must-not-be-used")
        settings = dict(REPEAT_START=2, REPEATS=1, STAGE="finetune", SETTINGS="cka_proj_final", DRY_RUN=0)
        missing = self.run_script(**settings)
        self.assertNotEqual(missing.returncode, 0)
        self.assertFalse(self.log.exists())
        adapter = self.output_for(2, "cka_proj", "pretrain") / "mm_projector.bin"
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b"correct-repeat-two-projector-checkpoint")
        result = self.run_script(**settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        command, = self.calls()
        self.assert_default_job(command, 2, "cka_proj_final", "finetune")
        self.assertFalse(self.output_for(2, "cka_proj", "finetune").exists())
        self.assertEqual(adapter.read_bytes(), b"correct-repeat-two-projector-checkpoint")
        for protected in (other_repeat, unused_final):
            self.assertEqual(protected.read_bytes(), b"unrelated-checkpoint-must-not-be-used")

    def test_repeat_start_counts_jobs_from_the_selected_repeat_number(self):
        result = self.run_script(REPEAT_START=2, REPEATS=2, SETTINGS="cka_proj", STAGE="pretrain")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 2)
        for command, repeat in zip(commands, (2, 3)):
            self.assert_default_job(command, repeat, "cka_proj", "pretrain")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_repeat_two_continuation_preserves_completed_first_repeat_and_its_own_adapter(self):
        protected = {}
        for setting in SETTINGS:
            for stage in ("pretrain", "finetune"):
                output = self.output_for(1, setting, stage)
                output.mkdir(parents=True)
                sentinel = output / "completed.txt"
                sentinel.write_text(f"finished first repeat: {setting}/{stage}")
                protected[sentinel] = sentinel.read_bytes()
        baseline_adapter = self.output_for(2, "baseline", "pretrain") / "mm_projector.bin"
        baseline_adapter.parent.mkdir(parents=True)
        baseline_adapter.write_bytes(b"completed-repeat-two-baseline-initialization")
        protected[baseline_adapter] = baseline_adapter.read_bytes()

        baseline = self.run_script(REPEAT_START=2, REPEATS=1, STAGE="finetune", SETTINGS="baseline", DRY_RUN=0)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        remaining = self.run_script(REPEAT_START=2, REPEATS=1, STAGE="all",
                                    SETTINGS="cka_proj cka_proj_final", DRY_RUN=0)
        self.assertEqual(remaining.returncode, 0, remaining.stderr)
        commands = self.calls()
        jobs = [(2, "baseline", "finetune"), (2, "cka_proj", "pretrain"),
                (2, "cka_proj", "finetune"), (2, "cka_proj_final", "finetune")]
        self.assertEqual(len(commands), len(jobs))
        for command, job in zip(commands, jobs):
            self.assert_default_job(command, *job)
        for sentinel, original in protected.items():
            self.assertEqual(sentinel.read_bytes(), original)

    def test_repeat_start_preflights_every_selected_repeat_before_launch(self):
        output = self.output_for(3, "baseline", "finetune")
        output.mkdir(parents=True)
        sentinel = output / "preserve.txt"
        sentinel.write_text("completed selected output")
        for dry_run in (0, 1):
            result = self.run_script(REPEAT_START=2, REPEATS=2, SETTINGS="baseline", DRY_RUN=dry_run)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.log.exists())
            self.assertFalse(self.output_for(2, "baseline", "pretrain").exists())
            self.assertEqual(sentinel.read_text(), "completed selected output")

    def test_finetune_checks_every_required_adapter_before_launching_any_job(self):
        missing = self.output_for(2, "cka_proj", "pretrain") / "mm_projector.bin"
        for repeat in (1, 2):
            for setting in ("baseline", "cka_proj"):
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
            setting = "cka_proj" if stage == "pretrain" else "cka_proj_final"
            output = self.output_for(2, setting, stage)
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
                         *({"REPEAT_START": value} for value in ("0", "-1", "01", "two", "")),
                         *({"PROJECTOR_INIT_SEED_BASE": value}
                           for value in ("", "-1", "abc", "01", str(2**63), str(2**63 - 1))),
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
        self.assertEqual(len(commands), 5)
        for command, (_, setting, _) in zip(commands, self.jobs()[:5]):
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
            MM_PROJECTOR_INIT_SEED="555",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 10)
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

    def test_native_argument_parser_accepts_all_ten_jobs(self):
        import transformers.utils
        import transformers.utils.import_utils

        with mock.patch.object(transformers.utils, "is_flash_attn_2_available", return_value=False), \
             mock.patch.object(transformers.utils.import_utils, "is_flash_attn_2_available", return_value=False):
            from transformers import HfArgumentParser
            from llava.train.train import DataArguments, ModelArguments, TrainingArguments

        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 10)
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        for command, (repeat, setting, _) in zip(commands, self.jobs()):
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
            self.assertEqual(model.mm_projector_init_seed, 1000 + repeat)


if __name__ == "__main__":
    unittest.main()
