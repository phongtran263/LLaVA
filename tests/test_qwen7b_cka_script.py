"""Verify the single-GPU Qwen7B CKA suite without training or GPU access."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests import test_qwen3b_cka_script as cka_fixtures


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/v1_5/7b/train_qwen2_5_7b_cka_ablations.sh"


class Qwen7bCkaScriptTests(unittest.TestCase):
    def setUp(self):
        # Reuse fake DeepSpeed and argv helpers, not the 3B test methods.
        cka_fixtures.Qwen3bCkaScriptTests.setUp(self)
        for key in ("GRADIENT_CHECKPOINTING", "PRETRAIN_GRADIENT_CHECKPOINTING",
                    "FINETUNE_GRADIENT_CHECKPOINTING"):
            self.env.pop(key, None)

    commands = staticmethod(cka_fixtures.Qwen3bCkaScriptTests.commands)
    flags = staticmethod(cka_fixtures.Qwen3bCkaScriptTests.flags)
    calls = cka_fixtures.Qwen3bCkaScriptTests.calls

    def run_script(self, *arguments, **settings):
        env = dict(self.env)
        env.update({key: str(value) for key, value in settings.items()})
        return subprocess.run(["bash", str(SCRIPT), *arguments], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=30)

    @staticmethod
    def jobs(repeats=(1, 2, 3)):
        return [(repeat, setting, stage) for repeat in repeats
                for setting, stage in (("baseline", "pretrain"), ("baseline", "finetune"),
                                       ("cka_proj", "pretrain"), ("cka_proj", "finetune"),
                                       ("cka_proj_final", "finetune"))]

    def output_for(self, repeat, setting, stage):
        return self.output / f"repeat-{repeat}" / setting / f"llava-{stage}"

    def assert_job(self, command, repeat, setting, stage):
        flags = self.flags(command)
        pretrain = stage == "pretrain"
        baseline = setting == "baseline"
        final = setting == "cka_proj_final"
        self.assertEqual(flags["--model_name_or_path"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(flags["--mm_projector_type"], "mlp2x_gelu")
        self.assertEqual(flags["--seed"], "42")
        self.assertNotIn("--mm_projector_init_seed", flags)
        self.assertEqual(flags["--run_name"], f"qwen2.5-7b-{setting}-repeat{repeat}-seed42-{stage}")
        self.assertEqual(flags["--cka_loss"], "False" if baseline else "True")
        self.assertEqual(float(flags["--cka_loss_projector_weight"]), 0 if baseline else 0.1)
        self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), 0.1 if final else 0)
        self.assertEqual(flags["--cka_loss_layers"], "final" if final else "-1")
        self.assertEqual(flags["--output_dir"], str(self.output_for(repeat, setting, stage)))
        self.assertEqual(command[command.index("--include") + 1], "localhost:3")
        self.assertEqual(flags["--per_device_train_batch_size"], "16" if pretrain else "1")
        self.assertEqual(flags["--gradient_accumulation_steps"], "16" if pretrain else "128")
        effective_batch = int(flags["--per_device_train_batch_size"]) * int(flags["--gradient_accumulation_steps"])
        self.assertEqual(effective_batch, 256 if pretrain else 128)
        self.assertEqual(flags["--gradient_checkpointing"], "False" if pretrain else "True")
        self.assertEqual(flags["--tune_mm_mlp_adapter"], "True" if pretrain else "False")
        for flag in ("--use_pcgrad", "--vsp_asymmetric_pcgrad", "--vsp_norm_cap",
                     "--vsp_apply_to_projector_only", "--vsp_gradient_diagnostics",
                     "--protected_cka_refinement", "--protected_stage2", "--cka_loss_final_mlp_head"):
            self.assertEqual(flags[flag], "False", flag)
        self.assertNotIn("--resume_from_checkpoint", flags)
        if pretrain:
            self.assertNotEqual(setting, "cka_proj_final")
            self.assertNotIn("--pretrain_mm_mlp_adapter", flags)
        else:
            producer = "baseline" if baseline else "cka_proj"
            self.assertEqual(flags["--pretrain_mm_mlp_adapter"],
                             str(self.output_for(repeat, producer, "pretrain") / "mm_projector.bin"))

    def test_default_dry_run_has_six_pretrains_and_nine_finetunes(self):
        self.env.pop("CONDA_PREFIX")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 15, result.stdout)
        for command, job in zip(commands, self.jobs()):
            self.assert_job(command, *job)
        flags = [self.flags(command) for command in commands]
        self.assertEqual(sum(entry["--tune_mm_mlp_adapter"] == "True" for entry in flags), 6)
        self.assertEqual(sum(entry["--tune_mm_mlp_adapter"] == "False" for entry in flags), 9)
        self.assertEqual(len({entry["--run_name"] for entry in flags}), 15)
        self.assertEqual(len({entry["--output_dir"] for entry in flags}), 15)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_fake_children_share_only_the_cka_adapter_within_each_repeat(self):
        result = self.run_script(DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        self.assertEqual(len(commands), 15)
        for command, job in zip(commands, self.jobs()):
            self.assert_job(command, *job)
        for repeat in (1, 2, 3):
            for setting in ("baseline", "cka_proj"):
                adapter = self.output_for(repeat, setting, "pretrain") / "mm_projector.bin"
                self.assertTrue(adapter.is_file())
            self.assertFalse(self.output_for(repeat, "cka_proj_final", "pretrain").exists())
        self.assertEqual(self.preflight_log.read_text().splitlines(), ["dependency-check"] * 9)

    def test_fixed_model_objectives_and_random_projector_override_inherited_settings(self):
        result = self.run_script(
            MODEL_NAME_OR_PATH="Qwen/Qwen2.5-3B-Instruct", RUN_NAME="old-model",
            MM_PROJECTOR_TYPE="coupling1x_gelu", SETTINGS="cka_proj_final", CKA_WEIGHT="0.5",
            PROJECTOR_INIT_SEED_MODE="seeded", PROJECTOR_INIT_SEED_BASE="unused-invalid-base",
            MM_PROJECTOR_INIT_SEED="1234", CKA_LOSS_ENABLED="False",
            CKA_PROJECTOR_WEIGHT="0.9", CKA_FINAL_HIDDEN_WEIGHT="0.8", CKA_LAYERS="all",
            CKA_FINAL_MLP_HEAD="True", VSP_PCGRAD="True", VSP_NORM_CAP="True",
            VSP_APPLY_TO_PROJECTOR_ONLY="True", VSP_DIAGNOSTICS="True",
            RESUME_FROM_CHECKPOINT="old-checkpoint",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 15)
        for command, job in zip(commands, self.jobs()):
            self.assert_job(command, *job)

    def test_repeat_start_and_shared_resource_overrides_remain_configurable(self):
        result = self.run_script(
            REPEAT_START=2, REPEATS=1, GPU_INCLUDE="localhost:1",
            PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE=2, PRETRAIN_GRADIENT_ACCUMULATION_STEPS=3,
            PER_DEVICE_TRAIN_BATCH_SIZE=4, GRADIENT_ACCUMULATION_STEPS=5,
            PRETRAIN_GRADIENT_CHECKPOINTING="True", FINETUNE_GRADIENT_CHECKPOINTING="False",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 5)
        for command, (_, setting, stage) in zip(commands, self.jobs((2,))):
            flags = self.flags(command)
            pretrain = stage == "pretrain"
            self.assertEqual(flags["--output_dir"], str(self.output_for(2, setting, stage)))
            self.assertEqual(flags["--seed"], "42")
            self.assertNotIn("--mm_projector_init_seed", flags)
            self.assertEqual(command[command.index("--include") + 1], "localhost:1")
            self.assertEqual(flags["--per_device_train_batch_size"], "2" if pretrain else "4")
            self.assertEqual(flags["--gradient_accumulation_steps"], "3" if pretrain else "5")
            self.assertEqual(flags["--gradient_checkpointing"], "True" if pretrain else "False")

    def test_stage_gpu_overrides_take_precedence_over_shared_gpu(self):
        result = self.run_script(REPEATS=1, GPU_INCLUDE="localhost:0",
                                 PRETRAIN_GPU_INCLUDE="localhost:1,2", FINETUNE_GPU_INCLUDE="localhost:2")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 5)
        for command, (_, _, stage) in zip(commands, self.jobs((1,))):
            self.assertEqual(command[command.index("--include") + 1],
                             "localhost:1,2" if stage == "pretrain" else "localhost:2")

    def test_wrapper_checkpoint_defaults_are_independent_of_shared_inherited_setting(self):
        for inherited in ("True", "False"):
            with self.subTest(inherited=inherited):
                result = self.run_script(REPEATS=1, GRADIENT_CHECKPOINTING=inherited)
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.commands(result.stdout)
                self.assertEqual(len(commands), 5)
                for command, job in zip(commands, self.jobs((1,))):
                    self.assert_job(command, *job)

    def test_child_checkpointing_defaults_and_fallbacks_remain_backward_compatible(self):
        for stage in ("pretrain", "finetune"):
            variable = f"{stage.upper()}_GRADIENT_CHECKPOINTING"
            cases = (({}, "False"), ({"GRADIENT_CHECKPOINTING": "True"}, "True"),
                     ({"GRADIENT_CHECKPOINTING": "False", variable: "True"}, "True"),
                     ({"GRADIENT_CHECKPOINTING": "True", variable: "False"}, "False"))
            for settings, expected in cases:
                with self.subTest(stage=stage, settings=settings):
                    env = dict(self.env)
                    env.update(settings)
                    result = subprocess.run(["bash", str(SCRIPT.with_name(f"{stage}_qwen2_5.sh"))],
                                            cwd=REPO, env=env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    command, = self.commands(result.stdout)
                    self.assertEqual(self.flags(command)["--gradient_checkpointing"], expected)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_pretrain_deduplicates_cka_and_finetune_requires_both_adapter_kinds(self):
        result = self.run_script(STAGE="pretrain")
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = [job for job in self.jobs() if job[2] == "pretrain"]
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 6)
        for command, job in zip(commands, jobs):
            self.assert_job(command, *job)

        protected = {}
        for repeat in (1, 2, 3):
            for setting in ("baseline", "cka_proj"):
                adapter = self.output_for(repeat, setting, "pretrain") / "mm_projector.bin"
                adapter.parent.mkdir(parents=True)
                protected[adapter] = f"preserve-repeat-{repeat}-{setting}".encode()
                if (repeat, setting) != (3, "cka_proj"):
                    adapter.write_bytes(protected[adapter])
        missing = self.run_script(STAGE="finetune", DRY_RUN=0)
        self.assertNotEqual(missing.returncode, 0)
        self.assertFalse(self.log.exists())
        adapter = self.output_for(3, "cka_proj", "pretrain") / "mm_projector.bin"
        adapter.write_bytes(protected[adapter])
        result = self.run_script(STAGE="finetune", DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        self.assertEqual(len(commands), 9)
        jobs = [job for job in self.jobs() if job[2] == "finetune"]
        for command, job in zip(commands, jobs):
            self.assert_job(command, *job)
        for path, content in protected.items():
            self.assertEqual(path.read_bytes(), content)

    def test_late_existing_output_blocks_all_launches_without_clobbering_checkpoints(self):
        output = self.output_for(3, "cka_proj_final", "finetune")
        output.parent.mkdir(parents=True)
        output.symlink_to(self.root / "missing-target")
        protected = {}
        for repeat in (1, 2, 3):
            for setting in ("baseline", "cka_proj"):
                adapter = self.output_for(repeat, setting, "pretrain") / "mm_projector.bin"
                adapter.parent.mkdir(parents=True)
                adapter.write_bytes(f"preserve-repeat-{repeat}-{setting}".encode())
                protected[adapter] = adapter.read_bytes()
        for dry_run in (0, 1):
            result = self.run_script(STAGE="finetune", DRY_RUN=dry_run,
                                     RESUME_FROM_CHECKPOINT="old-checkpoint")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.log.exists())
            for adapter, content in protected.items():
                self.assertEqual(adapter.read_bytes(), content)
        self.assertTrue(output.is_symlink())

    def test_non_42_training_seeds_and_invalid_repeats_reject_before_launch(self):
        for settings in ({"SEED": "43"}, {"SEEDS": "42 43"}, {"STAGE1_SEED": "43"},
                         {"REPEATS": "0"}, {"REPEAT_START": "0"}):
            with self.subTest(settings=settings):
                result = self.run_script(DRY_RUN=0, **settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.log.exists())
                self.assertFalse(self.output.exists())

    def test_child_failure_stops_the_remaining_jobs(self):
        for stage, count in (("pretrain", 1), ("finetune", 2)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "calls.jsonl"
                result = self.run_script(DRY_RUN=0, FAIL_STAGE=stage,
                                         EXPERIMENT_ROOT=Path(directory) / "outputs", CALL_LOG=log)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual(len(log.read_text().splitlines()), count)

    def test_default_output_root_distinguishes_7b_random_projector_runs(self):
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
        self.assertEqual(len(commands), 15)
        default_root = Path("checkpoints/cka-two-stage/qwen2.5-7b-mlp2x_gelu-random-projector-lambda0.1-seed42")
        for command in commands:
            self.assertTrue(Path(self.flags(command)["--output_dir"]).is_relative_to(default_root))
        self.assertFalse((isolated / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
