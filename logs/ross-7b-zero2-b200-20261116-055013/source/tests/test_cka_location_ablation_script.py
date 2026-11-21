"""Exercise the simple CKA launch contract without training or model downloads."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/v1_5/7b/train_cka_ablations.sh"
VARIANTS = ("baseline", "proj", "final", "full")


class CkaLocationAblationScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "experiment with spaces"
        self.log = self.root / "calls.jsonl"
        conda_bin = self.root / "fake conda/bin"
        conda_bin.mkdir(parents=True)
        # Run the actual Qwen shell launchers. Bypass only the finite dependency
        # version preflight, never arbitrary Python or training code.
        preflight_python = conda_bin / "python"
        preflight_python.write_text("#!" + sys.executable + "\n" + r'''
from pathlib import Path
import sys

if sys.argv[1:] != ["-"]:
    raise SystemExit("The test shim accepts only the Qwen dependency preflight")
source = sys.stdin.read()
required = ("from packaging import version", "import accelerate", "import transformers",
            "transformers.__version__", "accelerate.__version__", "4.51.3", "1.6.0")
if not all(fragment in source for fragment in required) or "train_mem" in source:
    raise SystemExit("Unexpected Python invocation in launcher contract test")
with Path(__file__).with_name("preflight_calls.txt").open("a") as stream:
    stream.write("dependency-check\n")
''')
        preflight_python.chmod(0o755)
        self.preflight_log = conda_bin / "preflight_calls.txt"
        launcher = conda_bin / "deepspeed"
        launcher.write_text("#!" + sys.executable + "\n" + r'''
import json
import os
from pathlib import Path
import sys

argv = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as stream:
    stream.write(json.dumps(argv) + "\n")
def value(flag):
    return argv[argv.index(flag) + 1]
stage1 = "--tune_mm_mlp_adapter" in argv and value("--tune_mm_mlp_adapter") == "True"
if os.environ.get("FAIL_STAGE") == ("pretrain" if stage1 else "finetune"):
    raise SystemExit(7)
output = Path(value("--output_dir"))
output.mkdir(parents=True, exist_ok=True)
artifact_mode = os.environ.get("ADAPTER_MODE", "normal")
if artifact_mode.startswith("cka-"):
    artifact_mode = artifact_mode[4:] if value("--cka_loss") == "True" else "normal"
if stage1 and artifact_mode != "missing":
    (output / "mm_projector.bin").write_bytes(
        b"" if artifact_mode == "empty" else b"shared-stage1-adapter")
''')
        launcher.chmod(0o755)
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith(("CKA_", "VSP_", "WANDB_")):
                self.env.pop(key)
        for key in (
            "STAGE", "VARIANTS", "SEEDS", "SEED", "STAGE1_SEED", "MODEL_NAME_OR_PATH",
            "RUN_NAME", "MM_PROJECTOR_TYPE", "EXPERIMENT_ROOT", "PRETRAIN_OUTPUT_DIR",
            "PRETRAIN_ADAPTER", "DRY_RUN", "GPU_INCLUDE", "PRETRAIN_GPU_INCLUDE",
            "FINETUNE_GPU_INCLUDE", "FAIL_STAGE", "ADAPTER_MODE", "INIT_MODE",
            "BASELINE_PRETRAIN_OUTPUT_DIR", "CKA_PRETRAIN_OUTPUT_DIR",
            "BASELINE_PRETRAIN_ADAPTER", "CKA_PRETRAIN_ADAPTER",
            "PRETRAIN_CKA_PROJECTOR_WEIGHT", "CKA_PROJECTOR_WEIGHT", "CKA_FINAL_HIDDEN_WEIGHT",
            "CONVERSATION_VERSION", "DATA_PATH", "IMAGE_FOLDER", "VISION_TOWER",
            "MM_VISION_SELECT_LAYER", "OUTPUT_DIR", "SAVE_STRATEGY", "LOGGING_STEPS",
            "RESUME_FROM_CHECKPOINT", "STOP_AFTER_STEP_RATIO", "DEEPSPEED_CONFIG",
            "PER_DEVICE_TRAIN_BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS",
            "PRETRAIN_PER_DEVICE_TRAIN_BATCH_SIZE", "PRETRAIN_GRADIENT_ACCUMULATION_STEPS",
        ):
            self.env.pop(key, None)
        self.env.update(
            CONDA_PREFIX=str(conda_bin.parent), EXPERIMENT_ROOT=str(self.output),
            CALL_LOG=str(self.log), DRY_RUN="1", INIT_MODE="shared",
        )

    def run_script(self, *arguments, **settings):
        env = dict(self.env)
        env.update({key: str(value) for key, value in settings.items()})
        return subprocess.run(["bash", str(SCRIPT), *arguments], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=30)

    @staticmethod
    def commands(output):
        result = []
        for line in output.splitlines():
            if "llava/train/train_mem.py" not in line or line.startswith("#"):
                continue
            argv = shlex.split(line)
            if "llava/train/train_mem.py" in argv:
                result.append(argv)
        return result

    @staticmethod
    def flags(argv):
        arguments = argv[argv.index("llava/train/train_mem.py") + 1:]
        return dict(zip(arguments[::2], arguments[1::2]))

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def stage1_output(self, seed=42):
        return self.output / f"stage1-seed{seed}" / "llava-pretrain"

    def stage2_output(self, variant, seed=42):
        return self.output / f"seed-{seed}" / variant / "llava-finetune"

    def paired_stage1_output(self, kind, seed=42):
        return self.output / "paired" / f"stage1-seed{seed}" / kind / "llava-pretrain"

    def paired_stage2_output(self, variant, seed=42):
        return self.output / "paired" / f"seed-{seed}" / variant / "llava-finetune"

    def assert_qwen_recipe(self, flags, *, stage1):
        self.assertEqual(flags["--model_name_or_path"], "lmsys/vicuna-7b-v1.5")
        self.assertEqual(flags["--mm_projector_type"], "mlp2x_gelu")
        self.assertEqual(flags["--per_device_train_batch_size"], "32" if stage1 else "8")
        self.assertEqual(flags["--gradient_accumulation_steps"], "8" if stage1 else "16")
        self.assertEqual(flags["--save_strategy"], "no")
        self.assertEqual(flags["--logging_steps"], "1" if stage1 else "10")

    def test_dry_run_prints_shared_stage1_and_only_four_objective_changes(self):
        self.env.pop("CONDA_PREFIX")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 5, result.stdout)
        stage1, *stage2 = [self.flags(command) for command in commands]
        self.assertEqual(stage1["--tune_mm_mlp_adapter"], "True")
        self.assertEqual(stage1["--version"], "plain")
        self.assertEqual(stage1["--cka_loss"], "True")
        self.assertEqual(float(stage1["--cka_loss_projector_weight"]), 0.1)
        self.assertEqual(float(stage1["--cka_loss_final_hidden_weight"]), 0.0)
        self.assertEqual(stage1["--cka_loss_layers"], "-1")
        self.assertEqual(stage1["--output_dir"], str(self.stage1_output()))
        self.assertNotIn("--pretrain_mm_mlp_adapter", stage1)
        self.assert_qwen_recipe(stage1, stage1=True)
        expected = (("False", 0, 0, "-1"), ("True", 0.01, 0, "-1"),
                    ("True", 0, 0.01, "final"), ("True", 0.01, 0.01, "final"))
        differing = {"--cka_loss", "--cka_loss_projector_weight", "--cka_loss_final_hidden_weight",
                     "--cka_loss_layers", "--output_dir", "--run_name"}
        common = {key: value for key, value in stage2[0].items() if key not in differing}
        for variant, flags, (enabled, projector, final, layers) in zip(VARIANTS, stage2, expected):
            with self.subTest(variant=variant):
                self.assertEqual(flags["--cka_loss"], enabled)
                self.assertEqual(float(flags["--cka_loss_projector_weight"]), projector)
                self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), final)
                self.assertEqual(flags["--cka_loss_layers"], layers)
                self.assertEqual(flags["--pretrain_mm_mlp_adapter"],
                                 str(self.stage1_output() / "mm_projector.bin"))
                self.assertEqual(flags["--output_dir"], str(self.stage2_output(variant)))
                self.assertEqual({key: value for key, value in flags.items() if key not in differing}, common)
                self.assert_qwen_recipe(flags, stage1=False)
        for flags in (stage1, *stage2):
            self.assertEqual(flags["--model_name_or_path"], "lmsys/vicuna-7b-v1.5")
            self.assertEqual(flags["--mm_projector_type"], "mlp2x_gelu")
            self.assertEqual(flags["--eval_strategy"], "no")
            self.assertNotIn("--evaluation_strategy", flags)
            for key, value in flags.items():
                self.assertNotIn("anchor", key)
                self.assertNotIn("subset", key)
                if key in ("--use_pcgrad", "--vsp_asymmetric_pcgrad", "--vsp_norm_cap",
                           "--vsp_apply_to_projector_only", "--vsp_gradient_diagnostics",
                           "--protected_cka_refinement", "--protected_stage2", "--cka_loss_final_mlp_head"):
                    self.assertEqual(value, "False", key)
            if "--cka_loss_geometry" in flags:
                self.assertEqual(flags["--cka_loss_geometry"], "linear")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_selection_seeds_and_gpu_overrides_keep_one_common_initialization(self):
        result = self.run_script(STAGE="finetune", VARIANTS="final full", SEEDS="43 44",
                                 GPU_INCLUDE="localhost:2", STAGE1_SEED=17)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 4, result.stdout)
        flags = [self.flags(command) for command in commands]
        self.assertEqual({entry["--seed"] for entry in flags}, {"43", "44"})
        self.assertEqual({entry["--pretrain_mm_mlp_adapter"] for entry in flags},
                         {str(self.stage1_output(17) / "mm_projector.bin")})
        self.assertEqual({entry["--output_dir"] for entry in flags},
                         {str(self.stage2_output(variant, seed)) for variant in ("final", "full") for seed in (43, 44)})
        for command in commands:
            self.assertEqual(command[command.index("--include") + 1], "localhost:2")
        pretrain = self.run_script(STAGE="pretrain", SEED=123, STAGE1_SEED=17)
        self.assertEqual(pretrain.returncode, 0, pretrain.stderr)
        command, = self.commands(pretrain.stdout)
        self.assertEqual(self.flags(command)["--seed"], "17")
        self.assertEqual(command[command.index("--include") + 1], "localhost:1,2")
        fallback = self.run_script(STAGE="finetune", VARIANTS="baseline", SEED=123)
        self.assertEqual(fallback.returncode, 0, fallback.stderr)
        command, = self.commands(fallback.stdout)
        self.assertEqual(self.flags(command)["--seed"], "123")
        self.assertEqual(command[command.index("--include") + 1], "localhost:1,2")

    def test_actual_pipeline_uses_the_generated_adapter_without_training(self):
        result = self.run_script(DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [self.flags(command) for command in self.calls()]
        self.assertEqual(len(calls), 5)
        adapter = self.stage1_output() / "mm_projector.bin"
        self.assertEqual(adapter.read_bytes(), b"shared-stage1-adapter")
        self.assertTrue(all(command["--pretrain_mm_mlp_adapter"] == str(adapter) for command in calls[1:]))
        self.assert_qwen_recipe(calls[0], stage1=True)
        for flags in calls[1:]:
            self.assert_qwen_recipe(flags, stage1=False)
        self.assertEqual(self.preflight_log.read_text().splitlines(), ["dependency-check"] * 4)

    def test_child_launchers_ignore_inherited_optional_techniques_and_checkpoint_overrides(self):
        result = self.run_script(
            CKA_LOSS="False", CKA_LOSS_ENABLED="False", CKA_LAYERS="all",
            CKA_LOSS_TAU="0.9", CKA_LOSS_START_RATIO="0.8", CKA_LOSS_GEOMETRY="row_normalized",
            CKA_FINAL_MLP_HEAD="True", CKA_ANCHOR_LAYER="12", CKA_VISION_ANCHOR_LAYER="20",
            CKA_PROJECTOR_VISION_ANCHOR_LAYER="21", CKA_FINAL_VISION_ANCHOR_LAYER="22",
            CKA_LOSS_SUBSET_SELECT_LAYER="9", CKA_LOSS_SUBSET_QUERY_TOKENS="instruction",
            VSP_PCGRAD="True", VSP_APPLY_TO_PROJECTOR_ONLY="True", VSP_NORM_CAP="True",
            VSP_DIAGNOSTICS="True", RESUME_FROM_CHECKPOINT="old-checkpoint", STOP_AFTER_STEP_RATIO="0.8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [self.flags(command) for command in self.commands(result.stdout)]
        self.assertEqual(len(commands), 5)
        self.assertEqual([flags["--cka_loss_layers"] for flags in commands], ["-1", "-1", "-1", "final", "final"])
        for flags in commands:
            self.assertEqual(flags["--cka_loss_geometry"], "linear")
            self.assertEqual(float(flags["--cka_loss_tau"]), 0)
            self.assertEqual(float(flags["--cka_loss_start_ratio"]), 0)
            for key in ("--use_pcgrad", "--vsp_gradient_diagnostics", "--vsp_asymmetric_pcgrad",
                        "--vsp_apply_to_projector_only", "--vsp_norm_cap", "--cka_loss_final_mlp_head"):
                self.assertEqual(flags[key], "False")
            for key in flags:
                self.assertNotIn("anchor", key)
                self.assertNotIn("subset", key)
            self.assertNotIn("--resume_from_checkpoint", flags)
            self.assertNotIn("--stop_after_step_ratio", flags)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.output.exists())

    def test_paired_dry_run_matches_stage1_recipe_and_routes_the_two_adapters(self):
        result = self.run_script(INIT_MODE="paired", STAGE1_SEED=17, SEEDS="43")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [self.flags(command) for command in self.commands(result.stdout)]
        self.assertEqual(len(commands), 6, result.stdout)
        baseline, structural, *stage2 = commands
        differing = {"--cka_loss", "--cka_loss_projector_weight", "--output_dir", "--run_name"}
        self.assertEqual({key: value for key, value in baseline.items() if key not in differing},
                         {key: value for key, value in structural.items() if key not in differing})
        for kind, flags, enabled, weight in (("baseline", baseline, "False", 0),
                                             ("cka", structural, "True", 0.1)):
            with self.subTest(kind=kind):
                self.assertEqual(flags["--cka_loss"], enabled)
                self.assertEqual(float(flags["--cka_loss_projector_weight"]), weight)
                self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), 0)
                self.assertEqual(flags["--cka_loss_layers"], "-1")
                self.assertEqual(flags["--tune_mm_mlp_adapter"], "True")
                self.assertEqual(flags["--seed"], "17")
                self.assertEqual(flags["--output_dir"], str(self.paired_stage1_output(kind, 17)))
                self.assertNotIn("--pretrain_mm_mlp_adapter", flags)
        for variant, flags in zip(VARIANTS, stage2):
            kind = "baseline" if variant == "baseline" else "cka"
            self.assertEqual(flags["--pretrain_mm_mlp_adapter"],
                             str(self.paired_stage1_output(kind, 17) / "mm_projector.bin"))
            self.assertEqual(flags["--seed"], "43")
            self.assertEqual(flags["--output_dir"], str(self.paired_stage2_output(variant, 43)))
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_paired_actual_pipeline_pretrains_both_before_any_finetune(self):
        result = self.run_script(INIT_MODE="paired", DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [self.flags(command) for command in self.calls()]
        self.assertEqual(len(calls), 6)
        self.assertEqual([entry["--tune_mm_mlp_adapter"] for entry in calls],
                         ["True", "True", "False", "False", "False", "False"])
        self.assertEqual([entry["--output_dir"] for entry in calls[:2]],
                         [str(self.paired_stage1_output(kind)) for kind in ("baseline", "cka")])
        for variant, flags in zip(VARIANTS, calls[2:]):
            kind = "baseline" if variant == "baseline" else "cka"
            adapter = self.paired_stage1_output(kind) / "mm_projector.bin"
            self.assertTrue(adapter.is_file())
            self.assertEqual(flags["--pretrain_mm_mlp_adapter"], str(adapter))
        for mode in ("cka-missing", "cka-empty"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                call_log = Path(directory) / "calls.jsonl"
                result = self.run_script(INIT_MODE="paired", DRY_RUN=0, ADAPTER_MODE=mode,
                                         EXPERIMENT_ROOT=Path(directory) / "outputs", CALL_LOG=call_log)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(call_log.read_text().splitlines()), 2)

    def test_paired_finetune_requires_only_adapters_for_selected_variants(self):
        for variants, kind in (("baseline", "baseline"), ("proj final full", "cka")):
            with self.subTest(variants=variants), tempfile.TemporaryDirectory() as directory:
                adapter = Path(directory) / "selected-adapter.bin"
                adapter.write_bytes(b"existing-stage1")
                call_log = Path(directory) / "calls.jsonl"
                settings = {f"{kind.upper()}_PRETRAIN_ADAPTER": adapter}
                result = self.run_script(INIT_MODE="paired", STAGE="finetune", DRY_RUN=0,
                                         VARIANTS=variants, EXPERIMENT_ROOT=Path(directory) / "outputs",
                                         CALL_LOG=call_log, **settings)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [self.flags(json.loads(line)) for line in call_log.read_text().splitlines()]
                self.assertEqual(len(calls), len(variants.split()))
                self.assertTrue(all(entry["--pretrain_mm_mlp_adapter"] == str(adapter) for entry in calls))

    def test_paired_finetune_validates_all_required_adapters_before_any_launch(self):
        for present, variants in (("baseline", "baseline full"), ("cka", "full baseline")):
            with self.subTest(present=present), tempfile.TemporaryDirectory() as directory:
                baseline = Path(directory) / "baseline.bin"
                structural = Path(directory) / "cka.bin"
                (baseline if present == "baseline" else structural).write_bytes(b"one-existing-adapter")
                call_log = Path(directory) / "calls.jsonl"
                output = Path(directory) / "outputs"
                result = self.run_script(INIT_MODE="paired", STAGE="finetune", DRY_RUN=0,
                                         VARIANTS=variants, EXPERIMENT_ROOT=output, CALL_LOG=call_log,
                                         BASELINE_PRETRAIN_ADAPTER=baseline, CKA_PRETRAIN_ADAPTER=structural)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(call_log.exists())
                self.assertFalse(output.exists())

    def test_paired_existing_destinations_prevent_all_launches(self):
        for output in (self.paired_stage1_output("baseline"), self.paired_stage1_output("cka"),
                       self.paired_stage2_output("full")):
            with self.subTest(output=output):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.symlink_to(self.root / "missing-target")
                for dry_run in (0, 1):
                    result = self.run_script(INIT_MODE="paired", DRY_RUN=dry_run)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(self.log.exists())
                output.unlink()

    def test_paired_overrides_are_explicit_and_ambiguous_or_aliased_adapters_reject(self):
        baseline_output, structural_output = self.root / "baseline stage1", self.root / "cka stage1"
        result = self.run_script(INIT_MODE="paired", STAGE="pretrain",
                                 BASELINE_PRETRAIN_OUTPUT_DIR=baseline_output,
                                 CKA_PRETRAIN_OUTPUT_DIR=structural_output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([self.flags(command)["--output_dir"] for command in self.commands(result.stdout)],
                         [str(baseline_output), str(structural_output)])
        for settings in ({"PRETRAIN_OUTPUT_DIR": self.root / "ambiguous"},
                         {"PRETRAIN_ADAPTER": self.root / "ambiguous.bin"},
                         {"BASELINE_PRETRAIN_ADAPTER": self.root / "unrelated-baseline.bin"},
                         {"CKA_PRETRAIN_ADAPTER": self.root / "unrelated-cka.bin"}):
            with self.subTest(settings=settings):
                self.assertNotEqual(self.run_script(INIT_MODE="paired", **settings).returncode, 0)
        baseline = self.root / "existing-baseline.bin"
        baseline.write_bytes(b"one-checkpoint")
        aliases = [baseline, self.root / "symlink.bin", self.root / "hardlink.bin"]
        aliases[1].symlink_to(baseline)
        os.link(baseline, aliases[2])
        for alias in aliases:
            with self.subTest(alias=alias):
                result = self.run_script(INIT_MODE="paired", STAGE="finetune", DRY_RUN=0,
                                         BASELINE_PRETRAIN_ADAPTER=baseline, CKA_PRETRAIN_ADAPTER=alias)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.log.exists())

    def test_default_initialization_mode_preserves_shared_paths_and_adapter(self):
        self.env.pop("INIT_MODE")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [self.flags(command) for command in self.commands(result.stdout)]
        self.assertEqual(len(commands), 5)
        self.assertEqual(commands[0]["--output_dir"], str(self.stage1_output()))
        self.assertTrue(all(entry["--pretrain_mm_mlp_adapter"] == str(self.stage1_output() / "mm_projector.bin")
                            for entry in commands[1:]))

    def test_custom_weights_apply_only_to_their_enabled_locations_in_both_modes(self):
        for mode in ("shared", "paired"):
            with self.subTest(mode=mode):
                result = self.run_script(INIT_MODE=mode, PRETRAIN_CKA_PROJECTOR_WEIGHT="2.5e-2",
                                         CKA_PROJECTOR_WEIGHT="0.2", CKA_FINAL_HIDDEN_WEIGHT="3e-1")
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = [self.flags(command) for command in self.commands(result.stdout)]
                stage1, stage2 = commands[:-4], commands[-4:]
                self.assertEqual(len(stage1), 1 if mode == "shared" else 2)
                for index, flags in enumerate(stage1):
                    baseline = mode == "paired" and index == 0
                    self.assertEqual(flags["--cka_loss"], "False" if baseline else "True")
                    self.assertEqual(float(flags["--cka_loss_projector_weight"]), 0 if baseline else 0.025)
                    self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), 0)
                for variant, flags, expected in zip(VARIANTS, stage2, ((0, 0), (0.2, 0), (0, 0.3), (0.2, 0.3))):
                    self.assertEqual((float(flags["--cka_loss_projector_weight"]),
                                      float(flags["--cka_loss_final_hidden_weight"])), expected)
                    if mode == "paired":
                        kind = "baseline" if variant == "baseline" else "cka"
                        adapter = self.paired_stage1_output(kind) / "mm_projector.bin"
                    else:
                        adapter = self.stage1_output() / "mm_projector.bin"
                    self.assertEqual(flags["--pretrain_mm_mlp_adapter"], str(adapter))
                    output = self.paired_stage2_output(variant) if mode == "paired" else self.stage2_output(variant)
                    self.assertEqual(flags["--output_dir"], str(output))
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_zero_weights_disable_unneeded_losses_and_decoder_hooks(self):
        for mode in ("shared", "paired"):
            with self.subTest(mode=mode):
                result = self.run_script(INIT_MODE=mode, PRETRAIN_CKA_PROJECTOR_WEIGHT="0",
                                         CKA_PROJECTOR_WEIGHT="0.0", CKA_FINAL_HIDDEN_WEIGHT="0e0")
                self.assertEqual(result.returncode, 0, result.stderr)
                for flags in map(self.flags, self.commands(result.stdout)):
                    self.assertEqual(flags["--cka_loss"], "False")
                    self.assertEqual(float(flags["--cka_loss_projector_weight"]), 0)
                    self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), 0)
                    self.assertEqual(flags["--cka_loss_layers"], "-1")
        for projector, final, expected in (
            ("0.2", "0", (("False", "-1"), ("True", "-1"), ("False", "-1"), ("True", "-1"))),
            ("0", "0.3", (("False", "-1"), ("False", "-1"), ("True", "final"), ("True", "final"))),
        ):
            with self.subTest(projector=projector, final=final):
                result = self.run_script(STAGE="finetune", CKA_PROJECTOR_WEIGHT=projector,
                                         CKA_FINAL_HIDDEN_WEIGHT=final)
                self.assertEqual(result.returncode, 0, result.stderr)
                flags = [self.flags(command) for command in self.commands(result.stdout)]
                self.assertEqual([(entry["--cka_loss"], entry["--cka_loss_layers"]) for entry in flags], list(expected))

    def test_invalid_weights_reject_before_any_launch_or_command_print(self):
        for variable in ("PRETRAIN_CKA_PROJECTOR_WEIGHT", "CKA_PROJECTOR_WEIGHT", "CKA_FINAL_HIDDEN_WEIGHT"):
            for value in ("-0.1", "nan", "inf", "-inf", "1e309", "not-a-number", ""):
                for dry_run in (0, 1):
                    with self.subTest(variable=variable, value=value, dry_run=dry_run):
                        result = self.run_script(DRY_RUN=dry_run, **{variable: value})
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(self.commands(result.stdout), [])
                        self.assertFalse(self.log.exists())
                        self.assertFalse(self.output.exists())

    def test_finetune_weight_overrides_preserve_existing_pretraining_adapter(self):
        adapter = self.root / "existing-pretrained-projector.bin"
        adapter.write_bytes(b"fixed-initialization")
        result = self.run_script(STAGE="finetune", DRY_RUN=0, PRETRAIN_ADAPTER=adapter,
                                 PRETRAIN_CKA_PROJECTOR_WEIGHT="0.9", CKA_PROJECTOR_WEIGHT="0.25",
                                 CKA_FINAL_HIDDEN_WEIGHT="0.4")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [self.flags(command) for command in self.calls()]
        self.assertEqual(len(commands), 4)
        self.assertEqual(adapter.read_bytes(), b"fixed-initialization")
        for flags in commands:
            self.assertEqual(flags["--tune_mm_mlp_adapter"], "False")
            self.assertEqual(flags["--pretrain_mm_mlp_adapter"], str(adapter))
            self.assertIn(float(flags["--cka_loss_projector_weight"]), (0, 0.25))
            self.assertIn(float(flags["--cka_loss_final_hidden_weight"]), (0, 0.4))

    def test_child_failure_preserves_status_and_prevents_later_jobs(self):
        for stage, expected_calls in (("pretrain", 1), ("finetune", 2)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                call_log = Path(directory) / "calls.jsonl"
                result = self.run_script(DRY_RUN=0, FAIL_STAGE=stage,
                                         EXPERIMENT_ROOT=Path(directory) / "outputs", CALL_LOG=call_log)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual(len(call_log.read_text().splitlines()), expected_calls)

    def test_missing_or_empty_adapter_stops_before_stage2(self):
        for mode in ("missing", "empty"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = self.run_script(DRY_RUN=0, EXPERIMENT_ROOT=Path(directory) / "outputs",
                                         ADAPTER_MODE=mode, CALL_LOG=Path(directory) / "calls.jsonl")
                self.assertNotEqual(result.returncode, 0)
                calls = (Path(directory) / "calls.jsonl").read_text().splitlines()
                self.assertEqual(len(calls), 1)
        result = self.run_script(STAGE="finetune", DRY_RUN=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())
        directory_adapter = self.root / "adapter-is-a-directory"
        directory_adapter.mkdir()
        result = self.run_script(STAGE="finetune", DRY_RUN=0, PRETRAIN_ADAPTER=directory_adapter)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_existing_selected_output_stops_all_launches_including_dangling_symlink(self):
        for stage, output in (("all", self.stage2_output("full")),
                              ("pretrain", self.stage1_output())):
            for kind in ("directory", "file", "symlink"):
                with self.subTest(stage=stage, kind=kind):
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if kind == "directory":
                        output.mkdir()
                    elif kind == "file":
                        output.write_text("preserve")
                    else:
                        output.symlink_to(self.root / "missing-target")
                    for dry_run in (0, 1):
                        result = self.run_script(STAGE=stage, DRY_RUN=dry_run)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertFalse(self.log.exists())
                    if kind == "directory":
                        output.rmdir()
                    else:
                        output.unlink()

    def test_invalid_selection_or_cli_override_never_launches(self):
        for settings in ({"STAGE": "unknown"}, {"VARIANTS": "full unknown"}, {"DRY_RUN": "yes"}):
            with self.subTest(settings=settings):
                result = self.run_script(**settings)
                self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(self.run_script("--max_steps", "1").returncode, 0)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.output.exists())

    def test_generated_arguments_parse_with_native_dataclasses_on_cpu(self):
        import transformers.utils
        import transformers.utils.import_utils

        with mock.patch.object(transformers.utils, "is_flash_attn_2_available", return_value=False), \
             mock.patch.object(transformers.utils.import_utils, "is_flash_attn_2_available", return_value=False):
            from transformers import HfArgumentParser
            from llava.train.train import DataArguments, ModelArguments, TrainingArguments

        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        for settings in ({}, {"INIT_MODE": "paired", "PRETRAIN_CKA_PROJECTOR_WEIGHT": "3e-2",
                              "CKA_PROJECTOR_WEIGHT": "0.25", "CKA_FINAL_HIDDEN_WEIGHT": "0.5"}):
            result = self.run_script(**settings)
            self.assertEqual(result.returncode, 0, result.stderr)
            for command in self.commands(result.stdout):
                argv = command[command.index("llava/train/train_mem.py") + 1:]
                flags = self.flags(command)
                with mock.patch.object(TrainingArguments, "__post_init__", return_value=None):
                    model, _, training = parser.parse_args_into_dataclasses(argv, look_for_args_file=False)
                self.assertEqual(model.cka_loss_projector_weight, float(flags["--cka_loss_projector_weight"]))
                self.assertEqual(model.cka_loss_final_hidden_weight, float(flags["--cka_loss_final_hidden_weight"]))
                self.assertFalse(model.use_pcgrad)
                self.assertFalse(model.vsp_asymmetric_pcgrad)
                self.assertFalse(model.vsp_norm_cap)
                self.assertFalse(model.vsp_apply_to_projector_only)
                self.assertFalse(model.cka_loss_final_mlp_head)
                self.assertFalse(training.protected_cka_refinement)
                self.assertFalse(training.protected_stage2)
                self.assertEqual(model.cka_loss_geometry, "linear")
                self.assertIsNone(model.cka_loss_anchor_layer)
                self.assertIsNone(model.cka_loss_subset_select_layer)
                self.assertEqual(training.eval_strategy, "no")


if __name__ == "__main__":
    unittest.main()
