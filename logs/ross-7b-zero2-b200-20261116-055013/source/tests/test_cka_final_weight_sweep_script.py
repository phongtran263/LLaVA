"""Check the final-CKA coefficient sweep without model loading or GPU training."""

import json
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from tests import test_cka_location_ablation_script as launch_fixtures


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/v1_5/7b/finetune_cka_final_weight_sweep.sh"
WEIGHTS = ("0.05", "0.1", "0.25", "0.5")


class CkaFinalWeightSweepScriptTests(unittest.TestCase):
    def setUp(self):
        launch_fixtures.CkaLocationAblationScriptTests.setUp(self)
        for key in ("INIT_MODE", "PRETRAIN_CHECKPOINT_DIR", "FINAL_CKA_WEIGHTS", "MM_PROJECTOR_INIT_SEED"):
            self.env.pop(key, None)
        self.checkpoint = self.root / "shared stage1 checkpoint"
        self.checkpoint.mkdir()
        self.adapter = self.checkpoint / "mm_projector.bin"
        self.adapter.write_bytes(b"existing-structural-projector")
        self.config = {
            "_name_or_path": "lmsys/vicuna-7b-v1.5", "model_type": "llava_llama",
            "hidden_size": 4096, "num_hidden_layers": 32, "mm_projector_type": "mlp2x_gelu",
            "mm_vision_tower": "openai/clip-vit-large-patch14-336", "mm_vision_select_layer": -2,
            "mm_hidden_size": 1024, "mm_patch_merge_type": "flat",
            "cka_loss": True, "cka_loss_projector_weight": 0.1,
            "cka_loss_final_hidden_weight": 0.0, "cka_loss_layers": [-1],
        }
        self.config_path = self.checkpoint / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.env["PRETRAIN_CHECKPOINT_DIR"] = str(self.checkpoint)

    commands = staticmethod(launch_fixtures.CkaLocationAblationScriptTests.commands)
    flags = staticmethod(launch_fixtures.CkaLocationAblationScriptTests.flags)
    calls = launch_fixtures.CkaLocationAblationScriptTests.calls

    def run_script(self, *arguments, **settings):
        env = dict(self.env)
        env.update({key: str(value) for key, value in settings.items()})
        return subprocess.run(["bash", str(SCRIPT), *arguments], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=30)

    def output_for(self, weight):
        return self.output / f"final-{weight}" / "llava-finetune"

    def assert_job(self, command, weight):
        flags = self.flags(command)
        expected = {
            "--model_name_or_path": "lmsys/vicuna-7b-v1.5", "--mm_projector_type": "mlp2x_gelu",
            "--pretrain_mm_mlp_adapter": str(self.adapter), "--tune_mm_mlp_adapter": "False",
            "--seed": "42", "--cka_loss": "True", "--cka_loss_layers": "final",
            "--per_device_train_batch_size": "4", "--gradient_accumulation_steps": "16",
            "--output_dir": str(self.output_for(weight)), "--eval_strategy": "no",
        }
        for key, value in expected.items():
            self.assertEqual(flags[key], value, key)
        self.assertEqual(float(flags["--cka_loss_projector_weight"]), 0.1)
        self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), float(weight))
        self.assertNotIn("--mm_projector_init_seed", flags)
        self.assertEqual(command[command.index("--include") + 1], "localhost:2,3")

    def test_dry_run_prints_four_matched_jobs_without_writing_outputs(self):
        self.env.pop("CONDA_PREFIX")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 4, result.stdout)
        differing = {"--cka_loss_final_hidden_weight", "--output_dir", "--run_name"}
        shared = None
        for command, weight in zip(commands, WEIGHTS):
            self.assert_job(command, weight)
            common = {key: value for key, value in self.flags(command).items() if key not in differing}
            shared = common if shared is None else shared
            self.assertEqual(common, shared)
        self.assertEqual(len({self.flags(command)["--run_name"] for command in commands}), 4)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.log.exists())

    def test_actual_child_runs_preserve_the_shared_pretraining_adapter(self):
        result = self.run_script(DRY_RUN=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        self.assertEqual(len(commands), 4)
        for command, weight in zip(commands, WEIGHTS):
            self.assert_job(command, weight)
        self.assertEqual(self.adapter.read_bytes(), b"existing-structural-projector")
        self.assertEqual(self.preflight_log.read_text().splitlines(), ["dependency-check"] * 4)

    def test_custom_weight_subset_is_canonical_and_preserves_requested_order(self):
        result = self.run_script(FINAL_CKA_WEIGHTS="2.5e-1 5e-2", GPU_INCLUDE="localhost:0",
                                 FINETUNE_GPU_INCLUDE="localhost:1")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 2)
        for command, weight in zip(commands, ("0.25", "0.05")):
            flags = self.flags(command)
            self.assertEqual(flags["--output_dir"], str(self.output_for(weight)))
            self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), float(weight))
            self.assertEqual(float(flags["--cka_loss_projector_weight"]), 0.1)
            self.assertEqual(flags["--pretrain_mm_mlp_adapter"], str(self.adapter))
            self.assertEqual(command[command.index("--include") + 1], "localhost:1")
        self.assertFalse(self.output.exists())

    def test_invalid_weights_and_seed_overrides_reject_before_any_launch(self):
        settings = [{"FINAL_CKA_WEIGHTS": value} for value in
                    ("", "0", "-0.1", "nan", "inf", "1e309", "abc", "0.1 1e-1")]
        settings.extend(({"SEED": "43"}, {"SEEDS": "42 43"}, {"DRY_RUN": "yes"}))
        for values in settings:
            with self.subTest(settings=values):
                result = self.run_script(**values)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.log.exists())
                self.assertFalse(self.output.exists())
        self.assertNotEqual(self.run_script("--cka_loss_final_hidden_weight", "0.9").returncode, 0)

    def test_existing_last_output_blocks_all_jobs_in_dry_and_actual_modes(self):
        output = self.output_for("0.5")
        for kind in ("directory", "file", "dangling-symlink"):
            with self.subTest(kind=kind):
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
                    self.assertFalse(self.output_for("0.05").exists())
                output.rmdir() if kind == "directory" else output.unlink()

    def test_child_failure_preserves_status_and_prevents_later_weights(self):
        result = self.run_script(DRY_RUN=0, FAIL_STAGE="finetune")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertFalse(self.output_for("0.1").exists())

    def test_missing_empty_or_nonfile_adapter_rejects_even_during_dry_run(self):
        self.adapter.unlink()
        for kind in ("missing", "empty", "directory"):
            with self.subTest(kind=kind):
                if kind == "empty":
                    self.adapter.touch()
                elif kind == "directory":
                    self.adapter.mkdir()
                for dry_run in (0, 1):
                    result = self.run_script(DRY_RUN=dry_run)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(self.log.exists())
                    self.assertFalse(self.output.exists())
                if kind == "empty":
                    self.adapter.unlink()
                elif kind == "directory":
                    self.adapter.rmdir()

    def test_checkpoint_config_is_required_and_must_match_the_vicuna_projector(self):
        self.config_path.unlink()
        self.assertNotEqual(self.run_script().returncode, 0)
        self.config_path.write_text("not-json")
        self.assertNotEqual(self.run_script().returncode, 0)
        for field, value in (("model_type", "llava_qwen"), ("hidden_size", 3584),
                             ("num_hidden_layers", 28), ("mm_projector_type", "coupling1x_gelu"),
                             ("mm_vision_tower", "other-vision-tower"), ("mm_vision_select_layer", -1),
                             ("mm_hidden_size", 768), ("mm_patch_merge_type", "spatial_unpad")):
            with self.subTest(field=field):
                self.config_path.write_text(json.dumps({**self.config, field: value}))
                result = self.run_script()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.log.exists())
                self.assertFalse(self.output.exists())

    def test_optional_environment_cannot_override_the_plain_sweep_objective(self):
        result = self.run_script(
            CKA_LOSS_ENABLED="False", CKA_PROJECTOR_WEIGHT="0.9", CKA_FINAL_HIDDEN_WEIGHT="0.8",
            CKA_LAYERS="all", CKA_LOSS_TAU="0.5", CKA_LOSS_START_RATIO="0.8", CKA_FINAL_MLP_HEAD="True",
            CKA_ANCHOR_LAYER="12", CKA_VISION_ANCHOR_LAYER="20", CKA_PROJECTOR_VISION_ANCHOR_LAYER="21",
            CKA_FINAL_VISION_ANCHOR_LAYER="22", CKA_LOSS_SUBSET_SELECT_LAYER="8",
            VSP_PCGRAD="True", VSP_NORM_CAP="True", VSP_APPLY_TO_PROJECTOR_ONLY="True",
            VSP_DIAGNOSTICS="True", RESUME_FROM_CHECKPOINT="old-checkpoint", STOP_AFTER_STEP_RATIO="0.8",
            WANDB_RUN_ID="old-run", WANDB_RESUME="must", PRETRAIN_ADAPTER=self.root / "wrong-adapter.bin",
            MM_PROJECTOR_INIT_SEED="123", MM_PROJECTOR_TYPE="coupling1x_gelu",
            MODEL_NAME_OR_PATH="Qwen/Qwen2.5-7B-Instruct", RUN_NAME="old-qwen-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 4)
        for command, weight in zip(commands, WEIGHTS):
            self.assert_job(command, weight)
            flags = self.flags(command)
            for key in ("--use_pcgrad", "--vsp_gradient_diagnostics", "--vsp_asymmetric_pcgrad",
                        "--vsp_apply_to_projector_only", "--vsp_norm_cap", "--cka_loss_final_mlp_head"):
                self.assertEqual(flags[key], "False")
            self.assertEqual(flags["--cka_loss_geometry"], "linear")
            self.assertEqual(float(flags["--cka_loss_tau"]), 0)
            self.assertEqual(float(flags["--cka_loss_start_ratio"]), 0)
            self.assertNotIn("--resume_from_checkpoint", flags)
            self.assertNotIn("--stop_after_step_ratio", flags)
            self.assertTrue(all("anchor" not in key and "subset" not in key for key in flags))

    def test_all_four_jobs_parse_with_native_training_arguments_on_cpu(self):
        import transformers.utils
        import transformers.utils.import_utils

        with mock.patch.object(transformers.utils, "is_flash_attn_2_available", return_value=False), \
             mock.patch.object(transformers.utils.import_utils, "is_flash_attn_2_available", return_value=False):
            from transformers import HfArgumentParser
            from llava.train.train import DataArguments, ModelArguments, TrainingArguments

        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 4)
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        for command, weight in zip(commands, WEIGHTS):
            argv = command[command.index("llava/train/train_mem.py") + 1:]
            with mock.patch.object(TrainingArguments, "__post_init__", return_value=None):
                model, _, training = parser.parse_args_into_dataclasses(argv, look_for_args_file=False)
            self.assertEqual(model.model_name_or_path, "lmsys/vicuna-7b-v1.5")
            self.assertEqual(model.cka_loss_projector_weight, 0.1)
            self.assertEqual(model.cka_loss_final_hidden_weight, float(weight))
            self.assertEqual(model.cka_loss_layers, "final")
            self.assertIsNone(model.mm_projector_init_seed)
            self.assertEqual(training.seed, 42)


if __name__ == "__main__":
    unittest.main()
