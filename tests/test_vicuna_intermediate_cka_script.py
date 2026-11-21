"""Check the Vicuna intermediate-CKA launch contract on CPU, without training."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/v1_5/7b/finetune_vicuna_7b_intermediate_cka.sh"
SETTINGS = ("intermediate_only", "projector_intermediate")
PARENT_CONFIG = {
    "_name_or_path": "lmsys/vicuna-7b-v1.5",
    "model_type": "llava_llama",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "mm_hidden_size": 1024,
    "mm_projector_type": "mlp2x_gelu",
    "mm_vision_tower": "openai/clip-vit-large-patch14-336",
    "mm_vision_select_layer": -2,
    "mm_vision_select_feature": "patch",
}


class VicunaIntermediateCkaScriptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repository with spaces"
        scripts = self.repo / "scripts/v1_5/7b"
        scripts.mkdir(parents=True)
        self.script = scripts / SCRIPT.name
        shutil.copyfile(SCRIPT, self.script)
        shutil.copyfile(SCRIPT.with_name("finetune_qwen2_5.sh"),
                        scripts / "finetune_qwen2_5.sh")
        self.baseline = self.make_parent(
            self.repo / "checkpoints/full-ft/baseline-7b/llava-pretrain")
        self.cka = self.make_parent(
            self.repo / "checkpoints/full-ft/cka-proj-7b/llava-pretrain")
        self.output = self.repo / "checkpoints/full-ft/vicuna-7b-intermediate-cka"
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith(("CKA_", "VSP_", "WANDB_", "PROTECTED_")):
                self.env.pop(key)
        for key in (
            "BASELINE_PRETRAIN_DIR", "CONDA_PREFIX", "CONVERSATION_VERSION",
            "DATA_PATH", "DATALOADER_NUM_WORKERS", "DEEPSPEED_CONFIG", "DRY_RUN",
            "FINETUNE_GRADIENT_CHECKPOINTING", "GPU_INCLUDE",
            "GRADIENT_ACCUMULATION_STEPS", "GRADIENT_CHECKPOINTING", "IMAGE_FOLDER",
            "LOGGING_STEPS", "MASTER_PORT", "MM_PROJECTOR_INIT_SEED",
            "MM_PROJECTOR_TYPE", "MM_VISION_SELECT_LAYER", "MODEL_NAME_OR_PATH",
            "OUTPUT_DIR", "OUTPUT_ROOT", "PER_DEVICE_TRAIN_BATCH_SIZE",
            "PRETRAIN_ADAPTER", "REPORT_TO", "RESUME_FROM_CHECKPOINT", "RUN_NAME",
            "SAVE_STRATEGY", "SEED", "SETTINGS", "STOP_AFTER_STEP_RATIO",
            "TRAIN_ENV", "VISION_TOWER",
            "OPTIM", "DATALOADER_PREFETCH_FACTOR", "DATALOADER_PERSISTENT_WORKERS",
            "ACCELERATOR_CONFIG",
        ):
            self.env.pop(key, None)
        self.env.update(DRY_RUN="1", PYTHON_CHECK=sys.executable)

    @staticmethod
    def make_parent(directory, **overrides):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(PARENT_CONFIG | overrides))
        # The launcher must only inspect metadata and a nonempty artifact; it
        # must not import torch or load projector tensors during preflight.
        (directory / "mm_projector.bin").write_bytes(b"fake-stage1-projector")
        return directory

    def run_script(self, *arguments, **settings):
        env = self.env | {key: str(value) for key, value in settings.items()}
        return subprocess.run(["bash", str(self.script), *arguments],
                              cwd=self.root, env=env, capture_output=True,
                              text=True, timeout=30)

    @staticmethod
    def commands(output):
        commands = []
        for line in output.splitlines():
            if "llava/train/train_mem.py" not in line or line.startswith("#"):
                continue
            argv = shlex.split(line)
            if "llava/train/train_mem.py" in argv:
                commands.append(argv)
        return commands

    @staticmethod
    def flags(command):
        arguments = command[command.index("llava/train/train_mem.py") + 1:]
        return dict(zip(arguments[::2], arguments[1::2]))

    def output_for(self, setting, layer=16, root=None):
        return (root or self.output) / f"layer-{layer}" / setting / "llava-finetune"

    def assert_rejected_before_launch(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.commands(result.stdout), [], result.stdout)

    def assert_recipe(self, command, setting, *, layer=16, weight=0.1,
                      parent=None, output_root=None):
        flags = self.flags(command)
        expected_parent = parent or (self.baseline if setting == "intermediate_only" else self.cka)
        self.assertEqual(flags["--pretrain_mm_mlp_adapter"],
                         str(expected_parent / "mm_projector.bin"))
        self.assertEqual(flags["--output_dir"],
                         str(self.output_for(setting, layer, output_root)))
        self.assertEqual(flags["--model_name_or_path"], "lmsys/vicuna-7b-v1.5")
        self.assertEqual(flags["--version"], "v1")
        self.assertEqual(flags["--mm_projector_type"], "mlp2x_gelu")
        self.assertEqual(flags["--vision_tower"], "openai/clip-vit-large-patch14-336")
        self.assertEqual(flags["--mm_vision_select_layer"], "-2")
        self.assertEqual(flags["--mm_vision_select_feature"], "patch")
        self.assertEqual(flags["--cka_loss"], "True")
        self.assertEqual(flags["--cka_loss_layers"], str(layer))
        self.assertEqual(float(flags["--cka_loss_final_hidden_weight"]), weight)
        self.assertEqual(float(flags["--cka_loss_projector_weight"]),
                         weight if setting == "projector_intermediate" else 0)
        self.assertEqual(float(flags["--cka_loss_tau"]), 0)
        self.assertEqual(float(flags["--cka_loss_start_ratio"]), 0)
        self.assertEqual(flags["--cka_loss_geometry"], "linear")
        self.assertEqual(flags["--save_strategy"], "no")
        for flag in (
            "--tune_mm_mlp_adapter", "--freeze_backbone", "--freeze_mm_mlp_adapter",
            "--lora_enable", "--cka_loss_final_mlp_head", "--use_pcgrad",
            "--vsp_asymmetric_pcgrad", "--vsp_norm_cap", "--vsp_apply_to_projector_only",
            "--vsp_gradient_diagnostics", "--protected_cka_refinement",
            "--protected_stage2", "--log_gradient_norms",
        ):
            self.assertEqual(flags[flag], "False", flag)
        for flag in flags:
            self.assertNotIn("anchor", flag)
            self.assertNotIn("subset", flag)
        for flag in ("--resume_from_checkpoint", "--stop_after_step_ratio",
                     "--mm_projector_init_seed"):
            self.assertNotIn(flag, flags)

    def test_defaults_use_two_saved_parents_and_safe_single_gpu_recipe(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 2, result.stdout)
        for command, setting in zip(commands, SETTINGS):
            self.assert_recipe(command, setting)
            flags = self.flags(command)
            self.assertEqual(command[command.index("--include") + 1], "localhost:3")
            self.assertEqual(flags["--per_device_train_batch_size"], "4")
            self.assertEqual(flags["--gradient_accumulation_steps"], "32")
            self.assertEqual(int(flags["--per_device_train_batch_size"]) *
                             int(flags["--gradient_accumulation_steps"]), 128)
            self.assertEqual(flags["--gradient_checkpointing"], "True")
        self.assertEqual(len({self.flags(command)["--run_name"] for command in commands}), 2)
        self.assertFalse(self.output.exists())
        for parent in (self.baseline, self.cka):
            self.assertEqual((parent / "mm_projector.bin").read_bytes(), b"fake-stage1-projector")

    def test_setting_subsets_and_reordering_preserve_parent_selection(self):
        for selected in ((SETTINGS[0],), (SETTINGS[1],), SETTINGS[::-1]):
            with self.subTest(selected=selected):
                result = self.run_script(SETTINGS=" ".join(selected))
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = self.commands(result.stdout)
                self.assertEqual(len(commands), len(selected), result.stdout)
                for command, setting in zip(commands, selected):
                    self.assert_recipe(command, setting)

    def test_custom_parent_directories_and_output_root_accept_spaces(self):
        baseline = self.make_parent(self.root / "custom baseline parent")
        cka = self.make_parent(self.root / "custom cka parent")
        output = self.root / "custom output root"
        result = self.run_script(BASELINE_PRETRAIN_DIR=baseline,
                                 CKA_PRETRAIN_DIR=cka, OUTPUT_ROOT=output)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 2, result.stdout)
        for command, setting, parent in zip(commands, SETTINGS, (baseline, cka)):
            self.assert_recipe(command, setting, parent=parent, output_root=output)
        self.assertFalse(output.exists())

    def test_only_selected_parent_and_output_are_required_to_be_usable(self):
        for selected, unused_parent, unselected in (
            (SETTINGS[0], "CKA_PRETRAIN_DIR", SETTINGS[1]),
            (SETTINGS[1], "BASELINE_PRETRAIN_DIR", SETTINGS[0]),
        ):
            with self.subTest(selected=selected):
                output = self.root / selected
                self.output_for(unselected, root=output).mkdir(parents=True)
                result = self.run_script(SETTINGS=selected, OUTPUT_ROOT=output,
                                         **{unused_parent: self.root / "missing parent"})
                self.assertEqual(result.returncode, 0, result.stderr)
                command, = self.commands(result.stdout)
                self.assert_recipe(command, selected, output_root=output)
                self.assertFalse(self.output_for(selected, root=output).exists())

    def test_layer_weight_and_shared_resource_overrides(self):
        result = self.run_script(CKA_LAYER=31, CKA_WEIGHT="2.5e-1", GPU_INCLUDE="localhost:2",
                                 PER_DEVICE_TRAIN_BATCH_SIZE=2, GRADIENT_ACCUMULATION_STEPS=64,
                                 FINETUNE_GRADIENT_CHECKPOINTING="False", MASTER_PORT=29617)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 2, result.stdout)
        for command, setting in zip(commands, SETTINGS):
            self.assert_recipe(command, setting, layer=31, weight=0.25)
            flags = self.flags(command)
            self.assertEqual(command[command.index("--include") + 1], "localhost:2")
            self.assertEqual(command[command.index("--master_port") + 1], "29617")
            self.assertLess(command.index("--master_port"), command.index("llava/train/train_mem.py"))
            self.assertEqual(flags["--per_device_train_batch_size"], "2")
            self.assertEqual(flags["--gradient_accumulation_steps"], "64")
            self.assertEqual(flags["--gradient_checkpointing"], "False")

    def test_first_intermediate_block_is_allowed(self):
        result = self.run_script(SETTINGS="intermediate_only", CKA_LAYER=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        command, = self.commands(result.stdout)
        self.assert_recipe(command, "intermediate_only", layer=1)

    def test_opt_in_b200_settings_preserve_objective_and_global_batch(self):
        result = self.run_script(
            OPTIM="adamw_torch_fused", DEEPSPEED_CONFIG="./scripts/zero2_b200.json",
            FINETUNE_GRADIENT_CHECKPOINTING="False", DATALOADER_PREFETCH_FACTOR=4,
            DATALOADER_PERSISTENT_WORKERS="True", ACCELERATOR_CONFIG='{"non_blocking":true}')
        self.assertEqual(result.returncode, 0, result.stderr)
        for command, setting in zip(self.commands(result.stdout), SETTINGS):
            self.assert_recipe(command, setting)
            flags = self.flags(command)
            self.assertEqual(flags["--optim"], "adamw_torch_fused")
            self.assertEqual(flags["--deepspeed"], "./scripts/zero2_b200.json")
            self.assertEqual(flags["--gradient_checkpointing"], "False")
            self.assertEqual(flags["--dataloader_prefetch_factor"], "4")
            self.assertEqual(flags["--dataloader_persistent_workers"], "True")
            self.assertEqual(json.loads(flags["--accelerator_config"]), {"non_blocking": True})
            self.assertEqual(int(flags["--per_device_train_batch_size"]) *
                             int(flags["--gradient_accumulation_steps"]), 128)

    def test_fixed_recipe_isolated_from_inherited_objective_and_checkpoint_overrides(self):
        result = self.run_script(
            MODEL_NAME_OR_PATH="wrong-model", CONVERSATION_VERSION="auto",
            MM_PROJECTOR_TYPE="linear", MM_PROJECTOR_INIT_SEED="17",
            VISION_TOWER="wrong-vision-tower", MM_VISION_SELECT_LAYER="-1",
            CKA_LOSS="False", CKA_LOSS_ENABLED="False", CKA_LAYERS="all",
            CKA_PROJECTOR_WEIGHT="0.9", CKA_FINAL_HIDDEN_WEIGHT="0.8",
            CKA_LOSS_TAU="0.5", CKA_LOSS_START_RATIO="0.75",
            CKA_FINAL_MLP_HEAD="True", CKA_LOSS_GEOMETRY="row_normalized",
            CKA_ANCHOR_LAYER="12", CKA_VISION_ANCHOR_LAYER="20",
            CKA_PROJECTOR_VISION_ANCHOR_LAYER="21", CKA_FINAL_VISION_ANCHOR_LAYER="22",
            CKA_LOSS_SUBSET_SELECT_LAYER="9", CKA_LOSS_SUBSET_QUERY_TOKENS="instruction",
            VSP_PCGRAD="True", VSP_NORM_CAP="True", VSP_APPLY_TO_PROJECTOR_ONLY="True",
            VSP_DIAGNOSTICS="True", GRADIENT_CHECKPOINTING="False",
            PRETRAIN_ADAPTER="wrong-adapter.bin", OUTPUT_DIR="wrong-output",
            RUN_NAME="wrong-run", WANDB_RUN_NAME="wrong-run",
            WANDB_RUN_ID="old-run-id", WANDB_RESUME="must",
            RESUME_FROM_CHECKPOINT="old-checkpoint", STOP_AFTER_STEP_RATIO="0.8",
            SAVE_STRATEGY="steps",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands(result.stdout)
        self.assertEqual(len(commands), 2, result.stdout)
        for command, setting in zip(commands, SETTINGS):
            self.assert_recipe(command, setting)
            flags = self.flags(command)
            self.assertEqual(flags["--gradient_checkpointing"], "True")
            self.assertEqual(flags["--deepspeed"], "./scripts/zero2.json")
            self.assertNotEqual(flags["--run_name"], "wrong-run")
            self.assertNotIn("old-run-id", command)
            self.assertNotIn("must", command)

    def test_invalid_layers_fail_before_any_child_launch(self):
        for layer in ("0", "32", "33", "-1", "final", "all", "16.0", "1,2", "1 2", "bad"):
            with self.subTest(layer=layer):
                result = self.run_script(CKA_LAYER=layer)
                self.assert_rejected_before_launch(result)
                self.assertIn("CKA_LAYER", result.stderr)
        self.assertFalse(self.output.exists())

    def test_weights_must_be_finite_and_positive(self):
        for weight in ("0", "-0.1", "nan", "NaN", "inf", "-inf", "1e999", "bad"):
            with self.subTest(weight=weight):
                result = self.run_script(CKA_WEIGHT=weight)
                self.assert_rejected_before_launch(result)
                self.assertIn("CKA_WEIGHT", result.stderr)
        self.assertFalse(self.output.exists())

    def test_empty_unknown_and_duplicate_settings_are_rejected(self):
        for settings in ("", " ", "baseline", "intermediate_only unknown",
                         "intermediate_only intermediate_only",
                         "projector_intermediate projector_intermediate"):
            with self.subTest(settings=settings):
                self.assert_rejected_before_launch(self.run_script(SETTINGS=settings))
        self.assertFalse(self.output.exists())

    def test_all_selected_parent_metadata_is_validated_before_any_launch(self):
        incompatible = {
            "_name_or_path": "lmsys/vicuna-13b-v1.5", "num_hidden_layers": 40,
            "hidden_size": 5120, "mm_hidden_size": 1280, "mm_projector_type": "linear",
            "mm_vision_tower": "openai/clip-vit-large-patch14",
            "mm_vision_select_layer": -1, "mm_vision_select_feature": "cls_patch",
        }
        for key, value in incompatible.items():
            with self.subTest(key=key):
                parent = self.make_parent(self.root / key, **{key: value})
                result = self.run_script(CKA_PRETRAIN_DIR=parent)
                self.assert_rejected_before_launch(result)
                self.assertIn(key, result.stderr)
        self.assertFalse(self.output.exists())

    def test_missing_invalid_or_incomplete_config_is_rejected(self):
        for name, contents in (("missing", None), ("invalid", "not-json"),
                               ("incomplete", "{}")):
            with self.subTest(config=name):
                parent = self.make_parent(self.root / name)
                config = parent / "config.json"
                if contents is None:
                    config.unlink()
                else:
                    config.write_text(contents)
                self.assert_rejected_before_launch(self.run_script(CKA_PRETRAIN_DIR=parent))
        self.assertFalse(self.output.exists())

    def test_missing_empty_or_directory_adapter_is_rejected_for_either_parent(self):
        for variable in ("BASELINE_PRETRAIN_DIR", "CKA_PRETRAIN_DIR"):
            for mode in ("missing", "empty", "directory"):
                with self.subTest(variable=variable, mode=mode):
                    parent = self.make_parent(self.root / f"{variable}-{mode}")
                    adapter = parent / "mm_projector.bin"
                    if mode == "empty":
                        adapter.write_bytes(b"")
                    else:
                        adapter.unlink()
                        if mode == "directory":
                            adapter.mkdir()
                    result = self.run_script(**{variable: parent})
                    self.assert_rejected_before_launch(result)
                    self.assertIn("mm_projector.bin", result.stderr)
        self.assertFalse(self.output.exists())

    def test_any_existing_planned_output_blocks_every_launch_without_modification(self):
        for setting in SETTINGS:
            for kind in ("directory", "file", "dangling-symlink"):
                with self.subTest(setting=setting, kind=kind):
                    output_root = self.root / f"output-{setting}-{kind}"
                    output = self.output_for(setting, root=output_root)
                    output.parent.mkdir(parents=True)
                    if kind == "directory":
                        output.mkdir()
                        (output / "keep.txt").write_bytes(b"existing-checkpoint")
                    elif kind == "file":
                        output.write_bytes(b"existing-checkpoint")
                    else:
                        output.symlink_to(self.root / "nonexistent-checkpoint")
                    result = self.run_script(OUTPUT_ROOT=output_root)
                    self.assert_rejected_before_launch(result)
                    self.assertIn(str(output), result.stderr)
                    if kind == "directory":
                        self.assertEqual((output / "keep.txt").read_bytes(), b"existing-checkpoint")
                    elif kind == "file":
                        self.assertEqual(output.read_bytes(), b"existing-checkpoint")
                    else:
                        self.assertTrue(output.is_symlink())
                        self.assertEqual(output.readlink(), self.root / "nonexistent-checkpoint")
                    other = SETTINGS[1] if setting == SETTINGS[0] else SETTINGS[0]
                    self.assertFalse(self.output_for(other, root=output_root).exists())


if __name__ == "__main__":
    unittest.main()
