"""CPU-only training, checkpoint, and shell plumbing for final-CKA heads."""

import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
from transformers import Trainer

from llava.train.llava_trainer import CKA_FINAL_MLP_HEAD_NAME, LLaVATrainer
from llava.train.train import (
    ModelArguments,
    TrainingArguments,
    find_all_linear_names,
    safe_save_model_for_hf_trainer,
    set_cka_final_mlp_head_config,
    validate_cka_final_mlp_head_args,
    validate_cka_final_mlp_head_layers,
)
from tests.test_cka_final_mlp_head import _tiny_model


class FinalMlpHeadConfigTests(unittest.TestCase):
    def arguments(self, **overrides):
        values = dict(cka_loss=True, vision_tower="local-fixture", cka_loss_final_mlp_head=True)
        values.update(overrides)
        return ModelArguments(**values)

    def test_disabled_is_default_and_does_not_add_validation(self):
        args = ModelArguments()
        self.assertFalse(args.cka_loss_final_mlp_head)
        validate_cka_final_mlp_head_args(args, SimpleNamespace(protected_stage2=True), family="mpt")
        validate_cka_final_mlp_head_layers(SimpleNamespace(cka_loss_layers=[-1]), 0)

    def test_disabled_preserves_legacy_config_keys_and_can_disable_existing_head(self):
        config = SimpleNamespace(cka_loss=False)
        original = vars(config).copy()
        set_cka_final_mlp_head_config(config, False)
        self.assertEqual(vars(config), original)
        set_cka_final_mlp_head_config(config, True)
        self.assertTrue(config.cka_loss_final_mlp_head)
        set_cka_final_mlp_head_config(config, False)
        self.assertFalse(config.cka_loss_final_mlp_head)

    def test_enabled_requires_standard_multimodal_cka(self):
        for changes, training, family, error in (
            ({"cka_loss": False}, {}, None, "cka_loss True"),
            ({"vision_tower": None}, {}, None, "vision_tower"),
            ({}, {"protected_cka_refinement": True}, None, "Protected CKA"),
            ({}, {"protected_stage2": True}, None, "Protected CKA"),
            ({}, {}, "mpt", "MPT"),
            ({"cka_loss_final_hidden_weight": 0}, {}, None, "nonzero"),
            ({"cka_loss_weight": 0}, {}, None, "nonzero"),
        ):
            with self.subTest(changes=changes, training=training, family=family):
                with self.assertRaisesRegex(ValueError, error):
                    validate_cka_final_mlp_head_args(self.arguments(**changes), SimpleNamespace(**training), family=family)

    def test_projector_only_norm_cap_cannot_suppress_auxiliary_only_head(self):
        args = self.arguments(tune_mm_mlp_adapter=True, vsp_norm_cap=True)
        for lora in (False, True):
            with self.assertRaisesRegex(ValueError, "vsp_norm_cap False"):
                validate_cka_final_mlp_head_args(args, SimpleNamespace(lora_enable=lora))
        args.vsp_norm_cap = False
        validate_cka_final_mlp_head_args(args, SimpleNamespace())
        validate_cka_final_mlp_head_args(self.arguments(vsp_norm_cap=True), SimpleNamespace())
        # freeze_backbone retains the CE-trained language-model output head.
        validate_cka_final_mlp_head_args(self.arguments(freeze_backbone=True, vsp_norm_cap=True), SimpleNamespace())

    def test_explicit_final_weight_overrides_legacy_zero_weight(self):
        validate_cka_final_mlp_head_args(
            self.arguments(cka_loss_weight=0, cka_loss_final_hidden_weight=0.1), SimpleNamespace()
        )

    def test_quantized_non_lora_norm_cap_cannot_suppress_auxiliary_only_head(self):
        args = self.arguments(vsp_norm_cap=True)
        for bits in (4, 8):
            with self.subTest(bits=bits):
                with self.assertRaisesRegex(ValueError, "vsp_norm_cap False"):
                    validate_cka_final_mlp_head_args(args, SimpleNamespace(bits=bits, lora_enable=False))
                validate_cka_final_mlp_head_args(args, SimpleNamespace(bits=bits, lora_enable=True))
                validate_cka_final_mlp_head_args(self.arguments(vsp_norm_cap=False), SimpleNamespace(bits=bits, lora_enable=False))

    def test_lora_best_model_restore_is_rejected_until_head_restore_supported(self):
        with self.assertRaisesRegex(ValueError, "load_best_model_at_end"):
            validate_cka_final_mlp_head_args(self.arguments(), SimpleNamespace(lora_enable=True, load_best_model_at_end=True))
        validate_cka_final_mlp_head_args(self.arguments(), SimpleNamespace(lora_enable=False, load_best_model_at_end=True))
        validate_cka_final_mlp_head_args(ModelArguments(), SimpleNamespace(lora_enable=True, load_best_model_at_end=True))

    def test_final_layer_selection_validation(self):
        for selection in ("final", "last", "all", "3", [3], [1, 3], [1, "final"]):
            with self.subTest(selection=selection):
                validate_cka_final_mlp_head_layers(
                    SimpleNamespace(cka_loss_final_mlp_head=True, cka_loss_layers=selection), 3
                )
        for selection in ([-1], "-1", [1, 2], "1,2", [999], [], None):
            with self.subTest(selection=selection):
                with self.assertRaisesRegex(ValueError, "final decoder block"):
                    validate_cka_final_mlp_head_layers(
                        SimpleNamespace(cka_loss_final_mlp_head=True, cka_loss_layers=selection), 3
                    )

    def test_head_is_excluded_from_lora_linear_targets(self):
        model = nn.Module()
        model.q_proj = nn.Linear(4, 4)
        model.mm_projector = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        model.cka_final_mlp_head = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        self.assertEqual(find_all_linear_names(model), ["q_proj"])


class FinalMlpHeadCheckpointTests(unittest.TestCase):
    def make_trainer(self, output_dir, *, lora=False, projector_only=False):
        model = _tiny_model(multimodal=False)
        model.get_model().mm_projector = nn.Linear(16, 16)
        if lora:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
            model.get_model().cka_final_mlp_head.requires_grad_(True)
        args = TrainingArguments(output_dir=output_dir, use_cpu=True, report_to=[], lora_enable=lora)
        args.tune_mm_mlp_adapter = projector_only
        trainer = object.__new__(LLaVATrainer)
        trainer.model, trainer.args = model, args
        trainer.processing_class = trainer.data_collator = None
        trainer.is_fsdp_enabled = False
        trainer.optimizer = None
        return trainer

    def test_optimizer_includes_full_rank_head_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, lora=True)
            optimizer = trainer.create_optimizer()
            included = {id(p) for group in optimizer.param_groups for p in group["params"]}
            for param in trainer.model.get_model().cka_final_mlp_head.parameters():
                self.assertIn(id(param), included)

    def test_final_and_periodic_projector_saves_include_head(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, projector_only=True)
            with mock.patch("llava.train.train.maybe_zero_3", side_effect=lambda p, **_: p.detach().cpu()):
                safe_save_model_for_hf_trainer(trainer, directory)
            state = torch.load(Path(directory) / "mm_projector.bin", weights_only=True)
            self.assertEqual(sum("cka_final_mlp_head" in key for key in state), 4)
            self.assertTrue(any("mm_projector" in key for key in state))
            trainer.state = SimpleNamespace(global_step=7)
            trainer._get_output_dir = lambda trial: directory
            with mock.patch("llava.train.llava_trainer.maybe_zero_3", side_effect=lambda p, **_: p.detach().cpu()):
                trainer._save_checkpoint(trainer.model, trial=None)
            periodic = torch.load(Path(directory) / "checkpoint-7/mm_projector.bin", weights_only=True)
            self.assertEqual(periodic.keys(), state.keys())
            for key, tensor in state.items():
                torch.testing.assert_close(tensor, periodic[key])

    def test_lora_checkpoint_restores_head_and_preserves_trainability(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, lora=True)
            head = trainer.model.get_model().cka_final_mlp_head
            expected = {key: tensor.clone() for key, tensor in head.state_dict().items()}
            trainer._save(directory)
            self.assertTrue((Path(directory) / CKA_FINAL_MLP_HEAD_NAME).is_file())
            with torch.no_grad():
                for param in head.parameters():
                    param.zero_()
            trainer._load_from_checkpoint(directory)
            for key, tensor in head.state_dict().items():
                torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
            self.assertTrue(all(param.requires_grad for param in head.parameters()))

    def test_lora_restore_accepts_final_export_prefix_and_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, lora=True)
            head = trainer.model.get_model().cka_final_mlp_head
            state = {"model.cka_final_mlp_head." + key: tensor.clone() for key, tensor in head.state_dict().items()}
            with mock.patch.object(Trainer, "_load_from_checkpoint"):
                with self.assertRaisesRegex(ValueError, "model-only initialization"):
                    trainer._load_from_checkpoint(directory)
                torch.save(state, Path(directory) / "non_lora_trainables.bin")
                trainer._load_from_checkpoint(directory)
                state.pop(next(iter(state)))
                torch.save(state, Path(directory) / "non_lora_trainables.bin")
                with self.assertRaisesRegex(ValueError, "missing final CKA MLP head parameters"):
                    trainer._load_from_checkpoint(directory)

    def test_lora_gathered_state_is_used_without_zero_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, lora=True)
            state = {key: tensor.clone() for key, tensor in trainer.model.state_dict().items()}
            for param in trainer.model.get_model().cka_final_mlp_head.parameters():
                param.ds_id = 1
            with mock.patch.object(Trainer, "_save"), mock.patch.object(trainer.model, "state_dict", side_effect=AssertionError("must use gathered state")):
                trainer._save(directory, state_dict=state)
            saved = torch.load(Path(directory) / CKA_FINAL_MLP_HEAD_NAME, weights_only=True)
            self.assertEqual(len(saved), 4)


class FinalMlpHeadScriptTests(unittest.TestCase):
    def test_requested_launchers_forward_opt_in_without_changing_defaults(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            env_bin = Path(directory) / "bin"
            env_bin.mkdir()
            for name, body in (("deepspeed", 'printf "%s\\n" "$@"'), ("python", "exit 0")):
                path = env_bin / name
                path.write_text("#!/bin/bash\n" + body + "\n")
                path.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if not key.startswith(("CKA_", "VSP_"))}
            env.update(CONDA_PREFIX=directory, PATH=str(env_bin) + os.pathsep + env.get("PATH", ""))
            for stage in ("pretrain", "finetune"):
                script = root / "scripts/v1_5/7b" / f"{stage}_qwen2_5.sh"
                for overrides in ({}, {"CKA_FINAL_MLP_HEAD": "True", "CKA_LAYERS": "final", "VSP_NORM_CAP": "False"}):
                    with self.subTest(stage=stage, overrides=overrides):
                        result = subprocess.run(["bash", str(script)], env=dict(env, **overrides), capture_output=True, text=True, check=True)
                        args = result.stdout.splitlines()
                        for flag, expected in (("--cka_loss_final_mlp_head", overrides.get("CKA_FINAL_MLP_HEAD", "False")),
                                               ("--cka_loss_layers", overrides.get("CKA_LAYERS", "-1")),
                                               ("--vsp_norm_cap", overrides.get("VSP_NORM_CAP", "True"))):
                            self.assertEqual(args[args.index(flag) + 1], expected)


if __name__ == "__main__":
    unittest.main()
