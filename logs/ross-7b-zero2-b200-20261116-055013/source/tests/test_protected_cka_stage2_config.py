"""CPU-only Stage 2 configuration, holdout provenance and dry-run launch tests."""

import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llava.train import protected_cka_stage2_config as config


ROOT = Path(__file__).resolve().parents[1]


def arguments(**overrides):
    values = {"protected_stage2_" + key: value for key, value in config.STAGE2_DEFAULTS.items()}
    values.update(protected_stage2=True, cka_grad_scope="existing", max_steps=1000,
                  protected_stage2_min_phase_a_steps=200, protected_stage2_min_phase_b_steps=0,
                  protected_stage2_max_phase_b_steps=250, protected_stage2_refine_lr=1e-5,
                  protected_stage2_ce_tolerance=0.02, protected_stage2_text_kl_budget=0.05,
                  save_strategy="no", eval_strategy="no", dataloader_num_workers=0,
                  world_size=1, n_gpu=0, device=SimpleNamespace(type="cpu"))
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeDataset:
    def __init__(self, data_path, tokenizer, data_args):
        self.tokenizer, self.data_args = tokenizer, data_args
        with open(data_path, encoding="utf-8") as handle:
            self.list_data_dict = json.load(handle)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter = self.root / "stage1" / "mm_projector.bin"
        self.adapter.parent.mkdir()
        self.adapter.write_bytes(b"fixture")
        self.model = SimpleNamespace(tune_mm_mlp_adapter=False, vision_tower="fixture-vision",
                                     pretrain_mm_mlp_adapter=str(self.adapter), cka_loss=True,
                                     cka_loss_layers="8,final", cka_loss_projector_weight=0.2)
        self.data = SimpleNamespace()

    def validate(self, args=None):
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            config.validate_protected_cka_stage2_config(self.model, self.data, args or arguments())

    def source(self, *, own=False, metadata=None):
        source = self.root / "stage2-source"
        source.mkdir(exist_ok=True)
        (source / "config.json").write_text(json.dumps(metadata or {"tune_mm_mlp_adapter": False, "cka_loss": False}))
        (source / ("student_model.pt" if own else "model.safetensors")).write_bytes(b"fixture")
        if own:
            (source / config.STAGE2_STATE_NAME).write_bytes(b"runtime-validates-phase-fixture")
        return source

    def explicit_validation(self):
        path = self.root / "heldout.json"
        path.write_text("[]")
        self.data.protected_stage2_validation_data_path = str(path)

    def test_disabled_and_missing_flag_are_strict_noops(self):
        for args in (SimpleNamespace(), arguments(protected_stage2=False, deepspeed="zero3", max_steps=-1)):
            before = vars(args).copy()
            self.validate(args)
            self.assertEqual(vars(args), before)
        self.assertEqual(self.model.pretrain_mm_mlp_adapter, str(self.adapter))

    def test_stage1_initialization_allows_fresh_instruction_holdout(self):
        # Even a CKA-trained Stage 1 adapter is initialization, not a Stage 2 teacher.
        (self.adapter.parent / "config.json").write_text(json.dumps({"cka_loss": True, "tune_mm_mlp_adapter": True}))
        args = arguments()
        before_model = vars(self.model).copy()
        self.validate(args)
        self.assertEqual(args.protected_stage2_max_phase_a_steps, args.max_steps)
        self.assertIsNone(self.data._protected_cka_stage2_source_manifest)
        self.assertEqual(vars(self.model), before_model)

    def test_stage2_gates_never_inherit_stage1_thresholds(self):
        for key in ("ce_tolerance", "text_kl_budget"):
            args = arguments(**{"protected_stage2_" + key: None,
                                "protected_cka_" + key: 100.0})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "set explicitly"):
                self.validate(args)
        self.assertIsNone(config.STAGE2_DEFAULTS["drift_budget"])

    def test_preserves_full_finetuning_or_lora_and_modality_grouping(self):
        for lora in (False, True):
            args = arguments(lora_enable=lora, group_by_modality_length=True,
                             gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
            self.validate(args)
            self.assertEqual(args.lora_enable, lora)
            self.assertTrue(args.group_by_modality_length)
            self.assertTrue(args.gradient_checkpointing)

    def test_independent_configuration_exports_scope_and_thresholds(self):
        args = arguments(cka_grad_scope="projector_only", protected_stage2_proj_aux_ratio=0.03,
                         protected_stage2_llm_aux_ratio=0.07)
        self.validate(args)
        exported = config.protected_cka_stage2_config_dict(args, self.data)
        self.assertEqual(exported["cka_grad_scope"], "projector_only")
        self.assertEqual(exported["protected_stage2_proj_aux_ratio"], 0.03)
        self.assertEqual(exported["protected_stage2_llm_aux_ratio"], 0.07)
        self.assertNotIn("protected_cka_ce_tolerance", exported)

    def test_reserved_phase_b_defaults_and_budget_validation(self):
        self.assertEqual(config.STAGE2_DEFAULTS["min_phase_a_steps"], 1500)
        self.assertEqual(config.STAGE2_DEFAULTS["min_phase_b_steps"], 350)
        args = arguments(max_steps=2173, protected_stage2_min_phase_a_steps=1500,
                         protected_stage2_min_phase_b_steps=350, protected_stage2_max_phase_b_steps=500)
        self.validate(args)
        self.assertEqual(args.max_steps, 2173)
        self.assertEqual(args.protected_stage2_max_phase_b_steps, 500)
        with self.assertRaisesRegex(ValueError, "min_phase_a_steps \\+ min_phase_b_steps"):
            self.validate(arguments(max_steps=1800, protected_stage2_min_phase_a_steps=1500,
                                    protected_stage2_min_phase_b_steps=350, protected_stage2_max_phase_b_steps=500))
        with self.assertRaisesRegex(ValueError, "min_phase_b_steps cannot exceed max_phase_b_steps"):
            self.validate(arguments(protected_stage2_min_phase_b_steps=350))
        for invalid in (-1, True, 0.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "min_phase_b_steps"):
                self.validate(arguments(protected_stage2_min_phase_b_steps=invalid))

    def test_unsupported_backends_and_replay_options_fail(self):
        cases = dict(deepspeed="zero2", fsdp=["full_shard"], fp16=True, fp16_full_eval=True,
                     half_precision_backend="apex", torch_compile=True, bits=4, world_size=2, n_gpu=2,
                     dataloader_num_workers=1, save_strategy="steps", eval_strategy="steps",
                     device=SimpleNamespace(type="mps"), group_by_length=True, ignore_data_skip=True,
                     load_best_model_at_end=True, auto_find_batch_size=True, eval_on_start=True,
                     use_liger_kernel=True, label_smoothing_factor=0.1, optim="adafactor",
                     lr_scheduler_type="reduce_lr_on_plateau", stop_after_step_ratio=0.5,
                     accelerator_config={"use_seedable_sampler": False})
        for key, value in cases.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Protected CKA Stage 2"):
                self.validate(arguments(**{key: value}))
        with self.assertRaisesRegex(ValueError, "nonreentrant|use_reentrant"):
            self.validate(arguments(gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": True}))

    def test_rejects_stage1_and_legacy_gradient_controllers(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.validate(arguments(protected_cka_refinement=True))
        for key in ("tune_mm_mlp_adapter", "use_pcgrad", "vsp_gradient_diagnostics",
                    "vsp_asymmetric_pcgrad", "vsp_apply_to_projector_only", "vsp_norm_cap"):
            with self.subTest(key=key), patch.object(self.model, key, True, create=True), self.assertRaises(ValueError):
                self.validate()
        with patch.object(self.model, "freeze_backbone", True, create=True):
            with self.assertRaisesRegex(ValueError, "student LLM"):
                self.validate(arguments(lora_enable=False))
            self.validate(arguments(lora_enable=True))
        for key in ("cka_loss_start_ratio", "cka_loss_tau"):
            with self.subTest(key=key), patch.object(self.model, key, 0.1, create=True), self.assertRaisesRegex(ValueError, key):
                self.validate()

    def test_numeric_budgets_and_modes(self):
        for key, value in {"max_phase_b_steps": None, "refine_lr": 0., "ema_decay": 1.,
                           "proj_aux_ratio": -1., "llm_aux_ratio": float("nan"), "validation_fraction": 1.,
                           "max_retries": -1, "eval_interval": 0, "max_phase_a_steps": 1001,
                           "reference_device": "cuda:7", "continuation": "unspecified",
                           "drift_budget": -0.1}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(arguments(**{"protected_stage2_" + key: value}))
        with self.assertRaisesRegex(ValueError, "cka_grad_scope"):
            self.validate(arguments(cka_grad_scope="all_layers"))
        for value in (0.0, -0.1, float("nan"), float("inf"), None):
            with self.subTest(learning_rate=value), self.assertRaisesRegex(ValueError, "learning_rate"):
                self.validate(arguments(learning_rate=value))

    def test_legacy_lora_source_requires_complete_non_lora_weights_and_recipe(self):
        source = self.root / "lora-source"
        source.mkdir()
        (source / "adapter_model.safetensors").write_bytes(b"fixture")
        self.explicit_validation()
        args = arguments(protected_stage2_start_phase_b=True,
                         protected_stage2_source_stage2_checkpoint=str(source), lora_enable=True)
        with self.assertRaisesRegex(ValueError, "non_lora_trainables.bin is missing"):
            self.validate(args)
        (source / "non_lora_trainables.bin").write_bytes(b"runtime-validates-all-required-keys")
        args.lora_enable = False
        with self.assertRaisesRegex(ValueError, "original adapter recipe"):
            self.validate(args)
        args.lora_enable = True
        self.validate(args)

    def test_full_source_does_not_require_separate_non_lora_file(self):
        source = self.source()
        self.explicit_validation()
        self.validate(arguments(protected_stage2_start_phase_b=True,
                                protected_stage2_source_stage2_checkpoint=str(source)))

    def test_direct_phase_b_cannot_use_stage1_adapter_as_teacher(self):
        with self.assertRaisesRegex(ValueError, "not a Stage 1 teacher"):
            self.validate(arguments(protected_stage2_start_phase_b=True))

    def test_stage2_source_requires_provenance_or_explicit_holdout(self):
        source = self.source()
        args = arguments(protected_stage2_start_phase_b=True,
                         protected_stage2_source_stage2_checkpoint=str(source))
        with self.assertRaisesRegex(ValueError, "new auto-holdout"):
            self.validate(args)
        # Stage 1 manifests cannot establish instruction holdout provenance.
        (source / "protected_cka_split").mkdir()
        (source / "protected_cka_split" / "split.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "new auto-holdout"):
            self.validate(args)
        self.explicit_validation()
        self.validate(args)
        self.assertIsNone(self.model.pretrain_mm_mlp_adapter)

    def test_source_original_manifest_is_selected(self):
        source = self.source()
        (source / config.STAGE2_SPLIT_DIR).mkdir()
        manifest = source / config.STAGE2_SPLIT_DIR / "split.json"
        manifest.write_text("{}")
        self.validate(arguments(protected_stage2_start_phase_b=True,
                                protected_stage2_source_stage2_checkpoint=str(source)))
        self.assertEqual(self.data._protected_cka_stage2_source_manifest, str(manifest))

    def test_source_preserves_manifest_even_with_explicit_validation(self):
        source = self.source()
        (source / config.STAGE2_SPLIT_DIR).mkdir()
        manifest = source / config.STAGE2_SPLIT_DIR / "split.json"
        manifest.write_text("{}")
        self.explicit_validation()
        self.validate(arguments(protected_stage2_start_phase_b=True,
                                protected_stage2_source_stage2_checkpoint=str(source)))
        self.assertEqual(self.data._protected_cka_stage2_source_manifest, str(manifest))

    def test_short_direct_phase_b_run_does_not_reserve_another_phase_b(self):
        source = self.source()
        self.explicit_validation()
        args = arguments(protected_stage2_start_phase_b=True,
                         protected_stage2_source_stage2_checkpoint=str(source), max_steps=100,
                         protected_stage2_max_phase_b_steps=100,
                         protected_stage2_min_phase_a_steps=1500, protected_stage2_min_phase_b_steps=350)
        self.validate(args)
        self.assertEqual(args.max_steps, 100)
        self.assertEqual(args.protected_stage2_max_phase_b_steps, 100)

    def test_source_and_resume_are_mutually_exclusive_and_outputs_fresh(self):
        source = self.source(own=True)
        self.explicit_validation()
        args = arguments(protected_stage2_start_phase_b=True,
                         protected_stage2_source_stage2_checkpoint=str(source),
                         resume_from_checkpoint=str(source))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.validate(args)
        args.resume_from_checkpoint = None
        args.output_dir = str(source)
        with self.assertRaisesRegex(ValueError, "fresh, empty"):
            self.validate(args)
        output = self.root / "existing-output"
        output.mkdir()
        (output / "trainer_state.json").write_text("{}")
        args.output_dir = str(output)
        with self.assertRaisesRegex(ValueError, "fresh, empty"):
            self.validate(args)

    def test_source_stage1_and_cka_metadata_rejected(self):
        self.explicit_validation()
        for metadata in ({"tune_mm_mlp_adapter": True}, {"cka_loss": True}, {"protected_stage2": True}):
            source = self.source(metadata=metadata)
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.validate(arguments(protected_stage2_start_phase_b=True,
                                        protected_stage2_source_stage2_checkpoint=str(source)))

    def test_own_phase_a_source_defers_phase_check_to_runtime(self):
        source = self.source(own=True, metadata={"cka_loss": True, "protected_stage2": True})
        self.explicit_validation()
        self.validate(arguments(protected_stage2_start_phase_b=True,
                                protected_stage2_source_stage2_checkpoint=str(source)))

    def test_full_resume_keeps_original_split_and_no_old_stage1_dependency(self):
        source = self.source(own=True)
        (source / config.STAGE2_SPLIT_DIR).mkdir()
        manifest = source / config.STAGE2_SPLIT_DIR / "split.json"
        manifest.write_text("{}")
        self.model.pretrain_mm_mlp_adapter = "/missing/initial-stage1.bin"
        self.validate(arguments(resume_from_checkpoint=str(source)))
        self.assertIsNone(self.model.pretrain_mm_mlp_adapter)
        self.assertEqual(self.data._protected_cka_stage2_source_manifest, str(manifest))

    def test_legacy_resume_fails_instead_of_resplitting(self):
        source = self.source()
        with self.assertRaisesRegex(ValueError, "full protected Stage 2 checkpoint"):
            self.validate(arguments(resume_from_checkpoint=str(source)))


class InstructionSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "instructions.json"
        self.data = SimpleNamespace(data_path=str(self.path), image_folder=str(self.root / "images"),
                                    train_data_fraction=1.0, protected_stage2_validation_data_path=None,
                                    protected_stage2_validation_image_folder=None)
        self.args = arguments(output_dir=str(self.root / "run"), protected_stage2_validation_fraction=0.25,
                              protected_stage2_validation_max_samples=6)

    def rows(self, count=20):
        rows = []
        for index in range(count):
            row = {"id": index, "conversations": [{"from": "human", "value": f"question {index}"},
                                                   {"from": "gpt", "value": f"answer {index}"}]}
            if index % 2 == 0:
                row["image"] = f"{index}.jpg"
            rows.append(row)
        return rows

    def write(self, rows, path=None):
        (path or self.path).write_text(json.dumps(rows))

    def prepare(self):
        dataset = FakeDataset(str(self.path), None, self.data)
        with contextlib.redirect_stdout(io.StringIO()):
            return config.prepare_protected_cka_stage2_datasets(dataset, None, self.data, self.args, FakeDataset)

    def test_mixed_split_disjoint_complete_reproducible_and_stage2_named(self):
        rows = self.rows()
        self.write(rows)
        train, validation = self.prepare()
        self.assertEqual(len(train.list_data_dict) + len(validation.list_data_dict), len(rows))
        self.assertLessEqual(len(validation.list_data_dict), 6)
        config._assert_disjoint(train.list_data_dict, validation.list_data_dict, self.data.image_folder,
                                self.data.image_folder, config._InstructionIdentities())
        manifest = Path(self.args.protected_stage2_split_manifest)
        self.assertEqual(manifest.parent.name, "protected_cka_stage2_split")
        original = manifest.read_bytes()
        train2, validation2 = self.prepare()
        self.assertEqual(train.list_data_dict, train2.list_data_dict)
        self.assertEqual(validation.list_data_dict, validation2.list_data_dict)
        self.assertEqual(original, manifest.read_bytes())
        self.assertFalse((manifest.parent.parent / "protected_cka_split").exists())

    def test_text_duplicates_grouped_despite_different_ids_and_metadata(self):
        rows = self.rows()
        rows[3]["conversations"] = rows[1]["conversations"]
        rows[3]["source"] = "different corpus metadata"
        self.write(rows)
        train, validation = self.prepare()
        for subset in (train.list_data_dict, validation.list_data_dict):
            ids = {row["id"] for row in subset}
            self.assertEqual(1 in ids, 3 in ids)

    def test_images_aliases_and_ids_are_transitively_grouped(self):
        rows = self.rows()
        rows[2]["image"] = "./0.jpg"
        rows[3]["id"] = 2
        self.write(rows)
        train, validation = self.prepare()
        for subset in (train.list_data_dict, validation.list_data_dict):
            ids = [row["id"] for row in subset]
            self.assertEqual(0 in ids, 2 in ids)
            self.assertIn(ids.count(2), (0, 2))

    def test_all_text_dataset_supported_and_manifest_has_no_dummy_image(self):
        rows = [{key: value for key, value in row.items() if key != "image"} for row in self.rows()]
        self.write(rows)
        train, validation = self.prepare()
        self.assertTrue(train.list_data_dict and validation.list_data_dict)
        manifest = Path(self.args.protected_stage2_split_manifest).parent / "validation_manifest.json"
        entries = json.loads(manifest.read_text())
        self.assertTrue(all(row["image"] is None and len(row["text_fingerprint"]) == 64 for row in entries))

    def test_explicit_text_image_or_id_overlap_rejected(self):
        rows = self.rows()
        self.write(rows)
        validation = self.root / "validation.json"
        self.data.protected_stage2_validation_data_path = str(validation)
        cases = [dict(rows[1], id="new-text-id"), dict(rows[0], id="new-image-id", image="./0.jpg"),
                 {"id": "0", "conversations": [{"value": "different text"}]}]
        for row in cases:
            self.write([row], validation)
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "overlap"):
                self.prepare()

    def test_explicit_validation_keeps_own_folder_and_full_dataset(self):
        self.write(self.rows())
        validation = self.root / "validation.json"
        heldout = [{"id": "heldout", "image": "0.jpg", "conversations": [{"value": "new"}]}]
        self.write(heldout, validation)
        self.data.protected_stage2_validation_data_path = str(validation)
        self.data.protected_stage2_validation_image_folder = str(self.root / "heldout-images")
        self.data.train_data_fraction = 0.1
        train, val = self.prepare()
        self.assertEqual(val.list_data_dict, heldout)
        self.assertEqual(val.data_args.train_data_fraction, 1.0)
        self.assertEqual(self.data.train_data_fraction, 0.1)
        self.assertEqual(len(train.list_data_dict), 20)

    def test_symlink_image_alias_grouping_preserved(self):
        folder = Path(self.data.image_folder)
        folder.mkdir()
        (folder / "alias.jpg").symlink_to(folder / "0.jpg")
        rows = self.rows()
        rows[2]["image"] = "alias.jpg"
        self.write(rows)
        train, val = self.prepare()
        for subset in (train.list_data_dict, val.list_data_dict):
            ids = {row["id"] for row in subset}
            self.assertEqual(0 in ids, 2 in ids)

    def test_changed_dataset_seed_and_manifests_are_rejected(self):
        rows = self.rows()
        self.write(rows)
        self.prepare()
        self.args.protected_stage2_validation_seed += 1
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            self.prepare()
        self.args.protected_stage2_validation_seed -= 1
        rows[1]["conversations"][1]["value"] = "changed answer"
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            self.prepare()

    def test_source_manifest_mismatch_prevents_new_holdout(self):
        self.write(self.rows())
        self.prepare()
        self.data._protected_cka_stage2_source_manifest = self.args.protected_stage2_split_manifest
        self.args.output_dir = str(self.root / "same-source-continuation")
        self.prepare()
        self.args.output_dir = str(self.root / "wrong-new-split")
        self.args.protected_stage2_validation_seed += 1
        with self.assertRaisesRegex(ValueError, "never redefine"):
            self.prepare()
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_empty_and_single_group_split_fail(self):
        for rows in ([], [{"id": n, "conversations": [{"value": "same text"}]} for n in range(3)]):
            self.write(rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.prepare()

    def test_invalid_image_marker_is_not_silently_text_only(self):
        rows = self.rows()
        rows[0]["image"] = None
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "nonempty single image"):
            self.prepare()

    def test_disabled_helper_preserves_every_row_without_writes(self):
        rows = self.rows()
        self.write(rows)
        self.args.protected_stage2 = False
        train, val = self.prepare()
        self.assertEqual(train.list_data_dict, rows)
        self.assertIsNone(val)
        self.assertFalse(Path(self.args.output_dir).exists())


class LauncherTests(unittest.TestCase):
    def command(self, *, wrapper=False, ablation=False, check=True, extra_args=(), **updates):
        # Do not let unrelated user/session training knobs contaminate fixtures.
        environment = {key: value for key, value in os.environ.items()
                       if key in ("PATH", "HOME", "TMPDIR", "LANG", "PYTHONPATH")}
        environment.update(MODEL_NAME_OR_PATH="model/fixture", TRAIN_DATA_PATH="/data/instructions.json",
                           TRAIN_IMAGE_FOLDER="/data/images", OUTPUT_DIR="/output/stage2", MAX_STEPS="1000",
                           GPU_ID="0", DRY_RUN="1", PRETRAIN_ADAPTER="/output/stage1/mm_projector.bin",
                           STAGE2_REFINE_LR="1e-5", STAGE2_MAX_PHASE_B_STEPS="250",
                           STAGE2_CE_TOLERANCE="0.02", STAGE2_TEXT_KL_BUDGET="0.05")
        if ablation:
            environment.update(SOURCE_STAGE2_CHECKPOINT="/source/stage2-ce", ABLATION_OUTPUT_ROOT="/output/ablation")
        for key, value in updates.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        script = ("scripts/v1_5/7b/finetune_qwen2_5_3b_protected_ablation.sh" if ablation
                  else "scripts/v1_5/7b/finetune_qwen2_5_3b_protected_cka.sh" if wrapper
                  else "scripts/v1_5/finetune_protected_cka.sh")
        result = subprocess.run(["bash", str(ROOT / script), *extra_args], env=environment,
                                text=True, capture_output=True, check=check)
        if not check:
            return result
        return [shlex.split(line) for line in result.stdout.splitlines() if line.strip()] if ablation else shlex.split(result.stdout)

    def test_default_preserves_stage2_cka_and_full_finetuning(self):
        command = self.command()
        expected = {"--protected_stage2": "True", "--protected_cka_refinement": "False",
                    "--protected_stage2_min_phase_a_steps": "1500", "--protected_stage2_min_phase_b_steps": "350",
                    "--tune_mm_mlp_adapter": "False", "--lora_enable": "False", "--cka_loss": "True",
                    "--cka_loss_projector_weight": "0.1", "--cka_loss_final_hidden_weight": "0.1",
                    "--cka_loss_layers": "final", "--group_by_modality_length": "True",
                    "--learning_rate": "2e-5", "--dataloader_num_workers": "0"}
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--deepspeed", command)
        self.assertNotIn("--protected_stage2_drift_budget", command)
        self.assertNotIn("--save_total_limit", command)

    def test_optional_checkpoint_retention_is_forwarded_without_changing_default(self):
        command = self.command(SAVE_TOTAL_LIMIT="2")
        self.assertEqual(command[command.index("--save_total_limit") + 1], "2")
        self.assertEqual(command[command.index("--save_strategy") + 1], "no")

    def test_lora_and_selection_overrides_preserved(self):
        command = self.command(LORA_ENABLE="True", LORA_R="32", CKA_LAYERS="8,16",
                               CKA_GRAD_SCOPE="projector_only", GRADIENT_CHECKPOINTING="True")
        for flag, value in (("--lora_enable", "True"), ("--lora_r", "32"),
                            ("--cka_loss_layers", "8,16"), ("--cka_grad_scope", "projector_only"),
                            ("--gradient_checkpointing", "True")):
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_three_continuations_have_same_data_lr_budget_and_source(self):
        commands = [self.command(STAGE2_START_PHASE_B="True", SOURCE_STAGE2_CHECKPOINT="/source/stage2-ce",
                                  PRETRAIN_ADAPTER=None, STAGE2_CONTINUATION=mode)
                    for mode in ("ce_only", "ce_kl", "protected")]
        for flag in ("--data_path", "--learning_rate", "--max_steps", "--protected_stage2_max_phase_b_steps",
                     "--protected_stage2_refine_lr", "--protected_stage2_source_stage2_checkpoint"):
            self.assertEqual(len({command[command.index(flag) + 1] for command in commands}), 1)
        self.assertEqual(commands[0][commands[0].index("--learning_rate") + 1], "1e-5")

    def test_requested_pair_has_matched_budget_and_no_ignored_stage1_adapter(self):
        commands = self.command(ablation=True, CONTINUATION_MODES="ce_only protected", MAX_STEPS="350")
        self.assertEqual(len(commands), 2)
        self.assertEqual([command[command.index("--protected_stage2_continuation") + 1] for command in commands],
                         ["ce_only", "protected"])
        for command in commands:
            self.assertNotIn("--pretrain_mm_mlp_adapter", command)
            self.assertEqual(command[command.index("--max_steps") + 1], "350")
            self.assertEqual(command[command.index("--protected_stage2_max_phase_b_steps") + 1], "350")
        for flag in ("--data_path", "--image_folder", "--seed", "--learning_rate", "--lr_scheduler_type",
                     "--warmup_ratio", "--protected_stage2_warmup_steps", "--protected_stage2_source_stage2_checkpoint"):
            self.assertEqual(commands[0][commands[0].index(flag) + 1], commands[1][commands[1].index(flag) + 1], flag)

    def test_ablation_rejects_unknown_or_duplicate_modes(self):
        for modes in ("ce_only ce_only", "protected unknown", " "):
            with self.subTest(modes=modes):
                result = self.command(ablation=True, check=False, CONTINUATION_MODES=modes)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(result.stdout.strip())

    def test_individual_source_rejects_resume_or_non_phase_b_before_launch(self):
        for settings in ({"STAGE2_START_PHASE_B": "False"},
                         {"STAGE2_START_PHASE_B": "True", "RESUME_FROM_CHECKPOINT": "/old/checkpoint-100"}):
            with self.subTest(settings=settings):
                result = self.command(check=False, SOURCE_STAGE2_CHECKPOINT="/source/ce-only", **settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(result.stdout.strip())

    def test_stage1_thresholds_cannot_satisfy_required_stage2_gates(self):
        result = self.command(check=False, STAGE2_CE_TOLERANCE=None, CE_TOLERANCE="0.005")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STAGE2_CE_TOLERANCE", result.stderr)

    def test_explicit_distributed_requests_fail_not_ignore(self):
        for values in ({"GPU_ID": "0,1"}, {"WORLD_SIZE": "2"}, {"DEEPSPEED_CONFIG": "zero2.json"},
                       {"GPU_INCLUDE": "localhost:0"}):
            with self.subTest(values=values):
                self.assertNotEqual(self.command(check=False, **values).returncode, 0)

    def test_disabled_launcher_cannot_silently_ignore_stage2_source(self):
        result = self.command(check=False, PROTECTED_CKA_STAGE2="False", SOURCE_STAGE2_CHECKPOINT="/source/ce-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STAGE2_CONTINUATION=ce_only", result.stderr)

    def test_qwen_wrapper_does_not_guess_stage1_checkpoint(self):
        command = self.command(wrapper=True, MODEL_NAME_OR_PATH=None)
        self.assertEqual(command[command.index("--model_name_or_path") + 1], "Qwen/Qwen2.5-3B-Instruct")
        result = self.command(wrapper=True, check=False, PRETRAIN_ADAPTER=None)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PRETRAIN_ADAPTER", result.stderr)

    def test_ablation_prints_three_matched_continuations_and_fresh_outputs(self):
        commands = self.command(ablation=True, SEED="17", MAX_STEPS="800")
        self.assertEqual(len(commands), 3)
        modes = [command[command.index("--protected_stage2_continuation") + 1] for command in commands]
        self.assertEqual(modes, ["ce_only", "ce_kl", "protected"])
        for flag in ("--data_path", "--image_folder", "--protected_stage2_source_stage2_checkpoint",
                     "--learning_rate", "--protected_stage2_refine_lr", "--max_steps",
                     "--protected_stage2_max_phase_b_steps", "--seed", "--per_device_train_batch_size",
                     "--gradient_accumulation_steps"):
            self.assertEqual(len({command[command.index(flag) + 1] for command in commands}), 1, flag)
        outputs = [command[command.index("--output_dir") + 1] for command in commands]
        self.assertEqual(outputs, ["/output/ablation/" + mode for mode in modes])
        self.assertEqual(len(set(outputs)), 3)
        for command in commands:
            self.assertEqual(command[command.index("--protected_stage2") + 1], "True")
            self.assertEqual(command[command.index("--protected_stage2_start_phase_b") + 1], "True")
            self.assertEqual(command[command.index("--protected_stage2_max_phase_b_steps") + 1], "800")
            self.assertEqual(command[command.index("--learning_rate") + 1], "1e-5")
            self.assertEqual(command[command.index("--seed") + 1], "17")
            self.assertNotIn("--resume_from_checkpoint", command)

    def test_ablation_old_stage1_knobs_cannot_replace_stage2_controls(self):
        commands = self.command(ablation=True, START_PHASE_B="False", REFINE_LR="99",
                                CE_TOLERANCE="99", TEXT_KL_BUDGET="99", VALIDATION_DATA_PATH="/stage1/captions.json")
        for command in commands:
            self.assertEqual(command[command.index("--protected_stage2_ce_tolerance") + 1], "0.02")
            self.assertEqual(command[command.index("--protected_stage2_text_kl_budget") + 1], "0.05")
            self.assertEqual(command[command.index("--protected_stage2_refine_lr") + 1], "1e-5")
            self.assertNotIn("--protected_stage2_validation_data_path", command)
        result = self.command(ablation=True, check=False, STAGE2_CE_TOLERANCE=None, CE_TOLERANCE="0.005")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STAGE2_CE_TOLERANCE", result.stderr)

    def test_ablation_rejects_resume_and_explicit_unsupported_backend(self):
        for values in ({"RESUME_FROM_CHECKPOINT": "/source/old-run"}, {"DEEPSPEED_CONFIG": "zero2.json"},
                       {"GPU_INCLUDE": "localhost:0,1"}, {"GPU_ID": "0,1"}, {"WORLD_SIZE": "2"}):
            with self.subTest(values=values):
                result = self.command(ablation=True, check=False, **values)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(result.stdout.strip())

    def test_ablation_rejects_cli_overrides_of_controlled_run_fields(self):
        cases = (("--output_dir", "/shared/output"), ("--output_dir=/shared/output",),
                 ("--output_d=/shared/output",), ("--protected_stage2_continu=ce_only",),
                 ("--resume_from_checkpoint", "/source/old-run"),
                 ("--protected_stage2", "False"),
                 ("--protected_stage2_continuation=ce_only",),
                 ("--protected_stage2_start_phase_b", "False"),
                 ("--protected_stage2_source_stage2_checkpoint=/different/source",),
                 ("--max_steps", "10"), ("--protected_stage2_max_phase_b_steps=10",),
                 ("--cka_grad_scope", "projector_only"))
        for extra in cases:
            with self.subTest(extra=extra):
                result = self.command(ablation=True, check=False, extra_args=extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(result.stdout.strip())

    def test_ablation_refuses_existing_or_symlink_outputs_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            (output / "ce_only").mkdir(parents=True)
            result = self.command(ablation=True, check=False, ABLATION_OUTPUT_ROOT=str(output),
                                  DRY_RUN="0", PYTHON_BIN="/nonexistent/no-training")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing", result.stderr)
            dangling = Path(temporary) / "dangling"
            dangling.mkdir()
            (dangling / "ce_kl").symlink_to(Path(temporary) / "absent-target")
            result = self.command(ablation=True, check=False, ABLATION_OUTPUT_ROOT=str(dangling),
                                  DRY_RUN="0", PYTHON_BIN="/nonexistent/no-training")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing", result.stderr)

    def test_ablation_stage1_source_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage1 = Path(temporary) / "stage1"
            stage1.mkdir()
            (stage1 / "config.json").write_text(json.dumps({"tune_mm_mlp_adapter": True}))
            (stage1 / "mm_projector.bin").write_bytes(b"fixture")
            validation = Path(temporary) / "heldout.json"
            validation.write_text("[]")
            commands = self.command(ablation=True, SOURCE_STAGE2_CHECKPOINT=str(stage1))
            for command in commands:
                args = arguments(protected_stage2_start_phase_b=True,
                                 protected_stage2_source_stage2_checkpoint=command[
                                     command.index("--protected_stage2_source_stage2_checkpoint") + 1])
                model = SimpleNamespace(tune_mm_mlp_adapter=False, vision_tower="fixture")
                data = SimpleNamespace(protected_stage2_validation_data_path=str(validation))
                with patch.dict(os.environ, {"WORLD_SIZE": "1"}), self.assertRaisesRegex(ValueError, "Stage 1 projector-only"):
                    config.validate_protected_cka_stage2_config(model, data, args)


if __name__ == "__main__":
    unittest.main()
