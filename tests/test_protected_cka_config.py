"""CPU-only config, split integrity, and launcher tests; never starts training."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("protected_config_under_test", ROOT / "llava/train/protected_cka_config.py")
CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG)


def args(**overrides):
    values = {"protected_cka_" + key: value for key, value in CONFIG.CONFIG_DEFAULTS.items()}
    values.update(protected_cka_refinement=True, protected_cka_refine_lr=1e-4,
                  protected_cka_min_phase_a_steps=200, protected_cka_min_phase_b_steps=0,
                  max_steps=1000, dataloader_num_workers=0, save_strategy="no", eval_strategy="no",
                  world_size=1, n_gpu=0, device=SimpleNamespace(type="cpu"))
    values.update(overrides)
    return SimpleNamespace(**values)


def model_args(**overrides):
    values = dict(tune_mm_mlp_adapter=True, vision_tower="test-vision")
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeDataset:
    def __init__(self, data_path, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args
        with open(data_path, encoding="utf-8") as handle:
            self.list_data_dict = json.load(handle)


class ConfigurationTests(unittest.TestCase):
    def validate(self, training=None, model=None, data=None):
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            CONFIG.validate_protected_cka_config(model or model_args(), data or SimpleNamespace(), training or args())

    def test_disabled_is_noop_even_with_unsupported_backend(self):
        training = args(protected_cka_refinement=False, deepspeed="zero3", max_steps=-1)
        before = vars(training).copy()
        self.validate(training, model=SimpleNamespace())
        self.assertEqual(vars(training), before)

    def test_default_starts_phase_a_and_derives_safe_limit(self):
        training = args()
        self.validate(training)
        self.assertFalse(training.protected_cka_start_phase_b)
        self.assertEqual(training.protected_cka_max_phase_a_steps, 1000)

    def test_new_experiment_defaults_reserve_phase_b_without_changing_budget(self):
        self.assertEqual(CONFIG.CONFIG_DEFAULTS["min_phase_a_steps"], 1500)
        self.assertEqual(CONFIG.CONFIG_DEFAULTS["min_phase_b_steps"], 350)
        training = args(max_steps=2173, protected_cka_min_phase_a_steps=1500,
                        protected_cka_min_phase_b_steps=350)
        self.validate(training)
        self.assertEqual(training.max_steps, 2173)
        self.assertEqual(training.protected_cka_max_phase_a_steps, 2173)
        with self.assertRaisesRegex(ValueError, "min_phase_a_steps \\+ min_phase_b_steps"):
            self.validate(args(max_steps=1800, protected_cka_min_phase_a_steps=1500,
                               protected_cka_min_phase_b_steps=350))
        for invalid in (-1, True, 0.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "min_phase_b_steps"):
                self.validate(args(protected_cka_min_phase_b_steps=invalid))

    def test_source_continuation_is_explicit_new_run_with_original_split(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "checkpoint-2173"
            (source / "protected_cka_split").mkdir(parents=True)
            for filename in ("protected_cka_state.pt", "mm_projector.bin"):
                (source / filename).write_bytes(b"runtime-validates-source")
            manifest = source / "protected_cka_split" / "split.json"
            manifest.write_text("{}")
            model = model_args(pretrain_mm_mlp_adapter="/missing/initial-adapter.bin")
            data = SimpleNamespace()
            training = args(protected_cka_start_phase_b=True,
                            protected_cka_source_stage1_checkpoint=str(source),
                            protected_cka_continuation="ce_only", max_steps=350,
                            protected_cka_min_phase_a_steps=1500, protected_cka_min_phase_b_steps=350,
                            output_dir=str(Path(temp) / "new-control"))
            self.validate(training, model, data)
            self.assertIsNone(model.pretrain_mm_mlp_adapter)
            self.assertEqual(data._protected_cka_source_split_manifest, str(manifest))
            self.assertEqual(training.max_steps, 350)
            training.resume_from_checkpoint = str(source)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                self.validate(training, model, data)
            training.resume_from_checkpoint = None
            training.output_dir = str(source)
            with self.assertRaisesRegex(ValueError, "fresh, empty"):
                self.validate(training, model, data)
            training.output_dir = str(Path(temp) / "new-refinement")
            manifest.unlink()
            with self.assertRaisesRegex(ValueError, "original protected_cka_split"):
                self.validate(training, model, data)

    def test_source_cannot_recreate_holdout_or_use_unverified_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "export"
            source.mkdir()
            (source / "mm_projector.bin").write_bytes(b"not-enough-provenance")
            validation = Path(temp) / "validation.json"
            validation.write_text("[]")
            with self.assertRaisesRegex(ValueError, "verified Stage 1 CE-only state"):
                self.validate(args(protected_cka_start_phase_b=True,
                                   protected_cka_source_stage1_checkpoint=str(source)),
                              data=SimpleNamespace(protected_cka_validation_data_path=str(validation)))

    def test_phase_b_requires_existing_adapter(self):
        training = args(protected_cka_start_phase_b=True)
        with self.assertRaisesRegex(ValueError, "existing --pretrain_mm_mlp_adapter"):
            self.validate(training)
        with tempfile.TemporaryDirectory() as temp:
            adapter = Path(temp) / "mm_projector.bin"
            adapter.write_bytes(b"fixture")
            validation = Path(temp) / "validation.json"
            validation.write_text("[]")
            self.validate(training, model=model_args(pretrain_mm_mlp_adapter=str(adapter)),
                          data=SimpleNamespace(protected_cka_validation_data_path=str(validation)))

    def test_old_adapter_cannot_retroactively_create_automatic_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = Path(temp) / "mm_projector.bin"
            adapter.write_bytes(b"fixture")
            for phase_b in (True, False):
                with self.subTest(phase_b=phase_b), self.assertRaisesRegex(ValueError, "already have seen"):
                    self.validate(args(protected_cka_start_phase_b=phase_b),
                                  model=model_args(pretrain_mm_mlp_adapter=str(adapter)))

    def test_full_resume_does_not_require_original_ce_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint-50"
            (checkpoint / "protected_cka_split").mkdir(parents=True)
            (checkpoint / "protected_cka_state.pt").write_bytes(b"fixture")
            (checkpoint / "protected_cka_split" / "split.json").write_text("{}")
            model = model_args(pretrain_mm_mlp_adapter="/missing/old-adapter.bin")
            data = SimpleNamespace()
            self.validate(args(protected_cka_start_phase_b=True, resume_from_checkpoint=str(checkpoint)), model, data)
            self.assertIsNone(model.pretrain_mm_mlp_adapter)
            self.assertEqual(data._protected_cka_source_split_manifest,
                             str(checkpoint / "protected_cka_split" / "split.json"))

    def test_rejects_unsupported_options(self):
        cases = dict(deepspeed="zero2", fsdp=["full_shard"], fp16=True, fp16_full_eval=True,
                     half_precision_backend="apex", torch_compile=True, torchdynamo="inductor",
                     gradient_checkpointing=True, lora_enable=True, bits=4,
                     dataloader_num_workers=1, max_steps=0, save_strategy="steps", eval_strategy="steps",
                     stop_after_step_ratio=0.5, save_at_step_ratio=0.5, n_gpu=2, world_size=2,
                     device=SimpleNamespace(type="mps"), freeze_mm_mlp_adapter=True,
                     group_by_length=True, group_by_modality_length=True, ignore_data_skip=True,
                     accelerator_config={"use_seedable_sampler": False}, load_best_model_at_end=True,
                     auto_find_batch_size=True, resume_model_only=True, optim="adafactor",
                     eval_on_start=True, use_liger_kernel=True, label_smoothing_factor=0.1,
                     lr_scheduler_type="reduce_lr_on_plateau")
        for key, value in cases.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Protected CKA"):
                self.validate(args(**{key: value}))

    def test_rejects_legacy_losses_and_stage2(self):
        for key in ("cka_loss", "use_pcgrad", "vsp_gradient_diagnostics", "vsp_asymmetric_pcgrad",
                    "vsp_apply_to_projector_only", "vsp_norm_cap", "mm_use_im_start_end"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(model=model_args(**{key: True}))
        with self.assertRaisesRegex(ValueError, "Stage 1"):
            self.validate(model=model_args(tune_mm_mlp_adapter=False))

    def test_numeric_validation(self):
        cases = dict(refine_lr=None, eps=0.0, margin=float("nan"), kl_weight=-1,
                     ema_decay=1.0, aux_backoff=1.0, validation_fraction=1.0,
                     validation_max_samples=0, eval_interval=0, max_retries=-1,
                     plateau_window=0, plateau_patience=0, plateau_min_delta=-0.1,
                     max_phase_a_steps=1001, min_phase_a_steps=1001)
        for key, value in cases.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(args(**{"protected_cka_" + key: value}))

    def test_dataclass_defaults_match_helper(self):
        tree = ast.parse((ROOT / "llava/train/train.py").read_text())
        training = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainingArguments")
        defaults = {}
        for node in training.body:
            if isinstance(node, ast.AnnAssign) and node.target.id.startswith("protected_cka_"):
                defaults[node.target.id] = ast.literal_eval(next(
                    keyword.value for keyword in node.value.keywords if keyword.arg == "default"))
        self.assertEqual(defaults, {"protected_cka_" + key: value for key, value in CONFIG.CONFIG_DEFAULTS.items()})


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.path = self.base / "train.json"
        self.data_args = SimpleNamespace(data_path=str(self.path), image_folder=str(self.base / "images"),
                                         train_data_fraction=1.0, protected_cka_validation_data_path=None,
                                         protected_cka_validation_image_folder=None)
        self.training = args(output_dir=str(self.base / "run"), protected_cka_validation_fraction=0.25,
                             protected_cka_validation_max_samples=4)

    def write(self, records, path=None):
        (path or self.path).write_text(json.dumps(records))

    def prepare(self):
        dataset = FakeDataset(str(self.path), None, self.data_args)
        return CONFIG.prepare_protected_cka_datasets(dataset, None, self.data_args, self.training, FakeDataset)

    def records(self, count=12):
        return [{"id": index, "image": f"{index}.jpg", "conversations": [{"value": f"caption {index}"}]} for index in range(count)]

    def test_grouped_split_disjoint_complete_and_reproducible(self):
        records = self.records()
        records[1]["image"] = "./0.jpg"
        records[2]["id"] = 1  # Transitively connected to images 0 and 2.
        self.write(records)
        train, validation = self.prepare()
        self.assertEqual(len(train.list_data_dict) + len(validation.list_data_dict), len(records))
        self.assertLessEqual(len(validation.list_data_dict), 4)
        CONFIG._assert_disjoint(train.list_data_dict, validation.list_data_dict,
                                self.data_args.image_folder, self.data_args.image_folder)
        manifest = Path(self.training.protected_cka_split_manifest)
        previous = manifest.read_bytes()
        train_again, validation_again = self.prepare()
        self.assertEqual(train.list_data_dict, train_again.list_data_dict)
        self.assertEqual(validation.list_data_dict, validation_again.list_data_dict)
        self.assertEqual(previous, manifest.read_bytes())
        self.assertEqual(len(self.training.protected_cka_split_fingerprint), 64)
        self.assertTrue((manifest.parent / "train_manifest.json").is_file())
        self.assertTrue((manifest.parent / "validation_manifest.json").is_file())

    def test_dataset_and_manifest_changes_rejected(self):
        records = self.records()
        self.write(records)
        self.prepare()
        records[0]["conversations"][0]["value"] = "changed caption"
        self.write(records)
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            self.prepare()

    def test_manifest_tampering_rejected(self):
        self.write(self.records())
        self.prepare()
        manifest = Path(self.training.protected_cka_split_manifest)
        value = json.loads(manifest.read_text())
        value["validation_indices"] = value["train_indices"][:1]
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            self.prepare()

    def test_seed_or_fraction_change_rejected_on_rerun(self):
        self.write(self.records())
        self.prepare()
        self.training.protected_cka_validation_seed += 1
        with self.assertRaises(ValueError):
            self.prepare()

    def test_source_checkpoint_split_matches_before_new_output_is_written(self):
        self.write(self.records())
        self.prepare()
        self.data_args._protected_cka_source_split_manifest = self.training.protected_cka_split_manifest
        self.training.output_dir = str(self.base / "new-run")
        self.prepare()
        self.training.output_dir = str(self.base / "changed-run")
        self.training.protected_cka_validation_seed += 1
        with self.assertRaisesRegex(ValueError, "source adapter/checkpoint split"):
            self.prepare()
        self.assertFalse(Path(self.training.output_dir).exists())

    def test_explicit_validation_uses_full_heldout_and_own_image_folder(self):
        self.write(self.records())
        validation_path = self.base / "validation.json"
        self.write([{"id": 99, "image": "99.jpg"}], validation_path)
        self.data_args.protected_cka_validation_data_path = str(validation_path)
        self.data_args.protected_cka_validation_image_folder = str(self.base / "validation-images")
        self.data_args.train_data_fraction = 0.1
        train, validation = self.prepare()
        self.assertEqual(len(train.list_data_dict), 12)
        self.assertEqual(len(validation.list_data_dict), 1)
        self.assertEqual(validation.data_args.train_data_fraction, 1.0)
        self.assertEqual(validation.data_args.image_folder, str(self.base / "validation-images"))
        self.assertEqual(self.data_args.train_data_fraction, 0.1)

    def test_explicit_image_and_id_overlap_rejected(self):
        self.write(self.records())
        validation_path = self.base / "validation.json"
        self.data_args.protected_cka_validation_data_path = str(validation_path)
        for record in ({"id": 99, "image": "./0.jpg"}, {"id": "0", "image": "99.jpg"}):
            with self.subTest(record=record):
                self.write([record], validation_path)
                with self.assertRaisesRegex(ValueError, "overlap"):
                    self.prepare()

    def test_symlink_aliases_stay_in_one_group(self):
        images = Path(self.data_args.image_folder)
        images.mkdir()
        (images / "alias.jpg").symlink_to(images / "0.jpg")
        records = self.records()
        records[1]["image"] = "alias.jpg"
        self.write(records)
        train, validation = self.prepare()
        for subset in (train.list_data_dict, validation.list_data_dict):
            identifiers = {record["id"] for record in subset}
            self.assertEqual(0 in identifiers, 1 in identifiers)

    def test_impossible_group_split_rejected(self):
        self.write([{"id": i, "image": "same.jpg"} for i in range(5)])
        with self.assertRaisesRegex(ValueError, "two independent"):
            self.prepare()

    def test_hard_cap_never_splits_large_image_group(self):
        records = [{"id": i, "image": "same.jpg"} for i in range(8)] + self.records()[8:]
        self.write(records)
        train, validation = self.prepare()
        self.assertLessEqual(len(validation.list_data_dict), 4)
        self.assertTrue(all(row["image"] != "same.jpg" for row in validation.list_data_dict))

    def test_disabled_data_helper_has_no_side_effects(self):
        self.write(self.records())
        self.training.protected_cka_refinement = False
        train, validation = self.prepare()
        self.assertEqual(len(train.list_data_dict), 12)
        self.assertIsNone(validation)
        self.assertFalse(Path(self.training.output_dir).exists())


class LauncherTests(unittest.TestCase):
    def command(self, **updates):
        environment = os.environ.copy()
        environment.update(MODEL_NAME_OR_PATH="model/test", TRAIN_DATA_PATH="/example/train.json",
                           TRAIN_IMAGE_FOLDER="/example/images", OUTPUT_DIR="/example/output",
                           REFINE_LR="1e-5", MAX_STEPS="1000", GPU_ID="0", DRY_RUN="1")
        environment.update(updates)
        result = subprocess.run(["bash", str(ROOT / "scripts/v1_5/pretrain_protected_cka.sh")],
                                env=environment, text=True, capture_output=True, check=True)
        return shlex.split(result.stdout)

    def test_default_auto_split_phase_a_single_process(self):
        command = self.command()
        self.assertNotIn("--deepspeed", command)
        self.assertNotIn("--pretrain_mm_mlp_adapter", command)
        self.assertNotIn("--protected_cka_validation_data_path", command)
        self.assertEqual(command[command.index("--learning_rate") + 1], "1e-3")
        for flag, expected in (("--protected_cka_refinement", "True"),
                               ("--protected_cka_start_phase_b", "False"),
                               ("--gradient_checkpointing", "False"),
                               ("--dataloader_num_workers", "0"), ("--save_strategy", "no"),
                               ("--eval_strategy", "no"), ("--cka_loss", "False")):
            self.assertEqual(command[command.index(flag) + 1], expected)

    def test_optional_adapter_heldout_and_baseline(self):
        command = self.command(PRETRAIN_ADAPTER="/example/ce.bin", START_PHASE_B="True",
                               VALIDATION_DATA_PATH="/example/val.json", VALIDATION_IMAGE_FOLDER="/example/val-images")
        self.assertEqual(command[command.index("--pretrain_mm_mlp_adapter") + 1], "/example/ce.bin")
        self.assertEqual(command[command.index("--protected_cka_start_phase_b") + 1], "True")
        self.assertEqual(command[command.index("--learning_rate") + 1], "1e-5")
        command = self.command(PROTECTED_CKA_REFINEMENT="False", PRETRAIN_ADAPTER="/example/ce.bin")
        self.assertEqual(command[command.index("--protected_cka_refinement") + 1], "False")
        self.assertEqual(command[command.index("--learning_rate") + 1], "1e-5")


if __name__ == "__main__":
    unittest.main()
