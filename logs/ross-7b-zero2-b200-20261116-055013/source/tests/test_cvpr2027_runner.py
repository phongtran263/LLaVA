import argparse
import csv
import json
import io
import os
import signal
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.cvpr2027 import run
from scripts.cvpr2027.calibration import match_strength
from scripts.cvpr2027.prepare_validation import split


class ControlledRunnerTests(unittest.TestCase):
    def args(self, *extra):
        return run.parser().parse_args(["train", "--backbone", "qwen2.5-7b", *extra])

    def test_factorial_changes_only_objective_and_run_identity(self):
        configs = [run.run_spec(self.args(), variant, 42, "train") for variant in run.MAIN]
        excluded = {"cka_loss", "cka_loss_projector_weight", "cka_loss_final_hidden_weight", "cka_loss_layers",
                    "output_dir", "run_name"}
        shared = [{k: v for k, v in cfg["flags"].items() if k not in excluded} for cfg in configs]
        self.assertTrue(all(flags == shared[0] for flags in shared))
        self.assertEqual(len({cfg["recipe_sha256"] for cfg in configs}), 1)
        final = configs[0]
        self.assertEqual(final["flags"]["cka_loss_layers"], "final")
        self.assertEqual(final["lambda_proj"], 0)
        self.assertEqual(final["lambda_final"], 0.1)
        self.assertFalse(final["flags"]["vsp_asymmetric_pcgrad"])

    def test_strong_control_requires_a_measurement(self):
        cfg = run.run_spec(self.args(), "proj_strong", 42, "train")
        self.assertIsNone(cfg["lambda_proj"])
        with tempfile.TemporaryDirectory() as directory:
            cfg["checkpoint_path"] = directory + "/new"
            with self.assertRaisesRegex(ValueError, "requires"):
                run.execute(cfg, [], self.args())
            self.assertFalse(Path(cfg["checkpoint_path"]).exists())

    def test_match_combines_vectors_and_reports_cancellation(self):
        # Opposing equal gradients: vector norm is .1, summed norms would be .3.
        result = match_strength([{"projector_raw_norm": 1.0, "full_projector_norm": 0.1}], 0.2)
        self.assertEqual(result["recommended_lambda_proj"], 0.1)
        self.assertFalse(result["is_stronger_than_original"])
        self.assertEqual(result["achieved_to_full_projector_norm_ratios"], [1.0])
        with self.assertRaises(ValueError):
            match_strength([{"projector_raw_norm": 0, "full_projector_norm": 1}], 0.1)

    def test_stage1_metadata_guards_architecture_and_method(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "mm_projector.bin"
            adapter.write_bytes(b"test checkpoint metadata only")
            config = adapter.parent / "config.json"
            config.write_text(json.dumps({"mm_projector_type": "mlp2x_gelu", "cka_loss": False,
                                          "_name_or_path": "base"}))
            with self.assertRaisesRegex(ValueError, "projector"):
                run.adapter_info(adapter, "base", "coupling1x_gelu")
            with self.assertRaisesRegex(ValueError, "structural"):
                run.adapter_info(adapter, "base", "mlp2x_gelu")
            self.assertTrue(run.adapter_info(adapter, "base", "mlp2x_gelu", True)["allow_unverified"])

    def test_planning_preserves_completed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            row = dict.fromkeys(run.FIELDS, "")
            row.update(run_id="test", status="completed", elapsed_seconds="123")
            run.manifest_update(directory, row)
            row.update(status="pending", elapsed_seconds="")
            returned = run.manifest_update(directory, row)
            self.assertEqual(returned["status"], "completed")
            saved = list(csv.DictReader((Path(directory) / "training_manifest.csv").open()))
            self.assertEqual(saved[0]["elapsed_seconds"], "123")

    def test_resume_rejects_checkpoint_from_another_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other" / "checkpoint-1"
            path.mkdir(parents=True)
            (path / "trainer_state.json").write_text("{}")
            args = self.args("--resume", str(path))
            cfg = run.run_spec(args, "final", 42, "train")
            cfg["checkpoint_path"] = directory + "/expected"
            with self.assertRaisesRegex(ValueError, "exact ablation"):
                run.command_for(cfg, args)

    def test_checkpointing_algorithm_is_matched_and_serialized_as_json(self):
        args = self.args("--gradient-checkpointing")
        for variant in ("baseline", "full"):
            spec = run.run_spec(args, variant, 42, "train")
            command = run.command_for(spec, args)
            value = command[command.index("--gradient_checkpointing_kwargs") + 1]
            self.assertEqual(json.loads(value), {"use_reentrant": False})

    def test_holdout_keeps_same_image_conversations_together(self):
        samples = [{"image": "a", "conversations": [1]}, {"image": "a", "conversations": [2]},
                   {"image": "b", "conversations": [3]}, {"image": "c", "conversations": [4]}]
        train, val = split(samples, 0.5, 2027)
        self.assertFalse({s["image"] for s in train} & {s["image"] for s in val})
        self.assertEqual((train, val), split(samples, 0.5, 2027))
        self.assertEqual(len(train) + len(val), len(samples))

    def test_stage1_respects_explicit_resource_overrides(self):
        args = self.args("--batch-size", "3", "--accumulation-steps", "7",
                         "--pretrain-data-path", "/tmp/stage1-data.json",
                         "--pretrain-image-folder", "/tmp/stage1-images")
        spec = run.run_spec(args, "full", 42, "stage1")
        self.assertEqual(spec["flags"]["per_device_train_batch_size"], 3)
        self.assertEqual(spec["flags"]["gradient_accumulation_steps"], 7)
        self.assertEqual(spec["flags"]["data_path"], "/tmp/stage1-data.json")
        self.assertEqual(spec["flags"]["image_folder"], "/tmp/stage1-images")

    def test_validation_identity_tracks_images_and_json_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "validation.json"
            data.write_text('[{"id":1}]')
            args = self.args("--validation-data-path", str(data), "--validation-image-folder", "/tmp/images-a")
            first = run.run_spec(args, "full", 42, "train")
            args.validation_image_folder = "/tmp/images-b"
            second = run.run_spec(args, "full", 42, "train")
            self.assertNotEqual(first["recipe_sha256"], second["recipe_sha256"])
            data.write_text('[{"id":2}]')
            third = run.run_spec(args, "full", 42, "train")
            self.assertNotEqual(second["recipe_sha256"], third["recipe_sha256"])

    def test_probe_flag_cannot_silently_replace_training(self):
        settings = run.read_json(run.DEFAULT_ROOT / "configs/ablation.json")
        settings["recipe"]["debug_compare_cka"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text(json.dumps(settings))
            args = self.args("--config", str(path))
            self.assertFalse(run.run_spec(args, "full", 42, "train")["flags"]["debug_compare_cka"])
            self.assertTrue(run.run_spec(args, "full", 42, "calibration")["flags"]["debug_compare_cka"])

    def test_child_environment_is_used_for_preflight_and_isolates_probe_settings(self):
        args = self.args("--gpu-include", "localhost:3")
        spec = run.run_spec(args, "full", 42, "calibration")
        with mock.patch.dict(os.environ, {"CVPR_VALIDATION_DATA": "unrecorded.json"}):
            env = run.child_environment(spec, args, {"stage1_sha256": "s", "data_sha256": "d"})
        self.assertNotIn("CVPR_VALIDATION_DATA", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        with mock.patch.object(run.subprocess, "run") as check:
            run.runtime_check(args, env)
        self.assertIs(check.call_args.kwargs["env"], env)

    def test_completed_status_requires_full_budget_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = {"checkpoint_path": directory, "mode": "train"}
            self.assertEqual(run.completed_artifacts(spec)[0], "failed")
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "trainer_state.json").write_text(json.dumps({"global_step": 2, "max_steps": 3}))
            self.assertEqual(run.completed_artifacts(spec)[0], "incomplete")
            (root / "trainer_state.json").write_text(json.dumps({"global_step": 3, "max_steps": 3}))
            self.assertEqual(run.completed_artifacts(spec)[0], "failed")
            (root / "model.safetensors").write_bytes(b"synthetic presence check")
            self.assertEqual(run.completed_artifacts(spec)[0], "completed")

    def test_interruption_stops_owned_worker_process_group(self):
        process = mock.Mock(pid=12345)
        process.wait.side_effect = [KeyboardInterrupt(), 0, 0]
        with mock.patch.object(run.subprocess, "Popen", return_value=process) as launch:
            with mock.patch.object(run.os, "killpg") as stop:
                with self.assertRaises(KeyboardInterrupt):
                    run.run_process(["fake-training"], io.StringIO(), {})
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(stop.call_args_list,
                         [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)])

    def test_export_index_requires_every_referenced_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"checkpoint_path": directory, "mode": "train"}
            (root / "config.json").write_text("{}")
            (root / "trainer_state.json").write_text(json.dumps({"global_step": 3, "max_steps": 3}))
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"layer1": "model-1.safetensors", "layer2": "model-2.safetensors"}}))
            (root / "model-1.safetensors").write_bytes(b"first shard")
            self.assertEqual(run.completed_artifacts(spec)[0], "failed")
            (root / "model-2.safetensors").write_bytes(b"second shard")
            self.assertEqual(run.completed_artifacts(spec)[0], "completed")


if __name__ == "__main__":
    unittest.main()
