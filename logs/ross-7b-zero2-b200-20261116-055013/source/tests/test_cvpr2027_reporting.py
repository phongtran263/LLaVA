"""CPU-only tests of ablation statistics, provenance, and immutable score import."""
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "cvpr2027"
sys.path.insert(0, str(SCRIPT_DIR))
import summarize as REPORT
import evaluate as EVAL


class ReportingFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = {}
        self.scores = []

    def add(self, variant, seed, score, **changes):
        run_id = f"{variant}-{seed}"
        record = dict(run_id=run_id, backbone="vicuna_7b", variant=variant, seed=str(seed), status="completed",
                      mode="train", model_name_or_path="base", recipe_sha256="recipe", stage1_sha256="stage1",
                      data_sha256="data", geometry="linear", checkpoint_path=str(self.root / run_id), elapsed_seconds="")
        weights = {"baseline": (0, 0), "proj": (.3, 0), "final": (0, .1), "full": (.3, .1)}
        record.update(zip(("lambda_proj", "lambda_final"), weights.get(variant, (.3, .1))))
        record.update(changes)
        self.manifest[run_id] = record
        self.scores.append(dict(run_id=run_id, backbone=record["backbone"], variant=variant, seed=str(seed),
                                benchmark="test_task", metric="acc,none", score=score, evaluation_protocol_sha256="a" * 64))
        return record

    def write_fixture(self):
        fields = list(next(iter(self.manifest.values()))) if self.manifest else ["run_id", "backbone", "variant", "seed"]
        REPORT.write_csv(self.root / "training_manifest.csv", fields, self.manifest.values())
        REPORT.write_csv(self.root / "benchmark_results.csv", REPORT.SCORE_FIELDS, self.scores)



class ReportingTests(ReportingFixture):
    def test_sample_sd_and_matched_seed_factorial_interaction(self):
        for seed, values in [(1, (10, 11, 12, 15)), (2, (20, 22, 24, 29))]:
            for variant, score in zip(REPORT.CORE, values):
                self.add(variant, seed, score)
        # A third Full seed without a quartet affects Full mean/SD, not interaction.
        self.add("full", 3, 90)
        self.write_fixture()
        rows = REPORT.summarize(self.root)
        full = next(row for row in rows if row["variant"] == "full")
        self.assertEqual(full["n_seeds"], 3)
        self.assertAlmostEqual(full["mean"], 134 / 3)
        self.assertEqual(full["paired_n_seeds"], 2)
        self.assertEqual(full["baseline_delta_mean"], 7)
        self.assertAlmostEqual(full["baseline_delta_std"], 8 ** .5)
        # Per-seed interaction is 2 and 3, with sample SD sqrt(.5).
        self.assertEqual(full["interaction_n_seeds"], 2)
        self.assertEqual(full["interaction_mean"], 2.5)
        self.assertAlmostEqual(full["interaction_std"], .5 ** .5)

    def test_single_seed_uncertainty_remains_blank(self):
        self.add("final", 42, .65)
        self.write_fixture()
        rows = REPORT.summarize(self.root)
        self.assertEqual(rows[0]["std"], "")
        self.assertEqual(rows[0]["baseline_delta_mean"], "")
        text = (self.root / "summary.md").read_text()
        self.assertIn("Unknown", text)
        self.assertNotIn("± 0", (self.root / "ablation_table.tex").read_text())
        for question in range(1, 7):
            self.assertIn(f"Q{question}:", text)

    def test_factorial_rejects_conflicting_provenance_and_geometry(self):
        for variant, score in zip(REPORT.CORE, (1, 2, 3, 4)):
            self.add(variant, 1, score)
        self.manifest["proj-1"]["stage1_sha256"] = "different-stage1"
        rows, _ = REPORT.build_summary(self.scores, self.manifest)
        full = next(row for row in rows if row["variant"] == "full")
        self.assertEqual(full["interaction_n_seeds"], 0)
        self.manifest["proj-1"]["stage1_sha256"] = "stage1"
        self.manifest["full-1"]["geometry"] = "row_normalized"
        rows, _ = REPORT.build_summary(self.scores, self.manifest)
        self.assertEqual(next(row for row in rows if row["variant"] == "full")["interaction_n_seeds"], 0)

    def test_factorial_does_not_mix_original_and_strength_control_coefficients(self):
        for variant, score in zip(REPORT.CORE, (1, 2, 3, 4)):
            self.add(variant, 1, score)
        self.manifest["proj-1"]["lambda_proj"] = .9
        rows, _ = REPORT.build_summary(self.scores, self.manifest)
        self.assertEqual(next(row for row in rows if row["variant"] == "full")["interaction_n_seeds"], 0)

    def comparison(self, left, right):
        _, index = REPORT.build_summary(self.scores, self.manifest)
        return REPORT.paired_comparisons(index, self.manifest, left, right)

    def test_final_comparison_requires_same_final_weight_and_no_projector_control(self):
        self.add("full", 1, .8)
        final = self.add("final", 1, .7)
        self.assertTrue(self.comparison("full", "final"))
        final["lambda_final"] = .2
        self.assertFalse(self.comparison("full", "final"))
        final["lambda_final"] = .1
        final["lambda_proj"] = .3
        self.assertFalse(self.comparison("full", "final"))

    def test_strength_comparison_requires_stronger_projector_and_no_final_control(self):
        self.add("full", 1, .8)
        strong = self.add("proj_strong", 1, .7, lambda_proj=.6, lambda_final=0)
        self.assertTrue(self.comparison("full", "proj_strong"))
        strong["lambda_proj"] = .3
        self.assertFalse(self.comparison("full", "proj_strong"))
        strong["lambda_proj"] = .6
        strong["lambda_final"] = .1
        self.assertFalse(self.comparison("full", "proj_strong"))

    def test_geometry_comparison_requires_equal_coefficients_and_expected_geometry(self):
        self.add("full", 1, .8)
        normalized = self.add("full_row_normalized", 1, .7, geometry="row_normalized")
        self.assertTrue(self.comparison("full", "full_row_normalized"))
        normalized["lambda_proj"] = .6
        self.assertFalse(self.comparison("full", "full_row_normalized"))
        normalized["lambda_proj"] = .3
        normalized["lambda_final"] = .2
        self.assertFalse(self.comparison("full", "full_row_normalized"))
        normalized["lambda_final"] = .1
        normalized["geometry"] = "linear"
        self.assertFalse(self.comparison("full", "full_row_normalized"))

    def test_baseline_delta_requires_zero_baseline_weights(self):
        baseline = self.add("baseline", 1, .7, lambda_proj=.3)
        self.add("full", 1, .8)
        rows, _ = REPORT.build_summary(self.scores, self.manifest)
        self.assertEqual(next(row for row in rows if row["variant"] == "full")["paired_n_seeds"], 0)
        baseline["lambda_proj"] = 0
        self.assertTrue(self.comparison("full", "baseline"))

    def test_research_judgment_describes_measured_data_without_hardcoded_verdict(self):
        self.add("full", 1, .8)
        self.add("final", 1, .7)
        self.write_fixture()
        (self.root / "metric_directions.json").write_text(json.dumps({"test_task": {"acc,none": "higher"}}))
        REPORT.summarize(self.root)
        text = (self.root / "summary.md").read_text()
        self.assertIn("Full versus Final: 1 favorable", text)
        self.assertIn("do not establish statistical significance", text)
        self.assertNotIn("not yet empirically established", text)

    def test_empty_package_does_not_fabricate_results(self):
        rows = REPORT.summarize(self.root)
        self.assertEqual(rows, [])
        self.assertEqual(REPORT.read_csv(self.root / "ablation_summary.csv"), [])
        self.assertIn("No measured benchmark scores", (self.root / "ablation_table.tex").read_text())
        self.assertIn("The evidence needed to assess", (self.root / "summary.md").read_text())

    def test_duplicate_seed_and_smoke_results_are_rejected(self):
        record = self.add("full", 1, 10)
        self.write_fixture()
        self.scores.append({**self.scores[0], "run_id": "repeat"})
        self.manifest["repeat"] = {**record, "run_id": "repeat"}
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "independent"):
            REPORT.summarize(self.root)
        self.scores.pop()
        self.manifest.pop("repeat")
        record["mode"] = "smoke"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "mode=train"):
            REPORT.summarize(self.root)

    def test_scores_require_a_nonempty_manifest(self):
        self.add("full", 1, .8)
        self.write_fixture()
        (self.root / "training_manifest.csv").unlink()
        with self.assertRaisesRegex(ValueError, "training_manifest"):
            REPORT.summarize(self.root)

    def test_aggregate_rejects_mixed_training_recipes_and_coefficients(self):
        self.add("full", 1, .8)
        second = self.add("full", 2, .7)
        second["recipe_sha256"] = "another-recipe"
        with self.assertRaisesRegex(ValueError, "different training recipes"):
            REPORT.build_summary(self.scores, self.manifest)
        second["recipe_sha256"] = "recipe"
        second["lambda_final"] = .2
        with self.assertRaisesRegex(ValueError, "different training recipes"):
            REPORT.build_summary(self.scores, self.manifest)

    def test_seed_aliases_are_not_independent_runs(self):
        self.add("full", "1", .8)
        self.add("full", "01", .7)
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "independent"):
            REPORT.summarize(self.root)

    def test_reused_checkpoint_is_not_an_independent_seed_or_variant(self):
        first = self.add("full", 1, .8)
        second = self.add("full", 2, .7)
        second["checkpoint_path"] = first["checkpoint_path"]
        with self.assertRaisesRegex(ValueError, "Repeated checkpoint"):
            REPORT.build_summary(self.scores, self.manifest)
        self.scores.pop()
        self.manifest.pop("full-2")
        final = self.add("final", 1, .7)
        final["checkpoint_path"] = first["checkpoint_path"]
        self.assertFalse(self.comparison("full", "final"))

    def test_symlinked_checkpoint_is_not_an_independent_seed(self):
        first = self.add("full", 1, .8)
        Path(first["checkpoint_path"]).mkdir()
        second = self.add("full", 2, .7)
        Path(second["checkpoint_path"]).symlink_to(first["checkpoint_path"], target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Repeated checkpoint"):
            REPORT.build_summary(self.scores, self.manifest)

    def test_unknown_or_different_evaluation_protocol_prevents_pair_claims(self):
        self.add("full", 1, .8)
        self.add("final", 1, .7)
        self.assertTrue(self.comparison("full", "final"))
        self.scores[1]["evaluation_protocol_sha256"] = ""
        self.assertFalse(self.comparison("full", "final"))
        self.scores[1]["evaluation_protocol_sha256"] = "b" * 64
        self.assertFalse(self.comparison("full", "final"))

    def test_multiple_seeds_cannot_mix_known_evaluation_protocols(self):
        self.add("full", 1, .8)
        self.add("full", 2, .7)
        self.scores[1]["evaluation_protocol_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different evaluation protocols"):
            REPORT.build_summary(self.scores, self.manifest)

    def test_factorial_and_baseline_claims_require_known_evaluation_protocol(self):
        for variant, score in zip(REPORT.CORE, (1, 2, 3, 4)):
            self.add(variant, 1, score)
        self.scores[0]["evaluation_protocol_sha256"] = ""
        rows, _ = REPORT.build_summary(self.scores, self.manifest)
        full = next(row for row in rows if row["variant"] == "full")
        self.assertEqual(full["paired_n_seeds"], 0)
        self.assertEqual(full["interaction_n_seeds"], 0)

    def test_full_baseline_claim_requires_two_enabled_losses(self):
        self.add("full", 1, .8, lambda_proj=0, lambda_final=0)
        self.add("baseline", 1, .7)
        rows, _ = REPORT.build_summary(self.scores, self.manifest)
        self.assertEqual(next(row for row in rows if row["variant"] == "full")["paired_n_seeds"], 0)

    def test_nonfinite_scores_are_rejected(self):
        self.add("final", 1, float("nan"))
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            REPORT.summarize(self.root)

    def test_latex_escaping(self):
        escaped = REPORT.latex_escape(r"Qwen_7B & 50% # $ {x} ~ ^ \ path")
        for literal in (r"\_", r"\&", r"\%", r"\#", r"\$", r"\{", r"\}", r"\textasciitilde{}", r"\textasciicircum{}", r"\textbackslash{}"):
            self.assertIn(literal, escaped)

    def test_metric_direction_does_not_assume_higher_is_better(self):
        text = REPORT.comparison_text({("base", "task", "error"): [-2.]}, {"task": {"error": "lower"}})
        self.assertIn("improvement", text)
        self.assertIn("uncertainty unknown", text)
        no_direction = REPORT.comparison_text({("base", "task", "error"): [-2.]}, {})
        self.assertIn("direction unspecified", no_direction)

    def test_training_cost_reads_only_completed_training(self):
        self.add("full", 1, .8, elapsed_seconds="120")
        directory = self.root / "training_logs" / "full-1"
        directory.mkdir(parents=True)
        (directory / "telemetry.jsonl").write_text(json.dumps({"cuda_peak_allocated_bytes": 4 * 2 ** 30, "throughput_examples_per_second": 3}) + "\n")
        (directory / "telemetry-rank1.jsonl").write_text(json.dumps({"cuda_peak_allocated_bytes": 5 * 2 ** 30, "throughput_examples_per_second": 9}) + "\n")
        (directory / "trainer_state.json").write_text(json.dumps({"log_history": [{"grad_norm/cka_loss/projector_params": 2}]}))
        costs = REPORT.read_training_evidence(self.root, self.manifest)
        self.assertEqual(costs[0]["elapsed_seconds"], 120)
        self.assertEqual(costs[0]["peak_allocated_gib"], 5)
        self.assertEqual(costs[0]["throughput_examples_per_second"], 3)
        self.assertEqual(costs[0]["gradient_measurements"], 1)


class EvaluationImportTests(ReportingFixture):
    @staticmethod
    def payload(run, score=.5, **config):
        return {"config": {"model": "llava", "model_args": f"pretrained={run['checkpoint_path']},conv_template=vicuna_v1", "limit": None, **config},
                "results": {"task": {"acc,none": score}},
                "configs": {"task": {"num_fewshot": 0, "generation_kwargs": {"temperature": 0}}},
                "versions": {"task": "1"}, "git_hash": "evaluator-version-1"}

    def test_lmms_metric_import_excludes_eval_stderr(self):
        run = self.add("full", 1, .5)
        payload = {"config": {"model_args": f"pretrained={run['checkpoint_path']}", "limit": None}, "results": {"task": {"alias": "task", "acc,none": .5,
                   "acc_stderr,none": .03, "bool": True, "missing": "N/A", "nan": float("nan")}}}
        rows = EVAL.result_rows(payload, run)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metric"], "acc,none")
        self.assertEqual(rows[0]["score"], .5)
        with self.assertRaisesRegex(ValueError, "Limited"):
            EVAL.result_rows({**payload, "config": {**payload["config"], "limit": 10}}, run)

    def test_import_preserves_source_is_idempotent_and_never_overwrites(self):
        run = self.add("full", 1, .5)
        source = self.root / "results.json"
        source.write_text(json.dumps(self.payload(run, .5)))
        self.assertEqual(EVAL.import_scores(self.root, run, source), 1)
        self.assertEqual(EVAL.import_scores(self.root, run, source), 0)
        self.assertEqual(len(list((self.root / "evaluation/full-1/imports").glob("*.json"))), 1)
        original = (self.root / "benchmark_results.csv").read_bytes()
        source.write_text(json.dumps(self.payload(run, .8)))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            EVAL.import_scores(self.root, run, source)
        self.assertEqual((self.root / "benchmark_results.csv").read_bytes(), original)

    def test_manual_csv_rejects_identity_conflict(self):
        run = self.add("full", 1, .5)
        source = self.root / "scores.csv"
        REPORT.write_csv(source, ["run_id", "benchmark", "metric", "score"], [dict(run_id="different", benchmark="task", metric="acc", score=.8)])
        with self.assertRaisesRegex(ValueError, "disagrees"):
            EVAL.manual_rows(source, run)

    def test_command_requires_explicit_mapping_and_preserves_spaces(self):
        run = self.add("full", 1, .5, checkpoint_path="/tmp/checkpoint with spaces")
        command, directory = EVAL.build_command(run, self.root, "task1,task2", "llava", "pretrained={checkpoint}", "python -m lmms_eval", "1")
        self.assertIn("pretrained=/tmp/checkpoint with spaces", command)
        self.assertEqual(command[:3], ["python", "-m", "lmms_eval"])
        with self.assertRaisesRegex(ValueError, "explicit"):
            EVAL.build_command(run, self.root, "", "", "", "python -m lmms_eval", "1")

    def test_wrong_checkpoint_is_rejected_for_command_and_json_import(self):
        run = self.add("full", 1, .5)
        payload = self.payload(run, model_args="pretrained=/tmp/unrelated-checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint disagrees"):
            EVAL.result_rows(payload, run)
        with self.assertRaisesRegex(ValueError, "checkpoint disagrees"):
            EVAL.build_command(run, self.root, "task", "llava", "pretrained=/tmp/unrelated-checkpoint", "python -m lmms_eval", "1")

    def test_missing_json_model_provenance_has_explicit_manual_route(self):
        run = self.add("full", 1, .5)
        for payload in ({"results": {"task": {"acc": .5}}}, {"config": {}, "results": {"task": {"acc": .5}}}):
            with self.assertRaisesRegex(ValueError, "manual|Manual"):
                EVAL.result_rows(payload, run)

    def test_custom_checkpoint_argument_binds_exact_manifest_path(self):
        run = self.add("full", 1, .5)
        payload = self.payload(run, model_args={"model_path": run["checkpoint_path"]})
        self.assertEqual(len(EVAL.result_rows(payload, run, checkpoint_arg="model_path")), 1)
        with self.assertRaisesRegex(ValueError, "pretrained"):
            EVAL.result_rows(payload, run)

    def test_protocol_normalizes_checkpoint_only_and_tracks_eval_settings(self):
        first = self.add("full", 1, .5)
        second = self.add("final", 1, .4)
        payload1, payload2 = self.payload(first), self.payload(second)
        protocol = EVAL.result_rows(payload1, first)[0]["evaluation_protocol_sha256"]
        self.assertEqual(protocol, EVAL.result_rows(payload2, second)[0]["evaluation_protocol_sha256"])
        for mutate in (
            lambda value: value["config"].update(model_args=f"pretrained={second['checkpoint_path']},conv_template=other"),
            lambda value: value["configs"]["task"].update(num_fewshot=5),
            lambda value: value["versions"].update(task="2"),
            lambda value: value.update(git_hash="evaluator-version-2"),
            lambda value: value["configs"]["task"]["generation_kwargs"].update(temperature=1),
        ):
            changed = self.payload(second)
            mutate(changed)
            self.assertNotEqual(protocol, EVAL.result_rows(changed, second)[0]["evaluation_protocol_sha256"])

    def test_source_preservation_failure_leaves_official_scores_untouched(self):
        run = self.add("full", 1, .5)
        source = self.root / "results.json"
        source.write_text(json.dumps(self.payload(run)))
        with mock.patch.object(EVAL, "preserve_source", side_effect=OSError("fixture copy failure")):
            with self.assertRaisesRegex(OSError, "copy failure"):
                EVAL.import_scores(self.root, run, source)
        self.assertFalse((self.root / "benchmark_results.csv").exists())

    def test_manual_csv_protocol_is_optional_but_missing_means_unknown(self):
        run = self.add("full", 1, .5)
        source = self.root / "official.csv"
        REPORT.write_csv(source, ["benchmark", "metric", "score"], [dict(benchmark="task", metric="acc", score=.5)])
        self.assertEqual(EVAL.manual_rows(source, run)[0]["evaluation_protocol_sha256"], "")
        protocol = {"evaluator": "official-v1", "num_fewshot": 0, "temperature": 0}
        self.assertEqual(len(EVAL.manual_rows(source, run, protocol)[0]["evaluation_protocol_sha256"]), 64)
        protocol_file = self.root / "protocol.json"
        protocol_file.write_text(json.dumps(protocol))
        self.assertEqual(EVAL.import_scores(self.root, run, source, manual=True, evaluation_protocol=protocol_file), 1)
        self.assertEqual(len(list((self.root / "evaluation/full-1/imports").glob("*.protocol.json"))), 1)

    def test_pending_and_calibration_runs_cannot_be_evaluated(self):
        run = self.add("proj_strong", 1, .5, mode="calibration")
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "mode=train"):
            EVAL.select_run(self.root, run["run_id"])
        run["mode"] = "train"
        run["status"] = "pending"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "status=completed"):
            EVAL.select_run(self.root, run["run_id"])


if __name__ == "__main__":
    unittest.main()
