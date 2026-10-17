#!/usr/bin/env python3
"""Summarize measured ablations without inventing scores or seed uncertainty.

Run: python scripts/cvpr2027/summarize.py --root cka_results/cvpr2027_ablations
Optional metric_directions.json: {"benchmark": {"metric": "higher" | "lower"}}.
CSV scores remain in the original metric units; no cross-benchmark average is made.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BASE_SCORE_FIELDS = ["run_id", "backbone", "variant", "seed", "benchmark", "metric", "score"]
SCORE_FIELDS = BASE_SCORE_FIELDS + ["evaluation_protocol_sha256"]
SUMMARY_FIELDS = ["backbone", "variant", "benchmark", "metric", "evaluation_protocol_sha256", "n_seeds", "seeds", "mean", "std",
                  "baseline_delta_mean", "baseline_delta_std", "paired_n_seeds",
                  "interaction_mean", "interaction_std", "interaction_n_seeds"]
CORE = ("baseline", "proj", "final", "full")
PROVENANCE = ("model_name_or_path", "recipe_sha256", "stage1_sha256", "data_sha256")


def read_csv(path, required=()):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {', '.join(required)}")
        return list(reader)


def canonical_seed(value):
    try:
        if value is None or isinstance(value, bool):
            raise ValueError
        return str(int(str(value).strip(), 10))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Training seed must be an integer: {value!r}") from exc


def load_manifest(root):
    rows = read_csv(root / "training_manifest.csv", ("run_id", "backbone", "variant", "seed"))
    result = {}
    for row in rows:
        run_id = row["run_id"]
        if not run_id or run_id in result:
            raise ValueError(f"Missing or duplicate manifest run_id: {run_id!r}")
        row["seed"] = canonical_seed(row["seed"])
        result[run_id] = row
    return result


def eligible_run(row):
    return row.get("status") == "completed" and row.get("mode") == "train"


def load_scores(root, manifest):
    rows = read_csv(root / "benchmark_results.csv", BASE_SCORE_FIELDS)
    return validate_scores(rows, manifest)


def validate_scores(rows, manifest):
    if rows and not manifest:
        raise ValueError("Measured scores require a nonempty training_manifest.csv with completed training runs")
    seen, independent = set(), set()
    for row in rows:
        if any(row.get(field) is None or not str(row[field]).strip() for field in BASE_SCORE_FIELDS):
            raise ValueError("Benchmark rows must contain every required field; blank scores are not measurements")
        try:
            row["score"] = float(row["score"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid score for {row['run_id']}") from exc
        if not math.isfinite(row["score"]):
            raise ValueError(f"Nonfinite score for {row['run_id']}")
        protocol = row.get("evaluation_protocol_sha256") or ""
        if protocol and (len(protocol) != 64 or any(char not in "0123456789abcdef" for char in protocol)):
            raise ValueError(f"Invalid evaluation protocol SHA256 for {row['run_id']}")
        row["evaluation_protocol_sha256"] = protocol
        row["seed"] = canonical_seed(row["seed"])
        key = (row["run_id"], row["benchmark"], row["metric"])
        seed_key = tuple(row[x] for x in ("backbone", "variant", "seed", "benchmark", "metric"))
        if key in seen or seed_key in independent:
            raise ValueError(f"Repeated evaluation/training seed cannot count as an independent run: {key}")
        seen.add(key)
        independent.add(seed_key)
        if manifest:
            run = manifest.get(row["run_id"])
            if run is None:
                raise ValueError(f"Unknown manifest run: {row['run_id']}")
            if not eligible_run(run):
                raise ValueError(f"Scores require completed mode=train runs: {row['run_id']}")
            for field in ("backbone", "variant", "seed"):
                expected = canonical_seed(run[field]) if field == "seed" else run[field]
                if row[field] != expected:
                    raise ValueError(f"Score {field} disagrees with manifest for {row['run_id']}")
    return rows


def mean_sd(values):
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else "") if values else ("", "")


def controlled_pair(rows, manifest, same_geometry=False):
    """Require actual recorded, identical initialization/data/recipe for comparisons."""
    if not manifest:
        return False
    protocols = [row.get("evaluation_protocol_sha256", "") for row in rows]
    if not all(protocols) or len(set(protocols)) != 1:
        return False
    runs = [manifest[row["run_id"]] for row in rows]
    checkpoints = defaultdict(set)
    for run in runs:
        if not run.get("checkpoint_path"):
            return False
        checkpoints[str(Path(run["checkpoint_path"]).expanduser().resolve())].add(run["run_id"])
    if any(len(run_ids) > 1 for run_ids in checkpoints.values()):
        return False
    for field in PROVENANCE:
        values = [run.get(field, "") for run in runs]
        if not all(values) or len(set(values)) != 1:
            return False
    if same_geometry and len({run.get("geometry", "") for run in runs}) != 1:
        return False
    return True


def pair_coefficients_match(left, right, manifest):
    """The shared recipe hash excludes coefficients, so check each comparison."""
    try:
        lhs, rhs = manifest[left["run_id"]], manifest[right["run_id"]]
        p, f = float(lhs["lambda_proj"]), float(lhs["lambda_final"])
        rp, rf = float(rhs["lambda_proj"]), float(rhs["lambda_final"])
    except (KeyError, ValueError, TypeError):
        return False
    if not all(math.isfinite(value) and value >= 0 for value in (p, f, rp, rf)):
        return False
    variants = left["variant"], right["variant"]
    if right["variant"] == "baseline":
        if rp != 0 or rf != 0:
            return False
        variant = left["variant"]
        if variant == "baseline":
            return p == f == 0
        if variant in ("proj", "proj_strong"):
            return p > 0 and f == 0 and lhs.get("geometry") == "linear"
        if variant == "final":
            return p == 0 and f > 0 and lhs.get("geometry") == "linear"
        if variant == "full_row_normalized":
            return p > 0 and f > 0 and lhs.get("geometry") == "row_normalized"
        if variant == "full" or variant.startswith("full_"):
            return p > 0 and f > 0 and lhs.get("geometry") == "linear"
        return True
    if variants == ("full", "final"):
        return p > 0 and f > 0 and rp == 0 and rf == f and lhs.get("geometry") == rhs.get("geometry") == "linear"
    if variants == ("full", "proj_strong"):
        return p > 0 and f > 0 and rp > p and rf == 0 and lhs.get("geometry") == rhs.get("geometry") == "linear"
    if variants == ("full", "full_row_normalized"):
        return p > 0 and f > 0 and (p, f) == (rp, rf) and lhs.get("geometry") == "linear" and rhs.get("geometry") == "row_normalized"
    return True


def original_geometry(row, manifest):
    return manifest.get(row["run_id"], {}).get("geometry") == "linear"


def factorial_coefficients_match(quartet, manifest):
    try:
        coefficients = [(float(manifest[row["run_id"]]["lambda_proj"]),
                         float(manifest[row["run_id"]]["lambda_final"])) for row in quartet]
    except (KeyError, ValueError, TypeError):
        return False
    baseline, proj, final, full = coefficients
    return (all(math.isfinite(value) for pair in coefficients for value in pair)
            and baseline == (0.0, 0.0) and proj[0] > 0 and proj[1] == 0
            and final[0] == 0 and final[1] > 0 and full == (proj[0], final[1]))


def aggregation_signature(row, manifest):
    run = manifest[row["run_id"]]
    coefficients = []
    for field in ("lambda_proj", "lambda_final"):
        try:
            value = float(run.get(field, ""))
        except (ValueError, TypeError):
            value = None
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid {field} for {row['run_id']}")
        coefficients.append(value)
    return tuple(run.get(field, "") for field in (*PROVENANCE, "geometry")) + tuple(coefficients)


def validate_aggregation(scores, manifest):
    signatures = {}
    counts = defaultdict(int)
    protocols = defaultdict(set)
    checkpoint_runs = defaultdict(dict)
    for row in scores:
        key = tuple(row[field] for field in ("backbone", "variant", "benchmark", "metric"))
        signature = aggregation_signature(row, manifest)
        checkpoint = manifest[row["run_id"]].get("checkpoint_path")
        checkpoint = str(Path(checkpoint).expanduser().resolve()) if checkpoint else None
        if checkpoint in checkpoint_runs[key] and checkpoint_runs[key][checkpoint] != row["run_id"]:
            raise ValueError(f"Repeated checkpoint cannot count as an independent training run: {key}")
        checkpoint_runs[key][checkpoint] = row["run_id"]
        if len(checkpoint_runs[key]) > 1 and None in checkpoint_runs[key]:
            raise ValueError(f"Multiple-seed aggregation requires distinct recorded checkpoint paths: {key}")
        protocol = row.get("evaluation_protocol_sha256", "")
        if protocol:
            protocols[key].add(protocol)
        if len(protocols[key]) > 1:
            raise ValueError(f"Cannot aggregate different evaluation protocols: {key}. Use separate report roots for different evaluation settings.")
        if key in signatures and signature != signatures[key]:
            raise ValueError(f"Cannot aggregate different training recipes, initialization, data, geometry, or coefficients: {key}. Use separate report roots for different protocols.")
        counts[key] += 1
        if counts[key] > 1 and (not all(signature[:len(PROVENANCE) + 1]) or any(value is None for value in signature[-2:])):
            raise ValueError(f"Multiple-seed aggregation requires complete training provenance and coefficients: {key}")
        signatures[key] = signature


def build_summary(scores, manifest):
    validate_scores(scores, manifest)
    validate_aggregation(scores, manifest)
    groups = defaultdict(dict)
    index = {}
    for row in scores:
        key = tuple(row[field] for field in ("backbone", "variant", "benchmark", "metric"))
        groups[key][row["seed"]] = row
        index[(row["backbone"], row["variant"], row["seed"], row["benchmark"], row["metric"])] = row
    summaries = []
    for key, seed_rows in sorted(groups.items()):
        backbone, variant, benchmark, metric = key
        average, deviation = mean_sd([row["score"] for row in seed_rows.values()])
        protocols = {row.get("evaluation_protocol_sha256", "") for row in seed_rows.values()}
        protocol = next(iter(protocols)) if len(protocols) == 1 else ""
        deltas, interactions = [], []
        for seed, row in sorted(seed_rows.items()):
            baseline = index.get((backbone, "baseline", seed, benchmark, metric))
            if baseline and controlled_pair([row, baseline], manifest) and pair_coefficients_match(row, baseline, manifest):
                deltas.append(row["score"] - baseline["score"])
            if variant == "full":
                quartet = [index.get((backbone, member, seed, benchmark, metric)) for member in CORE]
                if all(quartet) and all(original_geometry(item, manifest) for item in quartet) and controlled_pair(quartet, manifest, True) and factorial_coefficients_match(quartet, manifest):
                    baseline_row, proj, final, full = quartet
                    interactions.append(full["score"] - proj["score"] - final["score"] + baseline_row["score"])
        delta_mean, delta_sd = mean_sd(deltas)
        interaction_mean, interaction_sd = mean_sd(interactions)
        summaries.append(dict(zip(("backbone", "variant", "benchmark", "metric"), key),
                              evaluation_protocol_sha256=protocol, n_seeds=len(seed_rows), seeds=";".join(sorted(seed_rows)), mean=average, std=deviation,
                              baseline_delta_mean=delta_mean, baseline_delta_std=delta_sd, paired_n_seeds=len(deltas),
                              interaction_mean=interaction_mean, interaction_std=interaction_sd,
                              interaction_n_seeds=len(interactions)))
    return summaries, index


def paired_comparisons(index, manifest, left, right, geometry_equal=False):
    groups = defaultdict(list)
    for (backbone, variant, seed, benchmark, metric), row in sorted(index.items()):
        if variant != left:
            continue
        other = index.get((backbone, right, seed, benchmark, metric))
        if other and controlled_pair([row, other], manifest, geometry_equal) and pair_coefficients_match(row, other, manifest):
            groups[(backbone, benchmark, metric)].append(row["score"] - other["score"])
    return groups


def latex_escape(value):
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in str(value))


def number(value):
    return "--" if value == "" else f"{value:.6g}"


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(root, summaries):
    lines = [r"% Requires \usepackage{booktabs}. Scores retain original benchmark units.",
             r"\begin{tabular}{llllrrr}", r"\toprule",
             r"Backbone & Variant & Benchmark & Metric & Seeds & Score & $\Delta$ baseline \\", r"\midrule"]
    for row in summaries:
        score = number(row["mean"])
        if row["std"] != "":
            score += r" $\pm$ " + number(row["std"])
        delta = number(row["baseline_delta_mean"])
        if row["baseline_delta_std"] != "":
            delta += r" $\pm$ " + number(row["baseline_delta_std"])
        lines.append(" & ".join([latex_escape(row[x]) for x in ("backbone", "variant", "benchmark", "metric")]
                                + [str(row["n_seeds"]), score, delta]) + r" \\")
    if not summaries:
        lines.append(r"\multicolumn{7}{c}{No measured benchmark scores available} \\")
    lines += [r"\bottomrule", r"\end{tabular}", "% +/- denotes sample standard deviation across independent training seeds.",
              "% Single-seed SD is omitted. Deltas use only provenance-matched paired seeds."]
    (root / "ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_training_evidence(root, manifest):
    evidence = []
    for run_id, run in sorted(manifest.items()):
        if not eligible_run(run):
            continue
        row = {"run_id": run_id, "elapsed_seconds": "", "peak_allocated_gib": "", "throughput_examples_per_second": "", "gradient_measurements": 0}
        try:
            value = float(run.get("elapsed_seconds") or "nan")
            if math.isfinite(value) and value >= 0:
                row["elapsed_seconds"] = value
        except ValueError:
            pass
        directory = Path(run.get("log_path") or root / "training_logs" / run_id)
        paths = [directory / "trainer_state.json"]
        checkpoint = run.get("checkpoint_path")
        if checkpoint:
            paths.append(Path(checkpoint) / "trainer_state.json")
        history = []
        for path in paths:
            if path.is_file():
                try:
                    history = json.loads(path.read_text()).get("log_history", [])
                    break
                except (ValueError, AttributeError):
                    pass
        for record in history:
            if row["elapsed_seconds"] == "" and isinstance(record.get("train_runtime"), (float, int)):
                row["elapsed_seconds"] = record["train_runtime"]
            row["gradient_measurements"] += sum(key.startswith("grad_norm/") and key != "grad_norm/measured_global_step" for key in record)
        telemetry = directory / "telemetry.jsonl"
        peaks, speeds = [], []
        for telemetry_path in sorted(directory.glob("telemetry*.jsonl")):
            for line in telemetry_path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                bytes_value = record.get("cuda_peak_allocated_bytes")
                if isinstance(bytes_value, (int, float)) and math.isfinite(bytes_value):
                    peaks.append(bytes_value / 2 ** 30)
                for key in ("cuda_max_memory_allocated_gib", "peak_allocated_gib", "max_memory_allocated_gib"):
                    value = record.get(key)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        peaks.append(value)
                        break
                for key in ("examples_per_second", "throughput_examples_per_second", "throughput_examples_per_s"):
                    value = record.get(key)
                    if telemetry_path == telemetry and isinstance(value, (int, float)) and math.isfinite(value):
                        speeds.append(value)
                        break
        if peaks:
            row["peak_allocated_gib"] = max(peaks)
        if speeds:
            row["throughput_examples_per_second"] = statistics.mean(speeds)
        evidence.append(row)
    return evidence


def calibration_evidence(root, manifest):
    evidence = []
    for run_id, run in sorted(manifest.items()):
        if run.get("status") != "completed" or run.get("mode") != "calibration":
            continue
        directory = Path(run.get("log_path") or root / "training_logs" / run_id)
        match = directory / "strength_match.json"
        gradients = directory / "gradient_norms.csv"
        if not match.is_file() or not gradients.is_file():
            continue
        try:
            values = json.loads(match.read_text())
            coefficient = float(values["recommended_lambda_proj"])
            if not math.isfinite(coefficient):
                continue
            batches = read_csv(gradients)
        except (KeyError, ValueError, TypeError):
            continue
        evidence.append(f"- {run_id}: {len(batches)} measured calibration batches; recommended projector coefficient "
                        f"{number(coefficient)}; scope: {values.get('matching_scope', 'unspecified')}. "
                        f"{values.get('limitation', 'Training-time strength matching is unknown.')}")
    return evidence


def load_directions(root):
    path = root / "metric_directions.json"
    if not path.exists():
        return {}
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or any(not isinstance(metrics, dict) or any(direction not in ("higher", "lower") for direction in metrics.values()) for metrics in values.values()):
        raise ValueError("metric_directions.json must map benchmark -> metric -> higher/lower")
    return values


def comparison_text(groups, directions):
    if not groups:
        return "Unknown: no complete provenance-matched paired evaluations."
    lines = []
    for (backbone, benchmark, metric), values in sorted(groups.items()):
        average, deviation = mean_sd(values)
        detail = f"{backbone} / {benchmark} / {metric}: paired delta {number(average)}"
        if deviation != "":
            detail += f" ± {number(deviation)} sample SD"
        detail += f" (n={len(values)} training seeds)"
        direction = directions.get(benchmark, {}).get(metric)
        if direction:
            favorable = average * (1 if direction == "higher" else -1)
            detail += "; observed mean " + ("improvement" if favorable > 0 else "regression" if favorable < 0 else "tie")
        else:
            detail += "; metric direction unspecified"
        if len(values) == 1:
            detail += "; uncertainty unknown"
        lines.append(detail + ".")
    return "\n\n".join(lines)


def write_markdown(root, manifest, summaries, index, costs, directions):
    lines = ["# CVPR 2027 VSP ablation results", "", f"Measured benchmark rows summarize {len(summaries)} variant/benchmark/metric groups.", "",
             "Scores keep their original units. SD is the sample standard deviation across independent training seeds; it is blank for one seed. No aggregate mixes different benchmarks or metric scales.", "",
             "Paired comparisons require a known, identical evaluation-protocol SHA256 as well as the same seed, backbone, benchmark, metric, model initialization, recipe hash, stage-1 hash, and data hash. Q1 also requires matched final coefficients; Q2 requires a strictly stronger projector coefficient and no final term in its control; Q4 requires identical coefficients with only geometry changed. Missing or conflicting provenance or coefficients leaves comparisons unknown. Manual scores without a declared evaluation protocol have unverified evaluation settings and yield no paired claims. Scores from different known evaluation protocols or training recipes are never pooled. Positive raw deltas do not imply improvement for lower-is-better metrics.", ""]
    status_counts = defaultdict(int)
    for run in manifest.values():
        status_counts[(run.get("mode", "unknown"), run.get("status", "unknown"))] += 1
    lines += ["## Execution coverage", ""]
    lines += [f"- {mode} / {status}: {count} runs." for (mode, status), count in sorted(status_counts.items())] or ["No training manifest records are available."]
    lines += ["", "## Q1: Does Full outperform Final-only?", "",
              comparison_text(paired_comparisons(index, manifest, "full", "final", True), directions), "",
              "## Q2: Does Full outperform stronger projector-only regularization?", "",
              comparison_text(paired_comparisons(index, manifest, "full", "proj_strong", True), directions), "",
              "Coefficient size alone does not establish matched gradient strength. Confirm the representative-batch gradient measurements and strength_match.json before interpreting this comparison.", "",
              *calibration_evidence(root, manifest), "",
              "## Q3: Are gains consistent across backbones and seeds?", "",
              comparison_text(paired_comparisons(index, manifest, "full", "baseline", True), directions), "",
              "Consistency requires multiple independent training seeds and backbones. A one-seed result has no estimated training-seed uncertainty. These descriptive statistics do not establish statistical significance.", "",
              "## Q4: Is original centered linear CKA preferable to row-normalized CKA?", "",
              comparison_text(paired_comparisons(index, manifest, "full", "full_row_normalized"), directions), "",
              "The Q4 delta is original minus row-normalized, with the same training provenance except geometry.", "",
              "## Q5: Which benchmarks improve or regress?", ""]
    changes = []
    for row in summaries:
        if row["variant"] == "baseline" or not row["paired_n_seeds"]:
            continue
        direction = directions.get(row["benchmark"], {}).get(row["metric"])
        delta = row["baseline_delta_mean"]
        verdict = "direction unknown"
        if direction:
            signed = delta * (1 if direction == "higher" else -1)
            verdict = "improvement" if signed > 0 else "regression" if signed < 0 else "tie"
        changes.append(f"- {row['backbone']} / {row['variant']} / {row['benchmark']} / {row['metric']}: {number(delta)} ({verdict}; {row['paired_n_seeds']} paired seeds).")
    lines += changes or ["Unknown: no provenance-matched baseline deltas are available."]
    lines += ["", "## Q6: Does added training cost justify the gains?", ""]
    available = [row for row in costs if row["elapsed_seconds"] != ""]
    if available:
        lines += [f"- {row['run_id']}: {number(row['elapsed_seconds'])} seconds; peak allocated GPU memory per device (max logged rank) {number(row['peak_allocated_gib'])} GiB; logged mean throughput {number(row['throughput_examples_per_second'])} examples/s; {row['gradient_measurements']} logged gradient norms." for row in available]
        lines += ["", "Unknown as an overall judgment: compare matched hardware and training budgets with measured gains. Elapsed time alone can include different overhead; absent measurements remain unknown."]
    else:
        lines += ["Unknown: no completed training-cost measurements are available."]
    interactions = [row for row in summaries if row["interaction_n_seeds"]]
    lines += ["", "## Factorial interaction", "", "Interaction = Full − Proj − Final + Baseline, computed independently per complete, provenance-matched linear-CKA seed quartet before aggregation.", ""]
    lines += [f"- {row['backbone']} / {row['benchmark']} / {row['metric']}: {number(row['interaction_mean'])}; sample SD {number(row['interaction_std'])}; {row['interaction_n_seeds']} complete quartets." for row in interactions] or ["Unknown: no complete matched quartets are available."]
    placement = paired_comparisons(index, manifest, "full", "final", True)
    strength = paired_comparisons(index, manifest, "full", "proj_strong", True)
    if not placement and not strength:
        judgment = "Unknown: no controlled Full-versus-Final or Full-versus-stronger-projector benchmark comparisons are available. The evidence needed to assess the two-location design remains incomplete."
    else:
        observations = []
        for label, groups in (("Full versus Final", placement), ("Full versus stronger projector", strength)):
            favorable = unfavorable = tied = unspecified = 0
            for (_, benchmark, metric), values in groups.items():
                direction = directions.get(benchmark, {}).get(metric)
                if not direction:
                    unspecified += 1
                    continue
                signed = statistics.mean(values) * (1 if direction == "higher" else -1)
                favorable += signed > 0
                unfavorable += signed < 0
                tied += signed == 0
            observations.append(f"{label}: {favorable} favorable, {unfavorable} unfavorable, {tied} tied observed means; {unspecified} metric directions unspecified, across {len(groups)} controlled backbone/benchmark/metric comparisons.")
        judgment = " ".join(observations) + " These counts describe measured means; they do not establish statistical significance or generalization beyond the observed tasks, seeds, and backbones. Assess training cost and the calibration limitations alongside Q1–Q6."
    lines += ["", "## Research judgment", "", judgment,
              "Pending, failed, smoke, and calibration runs are not benchmark results. No scores or uncertainty are imputed.", ""]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_scores(root, summaries):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    groups = defaultdict(list)
    for row in summaries:
        groups[(row["backbone"], row["benchmark"], row["metric"])].append(row)
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    for counter, (key, rows) in enumerate(sorted(groups.items()), 1):
        figure, axes = plt.subplots(figsize=(max(6, len(rows)), 4))
        x = list(range(len(rows)))
        axes.bar(x, [row["mean"] for row in rows])
        for position, row in zip(x, rows):
            if row["std"] != "":
                axes.errorbar(position, row["mean"], yerr=row["std"], color="black", capsize=3)
        axes.set_xticks(x, [row["variant"] for row in rows], rotation=30, ha="right")
        axes.set_title(" / ".join(key))
        axes.set_ylabel("Original metric units (error bars: seed sample SD)")
        figure.tight_layout()
        figure.savefig(directory / f"ablation_{counter:03d}.png", dpi=160)
        plt.close(figure)
    return True


def plot_training(root, manifest):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    for counter, (run_id, run) in enumerate(sorted(manifest.items()), 1):
        logs = Path(run.get("log_path") or root / "training_logs" / run_id)
        path = logs / "trainer_state.json"
        if not eligible_run(run) or not path.is_file():
            continue
        try:
            history = json.loads(path.read_text()).get("log_history", [])
        except (ValueError, AttributeError):
            continue
        series = defaultdict(list)
        for record in history:
            step = record.get("step")
            if not isinstance(step, (int, float)) or not math.isfinite(step):
                continue
            for key, value in record.items():
                if (key == "loss" or key.startswith("loss/") or key.startswith("grad_norm/")) and key != "grad_norm/measured_global_step" and isinstance(value, (int, float)) and math.isfinite(value):
                    series[key].append((step, value))
        if not series:
            continue
        has_gradients = any(key.startswith("grad_norm/") for key in series)
        figure, axes = plt.subplots(2 if has_gradients else 1, 1, figsize=(8, 7 if has_gradients else 4), squeeze=False)
        for key, points in sorted(series.items()):
            axis = axes[1 if key.startswith("grad_norm/") else 0, 0]
            axis.plot([point[0] for point in points], [point[1] for point in points], label=key)
        axes[0, 0].set_ylabel("Logged training loss")
        if has_gradients:
            axes[1, 0].set_ylabel("Logged gradient L2 norm")
        for axis in axes.flat:
            axis.set_xlabel("Optimizer step")
            axis.legend(fontsize="x-small")
        figure.suptitle(run_id)
        figure.tight_layout()
        figure.savefig(directory / f"training_{counter:03d}.png", dpi=160)
        plt.close(figure)
    return True


def summarize(root, plots=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(root)
    scores = load_scores(root, manifest)
    summaries, index = build_summary(scores, manifest)
    directions = load_directions(root)
    costs = read_training_evidence(root, manifest)
    write_csv(root / "ablation_summary.csv", SUMMARY_FIELDS, summaries)
    write_tex(root, summaries)
    write_markdown(root, manifest, summaries, index, costs, directions)
    if plots:
        if summaries:
            plot_scores(root, summaries)
        plot_training(root, manifest)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("cka_results/cvpr2027_ablations"))
    parser.add_argument("--plots", action="store_true", help="Create plots if matplotlib is installed")
    args = parser.parse_args()
    try:
        rows = summarize(args.root, args.plots)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote report with {len(rows)} measured groups to {args.root}")


if __name__ == "__main__":
    main()
