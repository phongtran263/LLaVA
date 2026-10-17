#!/usr/bin/env python3
"""Plan/run an explicitly configured lmms-eval task set, or import measured scores.

Requires a completed mode=train manifest entry. No task/model adapter is guessed.
Planning is the default; --execute runs the command. Example:
  python scripts/cvpr2027/evaluate.py --run-id RUN --tasks TASK1,TASK2 \\
    --model llava --model-args 'pretrained={checkpoint},conv_template=vicuna_v1'
  python scripts/cvpr2027/evaluate.py --run-id RUN --import-results results.json
  python scripts/cvpr2027/evaluate.py --run-id RUN --import-csv official_scores.csv

Manual CSV requires benchmark,metric,score. Optional identity columns must match
this manifest run. Preserve each official metric's units. lmms-eval's metric
standard errors are not independent-training-seed uncertainty and are excluded.
lmms JSON imports require config.model_args bound to the manifest checkpoint.
For adapters using another checkpoint key, set --checkpoint-arg. External
CSV scores use manual attestation; supply --evaluation-protocol protocol.json
to permit protocol-matched comparisons. Without it, comparisons stay unknown.
Check the installed evaluator's --help and supported tasks before execution.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from summarize import BASE_SCORE_FIELDS, SCORE_FIELDS, canonical_seed, eligible_run, load_manifest, read_csv  # noqa: E402


def select_run(root, run_id):
    manifest = load_manifest(root)
    run = manifest.get(run_id)
    if run is None:
        raise ValueError(f"Unknown manifest run_id: {run_id}")
    if not eligible_run(run):
        raise ValueError("Evaluation requires status=completed and mode=train; smoke/calibration runs cannot produce paper scores")
    return run


def numeric(value):
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def parse_model_args(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model args are missing; checkpoint provenance cannot be verified. Use the documented manual CSV route for external scores.")
    if value.lstrip().startswith("{"):
        try:
            parsed = json.loads(value)
        except ValueError as exc:
            raise ValueError("JSON model args must be an object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("JSON model args must be an object")
        return parsed
    result = {}
    for part in value.split(","):
        key, separator, entry = part.partition("=")
        key, entry = key.strip(), entry.strip()
        if not separator or not key or key in result:
            raise ValueError("Model args must be unambiguous key=value pairs, or a JSON object")
        if len(entry) > 1 and entry[0] == entry[-1] and entry[0] in ("'", '"'):
            entry = entry[1:-1]
        result[key] = entry
    return result


def validate_checkpoint_binding(model_args, run, checkpoint_arg="pretrained"):
    args = parse_model_args(model_args)
    actual = args.get(checkpoint_arg)
    expected = run.get("checkpoint_path")
    if not isinstance(actual, str) or not actual.strip() or not expected:
        raise ValueError(f"Model args must identify the manifest checkpoint using {checkpoint_arg}=...; use --checkpoint-arg for another adapter key, or manual CSV for unverified external results")
    if Path(actual).expanduser().resolve() != Path(expected).expanduser().resolve():
        raise ValueError(f"Evaluation checkpoint disagrees with manifest: {actual!r} versus {expected!r}")


def protocol_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def lmms_protocol(payload, benchmark, run, checkpoint_arg):
    config = dict(payload["config"])
    model_args = dict(parse_model_args(config["model_args"]))
    model_args[checkpoint_arg] = "<manifest_checkpoint>"
    config["model_args"] = model_args
    # Paths and switches that only control output do not define the evaluation.
    for key in ("output_path", "log_samples", "log_samples_suffix", "verbosity", "show_config", "write_out"):
        config.pop(key, None)
    task_configs = payload.get("configs", {})
    versions = payload.get("versions", {})
    identity = {"schema": "lmms-eval-protocol-v1", "benchmark": benchmark,
                "checkpoint_arg": checkpoint_arg, "config": config,
                "task_config": task_configs.get(benchmark) if isinstance(task_configs, dict) else None,
                "task_version": versions.get(benchmark) if isinstance(versions, dict) else None,
                "evaluator": {key: payload[key] for key in ("git_hash", "lmms_eval_version", "lm_eval_version", "version") if key in payload}}
    return protocol_digest(identity)


def manual_protocol(protocol, benchmark):
    if protocol is None:
        return ""
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError("Manual evaluation protocol must be a nonempty JSON object describing evaluator/task/decoding settings")
    if "benchmarks" in protocol:
        entries = protocol["benchmarks"]
        if not isinstance(entries, dict) or benchmark not in entries:
            raise ValueError(f"Manual protocol has no settings for benchmark {benchmark!r}")
        protocol = {**{key: value for key, value in protocol.items() if key != "benchmarks"}, "benchmark_settings": entries[benchmark]}
    return protocol_digest({"schema": "manual-evaluation-protocol-v1", "benchmark": benchmark, "protocol": protocol})


def result_rows(payload, run, checkpoint_arg="pretrained"):
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError("Expected lmms-eval JSON containing a results object")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("lmms-eval JSON is missing its config/model_args; checkpoint provenance is unverified. Use manual CSV for external scores.")
    if config.get("limit") is not None:
        raise ValueError("Limited/smoke evaluation cannot be imported into benchmark_results.csv")
    validate_checkpoint_binding(config.get("model_args"), run, checkpoint_arg)
    rows = []
    for benchmark, metrics in sorted(payload["results"].items()):
        if not isinstance(metrics, dict):
            continue
        for metric, value in sorted(metrics.items()):
            if "stderr" in metric.lower() or metric in ("alias", "samples", "num_samples", "n_samples"):
                continue
            score = numeric(value)
            if score is None:
                continue
            rows.append({**{key: run[key] for key in ("run_id", "backbone", "variant", "seed")},
                         "benchmark": benchmark, "metric": metric, "score": score,
                         "evaluation_protocol_sha256": lmms_protocol(payload, benchmark, run, checkpoint_arg)})
    if not rows:
        raise ValueError("No finite numeric benchmark metrics found; no scores were imported")
    return rows


def manual_rows(path, run, protocol=None):
    records = read_csv(path, ("benchmark", "metric", "score"))
    rows = []
    for record in records:
        for key in ("run_id", "backbone", "variant", "seed"):
            if key in record and record[key] != run[key]:
                raise ValueError(f"Manual CSV {key} disagrees with manifest run {run['run_id']}")
        score = numeric(record["score"])
        if score is None or not (record["benchmark"] or "").strip() or not (record["metric"] or "").strip():
            raise ValueError("Manual CSV requires nonempty benchmark/metric and a finite measured score")
        fingerprint = manual_protocol(protocol, record["benchmark"])
        if record.get("evaluation_protocol_sha256") and record["evaluation_protocol_sha256"] != fingerprint:
            raise ValueError("CSV evaluation protocol hash must match --evaluation-protocol; an unattested hash cannot establish provenance")
        rows.append({**{key: run[key] for key in ("run_id", "backbone", "variant", "seed")},
                     "benchmark": record["benchmark"], "metric": record["metric"], "score": score,
                     "evaluation_protocol_sha256": fingerprint})
    if not rows:
        raise ValueError("Manual CSV contains no measured rows")
    return rows


def score_key(row):
    return row["run_id"], row["benchmark"], row["metric"]


def append_scores(root, rows):
    """Idempotent immutable import; reject conflicting scores and concurrent loss."""
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "benchmark_results.csv"
    with (root / ".benchmark_results.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = read_csv(destination, BASE_SCORE_FIELDS)
        for row in existing:
            row.setdefault("evaluation_protocol_sha256", "")
        by_key = {score_key(row): row for row in existing}
        seed_keys = {tuple(canonical_seed(row[field]) if field == "seed" else row[field] for field in ("backbone", "variant", "seed", "benchmark", "metric")): score_key(row) for row in existing}
        if len(by_key) != len(existing) or len(seed_keys) != len(existing):
            raise ValueError("Existing benchmark_results.csv contains duplicate measurements")
        added = []
        for row in rows:
            key = score_key(row)
            if key in by_key:
                old = by_key[key]
                if any(str(old[field]) != str(row[field]) for field in SCORE_FIELDS if field != "score") or numeric(old["score"]) != row["score"]:
                    raise ValueError(f"Refusing to overwrite an existing benchmark score: {key}")
                continue
            seed_key = tuple(canonical_seed(row[field]) if field == "seed" else str(row[field]) for field in ("backbone", "variant", "seed", "benchmark", "metric"))
            if seed_key in seed_keys:
                raise ValueError(f"Refusing to count the same training seed twice: {seed_key}")
            seed_keys[seed_key] = key
            by_key[key] = row
            added.append(row)
        if not added and destination.exists():
            return 0
        descriptor, temporary = tempfile.mkstemp(prefix=".benchmark_results-", suffix=".csv", dir=root)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
                writer.writeheader()
                writer.writerows(existing + added)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return len(added)


def preserve_source(directory, source_bytes, suffix):
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source_bytes).hexdigest()
    destination = directory / (digest + suffix)
    if not destination.exists():
        try:
            with destination.open("xb") as handle:
                handle.write(source_bytes)
        except FileExistsError:
            pass
    if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Preserved source is incomplete or corrupt: {destination}")
    return destination, digest


def import_scores(root, run, source, manual=False, checkpoint_arg="pretrained", evaluation_protocol=None):
    source = Path(source)
    source_bytes = source.read_bytes()
    if manual:
        # Preserve before parsing as well as before score mutation, so the data
        # that is parsed is the immutable archived snapshot, not a changing file.
        destination, digest = preserve_source(root / "evaluation" / run["run_id"] / "imports", source_bytes, source.suffix)
        protocol = None
        if evaluation_protocol is not None:
            protocol_bytes = Path(evaluation_protocol).read_bytes()
            protocol = json.loads(protocol_bytes)
            preserve_source(destination.parent, protocol_bytes, ".protocol.json")
        rows = manual_rows(destination, run, protocol)
    else:
        rows = result_rows(json.loads(source_bytes), run, checkpoint_arg)
        destination, digest = preserve_source(root / "evaluation" / run["run_id"] / "imports", source_bytes, source.suffix)
    added = append_scores(root, rows)
    event = dict(source=str(source.resolve()), preserved_source=str(destination), sha256=digest,
                 format="manual_csv" if manual else "lmms_eval_json", imported_rows=added,
                 checkpoint_arg=checkpoint_arg if not manual else None,
                 checkpoint_binding="verified_json_config" if not manual else "manual_attestation",
                 evaluation_protocol_sha256={row["benchmark"]: row["evaluation_protocol_sha256"] for row in rows},
                 timestamp=datetime.now(timezone.utc).isoformat())
    with (destination.parent.parent / "imports.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return added


def build_command(run, root, tasks, model, model_args, launcher, batch_size, checkpoint_arg="pretrained"):
    if not tasks or not model or not model_args:
        raise ValueError("Evaluation planning requires explicit --tasks, --model and --model-args; task/adapter mappings are not inferred")
    checkpoint = run.get("checkpoint_path", "")
    if not checkpoint:
        raise ValueError("Manifest is missing the evaluation checkpoint_path")
    try:
        expanded = model_args.format(checkpoint=checkpoint, model_name_or_path=run.get("model_name_or_path", ""))
    except (KeyError, ValueError) as exc:
        raise ValueError("--model-args placeholders supported: {checkpoint}, {model_name_or_path}; escape literal braces as {{ }}") from exc
    validate_checkpoint_binding(expanded, run, checkpoint_arg)
    directory = root / "evaluation" / run["run_id"]
    # An explicit output directory keeps source evaluator results separate from the
    # official per-run CSV. Import a selected results JSON after successful eval.
    prefix = shlex.split(launcher)
    if not prefix:
        raise ValueError("--launcher cannot be empty")
    command = prefix + ["--model", model, "--model_args", expanded, "--tasks", tasks,
                                     "--batch_size", batch_size, "--output_path", str(directory / "lmms")]
    return command, directory


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("cka_results/cvpr2027_ablations"))
    parser.add_argument("--run-id", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--import-results", type=Path, help="Import a full lmms-eval results JSON")
    source.add_argument("--import-csv", type=Path, help="Import official scores with benchmark,metric,score columns")
    parser.add_argument("--tasks", help="Explicit comma-separated task IDs supported by your installed evaluator")
    parser.add_argument("--model", help="Explicit lmms-eval model backend")
    parser.add_argument("--model-args", help="Explicit adapter args; {checkpoint} expands from the manifest")
    parser.add_argument("--evaluation-protocol", type=Path, help="Manual CSV protocol JSON; common settings apply to all tasks, or use a benchmarks object for per-task settings")
    parser.add_argument("--checkpoint-arg", default="pretrained", help="Model-args key that identifies the manifest checkpoint (default: pretrained)")
    parser.add_argument("--launcher", default=f"{shlex.quote(sys.executable)} -m lmms_eval", help="External CLI prefix, e.g. 'accelerate launch -m lmms_eval'")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        run = select_run(root, args.run_id)
        if args.evaluation_protocol and not args.import_csv:
            raise ValueError("--evaluation-protocol applies only to --import-csv; lmms JSON records its own settings")
        if args.import_results or args.import_csv:
            if args.execute:
                raise ValueError("--execute cannot be combined with importing scores")
            count = import_scores(root, run, args.import_results or args.import_csv, manual=bool(args.import_csv), checkpoint_arg=args.checkpoint_arg, evaluation_protocol=args.evaluation_protocol)
            print(f"Imported {count} new measured scores; source preserved under evaluation/{args.run_id}/imports")
            return
        command, directory = build_command(run, root, args.tasks, args.model, args.model_args, args.launcher, args.batch_size, args.checkpoint_arg)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        (directory / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n")
        print(shlex.join(command))
        if args.execute:
            checkpoint = Path(run["checkpoint_path"])
            if not checkpoint.is_dir():
                raise ValueError(f"Evaluation checkpoint is not a local directory: {checkpoint}")
            with (directory / "console.log").open("a", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            event = {"returncode": completed.returncode, "run_id": args.run_id,
                     "timestamp": datetime.now(timezone.utc).isoformat(), "command": command}
            with (directory / "execution.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            if completed.returncode:
                parser.exit(completed.returncode, f"Evaluation failed; inspect {directory / 'console.log'}\n")
            print(f"Evaluation finished. Import the actual full results JSON with --import-results; results directory: {directory / 'lmms'}")
        else:
            print("Command saved. Add --execute after verifying installed tasks, model adapter, and benchmark data.")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
