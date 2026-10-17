#!/usr/bin/env python3
"""Plan or launch matched VSP experiments. A launch always requires --execute."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import sys
import time
import threading

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "cka_results/cvpr2027_ablations"
FIELDS = ["run_id", "backbone", "variant", "seed", "lambda_proj", "lambda_final",
          "geometry", "model_name_or_path", "stage1_adapter", "git_revision",
          "status", "checkpoint_path", "log_path", "command_path", "mode",
          "recipe_sha256", "stage1_sha256", "data_sha256", "elapsed_seconds", "note"]
VARIANTS = {
    "baseline": (0, 0, "linear"), "proj": (1, 0, "linear"),
    "final": (0, 1, "linear"), "full": (1, 1, "linear"),
    "proj_strong": (None, 0, "linear"), "full_row_normalized": (1, 1, "row_normalized"),
    "full_proj_low": (0.5, 1, "linear"), "full_proj_high": (2, 1, "linear"),
    "full_final_low": (1, 0.5, "linear"), "full_final_high": (1, 2, "linear"),
}
MAIN = ["final", "baseline", "proj", "full"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def source_digest():
    # Includes dirty implementation changes: a commit alone does not identify this run.
    paths = sorted((REPO / "llava").rglob("*.py")) + sorted((REPO / "scripts/cvpr2027").glob("*.py"))
    return digest({str(p.relative_to(REPO)): file_digest(p) for p in paths})


def absolute(path):
    path = Path(path).expanduser()
    return str((REPO / path).resolve() if not path.is_absolute() else path.resolve())


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def manifest_update(root, row):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "training_manifest.csv"
        rows = list(csv.DictReader(path.open())) if path.exists() else []
        previous = next((r for r in rows if r["run_id"] == row["run_id"]), None)
        if previous and row["status"] == "pending" and previous["status"] != "pending":
            return previous
        rows = [r for r in rows if r["run_id"] != row["run_id"]] + [row]
        tmp = root / "training_manifest.csv.tmp"
        with tmp.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    return row


def adapter_info(path, model_name, projector_type, allow_unverified=False):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Stage-1 adapter not found: {path}. Supply --stage1-adapter or run stage1 first.")
    config_path = path.parent / "config.json"
    config = read_json(config_path) if config_path.exists() else {}
    stored_type = config.get("mm_projector_type")
    if stored_type and stored_type != projector_type:
        raise ValueError(f"Stage-1 projector is {stored_type}, recipe is {projector_type}; set --projector-type consistently.")
    stored_model = config.get("_name_or_path")
    if stored_model and stored_model != model_name and not allow_unverified:
        raise ValueError(f"Stage-1 language initialization {stored_model!r} differs from {model_name!r}.")
    if not config.get("cka_loss", False) and not allow_unverified:
        raise ValueError("Stage-1 config does not verify structural training. Supply the intended structural adapter; "
                         "--allow-unverified-stage1 explicitly records an alternative initialization experiment.")
    return {"path": str(path), "sha256": file_digest(path), "config": config,
            "config_sha256": file_digest(config_path) if config_path.exists() else "",
            "structural_config_verified": bool(config.get("cka_loss", False)),
            "allow_unverified": bool(allow_unverified)}


def load_settings(args, mode="train"):
    settings = read_json(args.config)
    recipe = dict(settings["recipe"])
    if mode == "stage1":
        recipe.update(settings["stage1_recipe"])
    for field in ("cka_loss_subset_select_layer", "cka_loss_anchor_layer", "cka_loss_vision_anchor_layer",
                  "cka_loss_projector_vision_anchor_layer", "cka_loss_final_vision_anchor_layer"):
        if recipe.get(field) is not None:
            raise ValueError(f"{field} is outside the controlled VSP method scope; remove it from the shared recipe.")
    backbone = dict(settings["backbones"][args.backbone])
    if args.model_name_or_path:
        backbone["model_name_or_path"] = args.model_name_or_path
    adapter = args.stage1_adapter or backbone.get("stage1_adapter")
    if adapter:
        adapter = absolute(adapter)
    if args.projector_type:
        recipe["mm_projector_type"] = args.projector_type
    elif adapter and (Path(adapter).parent / "config.json").exists():
        recipe["mm_projector_type"] = read_json(Path(adapter).parent / "config.json").get(
            "mm_projector_type", recipe["mm_projector_type"])
    data_override = (args.pretrain_data_path or args.data_path) if mode == "stage1" else args.data_path
    image_override = (args.pretrain_image_folder or args.image_folder) if mode == "stage1" else args.image_folder
    if data_override:
        recipe["data_path"] = data_override
    if image_override:
        recipe["image_folder"] = image_override
    recipe["data_path"] = absolute(recipe["data_path"])
    recipe["image_folder"] = absolute(recipe["image_folder"])
    if args.batch_size is not None:
        recipe["per_device_train_batch_size"] = args.batch_size
    if args.accumulation_steps is not None:
        recipe["gradient_accumulation_steps"] = args.accumulation_steps
    if args.gradient_checkpointing:
        recipe["gradient_checkpointing"] = True
    if recipe.get("gradient_checkpointing"):
        recipe["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return settings, recipe, backbone, adapter


def run_spec(args, variant, seed, mode):
    settings, recipe, backbone, adapter = load_settings(args, mode)
    p = positive(settings["lambda_proj"], "lambda_proj")
    f = positive(settings["lambda_final"], "lambda_final")
    calibration_identity = {
        "recipe": recipe, "model": backbone["model_name_or_path"],
        "stage1_adapter": adapter, "lambda_proj": p, "lambda_final": f,
        "source_sha256": source_digest(),
    }
    if mode == "stage1":
        adapter = None
        variant, p, f, geometry = "structural_stage1", p, 0.0, "linear"
    else:
        pm, fm, geometry = VARIANTS[variant]
        if variant == "proj_strong":
            if args.strong_projector_weight is not None:
                p = positive(args.strong_projector_weight, "strong projector coefficient")
            elif args.strength_match:
                match = read_json(args.strength_match)
                if match.get("calibration_identity_sha256") != digest(calibration_identity):
                    raise ValueError("Strength calibration used different initialization/recipe/coefficients.")
                if match.get("seed") != seed:
                    raise ValueError("Strength calibration seed differs. Use that seed for strength control, "
                                     "or explicitly freeze --strong-projector-weight for all seeds.")
                p = positive(match["recommended_lambda_proj"], "measured projector coefficient")
            else:
                p = None
            if p is not None and p <= float(settings["lambda_proj"]):
                raise ValueError("Measured coefficient is not stronger than original. Characterize cancellation "
                                 "in strength_match.json and choose an explicit stronger control if needed.")
            f = 0.0
        else:
            p, f = p * pm, f * fm
    for name in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        if not isinstance(recipe[name], int) or recipe[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    flags = dict(recipe)
    flags.update({"model_name_or_path": backbone["model_name_or_path"],
                  "version": "plain" if mode == "stage1" else backbone["version"],
                  "seed": seed, "data_seed": seed, "train_data_seed": seed,
                  "cka_loss": bool(p or f), "cka_loss_tau": 0.0,
                  "cka_loss_projector_weight": p, "cka_loss_final_hidden_weight": f,
                  "cka_loss_geometry": geometry, "cka_loss_start_ratio": 0.0,
                  "cka_loss_layers": "final" if f else "-1",
                  "freeze_backbone": False, "tune_mm_mlp_adapter": mode == "stage1",
                  "freeze_mm_mlp_adapter": False, "lora_enable": False,
                  "use_pcgrad": False, "vsp_asymmetric_pcgrad": False,
                  "vsp_norm_cap": False, "vsp_gradient_diagnostics": False,
                  "protected_cka_refinement": False, "protected_stage2": False,
                  "cka_loss_final_mlp_head": False, "debug_compare_cka": mode == "calibration"})
    if adapter:
        flags["pretrain_mm_mlp_adapter"] = adapter
    else:
        if mode != "stage1":
            flags["pretrain_mm_mlp_adapter"] = "STAGE1_ADAPTER_REQUIRED"
    if mode == "smoke":
        flags.update(max_steps=2, save_steps=1, per_device_train_batch_size=1,
                     gradient_accumulation_steps=1, dataloader_num_workers=0)
    if mode == "calibration":
        flags.update(debug_compare_cka=True, per_device_train_batch_size=args.calibration_batch_size,
                     gradient_accumulation_steps=1, dataloader_num_workers=0)
    if args.validation_data_path:
        # Selection uses a declared validation set, never benchmark test metrics.
        flags["eval_strategy"] = "steps"
        flags["eval_steps"] = settings.get("validation_steps", 500)
    run_id = f"{args.campaign}-{args.backbone}-{variant}-seed{seed}"
    if mode != "train":
        run_id += f"-{mode}"
    output = REPO / "checkpoints/cvpr2027_ablations" / args.campaign / args.backbone / variant / f"seed-{seed}" / mode
    log_dir = args.root / "training_logs" / run_id
    flags.update(output_dir=str(output), run_name=run_id, report_to=args.report_to)
    # Only loss placement, geometry, and coefficients vary. Hardware/world size is part of the recipe.
    validation = None
    if args.validation_data_path:
        validation_path = absolute(args.validation_data_path)
        validation = {"data_path": validation_path,
                      "data_sha256": file_digest(validation_path) if Path(validation_path).is_file() else "",
                      "image_folder": absolute(args.validation_image_folder or recipe["image_folder"])}
    recipe_id = digest({"recipe": recipe, "model": backbone, "gpu_include": args.gpu_include,
                        "deepspeed_sha256": file_digest(args.deepspeed_config),
                        "validation": validation,
                        "source_sha256": source_digest(), "protocol": "plain_sum_v1"})
    spec = {"run_id": run_id, "backbone": args.backbone, "variant": variant, "seed": seed,
            "mode": mode, "lambda_proj": p, "lambda_final": f, "geometry": geometry,
            "recipe_sha256": recipe_id, "git_revision": revision(), "flags": flags,
            "stage1_adapter": adapter, "checkpoint_path": str(output), "log_path": str(log_dir),
            "calibration_identity_sha256": digest(calibration_identity)}
    return spec


def command_for(spec, args):
    calibration = spec["mode"] == "calibration"
    if calibration:
        command = [args.python, "-m", "scripts.cvpr2027.entrypoint"]
    else:
        command = [args.python, "-m", "deepspeed.launcher.runner", "--include", args.gpu_include, "--master_port", str(args.master_port),
                   str(REPO / "scripts/cvpr2027/entrypoint.py"), "--deepspeed", absolute(args.deepspeed_config)]
    for key, value in spec["flags"].items():
        if value is not None:
            rendered = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            command.extend(["--" + key, rendered])
    if args.resume:
        if spec["mode"] not in ("train", "smoke", "stage1"):
            raise ValueError("Calibration cannot resume a trained optimizer.")
        output = Path(spec["checkpoint_path"])
        if args.resume == "latest":
            checkpoints = [p for p in output.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
            if not checkpoints:
                raise ValueError(f"No resumable checkpoints under {output}")
            checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
        else:
            checkpoint = Path(absolute(args.resume))
        if checkpoint.parent != output or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("Resume must name a saved checkpoint within this exact ablation run.")
        command.extend(["--resume_from_checkpoint", str(checkpoint)])
    return command


def runtime_check(args, env=None):
    # Inspect the selected interpreter, not the process that generated the plan.
    code = ("import json, sys, transformers, accelerate, torch; "
            "assert transformers.__version__ == '4.51.3', "
            "f'Activate the repository environment: expected transformers==4.51.3, got {transformers.__version__}'; "
            "assert torch.cuda.is_available(), 'CUDA training device unavailable'; "
            "import deepspeed, flash_attn; "
            "print(json.dumps({'python': sys.executable, 'torch': torch.__version__, "
            "'transformers': transformers.__version__, 'accelerate': accelerate.__version__, "
            "'deepspeed': deepspeed.__version__, 'flash_attn': flash_attn.__version__, "
            "'cuda_runtime': torch.version.cuda}))")
    output = ""
    try:
        result = subprocess.run([args.python, "-c", code], check=True, cwd=REPO, env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout
    except subprocess.CalledProcessError as error:
        output = error.stdout
        raise
    finally:
        if isinstance(output, str) and output and env and env.get("CVPR_LOG_DIR"):
            directory = Path(env["CVPR_LOG_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "preflight.log").open("a") as stream:
                stream.write(output + "\n")


def child_environment(spec, args, identity):
    env = os.environ.copy()
    # Only explicitly resolved experiment settings may steer the entrypoint.
    for key in list(env):
        if key.startswith("CVPR_"):
            env.pop(key)
    env.update(CVPR_LOG_DIR=spec["log_path"], CVPR_RUN_ID=spec["run_id"], CVPR_MODE=spec["mode"],
               CVPR_CALIBRATION_BATCHES=str(args.calibration_batches),
               CVPR_CALIBRATION_BATCH_SIZE=str(args.calibration_batch_size),
               CVPR_CALIBRATION_ID=spec["calibration_identity_sha256"],
               CVPR_CALIBRATION_SEED=str(spec["seed"]),
               CVPR_STAGE1_SHA256=identity["stage1_sha256"], CVPR_DATA_SHA256=identity["data_sha256"],
               PYTHONNOUSERSITE="1", PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
    if args.validation_data_path:
        env["CVPR_VALIDATION_DATA"] = absolute(args.validation_data_path)
        env["CVPR_VALIDATION_IMAGES"] = absolute(args.validation_image_folder or spec["flags"]["image_folder"])
    if spec["mode"] == "calibration":
        host, separator, devices = args.gpu_include.partition(":")
        if host != "localhost" or not separator or not devices.isdigit():
            raise ValueError("Calibration needs --gpu-include localhost:N for exactly one local GPU.")
        env["CUDA_VISIBLE_DEVICES"] = devices
    return env


def run_process(command, log, env):
    """Stop all workers owned by this launch before releasing the run lock."""
    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    main_thread = threading.current_thread() is threading.main_thread()
    previous_term = signal.getsignal(signal.SIGTERM) if main_thread else None

    def interrupted_by_term(_number, _frame):
        raise KeyboardInterrupt("launch interrupted by SIGTERM")

    if main_thread:
        signal.signal(signal.SIGTERM, interrupted_by_term)
    try:
        return process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # A launcher can exit before workers that ignore TERM. Sweep its whole
        # process group even when the launcher has already returned.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, previous_term)


def completed_artifacts(spec):
    """A clean process exit alone does not establish full training completion."""
    output = Path(spec["checkpoint_path"])
    if spec["mode"] == "calibration":
        directory = Path(spec["log_path"])
        required = [directory / name for name in ("gradient_norms.csv", "strength_match.json", "calibration_samples.json")]
        if not all(path.is_file() for path in required):
            return "failed", "calibration exited without measured probe artifacts"
        return "completed", "calibration measurements saved"
    state_path = output / "trainer_state.json"
    if not state_path.is_file() or not (output / "config.json").is_file():
        return "failed", "training exited without saved model configuration/trainer state"
    state = read_json(state_path)
    step, maximum = int(state.get("global_step", 0)), int(state.get("max_steps", 0))
    if maximum <= 0 or step < maximum:
        return "incomplete", f"training stopped at optimizer step {step}/{maximum}; resume before evaluation"
    patterns = ("mm_projector.bin",) if spec["mode"] == "stage1" else (
        "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
    valid_export = False
    for pattern in patterns:
        path = output / pattern
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if pattern.endswith(".index.json"):
            index = read_json(path)
            shards = set(index.get("weight_map", {}).values())
            if not shards:
                continue
            valid_export = all(isinstance(shard, str) and
                               (output / shard).resolve().is_relative_to(output.resolve()) and
                               (output / shard).is_file() and (output / shard).stat().st_size > 0
                               for shard in shards)
        else:
            valid_export = True
        if valid_export:
            break
    if not valid_export:
        return "failed", "training exited without exported model weights"
    return "completed", f"saved export at optimizer step {step}/{maximum}"


def execute(spec, command, args):
    output = Path(spec["checkpoint_path"])
    if output.exists() and not args.resume:
        raise ValueError(f"Refusing to overwrite {output}. Use --resume latest, or a new --campaign.")
    flags = spec["flags"]
    if spec["lambda_proj"] is None:
        raise ValueError("proj_strong requires --strength-match from calibration or explicit --strong-projector-weight.")
    if spec["mode"] != "stage1":
        if not spec["stage1_adapter"]:
            raise ValueError("This backbone has no verified structural Stage-1 adapter. Supply --stage1-adapter or run stage1.")
        info = adapter_info(spec["stage1_adapter"], flags["model_name_or_path"],
                            flags["mm_projector_type"], args.allow_unverified_stage1)
    else:
        info = None
    if not Path(flags["data_path"]).is_file() or not Path(flags["image_folder"]).is_dir():
        raise ValueError("Training JSON/image folder is missing; supply the local dataset paths.")
    data_hash = file_digest(flags["data_path"])
    validation_hash = file_digest(absolute(args.validation_data_path)) if args.validation_data_path else ""
    if validation_hash and validation_hash == data_hash:
        raise ValueError("Validation must be a held-out dataset, not the training JSON.")
    identity = {"recipe_sha256": spec["recipe_sha256"], "flags": flags,
                "stage1_sha256": info["sha256"] if info else "", "data_sha256": data_hash,
                "validation_sha256": validation_hash}
    record = output / "ablation_identity.json"
    if args.resume and (not record.exists() or read_json(record) != identity):
        raise ValueError("Resume identity differs from saved source/recipe/data/initialization. Preserve the old run and use a new campaign.")
    if spec["variant"] == "proj_strong" and args.strength_match:
        match = read_json(args.strength_match)
        if match.get("stage1_sha256") != identity["stage1_sha256"] or match.get("data_sha256") != data_hash:
            raise ValueError("Strength calibration adapter/dataset content differs from this run.")
    if args.resume and spec["mode"] in ("train", "stage1"):
        rows = list(csv.DictReader((args.root / "training_manifest.csv").open()))
        previous = next((row for row in rows if row["run_id"] == spec["run_id"]), None)
        if previous and previous["status"] == "completed":
            raise ValueError("This training run is complete. Use a new campaign to preserve its exported weights and benchmark identity.")
    env = child_environment(spec, args, identity)
    runtime_check(args, env=env)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(record, identity)
    log_dir = Path(spec["log_path"])
    write_json(log_dir / "stage1_provenance.json", info)
    spec.update(stage1_sha256=identity["stage1_sha256"], data_sha256=data_hash)
    write_json(log_dir / "resolved_config.json", spec)
    row = manifest_row(spec, command, "running")
    manifest_update(args.root, row)
    started = time.monotonic()
    history_path = log_dir / "attempts.jsonl"
    try:
        with (log_dir / "console.log").open("a" if args.resume else "w") as log:
            returncode = run_process(command, log, env)
        if returncode == 0:
            status, row["note"] = completed_artifacts(spec)
        else:
            status, row["note"] = "failed", f"exit_code={returncode}"
    except KeyboardInterrupt:
        status = "interrupted"
        row["note"] = "KeyboardInterrupt; resume from saved checkpoint"
    finally:
        attempt_seconds = round(time.monotonic() - started, 3)
        past_seconds = 0.0
        if history_path.exists():
            for line in history_path.read_text().splitlines():
                previous = json.loads(line)
                past_seconds += float(previous.get("attempt_seconds", previous.get("elapsed_seconds", 0)) or 0)
        row["elapsed_seconds"] = round(past_seconds + attempt_seconds, 3)
        row["status"] = locals().get("status", "failed")
        with history_path.open("a") as history:
            history.write(json.dumps({**row, "attempt_seconds": attempt_seconds}) + "\n")
        manifest_update(args.root, row)
    print(f"{spec['run_id']}: {row['status']}. Log: {log_dir / 'console.log'}", flush=True)
    return 0 if row["status"] == "completed" else 1


def manifest_row(spec, command, status):
    row = {key: spec.get(key, "") for key in FIELDS}
    row.update(model_name_or_path=spec["flags"]["model_name_or_path"], status=status,
               command_path=str(Path(spec["log_path"]) / "command.sh"))
    if status == "pending":
        requirements = []
        if not spec["stage1_adapter"] and spec["mode"] != "stage1":
            requirements.append("requires structural stage1 adapter")
        if spec["lambda_proj"] is None:
            requirements.append("requires measured or explicitly characterized stronger coefficient")
        row["note"] = "; ".join(requirements)
    return row


def relaunch_command(spec, args):
    action = "calibrate" if spec["mode"] == "calibration" else spec["mode"]
    command = [args.python, str(REPO / "scripts/cvpr2027/run.py"), action,
               "--config", str(args.config.resolve()), "--root", str(args.root),
               "--campaign", args.campaign, "--backbone", args.backbone,
               "--variant", "full" if spec["mode"] == "stage1" else spec["variant"],
               "--seed", str(spec["seed"]), "--execute"]
    options = ["stage1_adapter", "projector_type", "model_name_or_path", "data_path", "image_folder",
               "pretrain_data_path", "pretrain_image_folder", "validation_data_path", "validation_image_folder",
               "gpu_include", "batch_size", "accumulation_steps", "deepspeed_config", "python", "master_port",
               "report_to", "resume", "calibration_batches", "calibration_batch_size", "strength_match",
               "strong_projector_weight"]
    for name in options:
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    for name in ("allow_unverified_stage1", "gradient_checkpointing"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    return command


def plan_one(args, variant, seed, mode):
    spec = run_spec(args, variant, seed, mode)
    command = command_for(spec, args)
    log_dir = Path(spec["log_path"])
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Planning must not overwrite recorded executed configurations.
        if not (log_dir / "resolved_config.json").exists():
            write_json(log_dir / "planned_config.json", spec)
            text = "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(REPO)) + "\n"
            text += "# Underlying training command for review:\n# " + shlex.join(command) + "\n"
            text += "# Preserve launch checks and manifest logging when replaying this script.\n"
            text += shlex.join(relaunch_command(spec, args)) + "\n"
            (log_dir / "command.sh").write_text(text)
        manifest_update(args.root, manifest_row(spec, command, "pending"))
        print(shlex.join(command), flush=True)
        if args.execute:
            try:
                return execute(spec, command, args)
            except (ValueError, OSError, subprocess.CalledProcessError) as error:
                # A rejected launch never changes a completed run's manifest row.
                rows = list(csv.DictReader((args.root / "training_manifest.csv").open()))
                previous = next((r for r in rows if r["run_id"] == spec["run_id"]), None)
                if previous and previous["status"] not in ("completed", "running"):
                    row = {**previous, "status": "failed", "note": str(error)}
                    manifest_update(args.root, row)
                print(f"Launch failed: {error}", file=sys.stderr)
                return 1
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "train", "smoke", "calibrate", "stage1", "suite"])
    p.add_argument("--config", type=Path, default=DEFAULT_ROOT / "configs/ablation.json")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--campaign", default="controlled-v1")
    p.add_argument("--backbone", choices=["vicuna-7b", "qwen2.5-3b", "qwen2.5-7b"], default="vicuna-7b")
    p.add_argument("--variant", choices=list(VARIANTS), default="final")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--execute", action="store_true", help="Launch; otherwise only record/print the plan.")
    p.add_argument("--stage1-adapter")
    p.add_argument("--projector-type")
    p.add_argument("--model-name-or-path")
    p.add_argument("--allow-unverified-stage1", action="store_true")
    p.add_argument("--data-path")
    p.add_argument("--image-folder")
    p.add_argument("--pretrain-data-path")
    p.add_argument("--pretrain-image-folder")
    p.add_argument("--validation-data-path")
    p.add_argument("--validation-image-folder")
    p.add_argument("--gpu-include", default="localhost:0,1")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulation-steps", type=int)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--deepspeed-config", default=str(REPO / "scripts/zero2.json"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--master-port", type=int, default=29571)
    p.add_argument("--report-to", choices=["none", "wandb"], default="none")
    p.add_argument("--resume", nargs="?", const="latest")
    p.add_argument("--calibration-batches", type=int, default=8)
    p.add_argument("--calibration-batch-size", type=int, default=1)
    p.add_argument("--strength-match", type=Path)
    p.add_argument("--strong-projector-weight", type=float)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.root = args.root.resolve()
    if args.campaign in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.campaign):
        raise ValueError("Campaign must be a simple directory name.")
    if any(seed < 0 or seed >= 2 ** 32 for seed in [args.seed, *args.seeds]):
        raise ValueError("Seeds must be integers in [0, 2**32).")
    if args.calibration_batches < 1 or args.calibration_batch_size < 1:
        raise ValueError("Calibration needs a positive batch count/size.")
    if args.action in ("plan", "suite"):
        if args.action == "plan" and args.execute:
            raise ValueError("Use suite --execute to launch the plan.")
        if args.resume:
            raise ValueError("Resume one specific run using train/smoke/stage1.")
        return max(plan_one(args, variant, seed, "train")
                   for variant in (args.variants or MAIN) for seed in args.seeds)
    mode = {"train": "train", "smoke": "smoke", "calibrate": "calibration", "stage1": "stage1"}[args.action]
    variant = "full" if mode == "calibration" else args.variant
    return plan_one(args, variant, args.seed, mode)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        sys.exit(str(error))
