"""Opt-in Protected CKA configuration and deterministic held-out data handling.

This module deliberately has no model/torch imports: invalid jobs fail before a
model download and split invariants can be tested without GPUs.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import tempfile
import time


CONFIG_DEFAULTS = {
    "refinement": False,
    "start_phase_b": False,
    "source_stage1_checkpoint": None,
    "continuation": "protected",
    "margin": 0.02,
    "kl_weight": 0.10,
    "ema_decay": 0.99,
    "aux_ratio": 0.10,
    "lambda_max": 1.0,
    "ce_tolerance": 0.005,
    "text_kl_budget": 0.01,
    "drift_budget": 0.01,
    "refine_lr": None,
    "warmup_steps": 100,
    "ce_gradient_calibration_batches": 8,
    "eval_interval": 50,
    "max_retries": 3,
    "kl_chunk_size": 128,
    "aux_backoff": 0.5,
    "eps": 1e-8,
    "min_phase_a_steps": 1500,
    "min_phase_b_steps": 350,
    "plateau_patience": 3,
    "plateau_min_delta": 0.001,
    "max_phase_a_steps": None,
    "plateau_window": 3,
    "validation_fraction": 0.005,
    "validation_max_samples": 2048,
    "validation_seed": 42,
}


def protected_cka_config_dict(training_args, data_args=None):
    """Return JSON-safe feature settings, using the public CLI names."""
    result = {
        "protected_cka_" + name: getattr(training_args, "protected_cka_" + name, default)
        for name, default in CONFIG_DEFAULTS.items()
    }
    if data_args is not None:
        for name in ("validation_data_path", "validation_image_folder"):
            key = "protected_cka_" + name
            result[key] = getattr(data_args, key, None)
    return result


def _strategy(value):
    return getattr(value, "value", value)


def _protected_resume_directory(training_args):
    requested = getattr(training_args, "resume_from_checkpoint", None)
    if requested and requested is not True:
        directory = Path(requested)
        if not (directory / "protected_cka_state.pt").is_file():
            raise ValueError("Protected CKA resume requires a full protected checkpoint; use pretrain_mm_mlp_adapter for CE-only weights.")
        return directory
    output_dir = getattr(training_args, "output_dir", None)
    if output_dir:
        candidates = [path for path in Path(output_dir).glob("checkpoint-*")
                      if path.is_dir() and path.name.split("-")[-1].isdigit()]
        if candidates:
            latest = max(candidates, key=lambda path: int(path.name.split("-")[-1]))
            if not (latest / "protected_cka_state.pt").is_file():
                raise ValueError("Protected CKA cannot auto-resume a legacy checkpoint; choose a fresh output directory.")
            return latest
    if requested is True:
        raise ValueError("Protected CKA found no full protected checkpoint to resume.")
    return None


def _source_split_manifest(adapter):
    adapter = Path(adapter).resolve()
    for directory in (adapter.parent, adapter.parent.parent):
        manifest = directory / "protected_cka_split" / "split.json"
        if manifest.is_file():
            return manifest
    return None


def _require_fresh_continuation_output(training_args, source, label):
    """New-run initialization is never an implicit resume or source overwrite."""
    if getattr(training_args, "resume_from_checkpoint", None):
        raise ValueError(label + ": source checkpoint and resume_from_checkpoint are mutually exclusive.")
    output = getattr(training_args, "output_dir", None)
    if output:
        destination = Path(output)
        if destination.resolve() == Path(source).resolve() or destination.is_symlink() or (
                destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
            raise ValueError(label + ": continuation requires a fresh, empty output directory; "
                             "use exact resume for an existing run.")


def validate_protected_cka_config(model_args, data_args, training_args):
    """Validate the supported first-version contract; disabled is a strict no-op."""
    if not getattr(training_args, "protected_cka_refinement", False):
        return

    def reject(condition, message):
        if condition:
            raise ValueError("Protected CKA: " + message)

    reject(not getattr(model_args, "tune_mm_mlp_adapter", False),
           "only Stage 1 (--tune_mm_mlp_adapter True) is supported; Stage 2 is unchanged.")
    reject(not getattr(model_args, "vision_tower", None), "--vision_tower is required.")
    reject(getattr(training_args, "freeze_mm_mlp_adapter", False), "the projector must be trainable.")
    reject(getattr(model_args, "mm_use_im_start_end", False),
           "--mm_use_im_start_end must be False (only projector parameters may train).")
    for name in ("cka_loss", "use_pcgrad", "vsp_gradient_diagnostics", "vsp_asymmetric_pcgrad",
                 "vsp_apply_to_projector_only", "vsp_norm_cap"):
        reject(bool(getattr(model_args, name, False)), "cannot combine with legacy --" + name + ".")
    for name in ("deepspeed", "fsdp", "fp16", "fp16_full_eval", "torch_compile", "torchdynamo",
                 "gradient_checkpointing", "lora_enable", "use_ipex", "debug_compare_cka",
                 "group_by_modality_length", "group_by_length", "ignore_data_skip",
                 "load_best_model_at_end", "auto_find_batch_size", "resume_model_only",
                 "eval_on_start", "use_liger_kernel"):
        reject(bool(getattr(training_args, name, False)), "unsupported backend/option --" + name + ".")
    accelerator_config = getattr(training_args, "accelerator_config", None)
    seedable = (accelerator_config.get("use_seedable_sampler", True) if isinstance(accelerator_config, dict)
                else getattr(accelerator_config, "use_seedable_sampler", True))
    reject(not seedable, "accelerator_config.use_seedable_sampler must be True for exact replay.")
    reject(_strategy(getattr(training_args, "optim", "adamw_torch")) not in ("adamw_torch", "adamw_torch_fused"),
           "only optim=adamw_torch or adamw_torch_fused is supported.")
    reject(_strategy(getattr(training_args, "lr_scheduler_type", None)) == "reduce_lr_on_plateau",
           "reduce_lr_on_plateau is unsupported; the protected gate owns validation.")
    reject(float(getattr(training_args, "label_smoothing_factor", 0.0)) != 0.0,
           "label_smoothing_factor must be zero for the protected caption CE objective.")
    reject(getattr(training_args, "half_precision_backend", None) == "apex", "Apex is unsupported.")
    reject(getattr(training_args, "bits", 16) != 16, "quantized training is unsupported; --bits must be 16.")
    reject(int(os.environ.get("WORLD_SIZE", "1")) != 1 or
           int(getattr(training_args, "world_size", 1)) != 1,
           "only a single process is supported (no DDP).")
    reject(int(getattr(training_args, "n_gpu", 0)) > 1,
           "expose only one CUDA device; DataParallel is unsupported.")
    device = getattr(training_args, "device", None)
    device_type = getattr(device, "type", str(device).split(":")[0] if device is not None else None)
    reject(device_type not in (None, "cpu", "cuda"), "only CPU/CUDA FP32 or BF16 is supported.")
    reject(int(getattr(training_args, "dataloader_num_workers", 0)) != 0,
           "--dataloader_num_workers must be 0 for exact rollback/resume.")
    reject(int(getattr(training_args, "max_steps", -1)) <= 0, "--max_steps must be positive.")
    for name in ("save_strategy", "eval_strategy", "evaluation_strategy"):
        value = getattr(training_args, name, None)
        reject(value is not None and _strategy(value) != "no",
               "--" + name + " must be no; the protected runtime owns validation/checkpoints.")
    for name in ("stop_after_step_ratio", "save_at_step_ratio"):
        reject(getattr(training_args, name, None) is not None, "cannot combine with --" + name + ".")

    config = protected_cka_config_dict(training_args)
    reject(config["protected_cka_continuation"] not in ("protected", "ce_kl", "ce_only"),
           "continuation must be protected, ce_kl, or ce_only.")
    for name in ("margin", "kl_weight", "aux_ratio", "lambda_max", "ce_tolerance",
                 "text_kl_budget", "drift_budget", "plateau_min_delta"):
        value = float(config["protected_cka_" + name])
        reject(not math.isfinite(value) or value < 0, "--protected_cka_" + name + " must be finite and nonnegative.")
    for name in ("refine_lr", "eps"):
        value = config["protected_cka_" + name]
        reject(value is None or not math.isfinite(float(value)) or float(value) <= 0,
               "--protected_cka_" + name + " is required and must be finite and positive.")
    reject(not 0 <= float(config["protected_cka_ema_decay"]) < 1,
           "--protected_cka_ema_decay must be in [0, 1).")
    reject(not 0 < float(config["protected_cka_aux_backoff"]) < 1,
           "--protected_cka_aux_backoff must be in (0, 1).")
    reject(not 0 < float(config["protected_cka_validation_fraction"]) < 1,
           "--protected_cka_validation_fraction must be in (0, 1).")
    for name in ("ce_gradient_calibration_batches", "eval_interval", "kl_chunk_size",
                 "plateau_patience", "plateau_window", "validation_max_samples"):
        value = config["protected_cka_" + name]
        reject(isinstance(value, bool) or not isinstance(value, int) or value <= 0,
               "--protected_cka_" + name + " must be a positive integer.")
    for name in ("warmup_steps", "max_retries", "min_phase_a_steps", "min_phase_b_steps"):
        value = config["protected_cka_" + name]
        reject(isinstance(value, bool) or not isinstance(value, int) or value < 0,
               "--protected_cka_" + name + " must be a nonnegative integer.")
    if config["protected_cka_max_phase_a_steps"] is None:
        training_args.protected_cka_max_phase_a_steps = int(training_args.max_steps)
    maximum = training_args.protected_cka_max_phase_a_steps
    reject(isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0,
           "--protected_cka_max_phase_a_steps must be a positive integer.")
    reject(maximum > training_args.max_steps, "max_phase_a_steps cannot exceed max_steps.")
    source = config["protected_cka_source_stage1_checkpoint"]
    if source:
        _require_fresh_continuation_output(training_args, source, "Protected CKA")
    resume_directory = _protected_resume_directory(training_args)
    if not config["protected_cka_start_phase_b"]:
        reject(config["protected_cka_min_phase_a_steps"] > maximum,
               "min_phase_a_steps cannot exceed max_phase_a_steps.")
        reject(config["protected_cka_min_phase_a_steps"] + config["protected_cka_min_phase_b_steps"] > training_args.max_steps,
               "min_phase_a_steps + min_phase_b_steps cannot exceed max_steps.")
    elif resume_directory is None and not source:
        adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        reject(not adapter or not Path(adapter).is_file(),
               "--protected_cka_start_phase_b True requires an existing --pretrain_mm_mlp_adapter file.")
    validation_path = getattr(data_args, "protected_cka_validation_data_path", None)
    reject(bool(getattr(data_args, "protected_cka_validation_image_folder", None)) and not validation_path,
           "validation_image_folder requires validation_data_path.")
    if validation_path:
        reject(not Path(validation_path).is_file(), "validation_data_path must name an existing JSON file.")
    source_manifest = None
    if resume_directory is not None:
        source_manifest = resume_directory / "protected_cka_split" / "split.json"
        if not source_manifest.is_file():
            source_manifest = resume_directory.parent / "protected_cka_split" / "split.json"
        reject(not source_manifest.is_file(), "full resume requires the original protected_cka_split manifest.")
        # The full protected checkpoint restores its own projector/reference.
        # Do not depend on a possibly moved initialization-only CE adapter.
        model_args.pretrain_mm_mlp_adapter = None
    elif source:
        reject(not config["protected_cka_start_phase_b"],
               "source_stage1_checkpoint requires start_phase_b=True for an explicit new continuation.")
        checkpoint = Path(source)
        reject(not checkpoint.is_dir(), "source_stage1_checkpoint must name an existing checkpoint directory.")
        reject(not (checkpoint / "protected_cka_state.pt").is_file() or
               not (checkpoint / "mm_projector.bin").is_file(),
               "source_stage1_checkpoint requires verified Stage 1 CE-only state and mm_projector.bin; "
               "a standalone adapter cannot establish the original CE-only phase/provenance.")
        source_manifest = _source_split_manifest(checkpoint / "mm_projector.bin")
        reject(source_manifest is None, "Stage 1 continuation requires the original protected_cka_split manifest.")
        # Runtime validates the original frozen backbone/vision/tokenizer and
        # Phase A identity, then loads ONLY student weights into a fresh run.
        model_args.pretrain_mm_mlp_adapter = None
    elif getattr(model_args, "pretrain_mm_mlp_adapter", None) and not validation_path:
        source_manifest = _source_split_manifest(model_args.pretrain_mm_mlp_adapter)
        reject(source_manifest is None,
               "an initialized CE adapter may already have seen an automatic holdout. Provide genuinely held-out "
               "protected_cka_validation_data_path or the adapter's original protected_cka_split manifest; "
               "otherwise rerun Phase A with the automatic split.")
    data_args._protected_cka_source_split_manifest = str(source_manifest) if source_manifest else None
    data_args._protected_cka_preprocessing = {
        "conversation_version": getattr(model_args, "version", "plain"),
        "model_max_length": getattr(training_args, "model_max_length", None),
        "mm_use_im_start_end": getattr(model_args, "mm_use_im_start_end", False),
        "mm_use_im_patch_token": getattr(model_args, "mm_use_im_patch_token", True),
    }


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _image_path(record, image_folder):
    image = record.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError("Protected CKA requires a nonempty single image path for every Stage 1 sample.")
    return Path(image_folder or ".") / image


def _canonical_image(record, image_folder):
    return str(_image_path(record, image_folder).resolve())


class _ImagePathResolver:
    """Reuse canonical paths within one split preparation, never across runs.

    On shared storage, walking every parent directory for every caption is
    expensive. Resolve each parent once and inspect the leaf without following
    symlinks. Unusual paths still use Path.resolve's original semantics.
    """

    def __init__(self):
        self.paths = {}
        self.parents = {}

    def __call__(self, record, image_folder):
        path = _image_path(record, image_folder)
        key = str(path)
        if key not in self.paths:
            if not path.name or ".." in path.parts:
                resolved = str(path.resolve())
            else:
                parent_key = str(path.parent)
                if parent_key not in self.parents:
                    self.parents[parent_key] = path.parent.resolve()
                candidate = self.parents[parent_key] / path.name
                try:
                    is_link = stat.S_ISLNK(os.lstat(candidate).st_mode)
                except OSError:
                    # Missing files, inaccessible paths, etc. retain the
                    # non-strict/error behavior of the original resolver.
                    resolved = str(path.resolve())
                else:
                    resolved = str(path.resolve()) if is_link else str(candidate)
            self.paths[key] = resolved
        return self.paths[key]


def _identity(record, image_folder, resolve_image=None):
    keys = [("image", (resolve_image or _canonical_image)(record, image_folder))]
    if record.get("id") is not None:
        keys.append(("id", str(record["id"])))
    return keys


def _fingerprint(records, image_folder, data_path, resolve_image=None):
    digest = hashlib.sha256()
    digest.update(_json_bytes({"records": records, "image_paths": [
        (resolve_image or _canonical_image)(record, image_folder) for record in records]}))
    if data_path:
        path = Path(data_path).resolve()
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _groups(records, image_folder, resolve_image=None):
    """Union image and ID identities, including transitive duplicate captions."""
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, record in enumerate(records):
        for key in _identity(record, image_folder, resolve_image):
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
    grouped = {}
    for index in range(len(records)):
        grouped.setdefault(root(index), []).append(index)
    return list(grouped.values())


def _assert_disjoint(train_records, validation_records, train_folder, validation_folder, resolve_image=None):
    identities = {key for record in train_records for key in _identity(record, train_folder, resolve_image)}
    overlap = identities.intersection(
        key for record in validation_records for key in _identity(record, validation_folder, resolve_image))
    if overlap:
        kind, value = sorted(overlap)[0]
        raise ValueError(f"Protected CKA training/validation overlap by {kind}: {value}")


def _manifest(records, indices, image_folder, resolve_image=None):
    return [{"index": index, "id": records[index].get("id"),
             "image": (resolve_image or _canonical_image)(records[index], image_folder)} for index in indices]


def _prime_image_paths(records, image_folder, resolve_image, label):
    started = last_log = time.monotonic()
    print(f"[Protected CKA split] Resolving {label} image paths: 0/{len(records):,}", flush=True)
    for count, record in enumerate(records, 1):
        resolve_image(record, image_folder)
        now = time.monotonic()
        if now - last_log >= 10.0 or count == len(records):
            print(f"[Protected CKA split] Resolving {label} image paths: "
                  f"{count:,}/{len(records):,} ({now - started:.1f}s)", flush=True)
            last_log = now


def _persist_or_validate(path, value):
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != value:
            raise ValueError(f"Protected CKA split manifest/fingerprint changed: {path}; use a new output directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_protected_cka_datasets(train_dataset, tokenizer, data_args, training_args, dataset_cls):
    """Create a fixed disjoint validation dataset before the first training step.

    Image groups (and duplicate IDs) are indivisible. The requested fraction is
    a target; a group may exceed it, but the hard maximum is never exceeded.
    """
    if not getattr(training_args, "protected_cka_refinement", False):
        return train_dataset, None
    records = train_dataset.list_data_dict
    if not records:
        raise ValueError("Protected CKA requires a nonempty training dataset.")
    started = time.monotonic()
    resolve_image = _ImagePathResolver()
    folder = data_args.image_folder
    validation_args = copy.deepcopy(data_args)
    validation_args.train_data_fraction = 1.0
    validation_path = getattr(data_args, "protected_cka_validation_data_path", None)
    _prime_image_paths(records, folder, resolve_image, "training")
    print("[Protected CKA split] Fingerprinting training data...", flush=True)
    metadata = {"version": 1, "train_fingerprint": _fingerprint(
                    records, folder, data_args.data_path, resolve_image),
                "preprocessing": dict(getattr(data_args, "_protected_cka_preprocessing", {}),
                                      image_aspect_ratio=getattr(data_args, "image_aspect_ratio", "square"))}
    if validation_path:
        validation_args.data_path = validation_path
        validation_args.image_folder = getattr(data_args, "protected_cka_validation_image_folder", None) or folder
        validation_dataset = dataset_cls(data_path=validation_path, tokenizer=tokenizer, data_args=validation_args)
        validation_records = validation_dataset.list_data_dict
        train_indices = list(range(len(records)))
        validation_indices = list(range(len(validation_records)))
        _prime_image_paths(validation_records, validation_args.image_folder, resolve_image, "validation")
        metadata.update(mode="explicit", validation_fingerprint=_fingerprint(
            validation_records, validation_args.image_folder, validation_path, resolve_image))
    else:
        print("[Protected CKA split] Grouping duplicate images/IDs...", flush=True)
        groups = _groups(records, folder, resolve_image)
        if len(groups) < 2:
            raise ValueError("Protected CKA auto split needs at least two independent image/ID groups.")
        seed = int(getattr(training_args, "protected_cka_validation_seed", 42))
        fraction = float(getattr(training_args, "protected_cka_validation_fraction", 0.005))
        maximum = int(getattr(training_args, "protected_cka_validation_max_samples", 2048))
        target = min(maximum, max(1, math.ceil(len(records) * fraction)))
        random.Random(seed).shuffle(groups)
        selected = []
        for group in groups:
            if len(selected) >= target:
                break
            if len(selected) + len(group) <= maximum and len(selected) + len(group) < len(records):
                selected.extend(group)
        if not selected:
            raise ValueError("Protected CKA cannot form a nonempty disjoint split within validation_max_samples.")
        validation_indices = sorted(selected)
        selected = set(selected)
        train_indices = [index for index in range(len(records)) if index not in selected]
        validation_dataset = copy.copy(train_dataset)
        validation_dataset.data_args = validation_args
        validation_records = records
        metadata.update(mode="auto", seed=seed, fraction=fraction, max_samples=maximum)
    if not validation_indices:
        raise ValueError("Protected CKA requires a nonempty held-out validation dataset.")
    train_records = [records[index] for index in train_indices]
    held_out_records = [validation_records[index] for index in validation_indices]
    print("[Protected CKA split] Checking train/validation separation...", flush=True)
    _assert_disjoint(train_records, held_out_records, folder, validation_args.image_folder, resolve_image)
    metadata.update(train_indices=train_indices, validation_indices=validation_indices)
    source_manifest = getattr(data_args, "_protected_cka_source_split_manifest", None)
    if source_manifest:
        with open(source_manifest, encoding="utf-8") as handle:
            source_metadata = json.load(handle)
        if source_metadata != metadata:
            raise ValueError("Protected CKA source adapter/checkpoint split does not match this dataset, preprocessing, "
                             "or split configuration; do not redefine previously held-out samples.")
    split_dir = Path(training_args.output_dir) / "protected_cka_split"
    print(f"[Protected CKA split] Writing/verifying manifests: {split_dir}", flush=True)
    _persist_or_validate(split_dir / "split.json", metadata)
    _persist_or_validate(split_dir / "train_manifest.json", _manifest(records, train_indices, folder, resolve_image))
    _persist_or_validate(split_dir / "validation_manifest.json", _manifest(
        validation_records, validation_indices, validation_args.image_folder, resolve_image))
    train_dataset.list_data_dict = train_records
    validation_dataset.list_data_dict = held_out_records
    training_args.protected_cka_split_fingerprint = hashlib.sha256(_json_bytes(metadata)).hexdigest()
    training_args.protected_cka_split_manifest = str(split_dir / "split.json")
    print(f"[Protected CKA split] Ready: {len(train_records):,} train, {len(held_out_records):,} validation "
          f"samples ({time.monotonic() - started:.1f}s)", flush=True)
    return train_dataset, validation_dataset
