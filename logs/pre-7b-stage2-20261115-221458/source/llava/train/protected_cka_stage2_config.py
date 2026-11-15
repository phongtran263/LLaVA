"""Independent opt-in Stage 2 settings and leakage-safe instruction holdouts.

Stage 1 settings and split files are intentionally not used as Stage 2 defaults
or provenance. Only the pure path/serialization utilities are shared.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from llava.train.protected_cka_config import (
    _ImagePathResolver,
    _json_bytes,
    _persist_or_validate,
    _require_fresh_continuation_output,
)


STAGE2_DEFAULTS = {
    "start_phase_b": False,
    "source_stage2_checkpoint": None,
    "continuation": "protected",
    "reference_device": "same",
    "margin": 0.02,
    "kl_weight": 0.10,
    "ema_decay": 0.99,
    "proj_aux_ratio": 0.10,
    "llm_aux_ratio": 0.10,
    "lambda_max": 1.0,
    # These gates must be chosen explicitly for instruction data, not inherited
    # from Stage 1 caption thresholds.
    "ce_tolerance": None,
    "text_kl_budget": None,
    "drift_budget": None,
    "refine_lr": None,
    "warmup_steps": 100,
    "ce_gradient_calibration_batches": 8,
    "eval_interval": 100,
    "max_retries": 3,
    "kl_chunk_size": 128,
    "aux_backoff": 0.5,
    "eps": 1e-8,
    "min_phase_a_steps": 1500,
    "min_phase_b_steps": 350,
    "plateau_patience": 3,
    "plateau_min_delta": 0.001,
    "max_phase_a_steps": None,
    "max_phase_b_steps": None,
    "plateau_window": 3,
    "validation_fraction": 0.005,
    "validation_max_samples": 2048,
    "validation_seed": 42,
}

STAGE2_STATE_NAME = "protected_cka_stage2_state.pt"
STAGE2_SPLIT_DIR = "protected_cka_stage2_split"


def protected_cka_stage2_config_dict(training_args, data_args=None):
    result = {"protected_stage2": bool(getattr(training_args, "protected_stage2", False)),
              "cka_grad_scope": getattr(training_args, "cka_grad_scope", "existing")}
    result.update({"protected_stage2_" + name: getattr(
        training_args, "protected_stage2_" + name, default)
        for name, default in STAGE2_DEFAULTS.items()})
    if data_args is not None:
        for name in ("validation_data_path", "validation_image_folder"):
            key = "protected_stage2_" + name
            result[key] = getattr(data_args, key, None)
    return result


def _value(value):
    return getattr(value, "value", value)


def _resume_directory(training_args):
    requested = getattr(training_args, "resume_from_checkpoint", None)
    if requested and requested is not True:
        directory = Path(requested)
        if not (directory / STAGE2_STATE_NAME).is_file():
            raise ValueError("Protected CKA Stage 2 resume requires a full protected Stage 2 checkpoint; "
                             "use protected_stage2_source_stage2_checkpoint for a CE-only continuation.")
        return directory
    output = getattr(training_args, "output_dir", None)
    if output:
        candidates = [path for path in Path(output).glob("checkpoint-*")
                      if path.is_dir() and path.name.split("-")[-1].isdigit()]
        if candidates:
            directory = max(candidates, key=lambda path: int(path.name.split("-")[-1]))
            if not (directory / STAGE2_STATE_NAME).is_file():
                raise ValueError("Protected CKA Stage 2 cannot auto-resume a legacy/Stage 1 checkpoint; "
                                 "choose a fresh output directory.")
            return directory
    if requested is True:
        raise ValueError("No protected Stage 2 checkpoint is available to resume.")
    return None


def _source_manifest(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    for directory in (checkpoint, checkpoint.parent):
        manifest = directory / STAGE2_SPLIT_DIR / "split.json"
        if manifest.is_file():
            return manifest
    return None


def _validate_stage2_source(checkpoint):
    """Check checkpoint identity without loading potentially very large tensors."""
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "config.json"
    config = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError("Protected CKA Stage 2 source config.json must contain an object.")
    if config.get("tune_mm_mlp_adapter") or config.get("protected_cka_refinement"):
        raise ValueError("Protected CKA Stage 2 source is a Stage 1 projector-only checkpoint, not Stage 2 CE-only weights.")
    protected_source = (checkpoint / STAGE2_STATE_NAME).is_file()
    if protected_source:
        if not (checkpoint / "student_model.pt").is_file():
            raise ValueError("Protected CKA Stage 2 source is missing student_model.pt.")
        # The runtime checks the saved phase is A before accepting these tensors.
        return "protected"
    if config.get("cka_loss") or config.get("protected_stage2"):
        raise ValueError("Protected CKA Stage 2 continuation requires CE-only source weights; "
                         "source metadata enables CKA or lacks a verifiable protected Phase A state.")
    full_names = ("pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json",
                  "model.safetensors.index.json")
    if any((checkpoint / name).is_file() for name in full_names):
        return "full"
    if any((checkpoint / name).is_file() for name in ("adapter_model.bin", "adapter_model.safetensors")):
        if not (checkpoint / "non_lora_trainables.bin").is_file():
            raise ValueError("Protected CKA Stage 2 LoRA source is incomplete: non_lora_trainables.bin is missing, "
                             "so the trained projector/other non-LoRA modules cannot be restored. "
                             "Choose a complete Stage 2 export, not an adapter-only intermediate checkpoint.")
        return "lora"
    raise ValueError("Protected CKA Stage 2 source must contain full LLM weights or complete LoRA weights; "
                     "mm_projector.bin alone is only Stage 1 initialization.")


def validate_protected_cka_stage2_config(model_args, data_args, training_args):
    """Fail before loading models; a missing/false enable flag is a strict no-op."""
    if not getattr(training_args, "protected_stage2", False):
        return

    def reject(condition, message):
        if condition:
            raise ValueError("Protected CKA Stage 2: " + message)

    def option(name):
        return getattr(training_args, "protected_stage2_" + name, STAGE2_DEFAULTS[name])

    reject(getattr(training_args, "protected_cka_refinement", False), "Stage 1 and Stage 2 protection are mutually exclusive.")
    reject(getattr(model_args, "tune_mm_mlp_adapter", False),
           "Stage 2 must retain its full fine-tuning/LoRA recipe, not projector-only Stage 1 tuning.")
    reject(getattr(model_args, "freeze_backbone", False) and not getattr(training_args, "lora_enable", False),
           "full fine-tuning must not freeze the student LLM; use the existing LoRA recipe explicitly if desired.")
    reject(not getattr(model_args, "vision_tower", None), "a configured frozen --vision_tower is required.")
    for name in ("use_pcgrad", "vsp_gradient_diagnostics", "vsp_asymmetric_pcgrad",
                 "vsp_apply_to_projector_only", "vsp_norm_cap"):
        reject(bool(getattr(model_args, name, False)), "cannot combine gradient controllers with --" + name + ".")
    for name in ("cka_loss_start_ratio", "cka_loss_tau"):
        reject(float(getattr(model_args, name, 0.0)) != 0.0,
               name + " must be zero: Stage 2 plateau switching and target-margin CKA own this schedule/objective.")
    # Existing CKA weights/layers/token selection are deliberately allowed and
    # left untouched; they define the Stage 2 method to be protected.
    for name in ("deepspeed", "fsdp", "fp16", "fp16_full_eval", "torch_compile", "torchdynamo",
                 "use_ipex", "debug_compare_cka", "group_by_length", "ignore_data_skip",
                 "load_best_model_at_end", "auto_find_batch_size", "resume_model_only",
                 "eval_on_start", "use_liger_kernel"):
        reject(bool(getattr(training_args, name, False)), "unsupported backend/option --" + name + ".")
    reject(getattr(training_args, "half_precision_backend", None) == "apex", "Apex/scaled FP16 is unsupported.")
    reject(getattr(training_args, "bits", 16) != 16, "only unquantized full fine-tuning/LoRA is supported (--bits 16).")
    reject(int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(getattr(training_args, "world_size", 1)) != 1,
           "only one process is supported; DDP/distributed reduction is not implemented.")
    reject(int(getattr(training_args, "n_gpu", 0)) > 1, "expose a single CUDA device; DataParallel is unsupported.")
    device = getattr(training_args, "device", None)
    device_type = getattr(device, "type", str(device).split(":")[0] if device is not None else None)
    reject(device_type not in (None, "cpu", "cuda"), "only CPU/CUDA FP32 or BF16 is supported.")
    reject(int(getattr(training_args, "dataloader_num_workers", 0)) != 0,
           "dataloader_num_workers must be zero for exact rollback/replay.")
    checkpointing = getattr(training_args, "gradient_checkpointing_kwargs", None) or {}
    reject(getattr(training_args, "gradient_checkpointing", False) and bool(checkpointing.get("use_reentrant", False)),
           "gradient checkpointing must use use_reentrant=False for autograd.grad and reference switching.")
    accelerator = getattr(training_args, "accelerator_config", None)
    seedable = (accelerator.get("use_seedable_sampler", True) if isinstance(accelerator, dict)
                else getattr(accelerator, "use_seedable_sampler", True))
    reject(not seedable, "accelerator_config.use_seedable_sampler must be True for replay.")
    reject(_value(getattr(training_args, "optim", "adamw_torch")) not in ("adamw_torch", "adamw_torch_fused"),
           "only adamw_torch/adamw_torch_fused is supported for controlled Phase B optimizer reset.")
    learning_rate = getattr(training_args, "learning_rate", 2e-5)
    reject(learning_rate is None or not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0,
           "learning_rate must be finite and positive to preserve projector/LLM LR ratios at Phase B reset.")
    reject(_value(getattr(training_args, "lr_scheduler_type", None)) == "reduce_lr_on_plateau",
           "reduce_lr_on_plateau is unsupported because validation belongs to the protected runtime.")
    reject(float(getattr(training_args, "label_smoothing_factor", 0.0)) != 0.0,
           "label_smoothing_factor must be zero for instruction CE accounting.")
    for name in ("save_strategy", "eval_strategy", "evaluation_strategy"):
        value = getattr(training_args, name, None)
        reject(value is not None and _value(value) != "no", name + " must be no; the runtime owns safe checkpoints/gates.")
    for name in ("stop_after_step_ratio", "save_at_step_ratio"):
        reject(getattr(training_args, name, None) is not None, "cannot combine with " + name + ".")
    scope = getattr(training_args, "cka_grad_scope", "existing")
    reject(scope not in ("existing", "projector_only"), "cka_grad_scope must be existing or projector_only.")
    reject(scope == "projector_only" and getattr(training_args, "freeze_mm_mlp_adapter", False),
           "projector_only auxiliary routing requires a trainable projector.")
    reject(option("continuation") not in ("protected", "ce_kl", "ce_only"),
           "continuation must be protected, ce_kl, or ce_only.")
    reject(option("reference_device") not in ("same", "cpu"), "reference_device must be same or cpu.")
    for name in ("margin", "kl_weight", "proj_aux_ratio", "llm_aux_ratio", "lambda_max", "plateau_min_delta"):
        value = float(option(name))
        reject(not math.isfinite(value) or value < 0, name + " must be finite and nonnegative.")
    for name in ("ce_tolerance", "text_kl_budget"):
        value = option(name)
        reject(value is None or not math.isfinite(float(value)) or float(value) < 0,
               "protected_stage2_" + name + " must be set explicitly to a finite nonnegative Stage 2 threshold.")
    if option("drift_budget") is not None:
        reject(not math.isfinite(float(option("drift_budget"))) or float(option("drift_budget")) < 0,
               "drift_budget must be finite/nonnegative when explicitly enabling the optional drift gate.")
    for name in ("refine_lr", "eps"):
        value = option(name)
        reject(value is None or not math.isfinite(float(value)) or float(value) <= 0,
               name + " is required and must be finite and positive.")
    reject(not 0 <= float(option("ema_decay")) < 1, "ema_decay must be in [0, 1).")
    reject(not 0 < float(option("aux_backoff")) < 1, "aux_backoff must be in (0, 1).")
    reject(not 0 < float(option("validation_fraction")) < 1, "validation_fraction must be in (0, 1).")
    for name in ("ce_gradient_calibration_batches", "eval_interval", "kl_chunk_size", "plateau_patience",
                 "plateau_window", "validation_max_samples", "max_phase_b_steps"):
        value = option(name)
        reject(isinstance(value, bool) or not isinstance(value, int) or value <= 0,
               name + " must be an explicitly configured positive integer.")
    for name in ("warmup_steps", "max_retries", "min_phase_a_steps", "min_phase_b_steps"):
        value = option(name)
        reject(isinstance(value, bool) or not isinstance(value, int) or value < 0, name + " must be a nonnegative integer.")
    maximum = getattr(training_args, "max_steps", -1)
    reject(not isinstance(maximum, int) or maximum <= 0, "max_steps must be a positive total Stage 2 budget.")
    reject(option("max_phase_b_steps") > maximum, "max_phase_b_steps cannot exceed total max_steps.")
    if not option("start_phase_b"):
        reject(option("min_phase_b_steps") > option("max_phase_b_steps"),
               "min_phase_b_steps cannot exceed max_phase_b_steps.")
    if option("max_phase_a_steps") is None:
        training_args.protected_stage2_max_phase_a_steps = maximum
    phase_a_limit = training_args.protected_stage2_max_phase_a_steps
    reject(not isinstance(phase_a_limit, int) or isinstance(phase_a_limit, bool) or not 0 < phase_a_limit <= maximum,
           "max_phase_a_steps must be positive and no greater than total max_steps.")
    if not option("start_phase_b"):
        reject(option("min_phase_a_steps") > phase_a_limit, "min_phase_a_steps cannot exceed max_phase_a_steps.")
        reject(option("min_phase_a_steps") + option("min_phase_b_steps") > maximum,
               "min_phase_a_steps + min_phase_b_steps cannot exceed max_steps.")

    source = option("source_stage2_checkpoint")
    if source:
        _require_fresh_continuation_output(training_args, source, "Protected CKA Stage 2")
    resume = _resume_directory(training_args)
    validation_path = getattr(data_args, "protected_stage2_validation_data_path", None)
    validation_folder = getattr(data_args, "protected_stage2_validation_image_folder", None)
    reject(bool(validation_folder) and not validation_path, "validation_image_folder requires validation_data_path.")
    if validation_path:
        reject(not Path(validation_path).is_file(), "validation_data_path must be an existing instruction JSON file.")
    source_manifest = None
    if resume is not None:
        source_manifest = _source_manifest(resume)
        reject(source_manifest is None, "full resume requires its original protected_cka_stage2_split manifest.")
        model_args.pretrain_mm_mlp_adapter = None
    elif source:
        reject(not option("start_phase_b"), "source_stage2_checkpoint requires start_phase_b=True for an explicit continuation.")
        reject(not Path(source).is_dir(), "source_stage2_checkpoint must name an existing Stage 2 checkpoint directory.")
        source_kind = _validate_stage2_source(source)
        reject(source_kind == "lora" and not getattr(training_args, "lora_enable", False),
               "a LoRA source requires lora_enable=True with its original adapter recipe; full fine-tuning is not silently substituted.")
        source_manifest = _source_manifest(source)
        if not validation_path:
            reject(source_manifest is None,
                   "a trained Stage 2 checkpoint may have seen a new auto-holdout. Supply its original "
                   "protected_cka_stage2_split manifest or genuinely held-out instruction validation JSON.")
        model_args.pretrain_mm_mlp_adapter = None
    else:
        reject(option("start_phase_b"), "direct Phase B requires source_stage2_checkpoint, not a Stage 1 teacher.")
        adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        reject(not adapter or not Path(adapter).is_file(),
               "fresh Stage 2 requires the configured Stage 1 pretrain_mm_mlp_adapter file.")
        # A Stage 1 caption adapter is valid initialization, not Stage 2 split
        # provenance and never the teacher for Stage 2 refinement.
    data_args._protected_cka_stage2_source_manifest = str(source_manifest) if source_manifest else None
    data_args._protected_cka_stage2_preprocessing = {
        "conversation_version": getattr(model_args, "version", "auto"),
        "model_max_length": getattr(training_args, "model_max_length", None),
        "mm_use_im_start_end": getattr(model_args, "mm_use_im_start_end", False),
        "mm_use_im_patch_token": getattr(model_args, "mm_use_im_patch_token", True),
    }


def _text_content_fingerprint(record):
    # Duplicate assistant supervision must stay together even when dataset IDs
    # differ. Ancillary source metadata does not change conversation content.
    content = ({"conversations": record["conversations"]} if "conversations" in record
               else {key: value for key, value in record.items() if key not in ("id", "image")})
    return hashlib.sha256(_json_bytes(content)).hexdigest()


class _InstructionIdentities:
    def __init__(self):
        self.images = _ImagePathResolver()
        self.texts = {}

    def image(self, record, folder):
        return self.images(record, folder) if "image" in record else None

    def text(self, record):
        key = id(record)
        if key not in self.texts:
            self.texts[key] = _text_content_fingerprint(record)
        return self.texts[key]

    def keys(self, record, folder):
        image = self.image(record, folder)
        keys = [("image", image)] if image is not None else [("text", self.text(record))]
        if record.get("id") is not None:
            keys.append(("id", str(record["id"])))
        return keys


def _fingerprint(records, folder, data_path, identities):
    digest = hashlib.sha256(_json_bytes({"records": records, "image_paths": [
        identities.image(record, folder) for record in records]}))
    if data_path:
        path = Path(data_path).resolve()
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _groups(records, folder, identities):
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, record in enumerate(records):
        for key in identities.keys(record, folder):
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
    groups = {}
    for index in range(len(records)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def _assert_disjoint(train, validation, train_folder, validation_folder, identities):
    keys = {key for record in train for key in identities.keys(record, train_folder)}
    overlap = keys.intersection(key for record in validation for key in identities.keys(record, validation_folder))
    if overlap:
        kind, key = sorted(overlap)[0]
        raise ValueError(f"Protected CKA Stage 2 training/validation overlap by {kind}: {key}")


def _manifest(records, indices, folder, identities):
    return [{"index": index, "id": records[index].get("id"),
             "image": identities.image(records[index], folder),
             "text_fingerprint": identities.text(records[index]) if "image" not in records[index] else None}
            for index in indices]


def prepare_protected_cka_stage2_datasets(train_dataset, tokenizer, data_args, training_args, dataset_cls):
    """Persist an instruction holdout, grouping image aliases, IDs and text copies."""
    if not getattr(training_args, "protected_stage2", False):
        return train_dataset, None
    started = last_log = time.monotonic()
    records = train_dataset.list_data_dict
    if not records:
        raise ValueError("Protected CKA Stage 2 needs nonempty instruction training data.")
    identities = _InstructionIdentities()
    folder = data_args.image_folder
    print(f"[Protected CKA Stage 2 split] Indexing {len(records):,} instruction samples...", flush=True)
    for index, record in enumerate(records, 1):
        identities.keys(record, folder)
        now = time.monotonic()
        if now - last_log >= 10:
            print(f"[Protected CKA Stage 2 split] Indexed {index:,}/{len(records):,} ({now - started:.1f}s)", flush=True)
            last_log = now
    metadata = {"version": 1, "stage": 2,
                "train_fingerprint": _fingerprint(records, folder, data_args.data_path, identities),
                "preprocessing": dict(getattr(data_args, "_protected_cka_stage2_preprocessing", {}),
                                      image_aspect_ratio=getattr(data_args, "image_aspect_ratio", "square"))}
    validation_args = copy.deepcopy(data_args)
    validation_args.train_data_fraction = 1.0
    validation_path = getattr(data_args, "protected_stage2_validation_data_path", None)
    if validation_path:
        validation_args.data_path = validation_path
        validation_args.image_folder = getattr(data_args, "protected_stage2_validation_image_folder", None) or folder
        validation = dataset_cls(data_path=validation_path, tokenizer=tokenizer, data_args=validation_args)
        validation_records = validation.list_data_dict
        train_indices = list(range(len(records)))
        validation_indices = list(range(len(validation_records)))
        metadata.update(mode="explicit", validation_fingerprint=_fingerprint(
            validation_records, validation_args.image_folder, validation_path, identities))
    else:
        groups = _groups(records, folder, identities)
        if len(groups) < 2:
            raise ValueError("Protected CKA Stage 2 auto split needs at least two independent image/text/ID groups.")
        seed = int(getattr(training_args, "protected_stage2_validation_seed", 42))
        fraction = float(getattr(training_args, "protected_stage2_validation_fraction", 0.005))
        cap = int(getattr(training_args, "protected_stage2_validation_max_samples", 2048))
        target = min(cap, max(1, math.ceil(len(records) * fraction)))
        random.Random(seed).shuffle(groups)
        selected = []
        for group in groups:
            if len(selected) >= target:
                break
            if len(selected) + len(group) <= cap and len(selected) + len(group) < len(records):
                selected.extend(group)
        if not selected:
            raise ValueError("Cannot create a nonempty Stage 2 holdout within validation_max_samples without splitting a group.")
        validation_indices = sorted(selected)
        selected = set(selected)
        train_indices = [index for index in range(len(records)) if index not in selected]
        validation_records = records
        validation = copy.copy(train_dataset)
        validation.data_args = validation_args
        metadata.update(mode="auto", seed=seed, fraction=fraction, max_samples=cap)
    if not validation_indices:
        raise ValueError("Protected CKA Stage 2 requires a nonempty held-out instruction validation dataset.")
    train_rows = [records[index] for index in train_indices]
    validation_rows = [validation_records[index] for index in validation_indices]
    _assert_disjoint(train_rows, validation_rows, folder, validation_args.image_folder, identities)
    metadata.update(train_indices=train_indices, validation_indices=validation_indices)
    source = getattr(data_args, "_protected_cka_stage2_source_manifest", None)
    if source:
        with open(source, encoding="utf-8") as handle:
            if json.load(handle) != metadata:
                raise ValueError("Protected CKA Stage 2 source split/fingerprint mismatch; "
                                 "never redefine a trained checkpoint's instruction holdout.")
    destination = Path(training_args.output_dir) / STAGE2_SPLIT_DIR
    _persist_or_validate(destination / "split.json", metadata)
    _persist_or_validate(destination / "train_manifest.json", _manifest(records, train_indices, folder, identities))
    _persist_or_validate(destination / "validation_manifest.json", _manifest(
        validation_records, validation_indices, validation_args.image_folder, identities))
    training_args.protected_stage2_split_fingerprint = hashlib.sha256(_json_bytes(metadata)).hexdigest()
    training_args.protected_stage2_split_manifest = str(destination / "split.json")
    train_dataset.list_data_dict = train_rows
    validation.list_data_dict = validation_rows
    print(f"[Protected CKA Stage 2 split] Ready: {len(train_rows):,} train, "
          f"{len(validation_rows):,} held-out samples ({time.monotonic() - started:.1f}s)", flush=True)
    return train_dataset, validation
