#!/usr/bin/env python3
"""Read-only completion/fairness audit of the fixed Qwen3B two-stage protocol.

Use only with trusted local checkpoints: torch metadata is a pickle. Tensor
storages are CPU-mapped, never compared or scanned. Export checks inspect file
containers/headers, not numerical correctness or equivalence to checkpoints.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import re
import struct
import zipfile


DEFAULT_STAGE1 = "checkpoints/pretrain-protected-cka/qwen2.5-3b/continuation-350"
DEFAULT_STAGE2 = "checkpoints/finetune-protected-cka/qwen2.5-3b/paired-protected-cka"
MODES = ("ce_only", "protected")


def _json(path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _file(path):
    _require(path.is_file() and path.stat().st_size > 0, f"Missing/empty file: {path}")
    with path.open("rb") as handle:
        _require(bool(handle.read(1)), f"Unreadable/empty file: {path}")


def _torch_header(path):
    """Read ZIP directory only; never decompress tensor storage entries."""
    _file(path)
    with zipfile.ZipFile(path) as archive:
        _require(any(name.endswith("/data.pkl") for name in archive.namelist()),
                 f"Not a readable modern PyTorch archive: {path}")
    return {"file": str(path), "bytes": path.stat().st_size, "checked": "zip_directory_only"}


def _safetensors_header(path):
    _file(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        _require(len(length_bytes) == 8, f"Truncated safetensors header: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        _require(0 < length <= min(32 * 1024 * 1024, size - 8), f"Invalid safetensors header size: {path}")
        header = json.loads(handle.read(length))
    _require(isinstance(header, dict), f"Invalid safetensors header: {path}")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    _require(bool(tensors), f"No tensor entries in safetensors export: {path}")
    for key, value in tensors.items():
        _require(isinstance(value, dict), f"Invalid tensor header {key}: {path}")
        offsets = value.get("data_offsets", [])
        _require(isinstance(offsets, list) and len(offsets) == 2
                 and all(isinstance(x, int) for x in offsets)
                 and 0 <= offsets[0] <= offsets[1] <= size - 8 - length,
                 f"Missing/truncated tensor storage for {key}: {path}")
    return {"file": str(path), "bytes": size, "tensor_count": len(tensors), "checked": "header_and_offsets_only"}, tensors


def _final_export(directory, stage):
    _json(directory / "config.json")
    if stage == 1:
        return [_torch_header(directory / "mm_projector.bin")]
    for name, safe in (("model.safetensors.index.json", True), ("pytorch_model.bin.index.json", False)):
        index_path = directory / name
        if not index_path.is_file():
            continue
        mapping = _json(index_path).get("weight_map")
        _require(isinstance(mapping, dict) and bool(mapping), f"Empty/missing weight_map: {index_path}")
        reports = []
        _require(all(isinstance(value, str) for value in mapping.values()), f"Invalid shard filenames: {index_path}")
        for filename in sorted(set(mapping.values())):
            _require(isinstance(filename, str) and Path(filename).name == filename,
                     f"Invalid shard filename in {index_path}")
            shard = directory / filename
            if safe:
                report, entries = _safetensors_header(shard)
                required = {key for key, value in mapping.items() if value == filename}
                _require(required.issubset(entries), f"Index names missing from shard header: {shard}")
            else:
                report = _torch_header(shard)
            reports.append(report)
        return reports
    if (directory / "model.safetensors").is_file():
        return [_safetensors_header(directory / "model.safetensors")[0]]
    return [_torch_header(directory / "pytorch_model.bin")]


def _checkpoint(directory, stage):
    if stage == 1:
        return directory / "checkpoint-350"
    required = ("protected_cka_stage2_state.pt", "trainer_state.json", "student_model.pt",
                "optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json")
    candidates = [path for path in directory.glob("checkpoint-*")
                  if re.fullmatch(r"checkpoint-\d+", path.name) and path.is_dir()
                  and all((path / name).is_file() and (path / name).stat().st_size > 0 for name in required)]
    _require(bool(candidates), f"No complete Stage-2 checkpoint found: {directory}")
    return max(candidates, key=lambda path: int(path.name.split("-")[-1]))


def _arm(root, stage, mode, source_hashes):
    import torch

    directory = root / mode
    checkpoint = _checkpoint(directory, stage)
    state_name = "protected_cka_state.pt" if stage == 1 else "protected_cka_stage2_state.pt"
    for name in (state_name, "trainer_state.json", "config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        _file(checkpoint / name)
    state = torch.load(checkpoint / state_name, map_location="cpu", weights_only=False, mmap=True)
    _require(isinstance(state, dict), f"Invalid protected state: {checkpoint}")
    signature = state.get("signature", {})
    _require(isinstance(signature, dict), f"Invalid signature: {checkpoint}")
    config = signature.get("config", {})
    _require(isinstance(config, dict), f"Invalid training settings: {checkpoint}")
    prefix = "protected_cka_" if stage == 1 else "protected_stage2_"
    expected_step, expected_start = (350, 0) if stage == 1 else (5182, 4832)
    step = _json(checkpoint / "trainer_state.json").get("global_step")
    _require(state.get("version") == 1 and state.get("phase") == "B", f"Expected verified Phase B: {checkpoint}")
    _require(step == expected_step and config.get("max_steps") == expected_step,
             f"Incomplete/unexpected step: got {step}, require {expected_step}: {checkpoint}")
    _require(state.get("phase_b_start") == expected_start, f"Different phase boundary: {checkpoint}")
    _require(config.get(prefix + "continuation") == mode, f"Wrong continuation mode: {checkpoint}")
    _require(bool(state.get("refinement_accepted")) == (mode == "protected"),
             f"Incorrect refinement acceptance (protected must pass its gate): {checkpoint}")
    expected_reason = None if stage == 1 else "refinement_budget_complete"
    _require(state.get("finished_reason") == expected_reason, f"Stopped/exhausted before protocol completion: {checkpoint}")
    _require(state.get("reference") is not None if stage == 1 else state.get("has_reference") is True,
             f"Missing immutable reference metadata: {checkpoint}")
    _require(bool(config.get(prefix + "split_fingerprint")), f"Missing held-out fingerprint: {checkpoint}")
    recipe = {"seed": 42, "per_device_train_batch_size": 16 if stage == 1 else 8,
              "gradient_accumulation_steps": 16, "lr_scheduler_type": "cosine",
              "learning_rate": 1e-5 if stage == 1 else 2e-5, prefix + "refine_lr": 1e-5,
              prefix + "validation_max_samples": 1840 if stage == 1 else 2002}
    different = [key for key, value in recipe.items() if config.get(key) != value]
    _require(not different, f"Different fixed protocol settings ({', '.join(different)}): {checkpoint}")
    for key in ("model", "train_dataset", "eval_dataset"):
        _require(bool(signature.get(key)), f"Missing {key} identity: {checkpoint}")
    _require(signature["model"].get("_name_or_path") == "Qwen/Qwen2.5-3B-Instruct", f"Wrong protocol backbone: {checkpoint}")
    _torch_header(checkpoint / ("mm_projector.bin" if stage == 1 else "student_model.pt"))
    if stage == 2:
        _torch_header(checkpoint / "reference_model.pt")
        _require(config.get("protected_stage2_min_phase_a_steps") == 4832
                 and config.get("protected_stage2_max_phase_a_steps") == 4832
                 and config.get("protected_stage2_max_phase_b_steps") == 350,
                 f"Different Stage-2 phase-budget recipe: {checkpoint}")
    # The final export must have finished too, not merely its safe checkpoint.
    _require(_json(directory / "trainer_state.json").get("global_step") == expected_step,
             f"Final export training state is incomplete/stale: {directory}")
    report = {"checkpoint": str(checkpoint), "global_step": step, "phase_b_start": state["phase_b_start"],
              "phase_b_steps": step - state["phase_b_start"], "mode": mode,
              "refinement_accepted": bool(state["refinement_accepted"]), "retries": state.get("retries", 0),
              "finished_reason": state.get("finished_reason"), "split_fingerprint": config[prefix + "split_fingerprint"],
              "final_export": _final_export(directory, stage)}
    provenance = state.get("source_provenance")
    if stage == 1:
        _require(isinstance(provenance, dict) and provenance.get("checkpoint") and provenance.get("signature"),
                 f"Missing original CE-only source provenance: {checkpoint}")
        _require(provenance.get("global_step") == 2173, f"Unexpected original Stage-1 source step: {checkpoint}")
        source_file = Path(provenance["checkpoint"]) / "mm_projector.bin"
        source_key = str(source_file.resolve())
        if source_key not in source_hashes:
            _file(source_file)
            _require(source_file.stat().st_size <= 128 * 1024 * 1024, "Refusing to hash unexpectedly large source projector.")
            digest = hashlib.sha256()
            with source_file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            source_hashes[source_key] = digest.hexdigest()
        report["source_checkpoint"] = provenance["checkpoint"]
        report["source_global_step"] = provenance["global_step"]
        report["source_projector_sha256"] = source_hashes[source_key]
    return report, signature, provenance


def audit_pipeline(stage1_root=DEFAULT_STAGE1, stage2_root=DEFAULT_STAGE2):
    report = {"complete": False, "stages": {}, "errors": [],
              "verification_scope": "CPU-mapped metadata and export headers only; model tensor values/equivalence and Stage-2 initial-adapter linkage are not verified."}
    source_hashes = {}
    for stage, root in ((1, Path(stage1_root)), (2, Path(stage2_root))):
        name = f"stage{stage}"
        section = report["stages"][name] = {"root": str(root), "arms": {}, "matched_recorded_settings": False}
        values = {}
        for mode in MODES:
            try:
                arm, signature, provenance = _arm(root, stage, mode, source_hashes)
                section["arms"][mode] = arm
                values[mode] = signature, provenance
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError,
                    ImportError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
                report["errors"].append(f"{name}/{mode}: {error}")
        if len(values) == 2:
            left, right = (values[mode][0] for mode in MODES)
            key = "protected_cka_continuation" if stage == 1 else "protected_stage2_continuation"
            normalized = [dict(signature, config={k: v for k, v in signature["config"].items() if k != key})
                          for signature in (left, right)]
            if normalized[0] != normalized[1]:
                report["errors"].append(f"{name}: recorded signatures differ beyond continuation mode")
            elif stage == 1 and values["ce_only"][1] != values["protected"][1]:
                report["errors"].append("stage1: original CE-only source provenance differs")
            else:
                section["matched_recorded_settings"] = True
    report["complete"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", default=DEFAULT_STAGE1)
    parser.add_argument("--stage2-root", default=DEFAULT_STAGE2)
    args = parser.parse_args()
    report = audit_pipeline(args.stage1_root, args.stage2_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
