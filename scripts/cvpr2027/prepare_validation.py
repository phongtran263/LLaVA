#!/usr/bin/env python3
"""Write a deterministic image-grouped holdout without changing the source data."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def split(samples, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must be in (0, 1)")
    groups = {}
    for index, sample in enumerate(samples):
        # Keep all conversations involving the same image in one partition.
        key = "image:" + str(sample["image"]) if sample.get("image") else "text:" + hashlib.sha256(
            json.dumps(sample.get("conversations"), sort_keys=True).encode()).hexdigest()
        groups.setdefault(key, []).append(index)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need at least two independent image/text groups")
    random.Random(seed).shuffle(keys)
    validation_keys = set(keys[:max(1, min(len(keys) - 1, round(len(keys) * fraction)))])
    validation_indices = {i for key in validation_keys for i in groups[key]}
    training = [sample for i, sample in enumerate(samples) if i not in validation_indices]
    validation = [sample for i, sample in enumerate(samples) if i in validation_indices]
    return training, validation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fraction", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()
    raw = args.data_path.read_bytes()
    train, validation = split(json.loads(raw), args.fraction, args.seed)
    # An existing split is immutable; choose a new destination to change it.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("train.json", train), ("validation.json", validation)):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False) + "\n")
    manifest = {"source": str(args.data_path.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
                "fraction_of_groups": args.fraction, "seed": args.seed, "train_count": len(train),
                "validation_count": len(validation), "grouping": "image path or exact text conversation"}
    (args.output_dir / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
