#!/usr/bin/env python3
"""Run CPU ablation smoke tests without loading optional CUDA extensions.

The availability override is local to this process and occurs before importing
LLaVA. It permits CPU testing when an installed FlashAttention extension cannot
load against the host's CUDA runtime. It changes no packages, files, environment
variables, or training launcher settings and does not verify GPU compatibility.
Tests use tiny synthetic CPU models on the installed Transformers version. The
runner's target CUDA stack still requires its separate real-data smoke run to
verify model loading, training, and checkpoint resume.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import transformers.utils
    import transformers.utils.import_utils

    # Transformers checks this availability function when importing model
    # implementations. CPU-only tests never require the CUDA FlashAttention path.
    transformers.utils.is_flash_attn_2_available = lambda: False
    transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
    names = [
        "tests.test_cvpr2027_cka",
        "tests.test_cvpr2027_reporting",
        "tests.test_cvpr2027_runner",
        "tests.test_cvpr2027_runtime",
    ]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
    print(f"Synthetic CPU smoke on installed Transformers {transformers.__version__}.", flush=True)
    print("FlashAttention availability disabled only in this process; CUDA target load/resume requires a real-data smoke run.", flush=True)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
