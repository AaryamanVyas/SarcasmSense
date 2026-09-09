#!/usr/bin/env python
"""One command to get from a fresh clone to a working, verifiable installation.

    python scripts/quickstart.py

Runs, in order:

0. ``check_environment.py``   — verify the interpreter is 3.10-3.12 and that a
                                virtual environment is active, before anything
                                slow or confusing can fail
1. ``download_datasets.py``   — fetch the two auto-downloadable corpora and
                                validate whatever else is already on disk
2. ``prepare_datasets.py``    — normalise, deduplicate, split, verify no leakage
3. ``make_dev_assets.py``     — build the DEVELOPMENT/TEST-ONLY tiny encoder
4. ``build_knowledge_index.py --download-wordnet --sample``
                              — download WordNet and build the sample index
5. ``train.py --config configs/dev_smoke.yaml``   (unless --no-train)
                              — a ~1 minute development-scale training run
6. ``smoke_test.py``          — 11 end-to-end checks

Everything it builds is **development-scale and clearly labelled as such**. It
exists so you can verify the software works before committing GPU time to the
real thing. For research results, follow README sections 9–13 instead.

Use ``--skip`` to omit steps you have already run, e.g. on a machine where the
datasets are already prepared::

    python scripts/quickstart.py --skip datasets prepare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RULE = "=" * 78

STEPS: dict[str, tuple[str, list[str]]] = {
    "environment": (
        "Check the Python version and virtual environment",
        ["scripts/check_environment.py"],
    ),
    "datasets": (
        "Download + validate corpora",
        ["scripts/download_datasets.py"],
    ),
    "prepare": (
        "Normalise, split and verify the corpus",
        ["scripts/prepare_datasets.py"],
    ),
    "dev-assets": (
        "Build the DEVELOPMENT/TEST-ONLY tiny encoder",
        ["scripts/make_dev_assets.py"],
    ),
    "index": (
        "Download WordNet and build the sample knowledge index",
        ["scripts/build_knowledge_index.py", "--download-wordnet", "--sample"],
    ),
    "train": (
        "Development-scale training run (~1 minute on CPU)",
        ["scripts/train.py", "--config", "configs/dev_smoke.yaml", "--local-files-only"],
    ),
    "smoke": (
        "End-to-end smoke test",
        ["scripts/smoke_test.py"],
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=list(STEPS),
        help="Steps to skip (they have already been run).",
    )
    parser.add_argument(
        "--no-train", action="store_true",
        help="Skip the development training run (equivalent to --skip train).",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep going after a failing step instead of stopping.",
    )
    return parser.parse_args(argv)


def run_step(key: str, description: str, command: list[str]) -> tuple[int, float]:
    """Run one step as a subprocess; returns ``(exit_code, seconds)``."""
    print(f"\n{RULE}\n▶  {description}\n   python {' '.join(command)}\n{RULE}")
    started = time.perf_counter()
    result = subprocess.run([sys.executable, *command], cwd=ROOT, check=False)
    return result.returncode, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = parse_args(argv)
    skip = set(args.skip) | ({"train"} if args.no_train else set())

    print(f"\n{RULE}")
    print("QUICKSTART — development-scale setup and verification".center(78))
    print(RULE)
    print(f"  project: {ROOT}")
    print(f"  steps  : {', '.join(k for k in STEPS if k not in skip)}")
    if skip:
        print(f"  skipped: {', '.join(sorted(skip))}")
    print(
        "\n  Note: everything built here is DEVELOPMENT-SCALE. The tiny encoder and\n"
        "  the sample knowledge index verify that the software works; they are not\n"
        "  the research configuration. See README sections 9-13 for that."
    )

    outcomes: list[tuple[str, int, float]] = []
    for key, (description, command) in STEPS.items():
        if key in skip:
            continue
        # download_datasets exits 2 when a licence-gated corpus is absent, which
        # is expected and not a failure of the quickstart.
        code, seconds = run_step(key, description, command)
        acceptable = code == 0 or (key == "datasets" and code == 2)
        outcomes.append((key, 0 if acceptable else code, seconds))
        if not acceptable and not args.continue_on_error:
            print(f"\n✗ Step '{key}' failed with exit code {code}. Stopping.")
            print("  Re-run with --continue-on-error to push past it.")
            break

    print(f"\n{RULE}\nSUMMARY\n{RULE}")
    for key, code, seconds in outcomes:
        mark = "✓" if code == 0 else "✗"
        print(f"  {mark} {key:<12} {seconds:6.1f}s")
    failed = [key for key, code, _ in outcomes if code != 0]

    if failed:
        print(f"\n  Failed: {', '.join(failed)}")
        return 1

    print(
        "\n  Ready. Next:\n"
        "    pytest                                     # 324 tests\n"
        "    MODEL_PATH=checkpoints/dev_smoke/best_model \\\n"
        "      uvicorn backend.main:app --port 8000     # API + /docs\n"
        "    cd frontend && npm install && npm run dev  # UI on :5173\n"
        "\n  For research results, build a real knowledge index and train the real\n"
        "  model -- see README sections 10 and 11.\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
