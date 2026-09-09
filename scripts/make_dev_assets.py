#!/usr/bin/env python
"""Build the DEVELOPMENT/TEST-ONLY model assets.

Creates a tiny randomly-initialised RoBERTa plus a byte-level BPE tokenizer
trained on the project's own corpus, so that the test-suite, the smoke test and
CI can exercise every code path without downloading ``roberta-base``.

    python scripts/make_dev_assets.py

.. warning::
   The result has learned nothing. Metrics obtained with it describe the
   software pipeline, never the proposed architecture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dev_assets import DEV_WARNING, build_dev_assets  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402
from src.utils.paths import resolve  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402

RULE = "=" * 78


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="checkpoints/dev_encoder")
    parser.add_argument(
        "--corpus",
        default="data/processed/dataset.jsonl",
        help="Corpus used to train the BPE merges (falls back to a built-in sample).",
    )
    parser.add_argument("--corpus-limit", type=int, default=4000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Must be divisible by the fusion head count (8 by default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = parse_args(argv)
    configure_logging("INFO")
    set_global_seed(args.seed)

    corpus: list[str] | None = None
    corpus_path = resolve(args.corpus)
    if corpus_path.exists():
        corpus = [row["text"] for row in read_jsonl(corpus_path)][: args.corpus_limit]
        print(f"  training BPE on {len(corpus):,} utterances from {corpus_path}")
    else:
        print(
            f"  {corpus_path} not found; using the small built-in sample corpus. "
            f"Run scripts/prepare_datasets.py for a better dev tokenizer."
        )

    print(f"\n{RULE}\nBUILDING DEVELOPMENT/TEST ASSETS\n{RULE}")
    target = build_dev_assets(
        output_dir=args.output,
        corpus=corpus,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
    )
    print(f"\n  written to {target}\n")
    print(DEV_WARNING)
    print(
        "Next:\n"
        "  python scripts/build_knowledge_index.py --sample\n"
        "  python scripts/train.py --config configs/dev_smoke.yaml --local-files-only\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
