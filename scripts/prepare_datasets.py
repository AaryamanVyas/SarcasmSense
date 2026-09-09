#!/usr/bin/env python
"""Normalise every available corpus into one schema, then split it.

Pipeline::

    data/raw/**            (heterogeneous formats)
        |  dataset adapters + conservative preprocessing
        v
    data/processed/dataset.jsonl      (common schema, deduplicated)
        |  conversation-level stratified 70/15/15 split
        v
    data/splits/{train,val,test}.txt + manifest.json
        |  leakage verification
        v
    results/dataset_report.json

Examples
--------
    python scripts/prepare_datasets.py
    python scripts/prepare_datasets.py --datasets figlang2020_twitter mustard
    python scripts/prepare_datasets.py --sarc-max-rows 20000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import ALL_DATASET_KEYS, build_adapter  # noqa: E402
from src.data.preprocessing import PreprocessingOptions  # noqa: E402
from src.data.schema import records_to_dicts  # noqa: E402
from src.data.splitting import (  # noqa: E402
    apply_split,
    corpus_fingerprint,
    save_split,
    split_records,
    summarise_split,
)
from src.data.validation import check_leakage, deduplicate, validate_records  # noqa: E402
from src.utils.io import write_json, write_jsonl  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402

logger = logging.getLogger("prepare_datasets")
RULE = "=" * 78


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/processed/dataset.jsonl")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--report", default="results/dataset_report.json")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(ALL_DATASET_KEYS),
        choices=list(ALL_DATASET_KEYS),
        help="Corpora to include (missing ones are skipped with a warning).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context-turns", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--sarc-max-rows",
        type=int,
        default=25000,
        help=(
            "Cap on SARC rows. SARC has ~1M rows and would otherwise dominate the "
            "~50k target corpus. Set to 0 for no cap."
        ),
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not remove duplicate (context, text) pairs.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--replace-subreddits", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = parse_args(argv)
    configure_logging(args.log_level)
    set_global_seed(args.seed)

    options = PreprocessingOptions(replace_subreddits=args.replace_subreddits)
    report: dict[str, object] = {"seed": args.seed, "per_dataset": {}}

    # ------------------------------------------------------------- 1. load
    print(f"\n{RULE}\n1. LOADING CORPORA\n{RULE}")
    all_records = []
    skipped: list[str] = []
    for key in args.datasets:
        max_rows = args.sarc_max_rows if (key == "sarc" and args.sarc_max_rows) else None
        adapter = build_adapter(
            key,
            raw_dir=args.raw_dir,
            options=options,
            max_context_turns=args.max_context_turns,
            max_rows=max_rows,
            seed=args.seed,
        )
        if not adapter.available():
            skipped.append(key)
            print(f"  ‑ {key:<22} SKIPPED (files not found in {adapter.raw_dir})")
            continue
        records = list(adapter.load())
        all_records.extend(records)
        report["per_dataset"][key] = adapter.stats.as_dict()  # type: ignore[index]
        print(f"  ✓ {key:<22} {len(records):>7,} records   {adapter.stats.as_dict()}")

    if not all_records:
        print(
            "\nNo corpora available. Run  python scripts/download_datasets.py  first, "
            "and follow the manual instructions it prints for Kaggle-hosted datasets."
        )
        return 1
    if skipped:
        print(f"\n  Skipped (not downloaded): {', '.join(skipped)}")

    # -------------------------------------------------------- 2. deduplicate
    print(f"\n{RULE}\n2. DEDUPLICATION & QUALITY\n{RULE}")
    before = len(all_records)
    if args.keep_duplicates:
        print("  --keep-duplicates set; skipping deduplication")
        n_removed = n_conflicts = 0
    else:
        all_records, n_removed, n_conflicts = deduplicate(all_records)
        print(
            f"  removed {n_removed:,} duplicate rows "
            f"({n_conflicts:,} groups dropped entirely for conflicting labels)"
        )
    print(f"  corpus: {before:,} -> {len(all_records):,} records")

    merged_report = validate_records(all_records, name="MERGED CORPUS")
    print()
    print(merged_report.render())

    # ------------------------------------------------------------ 3. persist
    print(f"\n{RULE}\n3. WRITING NORMALISED CORPUS\n{RULE}")
    n_written = write_jsonl(args.output, records_to_dicts(all_records))
    print(f"  wrote {n_written:,} records -> {args.output}")

    # -------------------------------------------------------------- 4. split
    print(f"\n{RULE}\n4. CONVERSATION-LEVEL STRATIFIED SPLIT\n{RULE}")
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    assignment = split_records(all_records, ratios=ratios, seed=args.seed)
    splits = apply_split(all_records, assignment)
    print(summarise_split(splits))
    save_split(assignment, args.splits_dir)
    print(f"  split ids saved -> {args.splits_dir}")

    # ------------------------------------------------------------ 5. leakage
    print(f"\n{RULE}\n5. LEAKAGE VERIFICATION\n{RULE}")
    leakage = check_leakage(splits)
    print(leakage.render())

    report.update(
        {
            "n_records": len(all_records),
            "n_duplicates_removed": n_removed,
            "n_label_conflict_groups": n_conflicts,
            "corpus_fingerprint": corpus_fingerprint(all_records),
            "merged_report": merged_report.as_dict(),
            "split": assignment.as_dict(),
            "per_split_reports": {
                name: validate_records(rows, name=name).as_dict()
                for name, rows in splits.items()
            },
            "leakage": leakage.as_dict(),
            "skipped_datasets": skipped,
        }
    )
    path = write_json(args.report, report)
    print(f"\nFull report -> {path}")

    if not leakage.clean:
        print("\nLEAKAGE DETECTED -- do not train on this split.")
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
