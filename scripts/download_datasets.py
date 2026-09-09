#!/usr/bin/env python
"""Acquire and validate the raw sarcasm corpora.

Examples
--------
Download everything that can be fetched legally without credentials, then
report on all five corpora (including the two you must fetch by hand)::

    python scripts/download_datasets.py

Validate what is already on disk without touching the network::

    python scripts/download_datasets.py --validate-only

Fetch a single corpus and overwrite an existing copy::

    python scripts/download_datasets.py --datasets mustard --force

The script never contacts an unofficial mirror. For News-Headlines and SARC it
prints the exact manual steps, the exact expected filename and the exact
expected directory, and then verifies what you placed there.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make ``src`` importable when the script is run directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import download_dataset  # noqa: E402
from src.data.loaders import ALL_DATASET_KEYS, build_adapter  # noqa: E402
from src.data.sources import DATASET_SOURCES, get_source  # noqa: E402
from src.data.validation import validate_records  # noqa: E402
from src.utils.io import write_json  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402
from src.utils.paths import resolve  # noqa: E402

logger = logging.getLogger("download_datasets")

RULE = "=" * 78


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(ALL_DATASET_KEYS),
        choices=list(ALL_DATASET_KEYS),
        help="Which corpora to acquire/validate (default: all).",
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Where raw files live.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip all network access; only inspect files already on disk.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file exists."
    )
    parser.add_argument(
        "--report",
        default="results/dataset_acquisition_report.json",
        help="Where to write the machine-readable acquisition report.",
    )
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout (seconds).")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def print_manual_instructions(key: str) -> None:
    """Print the manual acquisition block for a credentials-gated dataset."""
    source = get_source(key)
    raw_dir = resolve("data/raw") / key
    print(f"\n{RULE}")
    print(f"MANUAL DOWNLOAD REQUIRED: {source.name}")
    print(RULE)
    print(f"Homepage : {source.homepage}")
    print(f"Licence  : {source.licence}")
    print(f"Target   : {raw_dir}")
    print(f"Expected : {', '.join(source.manual_files)}")
    print("-" * 78)
    print(source.manual_instructions)
    print(RULE)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = parse_args(argv)
    configure_logging(args.log_level)

    report: dict[str, object] = {"downloads": {}, "validation": {}, "missing": []}
    any_missing = False

    print(f"\n{RULE}\nDATASET ACQUISITION\n{RULE}")

    # ------------------------------------------------------------- download
    if not args.validate_only:
        for key in args.datasets:
            source = DATASET_SOURCES[key]
            if not source.auto_downloadable:
                continue
            results = download_dataset(
                source, raw_dir=args.raw_dir, force=args.force, timeout=args.timeout
            )
            report["downloads"][key] = [  # type: ignore[index]
                {
                    "path": str(r.path),
                    "status": r.status,
                    "bytes": r.bytes_written,
                    "error": r.error,
                }
                for r in results
            ]
            for r in results:
                marker = {"downloaded": "✓", "cached": "•", "failed": "✗"}[r.status]
                size = f"{r.bytes_written / 1e6:.2f} MB" if r.bytes_written else "-"
                print(f"  {marker} {key:<22} {r.path.name:<58} {size}")
                if r.status == "failed":
                    print(f"      error: {r.error}")

    # ------------------------------------------------------------- validate
    print(f"\n{RULE}\nVALIDATION\n{RULE}")
    for key in args.datasets:
        source = DATASET_SOURCES[key]
        adapter = build_adapter(key, raw_dir=args.raw_dir)
        if not adapter.available():
            any_missing = True
            report["missing"].append(key)  # type: ignore[union-attr]
            print(f"\n✗ {source.name}: NOT AVAILABLE")
            print(f"    expected one of: {[str(p) for p in adapter.expected_paths()]}")
            if not source.auto_downloadable:
                print_manual_instructions(key)
            continue

        records = list(adapter.load())
        dataset_report = validate_records(records, name=source.name)
        print()
        print(dataset_report.render())
        print(f"  source               : {source.homepage}")
        print(f"  licence              : {source.licence}")
        if source.expected_rows is not None:
            delta = dataset_report.n_rows - source.expected_rows
            status = "OK" if abs(delta) <= max(5, 0.02 * source.expected_rows) else "MISMATCH"
            print(
                f"  published row count  : {source.expected_rows:,} "
                f"(loaded {dataset_report.n_rows:,}, delta {delta:+,}) -> {status}"
            )
        print(f"  loader stats         : {adapter.stats.as_dict()}")
        if source.notes:
            print(f"  note                 : {source.notes.splitlines()[0]}")
        report["validation"][key] = {  # type: ignore[index]
            "report": dataset_report.as_dict(),
            "loader_stats": adapter.stats.as_dict(),
            "source": {
                "homepage": source.homepage,
                "licence": source.licence,
                "citation": source.citation,
            },
        }

    path = write_json(args.report, report)
    print(f"\nMachine-readable report written to {path}")

    if any_missing:
        print(
            "\nSome corpora are missing. The pipeline will still run on whatever is "
            "available -- see the manual instructions printed above to add the rest."
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
