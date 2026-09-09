#!/usr/bin/env python
"""Build the offline commonsense knowledge base and its FAISS index.

The index is built **once** and reused by training, evaluation and the API.
Nothing is retrieved from an external service at inference time.

Knowledge sources
-----------------
``wordnet``    WordNet 3.0 + SentiWordNet via NLTK. Downloadable without
               credentials, ~1.5 GB of RAM to build, ~250k facts. This is the
               default and the only source that always works offline.
``conceptnet`` The official ConceptNet 5.7 assertions dump (~1.2 GB gzipped).
               Downloaded from the URL published by the ConceptNet project.
``atomic``     ATOMIC / ATOMIC-2020 inferential tuples (COMET's training data),
               supplied by you as a TSV -- see --atomic-tsv.
``jsonl``      A previously exported facts file (``--facts-jsonl``).

Examples
--------
Smoke-test index (small, fast, clearly marked as a sample)::

    python scripts/build_knowledge_index.py --sample

Full WordNet index::

    python scripts/build_knowledge_index.py --download-wordnet --source wordnet

Research index over ConceptNet + ATOMIC::

    python scripts/build_knowledge_index.py --download-conceptnet \\
        --source conceptnet atomic --atomic-tsv knowledge/raw/atomic2020/train.tsv
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.encoder import build_sentence_encoder  # noqa: E402
from src.retrieval.faiss_index import KnowledgeIndex  # noqa: E402
from src.retrieval.knowledge_base import (  # noqa: E402
    KnowledgeFact,
    iter_atomic_facts,
    iter_conceptnet_facts,
    iter_wordnet_facts,
    read_facts,
    write_facts,
)
from src.utils.logging_utils import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir, resolve  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402

logger = logging.getLogger("build_knowledge_index")
RULE = "=" * 78

#: Official ConceptNet 5.7 assertions dump, as published by the ConceptNet project
#: (https://github.com/commonsense/conceptnet5/wiki/Downloads).
CONCEPTNET_URL = (
    "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
    "conceptnet-assertions-5.7.0.csv.gz"
)
CONCEPTNET_FILENAME = "conceptnet-assertions-5.7.0.csv.gz"

SAMPLE_NOTE = (
    "DEVELOPMENT/TEST ONLY -- this is a small sample index built from a subset of "
    "WordNet so the pipeline can be smoke-tested without a large download. It is "
    "NOT a research knowledge base and must not be used for reported results."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        nargs="+",
        default=["wordnet"],
        choices=["wordnet", "conceptnet", "atomic", "jsonl"],
        help="Knowledge sources to merge (default: wordnet).",
    )
    parser.add_argument("--output", default="knowledge/index", help="Index directory.")
    parser.add_argument(
        "--facts-out",
        default="knowledge/processed/facts.jsonl",
        help="Where the merged, verbalised facts are written.",
    )
    parser.add_argument("--facts-jsonl", default=None, help="Input for --source jsonl.")
    parser.add_argument("--raw-dir", default="knowledge/raw")
    parser.add_argument("--nltk-data-dir", default="knowledge/raw/nltk_data")
    parser.add_argument("--conceptnet-dump", default=None, help="Path to the .csv.gz dump.")
    parser.add_argument("--conceptnet-min-weight", type=float, default=1.0)
    parser.add_argument("--conceptnet-max-facts", type=int, default=1_500_000)
    parser.add_argument("--atomic-tsv", default=None, help="ATOMIC/ATOMIC-2020 TSV path.")
    parser.add_argument("--atomic-max-facts", type=int, default=500_000)
    parser.add_argument("--max-facts", type=int, default=None, help="Global cap.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help=(
            "Build a small WordNet-derived sample index for smoke tests "
            "(writes to knowledge/index_sample unless --output is given)."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--encoder-backend", default="auto", choices=["auto", "sentence-transformers", "hashing"])
    parser.add_argument(
        "--encoder-name", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--index-type", default="auto", choices=["auto", "flat", "ivf"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--store-embeddings", action="store_true")
    parser.add_argument("--download-wordnet", action="store_true")
    parser.add_argument("--download-conceptnet", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def download_wordnet(nltk_data_dir: str) -> None:
    """Download the WordNet + SentiWordNet corpora into the project tree."""
    import nltk

    target = ensure_dir(nltk_data_dir)
    for package in ("wordnet", "omw-1.4", "sentiwordnet"):
        logger.info("Downloading NLTK package %s -> %s", package, target)
        if not nltk.download(package, download_dir=str(target), quiet=False):
            raise RuntimeError(f"NLTK download failed for {package!r}")


def download_conceptnet(raw_dir: str) -> Path:
    """Download the official ConceptNet assertions dump (~1.2 GB)."""
    from src.data.download import RemoteFile, download_file

    target_dir = ensure_dir(Path(raw_dir) / "conceptnet")
    result = download_file(
        RemoteFile(url=CONCEPTNET_URL, relative_path=CONCEPTNET_FILENAME),
        target_dir,
        dataset_key="conceptnet",
        timeout=3600,
    )
    if result.status == "failed":
        raise RuntimeError(
            f"ConceptNet download failed: {result.error}\n"
            f"Download it manually from {CONCEPTNET_URL} and place it at "
            f"{target_dir / CONCEPTNET_FILENAME}, then re-run with "
            f"--conceptnet-dump {target_dir / CONCEPTNET_FILENAME}"
        )
    return result.path


def collect_facts(args: argparse.Namespace) -> list[KnowledgeFact]:
    """Materialise the merged fact list from every requested source."""
    facts: list[KnowledgeFact] = []

    if "wordnet" in args.source:
        max_synsets = args.sample_size if args.sample else None
        logger.info("Extracting WordNet facts (max_synsets=%s)", max_synsets)
        facts.extend(
            iter_wordnet_facts(
                nltk_data_dir=args.nltk_data_dir,
                max_synsets=max_synsets,
            )
        )

    if "conceptnet" in args.source:
        dump = args.conceptnet_dump
        if dump is None:
            candidate = resolve(args.raw_dir) / "conceptnet" / CONCEPTNET_FILENAME
            if not candidate.exists():
                raise SystemExit(
                    f"ConceptNet dump not found at {candidate}.\n"
                    f"Either run with --download-conceptnet, or download\n"
                    f"  {CONCEPTNET_URL}\n"
                    f"manually and pass --conceptnet-dump <path>."
                )
            dump = str(candidate)
        logger.info("Extracting ConceptNet facts from %s", dump)
        facts.extend(
            iter_conceptnet_facts(
                dump,
                min_weight=args.conceptnet_min_weight,
                max_facts=args.conceptnet_max_facts,
            )
        )

    if "atomic" in args.source:
        if not args.atomic_tsv:
            raise SystemExit(
                "--source atomic requires --atomic-tsv <path to ATOMIC train.tsv>.\n"
                "ATOMIC-2020 is distributed by AI2 at https://allenai.org/data/atomic-2020"
            )
        logger.info("Extracting ATOMIC facts from %s", args.atomic_tsv)
        facts.extend(iter_atomic_facts(args.atomic_tsv, max_facts=args.atomic_max_facts))

    if "jsonl" in args.source:
        if not args.facts_jsonl:
            raise SystemExit("--source jsonl requires --facts-jsonl <path>")
        facts.extend(read_facts(args.facts_jsonl))

    if args.max_facts:
        facts = list(itertools.islice(facts, args.max_facts))
    return facts


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = parse_args(argv)
    configure_logging(args.log_level)
    set_global_seed(args.seed)

    if args.sample and args.output == "knowledge/index":
        args.output = "knowledge/index_sample"
        args.facts_out = "knowledge/processed/sample_facts.jsonl"

    print(f"\n{RULE}\nBUILDING COMMONSENSE KNOWLEDGE INDEX\n{RULE}")
    if args.download_wordnet:
        download_wordnet(args.nltk_data_dir)
    if args.download_conceptnet:
        args.conceptnet_dump = str(download_conceptnet(args.raw_dir))

    facts = collect_facts(args)
    if not facts:
        print("No facts were produced. Check the --source arguments above.")
        return 1

    by_source: dict[str, int] = {}
    for fact in facts:
        by_source[fact.source] = by_source.get(fact.source, 0) + 1
    print(f"  facts collected : {len(facts):,}")
    for source, count in sorted(by_source.items()):
        print(f"    - {source:<12} {count:,}")

    written = write_facts(args.facts_out, facts)
    print(f"  facts written   : {written:,} -> {args.facts_out}")

    encoder = build_sentence_encoder(
        backend=args.encoder_backend, model_name=args.encoder_name, device=args.device
    )
    print(f"  encoder         : {encoder.describe()}")
    if encoder.is_fallback:
        print(
            "  ⚠  Using the LEXICAL fallback encoder. Retrieval quality is reduced; "
            "do not report research results from an index built this way."
        )

    index = KnowledgeIndex.build(
        facts,
        encoder,
        index_type=args.index_type,
        batch_size=args.batch_size,
        is_sample_index=args.sample,
        build_notes=SAMPLE_NOTE if args.sample else "",
        store_embeddings=args.store_embeddings,
    )
    target = index.save(args.output)

    print(f"\n{RULE}\nINDEX SUMMARY\n{RULE}")
    for key, value in index.describe().items():
        print(f"  {key:<22}: {value}")
    print(f"  directory             : {target}")
    if args.sample:
        print(f"\n  {SAMPLE_NOTE}")

    # A quick self-check so a broken index is caught here, not at serving time.
    probe = encoder.encode(["my laptop crashed right before the presentation"])
    scores, ids = index.search(probe, top_k=3)
    print("\n  sanity probe -> top-3 facts for a demo utterance:")
    for score, fact_id in zip(scores[0].tolist(), ids[0].tolist()):
        fact = index.fact_at(int(fact_id))
        if fact is not None:
            print(f"    {score:+.3f}  [{fact.source}/{fact.relation}] {fact.text[:90]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
