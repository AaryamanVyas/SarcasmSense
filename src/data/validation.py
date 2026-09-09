"""Corpus quality reporting and leakage checks.

Two responsibilities:

1. :func:`validate_records` -- produce a :class:`DatasetReport` (row counts,
   class balance, missing values, duplicates, context availability, length
   statistics). This is what ``scripts/download_datasets.py --validate-only``
   and ``scripts/prepare_datasets.py`` print.
2. :func:`check_leakage` -- verify that a train/val/test partition contains no
   shared conversations, no shared content hashes, and no context strings that
   appear as a target utterance in a different split.

Both are pure functions over :class:`~src.data.schema.SarcasmRecord` sequences,
which makes them trivial to unit-test.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from src.data.schema import LABEL_NAMES, SarcasmRecord

logger = logging.getLogger(__name__)


@dataclass
class DatasetReport:
    """Quality summary for one corpus (or for a merged corpus)."""

    name: str
    n_rows: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    class_balance: dict[str, float] = field(default_factory=dict)
    n_duplicate_content: int = 0
    duplicate_examples: list[str] = field(default_factory=list)
    n_with_context: int = 0
    context_availability: float = 0.0
    context_turn_histogram: dict[int, int] = field(default_factory=dict)
    n_missing_text: int = 0
    n_missing_label: int = 0
    n_empty_context_entries: int = 0
    text_length_chars: dict[str, float] = field(default_factory=dict)
    text_length_words: dict[str, float] = field(default_factory=dict)
    per_dataset_rows: dict[str, int] = field(default_factory=dict)
    per_domain_rows: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    n_conversations: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        from dataclasses import asdict

        return asdict(self)

    def render(self) -> str:
        """Render a compact, human-readable report block."""
        lines = [
            f"── {self.name} " + "─" * max(0, 60 - len(self.name)),
            f"  rows                 : {self.n_rows:,}",
            f"  conversations        : {self.n_conversations:,}",
            "  class distribution   : "
            + ", ".join(
                f"{k}={v:,} ({self.class_balance.get(k, 0.0):.1%})"
                for k, v in sorted(self.class_distribution.items())
            ),
            f"  duplicates (content) : {self.n_duplicate_content:,}",
            f"  with context         : {self.n_with_context:,} "
            f"({self.context_availability:.1%})",
            "  context turns        : "
            + ", ".join(f"{k}:{v:,}" for k, v in sorted(self.context_turn_histogram.items())),
            f"  missing text/label   : {self.n_missing_text:,} / {self.n_missing_label:,}",
            "  target length (words): "
            + ", ".join(f"{k}={v:.1f}" for k, v in self.text_length_words.items()),
        ]
        if self.per_dataset_rows:
            lines.append(
                "  per-dataset          : "
                + ", ".join(f"{k}={v:,}" for k, v in sorted(self.per_dataset_rows.items()))
            )
        if self.per_domain_rows:
            lines.append(
                "  per-domain           : "
                + ", ".join(f"{k}={v:,}" for k, v in sorted(self.per_domain_rows.items()))
            )
        for warning in self.warnings:
            lines.append(f"  ⚠  {warning}")
        return "\n".join(lines)


def _length_stats(values: Sequence[int]) -> dict[str, float]:
    """Mean / median / p95 / max helper that tolerates empty input."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[p95_index]),
        "max": float(ordered[-1]),
    }


def validate_records(
    records: Iterable[SarcasmRecord],
    name: str = "corpus",
    duplicate_examples: int = 3,
    min_class_ratio: float = 0.05,
) -> DatasetReport:
    """Compute a :class:`DatasetReport` for a stream of records.

    Args:
        records: The records to inspect (consumed once).
        name: Label used in the rendered report.
        duplicate_examples: How many duplicate texts to keep as examples.
        min_class_ratio: Emit a warning when either class falls below this
            fraction of the corpus.
    """
    report = DatasetReport(name=name)
    seen_hashes: dict[str, int] = defaultdict(int)
    char_lengths: list[int] = []
    word_lengths: list[int] = []
    conversations: set[str] = set()
    labels: Counter[str] = Counter()

    for record in records:
        report.n_rows += 1
        labels[LABEL_NAMES[record.label]] += 1
        report.per_dataset_rows[record.dataset] = (
            report.per_dataset_rows.get(record.dataset, 0) + 1
        )
        report.per_domain_rows[record.domain] = report.per_domain_rows.get(record.domain, 0) + 1
        report.languages[record.language] = report.languages.get(record.language, 0) + 1
        conversations.add(record.conversation_id or record.id)

        if not record.text.strip():
            report.n_missing_text += 1
        n_turns = len(record.context)
        report.context_turn_histogram[n_turns] = (
            report.context_turn_histogram.get(n_turns, 0) + 1
        )
        if n_turns:
            report.n_with_context += 1
        report.n_empty_context_entries += sum(1 for t in record.context if not t.strip())

        char_lengths.append(len(record.text))
        word_lengths.append(len(record.text.split()))

        digest = record.content_hash
        seen_hashes[digest] += 1
        if seen_hashes[digest] == 2:
            report.n_duplicate_content += 1
            if len(report.duplicate_examples) < duplicate_examples:
                report.duplicate_examples.append(record.text[:120])
        elif seen_hashes[digest] > 2:
            report.n_duplicate_content += 1

    report.class_distribution = dict(labels)
    report.n_conversations = len(conversations)
    if report.n_rows:
        report.class_balance = {k: v / report.n_rows for k, v in labels.items()}
        report.context_availability = report.n_with_context / report.n_rows
    report.text_length_chars = _length_stats(char_lengths)
    report.text_length_words = _length_stats(word_lengths)

    # ---------------------------------------------------------- warnings
    if report.n_rows == 0:
        report.warnings.append("corpus is empty")
    for label_name, ratio in report.class_balance.items():
        if ratio < min_class_ratio:
            report.warnings.append(
                f"class '{label_name}' is only {ratio:.1%} of the corpus; "
                f"enable training.class_weights=balanced"
            )
    if report.n_duplicate_content:
        report.warnings.append(
            f"{report.n_duplicate_content:,} duplicate (context, text) pairs found; "
            f"they are removed by prepare_datasets.py unless --keep-duplicates is set"
        )
    if report.n_missing_text:
        report.warnings.append(f"{report.n_missing_text} rows have empty target text")
    return report


def find_duplicate_groups(
    records: Sequence[SarcasmRecord],
) -> dict[str, list[SarcasmRecord]]:
    """Group records by content hash, keeping only groups with >1 member."""
    groups: dict[str, list[SarcasmRecord]] = defaultdict(list)
    for record in records:
        groups[record.content_hash].append(record)
    return {digest: rows for digest, rows in groups.items() if len(rows) > 1}


def deduplicate(
    records: Sequence[SarcasmRecord],
) -> tuple[list[SarcasmRecord], int, int]:
    """Drop duplicate (context, text) pairs, keeping the first occurrence.

    Returns:
        ``(kept, n_removed, n_label_conflicts)`` where ``n_label_conflicts``
        counts duplicate groups whose members disagree on the label -- those are
        removed entirely because they are unlearnable and would pollute the test
        set with guaranteed errors.
    """
    by_hash: dict[str, list[SarcasmRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.content_hash].append(record)

    kept: list[SarcasmRecord] = []
    removed = 0
    conflicts = 0
    for group in by_hash.values():
        labels = {record.label for record in group}
        if len(labels) > 1:
            conflicts += 1
            removed += len(group)
            continue
        kept.append(group[0])
        removed += len(group) - 1
    # Restore the original corpus ordering for reproducibility.
    order = {id(record): i for i, record in enumerate(records)}
    kept.sort(key=lambda r: order[id(r)])
    return kept, removed, conflicts


@dataclass
class LeakageReport:
    """Result of :func:`check_leakage`."""

    shared_conversations: dict[str, int] = field(default_factory=dict)
    shared_content_hashes: dict[str, int] = field(default_factory=dict)
    context_target_overlap: dict[str, int] = field(default_factory=dict)
    trivial_overlap: dict[str, int] = field(default_factory=dict)
    trivial_overlap_examples: list[str] = field(default_factory=list)
    min_overlap_words: int = 4
    clean: bool = True
    details: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        from dataclasses import asdict

        return asdict(self)

    def render(self) -> str:
        """Human-readable summary."""
        lines = (
            ["Leakage check: PASS (no shared conversations, duplicates or context overlap)"]
            if self.clean
            else ["Leakage check: FAIL", *(f"  - {d}" for d in self.details)]
        )
        if self.trivial_overlap:
            total = sum(self.trivial_overlap.values())
            lines.append(
                f"  note: {total} short utterance(s) (<{self.min_overlap_words} words, "
                f"e.g. {', '.join(repr(e) for e in self.trivial_overlap_examples)}) occur "
                f"in more than one split. These carry no thread-specific information and "
                f"are not treated as leakage."
            )
        return "\n".join(lines)


def check_leakage(
    splits: Mapping[str, Sequence[SarcasmRecord]], min_overlap_words: int = 4
) -> LeakageReport:
    """Verify that no information crosses split boundaries.

    Three distinct leaks are checked:

    * **conversation leakage** -- the same ``conversation_id`` in two splits,
      which would let the model memorise a thread it is later tested on;
    * **duplicate leakage** -- identical (context, text) content in two splits;
    * **context leakage** -- a string used as *context* in one split appearing
      as a *target utterance* in another. This is subtle but real: a model that
      memorises the target utterance from a training context gets a free win at
      test time.

    Args:
        splits: Mapping of split name to records.
        min_overlap_words: Strings shorter than this many words ("no.", "why?")
            are reported separately as *trivial* overlap rather than leakage.
            They are generic dialogue fillers that identify no particular
            thread, so forcing them into a single split would chain unrelated
            conversations into one giant cluster and destroy the 70/15/15
            proportions -- a worse outcome than the non-existent information
            they leak. The same threshold is used by
            :func:`src.data.splitting.build_conversation_groups`.
    """
    report = LeakageReport(min_overlap_words=min_overlap_words)
    names = list(splits)

    conv_sets = {
        name: {r.conversation_id or r.id for r in rows} for name, rows in splits.items()
    }
    hash_sets = {name: {r.content_hash for r in rows} for name, rows in splits.items()}
    target_sets = {
        name: {" ".join(r.text.split()).lower() for r in rows} for name, rows in splits.items()
    }
    context_sets = {
        name: {" ".join(t.split()).lower() for r in rows for t in r.context}
        for name, rows in splits.items()
    }

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pair = f"{a}|{b}"
            shared_conv = conv_sets[a] & conv_sets[b]
            if shared_conv:
                report.shared_conversations[pair] = len(shared_conv)
                report.details.append(
                    f"{len(shared_conv)} conversation id(s) shared between {a} and {b}"
                )
            shared_hash = hash_sets[a] & hash_sets[b]
            if shared_hash:
                report.shared_content_hashes[pair] = len(shared_hash)
                report.details.append(
                    f"{len(shared_hash)} duplicate record(s) shared between {a} and {b}"
                )
            overlap = (context_sets[a] & target_sets[b]) | (context_sets[b] & target_sets[a])
            substantive = {s for s in overlap if len(s.split()) >= min_overlap_words}
            trivial = overlap - substantive
            if substantive:
                report.context_target_overlap[pair] = len(substantive)
                report.details.append(
                    f"{len(substantive)} string(s) appear as context in one of {a}/{b} "
                    f"and as a target utterance in the other"
                )
            if trivial:
                report.trivial_overlap[pair] = len(trivial)
                for example in sorted(trivial)[:3]:
                    if len(report.trivial_overlap_examples) < 3:
                        report.trivial_overlap_examples.append(example)

    report.clean = not report.details
    return report
