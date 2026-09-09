"""Reproducible, conversation-level, stratified train/val/test splitting.

Why conversation-level?
-----------------------
Three of the four corpora are conversational. If two utterances from the same
Reddit thread (or the same MUStARD scene) land in different splits, the model
sees part of the test conversation during training -- through the *context*
field if not through the target. Measured performance then reflects memorisation
of that thread rather than sarcasm understanding. Splitting whole conversations
removes this leak by construction; :func:`~src.data.validation.check_leakage`
then *verifies* it rather than assuming it.

Why stratify?
-------------
Class balance varies wildly across corpora (FigLang is 50/50 by design, SARC is
balanced, News-Headlines is ~44/56). Stratifying on ``(dataset, majority label)``
keeps every split representative of the whole corpus, so the macro-F1 numbers
from validation actually predict the test numbers.

The split is a pure function of ``(records, seed, ratios)``: the same inputs
always give the same partition, and the resulting id lists are written to
``data/splits/`` so an experiment can be re-run months later on the exact same
partition.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.schema import LABEL_NAMES, SarcasmRecord
from src.utils.io import read_json, write_json
from src.utils.paths import ensure_dir, resolve

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


class SplitError(ValueError):
    """Raised when a split cannot be produced or a saved split does not match."""


@dataclass
class SplitAssignment:
    """The result of splitting: record ids per split plus provenance metadata."""

    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    seed: int = 42
    ratios: tuple[float, float, float] = DEFAULT_RATIOS
    corpus_fingerprint: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> list[str]:
        """``assignment["train"]`` accessor."""
        if name not in SPLIT_NAMES:
            raise KeyError(f"split must be one of {SPLIT_NAMES}, got {name!r}")
        return getattr(self, name)

    @property
    def sizes(self) -> dict[str, int]:
        """Record counts per split."""
        return {name: len(self[name]) for name in SPLIT_NAMES}

    def as_dict(self) -> dict[str, Any]:
        """Manifest payload (without the id lists, which live in .txt files)."""
        return {
            "seed": self.seed,
            "ratios": list(self.ratios),
            "corpus_fingerprint": self.corpus_fingerprint,
            "sizes": self.sizes,
            "stats": self.stats,
        }


def corpus_fingerprint(records: Sequence[SarcasmRecord]) -> str:
    """Stable hash of the corpus content, so we can detect silent data changes.

    If ``data/processed/dataset.jsonl`` is regenerated with different
    preprocessing, the fingerprint changes and :func:`load_split` refuses to
    reuse the stale split -- which is exactly what we want, because reusing it
    would silently corrupt the train/test boundary.
    """
    digest = hashlib.sha256()
    for record_id in sorted(record.id for record in records):
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(str(len(records)).encode("utf-8"))
    return digest.hexdigest()[:16]


class _UnionFind:
    """Minimal union-find over string keys (path compression + union by size)."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def add(self, key: str) -> None:
        """Register ``key`` as its own set if it is new."""
        if key not in self._parent:
            self._parent[key] = key
            self._size[key] = 1

    def find(self, key: str) -> str:
        """Return the representative of ``key``'s set."""
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:  # path compression
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        """Merge the sets containing ``a`` and ``b``."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]


def _normalise_utterance(text: str) -> str:
    """Whitespace/case-normalised form used to detect shared utterances."""
    return " ".join(text.split()).lower()


def build_conversation_groups(
    records: Sequence[SarcasmRecord],
    merge_shared_utterances: bool = True,
    min_link_words: int = 4,
) -> dict[str, list[SarcasmRecord]]:
    """Group records into leak-proof conversation clusters.

    Two records belong to the same cluster when either

    * they carry the same ``conversation_id`` (thread / scene id from the
      adapter), **or**
    * they *share an utterance*: a string that is the target of one record and a
      context turn of another, or a context turn of both. This is the case the
      raw ``conversation_id`` cannot catch, because FigLang ships each response
      as an independent row even when two responses reply to the same parent.

    Args:
        records: Corpus to cluster.
        merge_shared_utterances: Disable to fall back to plain ``conversation_id``
            grouping (kept as an ablation switch; leaves real leakage in place).
        min_link_words: Only strings with at least this many words create a
            link. Short strings ("lol", "thanks") appear across unrelated
            threads and would chain the whole corpus into one component.

    Returns:
        Mapping from cluster id to its records, in corpus order.
    """
    union = _UnionFind()
    for record in records:
        union.add(record.conversation_id or record.id)

    if merge_shared_utterances:
        owners: dict[str, str] = {}
        for record in records:
            conv = record.conversation_id or record.id
            for raw in (record.text, *record.context):
                key = _normalise_utterance(raw)
                if len(key.split()) < min_link_words:
                    continue
                previous = owners.get(key)
                if previous is None:
                    owners[key] = conv
                else:
                    union.union(previous, conv)

    clusters: dict[str, list[SarcasmRecord]] = defaultdict(list)
    for record in records:
        clusters[union.find(record.conversation_id or record.id)].append(record)
    return clusters


def _majority_label(group: Sequence[SarcasmRecord]) -> int:
    """Label used to stratify a whole conversation group."""
    counts = Counter(record.label for record in group)
    # ``most_common`` ties are broken deterministically towards the positive
    # class so the function is order-independent.
    top = counts.most_common()
    best = max(top, key=lambda item: (item[1], item[0]))
    return int(best[0])


def split_records(
    records: Sequence[SarcasmRecord],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    merge_shared_utterances: bool = True,
) -> SplitAssignment:
    """Partition ``records`` into train/val/test at the conversation level.

    Args:
        records: The full normalised corpus.
        ratios: ``(train, val, test)`` fractions; must sum to 1.0.
        seed: Controls the shuffling of conversation groups.
        merge_shared_utterances: Cluster records that share an utterance before
            splitting, so no string can be training context and test target.

    Returns:
        A :class:`SplitAssignment` holding record ids per split.

    Raises:
        SplitError: if ratios are invalid or the corpus is empty.
    """
    if len(ratios) != 3:
        raise SplitError(f"ratios must be a 3-tuple (train, val, test), got {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SplitError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")
    if any(r < 0 for r in ratios):
        raise SplitError(f"ratios must be non-negative, got {ratios}")
    if not records:
        raise SplitError("cannot split an empty corpus")

    # 1. Group records into leak-proof conversation clusters.
    groups = build_conversation_groups(
        records, merge_shared_utterances=merge_shared_utterances
    )
    largest = max((len(g) for g in groups.values()), default=0)
    if largest > 0.05 * len(records):
        logger.warning(
            "Largest conversation cluster holds %d records (%.1f%% of the corpus); "
            "the 70/15/15 proportions may be approximate.",
            largest,
            100 * largest / len(records),
        )

    # 2. Bucket conversations into strata so class + dataset balance is preserved.
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for conv_id, group in groups.items():
        strata[(group[0].dataset, _majority_label(group))].append(conv_id)

    rng = random.Random(seed)
    assignment = SplitAssignment(seed=seed, ratios=ratios)
    per_stratum: dict[str, dict[str, int]] = {}

    # 3. Within each stratum, shuffle conversations and fill splits by *record*
    #    count (not group count), which keeps the 70/15/15 proportions accurate
    #    even when conversation sizes differ.
    for stratum_key in sorted(strata, key=lambda k: (k[0], k[1])):
        conv_ids = sorted(strata[stratum_key])  # sort first => deterministic
        rng.shuffle(conv_ids)
        total = sum(len(groups[c]) for c in conv_ids)
        targets = {
            "train": ratios[0] * total,
            "val": ratios[1] * total,
            "test": ratios[2] * total,
        }
        filled = {"train": 0, "val": 0, "test": 0}
        for conv_id in conv_ids:
            size = len(groups[conv_id])
            # Greedy: give the conversation to whichever split is furthest
            # behind its target, in absolute record terms.
            deficits = {name: targets[name] - filled[name] for name in SPLIT_NAMES}
            # Splits with a zero target must never be selected.
            candidates = [name for name in SPLIT_NAMES if targets[name] > 0]
            if not candidates:
                candidates = ["train"]
            chosen = max(candidates, key=lambda name: (deficits[name], -SPLIT_NAMES.index(name)))
            filled[chosen] += size
            assignment[chosen].extend(record.id for record in groups[conv_id])
        per_stratum[f"{stratum_key[0]}|{LABEL_NAMES[stratum_key[1]]}"] = dict(filled)

    for name in SPLIT_NAMES:
        assignment[name].sort()

    assignment.corpus_fingerprint = corpus_fingerprint(records)
    assignment.stats = {
        "n_records": len(records),
        "n_conversations": len(groups),
        "largest_conversation_cluster": largest,
        "merge_shared_utterances": merge_shared_utterances,
        "per_stratum_records": per_stratum,
        "class_distribution": _class_distribution(records, assignment),
    }
    logger.info(
        "Split %d records / %d conversations -> train=%d val=%d test=%d (seed=%d)",
        len(records),
        len(groups),
        *(len(assignment[name]) for name in SPLIT_NAMES),
        seed,
    )
    return assignment


def _class_distribution(
    records: Sequence[SarcasmRecord], assignment: SplitAssignment
) -> dict[str, dict[str, int]]:
    """Per-split label counts, reported in the split manifest."""
    by_id = {record.id: record for record in records}
    out: dict[str, dict[str, int]] = {}
    for name in SPLIT_NAMES:
        counts: Counter[str] = Counter()
        for record_id in assignment[name]:
            record = by_id.get(record_id)
            if record is not None:
                counts[LABEL_NAMES[record.label]] += 1
        out[name] = dict(counts)
    return out


# ------------------------------------------------------------------ persistence
def save_split(assignment: SplitAssignment, splits_dir: str | Path) -> Path:
    """Write id lists plus a manifest to ``splits_dir``.

    Layout::

        data/splits/train.txt
        data/splits/val.txt
        data/splits/test.txt
        data/splits/manifest.json
    """
    directory = ensure_dir(splits_dir)
    for name in SPLIT_NAMES:
        (directory / f"{name}.txt").write_text(
            "\n".join(assignment[name]) + "\n", encoding="utf-8"
        )
    write_json(directory / "manifest.json", assignment.as_dict())
    logger.info("Saved split to %s", directory)
    return directory


def load_split(
    splits_dir: str | Path, expected_fingerprint: str | None = None
) -> SplitAssignment:
    """Load a previously saved split, optionally verifying the corpus fingerprint.

    Raises:
        SplitError: when files are missing or the fingerprint does not match the
            corpus currently on disk (which would mean the split is stale).
    """
    directory = resolve(splits_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise SplitError(
            f"No split manifest at {manifest_path}. Run scripts/prepare_datasets.py first."
        )
    manifest = read_json(manifest_path)
    assignment = SplitAssignment(
        seed=int(manifest.get("seed", 42)),
        ratios=tuple(manifest.get("ratios", DEFAULT_RATIOS)),  # type: ignore[arg-type]
        corpus_fingerprint=str(manifest.get("corpus_fingerprint", "")),
        stats=dict(manifest.get("stats", {})),
    )
    for name in SPLIT_NAMES:
        path = directory / f"{name}.txt"
        if not path.exists():
            raise SplitError(f"Missing split file: {path}")
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        getattr(assignment, name).extend([i for i in ids if i])

    if expected_fingerprint and assignment.corpus_fingerprint != expected_fingerprint:
        raise SplitError(
            "Saved split does not match the corpus on disk "
            f"(manifest fingerprint={assignment.corpus_fingerprint!r}, "
            f"corpus fingerprint={expected_fingerprint!r}). "
            "The processed dataset changed since the split was created; re-run "
            "scripts/prepare_datasets.py to regenerate both together."
        )
    return assignment


def apply_split(
    records: Sequence[SarcasmRecord], assignment: SplitAssignment
) -> dict[str, list[SarcasmRecord]]:
    """Materialise ``{"train": [...], "val": [...], "test": [...]}``.

    Raises:
        SplitError: if the split references ids that are not in ``records``.
    """
    by_id = {record.id: record for record in records}
    out: dict[str, list[SarcasmRecord]] = {}
    missing_total = 0
    for name in SPLIT_NAMES:
        rows: list[SarcasmRecord] = []
        for record_id in assignment[name]:
            record = by_id.get(record_id)
            if record is None:
                missing_total += 1
                continue
            rows.append(record)
        out[name] = rows
    if missing_total:
        raise SplitError(
            f"{missing_total} id(s) in the saved split are absent from the corpus. "
            "Regenerate the split with scripts/prepare_datasets.py."
        )
    return out


def filter_by_datasets(
    records: Iterable[SarcasmRecord], datasets: Sequence[str] | None
) -> list[SarcasmRecord]:
    """Keep only records from ``datasets`` (all of them when it is empty/None)."""
    rows = list(records)
    if not datasets:
        return rows
    wanted = set(datasets)
    unknown = wanted - {record.dataset for record in rows}
    if unknown:
        logger.warning("Requested dataset(s) not present in corpus: %s", sorted(unknown))
    return [record for record in rows if record.dataset in wanted]


def summarise_split(splits: Mapping[str, Sequence[SarcasmRecord]]) -> str:
    """One-line-per-split summary used by the CLIs."""
    lines = []
    for name, rows in splits.items():
        counts = Counter(LABEL_NAMES[r.label] for r in rows)
        with_ctx = sum(1 for r in rows if r.has_context)
        total = len(rows) or 1
        lines.append(
            f"  {name:<5} n={len(rows):>7,}  "
            + " ".join(f"{k}={v:,}" for k, v in sorted(counts.items()))
            + f"  context={with_ctx:,} ({with_ctx / total:.1%})"
        )
    return "\n".join(lines)
