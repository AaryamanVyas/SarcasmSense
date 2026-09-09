"""Base class shared by every dataset adapter.

An adapter's only job is: *read whatever shape this corpus happens to have on
disk, and yield validated* :class:`~src.data.schema.SarcasmRecord` *objects*.
Adapters never write files, never download anything, and never decide on
splits -- those concerns live in ``scripts/`` and ``src/data/splitting.py``.
"""

from __future__ import annotations

import abc
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.data.preprocessing import (
    PreprocessingOptions,
    clean_context,
    clean_text,
    coerce_label,
    is_valid_text,
)
from src.data.schema import SarcasmRecord
from src.data.sources import DatasetSource
from src.utils.paths import resolve

logger = logging.getLogger(__name__)


@dataclass
class LoadStats:
    """Counters describing what an adapter did, surfaced by the CLI report."""

    seen: int = 0
    emitted: int = 0
    #: When a ``max_rows`` cap was applied, how many usable records existed
    #: before subsampling. ``0`` when no cap was applied.
    sampled_from: int = 0
    dropped_empty_text: int = 0
    dropped_invalid_label: int = 0
    dropped_malformed: int = 0
    with_context: int = 0
    context_turn_histogram: dict[int, int] = field(default_factory=dict)

    def note_context(self, n_turns: int) -> None:
        """Record how many context turns a successfully emitted record had."""
        if n_turns:
            self.with_context += 1
        self.context_turn_histogram[n_turns] = (
            self.context_turn_histogram.get(n_turns, 0) + 1
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view for the acquisition report."""
        return {
            "seen": self.seen,
            "emitted": self.emitted,
            "sampled_from": self.sampled_from,
            "dropped_empty_text": self.dropped_empty_text,
            "dropped_invalid_label": self.dropped_invalid_label,
            "dropped_malformed": self.dropped_malformed,
            "with_context": self.with_context,
            "context_turn_histogram": dict(sorted(self.context_turn_histogram.items())),
        }


class DatasetAdapter(abc.ABC):
    """Convert one raw corpus into the common schema.

    Subclasses implement :meth:`iter_raw`, which yields ``(local_id, payload)``
    pairs, and :meth:`to_record`, which maps one payload onto a
    :class:`SarcasmRecord` (or ``None`` to drop the row).
    """

    #: Populated by each concrete subclass.
    source: DatasetSource

    def __init__(
        self,
        raw_dir: str | Path,
        options: PreprocessingOptions | None = None,
        max_context_turns: int = 3,
        max_rows: int | None = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            raw_dir: ``data/raw`` -- the adapter appends its own ``<key>/``.
            options: Preprocessing switches.
            max_context_turns: Truncate context to the most recent N turns.
            max_rows: Optional cap, used to subsample very large corpora (SARC).
                The cap is a **seeded random sample**, not a head slice -- see
                :meth:`load`.
            seed: Controls the subsample when ``max_rows`` is set.
        """
        self.raw_dir = resolve(raw_dir) / self.source.key
        self.options = options or PreprocessingOptions()
        self.max_context_turns = max_context_turns
        self.max_rows = max_rows
        self.seed = seed
        self.stats = LoadStats()

    # ------------------------------------------------------------- discovery
    @property
    def key(self) -> str:
        """The dataset key, e.g. ``"mustard"``."""
        return self.source.key

    def expected_paths(self) -> list[Path]:
        """Absolute paths this adapter needs in order to load."""
        names = [f.relative_path for f in self.source.remote_files] + list(
            self.source.manual_files
        )
        return [self.raw_dir / name for name in names]

    def available(self) -> bool:
        """Whether at least one acceptable input file is present on disk."""
        return any(path.exists() for path in self.expected_paths())

    def missing_paths(self) -> list[Path]:
        """Expected files that are not present."""
        return [path for path in self.expected_paths() if not path.exists()]

    # ---------------------------------------------------------------- loading
    @abc.abstractmethod
    def iter_raw(self) -> Iterator[tuple[str, Any]]:
        """Yield ``(local_id, raw_payload)`` for each row in the source files."""

    @abc.abstractmethod
    def to_record(self, local_id: str, payload: Any) -> SarcasmRecord | None:
        """Map one raw payload onto the common schema, or ``None`` to drop it."""

    def load(self) -> Iterator[SarcasmRecord]:
        """Yield validated records, updating :attr:`stats` as a side effect.

        When :attr:`max_rows` is set the corpus is subsampled by **seeded
        reservoir sampling**, not by taking the first N rows. This matters for
        SARC in particular: its distributed CSV is roughly date-ordered, so a
        head slice would silently give you the earliest ~25k Reddit comments
        rather than a representative sample of the corpus. Reservoir sampling
        keeps memory at O(max_rows) regardless of file size, and the fixed seed
        keeps it reproducible.

        Sampled records are emitted in their original file order so that the
        written corpus is byte-identical across runs with the same seed.
        """
        if not self.available():
            raise FileNotFoundError(
                f"No input files found for dataset {self.key!r}. Expected one of: "
                + ", ".join(str(p) for p in self.expected_paths())
            )
        self.stats = LoadStats()
        if self.max_rows is None:
            yield from self._iter_records()
            return
        yield from self._reservoir_sample(self._iter_records(), self.max_rows)

    def _iter_records(self) -> Iterator[SarcasmRecord]:
        """Convert every raw row into a record, skipping the unusable ones."""
        for local_id, payload in self.iter_raw():
            self.stats.seen += 1
            try:
                record = self.to_record(local_id, payload)
            except Exception:  # noqa: BLE001 - one bad row must not kill the run
                logger.debug("Malformed row %s in %s", local_id, self.key, exc_info=True)
                self.stats.dropped_malformed += 1
                continue
            if record is None:
                continue
            self.stats.emitted += 1
            self.stats.note_context(len(record.context))
            yield record

    def _reservoir_sample(
        self, records: Iterator[SarcasmRecord], k: int
    ) -> Iterator[SarcasmRecord]:
        """Algorithm R: a uniform sample of ``k`` records from an unbounded stream."""
        if k <= 0:
            return
        rng = random.Random(self.seed)
        reservoir: list[tuple[int, SarcasmRecord]] = []
        for index, record in enumerate(records):
            if index < k:
                reservoir.append((index, record))
                continue
            # Each seen record has probability k/(index+1) of being kept.
            position = rng.randint(0, index)
            if position < k:
                reservoir[position] = (index, record)
        self.stats.sampled_from = self.stats.emitted
        self.stats.emitted = len(reservoir)
        # The histogram was accumulated over every record the stream produced;
        # rebuild it from the sample so the reported statistics describe the
        # corpus that is actually written to disk.
        self.stats.with_context = 0
        self.stats.context_turn_histogram = {}
        reservoir.sort(key=lambda item: item[0])
        for _index, record in reservoir:
            self.stats.note_context(len(record.context))
            yield record

    # ------------------------------------------------------------- utilities
    def _build(
        self,
        local_id: str,
        text: Any,
        label: Any,
        context: Any = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SarcasmRecord | None:
        """Shared clean-validate-construct path used by every adapter.

        Returns ``None`` (and increments the appropriate drop counter) when the
        row cannot yield a usable training example.
        """
        cleaned_text = clean_text(text, self.options)
        if not is_valid_text(cleaned_text):
            self.stats.dropped_empty_text += 1
            return None
        label_id = coerce_label(label)
        if label_id is None:
            self.stats.dropped_invalid_label += 1
            return None
        turns = clean_context(context, self.max_context_turns, self.options)
        return SarcasmRecord(
            id=f"{self.key}:{local_id}",
            text=cleaned_text,
            label=label_id,
            dataset=self.key,
            context=turns,
            domain=self.source.domain,
            language="en",
            conversation_id=conversation_id or f"{self.key}:{local_id}",
            metadata=metadata or {},
        )

    def _first_existing(self, *names: str) -> Path:
        """Return the first of ``names`` that exists under ``raw_dir``."""
        for name in names:
            candidate = self.raw_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"{self.key}: none of {names} found in {self.raw_dir}. "
            f"See scripts/download_datasets.py --help"
        )
