"""Pre-computed retrieval cache.

Retrieval is deterministic for a fixed (index, encoder, query) triple, so
running it inside every training epoch would repeat identical work. This module
computes the top-k evidence **once** for the whole corpus and stores it as a
compact ``.npz``:

* ``record_ids``     -- ``[N]`` string array
* ``embedding_ids``  -- ``[N, K]`` int32, ``-1`` marks an empty slot
* ``scores``         -- ``[N, K]`` float32
* metadata JSON      -- index fingerprint, encoder identity, top-k, query mode

The metadata is checked on load: a cache built with a different encoder, a
different index or a different ``top_k`` is rejected rather than silently reused,
because stale evidence vectors would quietly change what the model is trained on.

Typical cost: ~4 seconds for 10k examples with MiniLM on CPU, versus ~4 seconds
*per epoch* without the cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.schema import SarcasmRecord
from src.retrieval.retriever import CommonsenseRetriever, RetrievalResult
from src.utils.paths import ensure_parent, resolve

logger = logging.getLogger(__name__)


class CacheMismatchError(RuntimeError):
    """Raised when a cache does not match the current retrieval configuration."""


@dataclass
class RetrievalCacheMeta:
    """Identity of the configuration a cache was built with."""

    top_k: int
    query_mode: str
    encoder_name: str
    encoder_backend: str
    index_fingerprint: str
    dimension: int
    n_records: int

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view."""
        from dataclasses import asdict

        return asdict(self)


def index_fingerprint(retriever: CommonsenseRetriever) -> str:
    """Short hash identifying the exact knowledge index in use."""
    if retriever.index is None:
        return "none"
    meta = retriever.index.metadata
    payload = json.dumps(
        {
            "n_facts": meta.n_facts,
            "dimension": meta.dimension,
            "index_type": meta.index_type,
            "encoder_name": meta.encoder_name,
            "sources": meta.sources,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class RetrievalCache:
    """Random-access store of pre-computed top-k evidence per record id."""

    def __init__(
        self,
        record_ids: Sequence[str],
        embedding_ids: np.ndarray,
        scores: np.ndarray,
        meta: RetrievalCacheMeta,
    ) -> None:
        self.meta = meta
        self.embedding_ids = embedding_ids.astype(np.int32)
        self.scores = scores.astype(np.float32)
        self._position = {record_id: i for i, record_id in enumerate(record_ids)}

    def __len__(self) -> int:
        """Number of cached records."""
        return len(self._position)

    def __contains__(self, record_id: str) -> bool:
        """Whether a record id is cached."""
        return record_id in self._position

    def get(self, record_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(embedding_ids, scores)`` for one record.

        Missing records yield empty arrays, which downstream code treats exactly
        like a retrieval failure -- an all-``False`` evidence mask.
        """
        position = self._position.get(record_id)
        if position is None:
            return (
                np.full(self.meta.top_k, -1, dtype=np.int32),
                np.zeros(self.meta.top_k, dtype=np.float32),
            )
        return self.embedding_ids[position], self.scores[position]

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        records: Sequence[SarcasmRecord],
        retriever: CommonsenseRetriever,
        batch_size: int = 64,
        progress_every: int = 2000,
    ) -> "RetrievalCache":
        """Run retrieval over every record and collect the results."""
        if retriever.index is None or retriever.encoder is None:
            raise CacheMismatchError(
                "Cannot build a retrieval cache without a loaded index and encoder"
            )
        k = retriever.top_k
        n = len(records)
        embedding_ids = np.full((n, k), -1, dtype=np.int32)
        scores = np.zeros((n, k), dtype=np.float32)

        for start in range(0, n, batch_size):
            chunk = records[start : start + batch_size]
            results: list[RetrievalResult] = retriever.retrieve_batch(
                [record.text for record in chunk],
                [record.context for record in chunk],
            )
            for offset, result in enumerate(results):
                for slot, evidence in enumerate(result.evidence[:k]):
                    embedding_ids[start + offset, slot] = evidence.embedding_id
                    scores[start + offset, slot] = evidence.score
            if progress_every and start and start % progress_every == 0:
                logger.info("Retrieval cache: %d/%d records", start, n)

        meta = RetrievalCacheMeta(
            top_k=k,
            query_mode=retriever.query_mode,
            encoder_name=retriever.encoder.name,
            encoder_backend=retriever.encoder.backend,
            index_fingerprint=index_fingerprint(retriever),
            dimension=retriever.dimension,
            n_records=n,
        )
        hit_rate = float((embedding_ids >= 0).any(axis=1).mean()) if n else 0.0
        logger.info(
            "Built retrieval cache for %d records (k=%d, hit rate %.1f%%)",
            n,
            k,
            100 * hit_rate,
        )
        return cls([record.id for record in records], embedding_ids, scores, meta)

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Write the cache to a ``.npz`` file."""
        target = ensure_parent(path)
        np.savez_compressed(
            target,
            record_ids=np.array(list(self._position.keys()), dtype=object),
            embedding_ids=self.embedding_ids,
            scores=self.scores,
            meta=json.dumps(self.meta.as_dict()),
        )
        logger.info("Saved retrieval cache -> %s", target)
        return target

    @classmethod
    def load(
        cls, path: str | Path, retriever: CommonsenseRetriever | None = None
    ) -> "RetrievalCache":
        """Load a cache, verifying it matches ``retriever`` when one is given.

        Raises:
            FileNotFoundError: when the cache file does not exist.
            CacheMismatchError: when the cache was built with a different
                encoder, index or ``top_k``.
        """
        source = resolve(path)
        if not source.exists():
            raise FileNotFoundError(f"Retrieval cache not found: {source}")
        payload = np.load(source, allow_pickle=True)
        meta = RetrievalCacheMeta(**json.loads(str(payload["meta"])))
        cache = cls(
            [str(x) for x in payload["record_ids"].tolist()],
            payload["embedding_ids"],
            payload["scores"],
            meta,
        )
        if retriever is not None:
            cache.verify(retriever)
        return cache

    def verify(self, retriever: CommonsenseRetriever) -> None:
        """Raise :class:`CacheMismatchError` if the cache is stale."""
        if retriever.encoder is None:
            raise CacheMismatchError("Retriever has no encoder to verify against")
        problems = []
        if self.meta.top_k != retriever.top_k:
            problems.append(f"top_k {self.meta.top_k} != {retriever.top_k}")
        if self.meta.encoder_name != retriever.encoder.name:
            problems.append(
                f"encoder {self.meta.encoder_name!r} != {retriever.encoder.name!r}"
            )
        current = index_fingerprint(retriever)
        if self.meta.index_fingerprint != current:
            problems.append(f"index fingerprint {self.meta.index_fingerprint} != {current}")
        if self.meta.query_mode != retriever.query_mode:
            problems.append(
                f"query_mode {self.meta.query_mode!r} != {retriever.query_mode!r}"
            )
        if problems:
            raise CacheMismatchError(
                "Retrieval cache does not match the current configuration: "
                + "; ".join(problems)
                + ". Delete the cache file so it is rebuilt."
            )


def ensure_retrieval_cache(
    records: Sequence[SarcasmRecord],
    retriever: CommonsenseRetriever,
    path: str | Path,
    rebuild: bool = False,
) -> RetrievalCache:
    """Load the cache if it is valid and complete, otherwise rebuild and save it."""
    target = resolve(path)
    if target.exists() and not rebuild:
        try:
            cache = RetrievalCache.load(target, retriever=retriever)
        except (CacheMismatchError, ValueError, KeyError) as exc:
            logger.warning("Rebuilding retrieval cache: %s", exc)
        else:
            missing = sum(1 for record in records if record.id not in cache)
            if missing == 0:
                logger.info("Reusing retrieval cache %s (%d records)", target, len(cache))
                return cache
            logger.warning(
                "Retrieval cache is missing %d/%d records; rebuilding.",
                missing,
                len(records),
            )
    cache = RetrievalCache.build(records, retriever)
    cache.save(target)
    return cache
