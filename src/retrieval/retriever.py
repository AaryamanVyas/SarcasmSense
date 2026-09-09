"""Commonsense retrieval: query construction, FAISS search, structured results.

Retrieval is **offline**: the knowledge base is embedded once by
``scripts/build_knowledge_index.py`` and afterwards only the query is embedded
at request time. No external API is contacted during inference, which is what
makes the ``<300 ms`` CPU target achievable and the system usable without
network access.

Failure is a first-class outcome. Every :class:`RetrievalResult` carries a
:class:`RetrievalStatus` and, when relevant, a human-readable warning that the
API surfaces and the UI displays. The classifier keeps working with an empty
evidence list -- it never crashes and never silently pretends evidence existed.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from src.retrieval.encoder import SentenceEncoder
from src.retrieval.faiss_index import KnowledgeIndex

logger = logging.getLogger(__name__)


class RetrievalStatus(str, Enum):
    """Why a retrieval call returned what it returned."""

    OK = "ok"
    NO_INDEX = "no_index"
    EMPTY_INDEX = "empty_index"
    NO_RESULTS_ABOVE_THRESHOLD = "no_results_above_threshold"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass(slots=True)
class RetrievedEvidence:
    """One retrieved commonsense fact, as returned by the API.

    Attributes:
        text: The verbalised fact shown to the user.
        source: ``"ConceptNet"`` / ``"ATOMIC"`` / ``"WordNet"``.
        relation: Original knowledge-graph relation.
        score: Cosine similarity to the query in ``[-1, 1]``.
        fact_id: Identifier within the knowledge base.
        embedding_id: Positional id inside the FAISS index.
        head / tail: The concepts of the underlying tuple.
    """

    text: str
    source: str
    relation: str
    score: float
    fact_id: str
    embedding_id: int
    head: str = ""
    tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return asdict(self)


@dataclass
class RetrievalResult:
    """Outcome of one retrieval call."""

    evidence: list[RetrievedEvidence] = field(default_factory=list)
    status: RetrievalStatus = RetrievalStatus.OK
    query: str = ""
    warning: str | None = None
    latency_ms: float = 0.0
    embedding_ids: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether usable evidence was returned."""
        return self.status is RetrievalStatus.OK and bool(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "evidence": [e.to_dict() for e in self.evidence],
            "status": self.status.value,
            "query": self.query,
            "warning": self.warning,
            "latency_ms": round(self.latency_ms, 2),
        }


def build_query(
    text: str, context: Sequence[str] | None = None, mode: str = "utterance_plus_context"
) -> str:
    """Build the retrieval query string.

    ``utterance_plus_context`` concatenates the target utterance with the most
    recent context turn. Using the *whole* context dilutes the query -- the
    sentence encoder averages over too much material and returns generic facts --
    while the immediately preceding turn is what establishes the situation the
    utterance reacts to, which is precisely what we want evidence about.
    """
    text = (text or "").strip()
    if mode == "utterance" or not context:
        return text
    return f"{context[-1].strip()} {text}".strip()


class CommonsenseRetriever:
    """Query the knowledge index and return structured, cited evidence."""

    def __init__(
        self,
        index: KnowledgeIndex | None,
        encoder: SentenceEncoder | None,
        top_k: int = 5,
        min_score: float = 0.0,
        query_mode: str = "utterance_plus_context",
        cache_size: int = 2048,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            index: A loaded :class:`KnowledgeIndex`, or ``None`` when the index
                is missing -- the retriever then degrades to
                :attr:`RetrievalStatus.NO_INDEX` instead of raising.
            encoder: Sentence encoder used for the *query*. Must be the same
                model that built the index.
            top_k: Number of facts to return (the proposal specifies 5).
            min_score: Cosine-similarity floor; facts below it are dropped.
            query_mode: See :func:`build_query`.
            cache_size: LRU cache entries for repeated queries (the demo UI and
                the benchmark script hit the same strings repeatedly).
            enabled: Master switch; ``False`` short-circuits to
                :attr:`RetrievalStatus.DISABLED`.
        """
        self.index = index
        self.encoder = encoder
        self.top_k = top_k
        self.min_score = min_score
        self.query_mode = query_mode
        self.enabled = enabled
        self._cache_size = max(0, cache_size)
        self._cache: OrderedDict[tuple[str, int], RetrievalResult] = OrderedDict()
        self.n_calls = 0
        self.n_failures = 0
        self.n_cache_hits = 0

    # ------------------------------------------------------------- properties
    @property
    def available(self) -> bool:
        """Whether retrieval can actually run."""
        return bool(
            self.enabled
            and self.index is not None
            and self.encoder is not None
            and not self.index.is_empty
        )

    @property
    def dimension(self) -> int:
        """Evidence-vector dimensionality (0 when unavailable)."""
        return self.index.dimension if self.index is not None else 0

    # ---------------------------------------------------------------- retrieve
    def retrieve(
        self, text: str, context: Sequence[str] | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        """Retrieve top-k commonsense evidence for one utterance."""
        return self.retrieve_batch([text], [list(context or [])], top_k=top_k)[0]

    def retrieve_batch(
        self,
        texts: Sequence[str],
        contexts: Sequence[Sequence[str]] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve for a batch of utterances.

        Batching matters: encoding 32 queries in one call is roughly an order of
        magnitude faster than 32 separate calls, which is what makes retrieval
        affordable inside the training loop.
        """
        k = self.top_k if top_k is None else top_k
        contexts = list(contexts or [[] for _ in texts])
        queries = [
            build_query(text, ctx, self.query_mode) for text, ctx in zip(texts, contexts)
        ]
        results: list[RetrievalResult | None] = [None] * len(queries)

        if not self.enabled:
            return [
                RetrievalResult(
                    status=RetrievalStatus.DISABLED,
                    query=q,
                    warning="Commonsense retrieval is disabled for this model configuration.",
                )
                for q in queries
            ]
        if self.index is None or self.encoder is None:
            self.n_failures += len(queries)
            return [
                RetrievalResult(
                    status=RetrievalStatus.NO_INDEX,
                    query=q,
                    warning=(
                        "Commonsense knowledge index is unavailable; the prediction "
                        "was generated without retrieved evidence."
                    ),
                )
                for q in queries
            ]
        if self.index.is_empty:
            self.n_failures += len(queries)
            return [
                RetrievalResult(
                    status=RetrievalStatus.EMPTY_INDEX,
                    query=q,
                    warning="Commonsense knowledge index is empty.",
                )
                for q in queries
            ]

        # 1. serve cache hits
        pending: list[int] = []
        for position, query in enumerate(queries):
            cached = self._cache_get(query, k)
            if cached is not None:
                self.n_cache_hits += 1
                results[position] = cached
            else:
                pending.append(position)

        # 2. encode + search the misses in one batch
        if pending:
            started = time.perf_counter()
            try:
                vectors = self.encoder.encode([queries[i] for i in pending])
                scores, ids = self.index.search(vectors, top_k=k)
            except Exception as exc:  # noqa: BLE001 - degrade, never crash the request
                logger.exception("Commonsense retrieval failed")
                self.n_failures += len(pending)
                for position in pending:
                    results[position] = RetrievalResult(
                        status=RetrievalStatus.ERROR,
                        query=queries[position],
                        warning=f"Commonsense retrieval failed ({type(exc).__name__}); "
                        f"the prediction was generated without retrieved evidence.",
                    )
            else:
                elapsed_ms = (time.perf_counter() - started) * 1000 / max(1, len(pending))
                for row, position in enumerate(pending):
                    result = self._materialise(
                        queries[position], scores[row], ids[row], elapsed_ms
                    )
                    self._cache_put(queries[position], k, result)
                    results[position] = result

        self.n_calls += len(queries)
        return [r for r in results if r is not None]

    def _materialise(
        self,
        query: str,
        scores: np.ndarray,
        ids: np.ndarray,
        latency_ms: float,
    ) -> RetrievalResult:
        """Turn raw FAISS output into a :class:`RetrievalResult`."""
        evidence: list[RetrievedEvidence] = []
        for score, fact_index in zip(scores.tolist(), ids.tolist()):
            if fact_index < 0:
                continue  # FAISS pads with -1 when k > index size
            if score < self.min_score:
                continue
            fact = self.index.fact_at(int(fact_index))  # type: ignore[union-attr]
            if fact is None:
                continue
            evidence.append(
                RetrievedEvidence(
                    text=fact.text,
                    source=fact.source,
                    relation=fact.relation,
                    score=round(float(score), 4),
                    fact_id=fact.id,
                    embedding_id=int(fact_index),
                    head=fact.head,
                    tail=fact.tail,
                )
            )
        if not evidence:
            self.n_failures += 1
            return RetrievalResult(
                status=RetrievalStatus.NO_RESULTS_ABOVE_THRESHOLD,
                query=query,
                latency_ms=latency_ms,
                warning=(
                    f"No commonsense fact scored above the similarity threshold "
                    f"({self.min_score:.2f}); the prediction was generated without "
                    f"retrieved evidence."
                ),
            )
        return RetrievalResult(
            evidence=evidence,
            status=RetrievalStatus.OK,
            query=query,
            latency_ms=latency_ms,
            embedding_ids=[e.embedding_id for e in evidence],
        )

    # ------------------------------------------------------------------ vectors
    def evidence_matrix(
        self, results: Sequence[RetrievalResult], top_k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(vectors, mask)`` for a batch of retrieval results.

        Args:
            results: One result per batch item.
            top_k: Slot count; defaults to :attr:`top_k`.

        Returns:
            ``vectors`` of shape ``[B, K, D]`` (zero-padded) and a boolean
            ``mask`` of shape ``[B, K]`` where ``True`` marks a real fact. The
            mask is what lets cross-attention ignore padding slots, so an
            example with two facts is not diluted by three zero vectors.
        """
        k = self.top_k if top_k is None else top_k
        dim = self.dimension or 1
        batch = len(results)
        vectors = np.zeros((batch, k, dim), dtype=np.float32)
        mask = np.zeros((batch, k), dtype=bool)
        if self.index is None:
            return vectors, mask
        for row, result in enumerate(results):
            ids = result.embedding_ids[:k]
            if not ids:
                continue
            found = self.index.reconstruct(ids)
            vectors[row, : len(ids)] = found
            mask[row, : len(ids)] = True
        return vectors, mask

    # -------------------------------------------------------------------- cache
    def _cache_get(self, query: str, k: int) -> RetrievalResult | None:
        if not self._cache_size:
            return None
        key = (query, k)
        result = self._cache.get(key)
        if result is not None:
            self._cache.move_to_end(key)
        return result

    def _cache_put(self, query: str, k: int, result: RetrievalResult) -> None:
        if not self._cache_size:
            return
        self._cache[(query, k)] = result
        self._cache.move_to_end((query, k))
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    # ----------------------------------------------------------------- metrics
    def stats(self) -> dict[str, Any]:
        """Counters for logging and ``GET /model/info``."""
        return {
            "available": self.available,
            "calls": self.n_calls,
            "failures": self.n_failures,
            "cache_hits": self.n_cache_hits,
            "cache_size": len(self._cache),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "query_mode": self.query_mode,
            "index": self.index.describe() if self.index is not None else None,
            "encoder": self.encoder.describe() if self.encoder is not None else None,
        }
