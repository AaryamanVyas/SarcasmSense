"""FAISS-backed nearest-neighbour index over the commonsense knowledge base.

The index stores L2-normalised embeddings in an **inner-product** index, so the
returned score *is* cosine similarity in ``[-1, 1]`` -- no post-hoc conversion,
and the number shown in the UI is directly interpretable.

Two index types are supported:

``flat`` (:class:`faiss.IndexFlatIP`)
    Exact search. Chosen automatically for knowledge bases below
    :data:`IVF_THRESHOLD` facts, which covers the WordNet and sample indices and
    is fast enough there (a few hundred microseconds for 100k facts).

``ivf`` (:class:`faiss.IndexIVFFlat`)
    Approximate search with an inverted file. Chosen automatically for large
    ConceptNet indices (millions of facts), where exact search would blow the
    latency budget. ``nlist``/``nprobe`` default to values that keep recall high
    while giving an order-of-magnitude speed-up.

Everything needed to reuse an index -- vectors, facts, encoder identity,
dimension, build parameters -- is written to disk together, and
:meth:`KnowledgeIndex.load` refuses to load an index that was built with a
different encoder than the one now in use. That check is what stops a silent
"embeddings from model A queried with model B" failure, which produces
plausible-looking but meaningless evidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.retrieval.knowledge_base import KnowledgeFact, read_facts, write_facts
from src.utils.io import read_json, write_json
from src.utils.paths import ensure_dir, resolve

logger = logging.getLogger(__name__)

INDEX_FILENAME = "knowledge.faiss"
FACTS_FILENAME = "facts.jsonl"
META_FILENAME = "index_meta.json"
EMBEDDINGS_FILENAME = "embeddings.npy"

#: Above this many facts we switch from exact to IVF search.
IVF_THRESHOLD = 200_000


class IndexError_(RuntimeError):
    """Raised for index build/load failures."""


@dataclass
class IndexMetadata:
    """Everything needed to validate and describe a built index."""

    n_facts: int
    dimension: int
    index_type: str
    encoder_backend: str
    encoder_name: str
    encoder_is_fallback: bool
    sources: dict[str, int]
    nlist: int | None = None
    nprobe: int | None = None
    is_sample_index: bool = False
    build_notes: str = ""

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view."""
        from dataclasses import asdict

        return asdict(self)


def _import_faiss():
    """Import FAISS with an actionable error message."""
    try:
        import faiss

        return faiss
    except ImportError as exc:  # pragma: no cover
        raise IndexError_(
            "faiss is not installed. Install it with:  pip install faiss-cpu==1.9.0.post1"
        ) from exc


class KnowledgeIndex:
    """A FAISS index plus the facts it indexes.

    Attributes:
        facts: The indexed facts, positionally aligned with the FAISS ids.
        metadata: Provenance and build parameters.
    """

    def __init__(
        self,
        index,  # faiss.Index -- untyped to keep faiss an optional import
        facts: Sequence[KnowledgeFact],
        metadata: IndexMetadata,
        embeddings: np.ndarray | None = None,
    ) -> None:
        self._index = index
        self.facts = list(facts)
        self.metadata = metadata
        self._embeddings = embeddings

    # -------------------------------------------------------------- properties
    def __len__(self) -> int:
        """Number of indexed facts."""
        return len(self.facts)

    @property
    def dimension(self) -> int:
        """Embedding dimensionality of the index."""
        return self.metadata.dimension

    @property
    def is_empty(self) -> bool:
        """Whether the index holds no facts (retrieval must degrade gracefully)."""
        return len(self.facts) == 0

    # ------------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        facts: Sequence[KnowledgeFact],
        encoder,  # SentenceEncoder
        index_type: str = "auto",
        batch_size: int = 256,
        nlist: int | None = None,
        nprobe: int = 16,
        show_progress: bool = True,
        is_sample_index: bool = False,
        build_notes: str = "",
        store_embeddings: bool = False,
    ) -> "KnowledgeIndex":
        """Embed ``facts`` and build a FAISS index over them.

        Args:
            facts: Knowledge facts to index.
            encoder: A :class:`~src.retrieval.encoder.SentenceEncoder`.
            index_type: ``"auto"``, ``"flat"`` or ``"ivf"``.
            batch_size: Encoder batch size.
            nlist: IVF cluster count; defaults to ``4 * sqrt(n)`` clipped to a
                sane range, the usual FAISS rule of thumb.
            nprobe: IVF clusters probed at query time (recall/speed trade-off).
            show_progress: Show the encoder progress bar.
            is_sample_index: Marks a smoke-test index so downstream code and the
                API can warn that it is not a research knowledge base.
            build_notes: Free-text note stored in the metadata.
            store_embeddings: Also persist the raw embedding matrix (useful for
                analysis; roughly ``n * dim * 4`` bytes).

        Raises:
            IndexError_: if ``facts`` is empty.
        """
        faiss = _import_faiss()
        if not facts:
            raise IndexError_("Cannot build an index over zero facts")

        texts = [fact.text for fact in facts]
        logger.info("Encoding %d facts with %s ...", len(texts), encoder.name)
        embeddings = encoder.encode(texts, batch_size=batch_size, show_progress=show_progress)
        if embeddings.shape[0] != len(facts):
            raise IndexError_(
                f"encoder returned {embeddings.shape[0]} vectors for {len(facts)} facts"
            )
        dimension = int(embeddings.shape[1])

        resolved_type = index_type
        if index_type == "auto":
            resolved_type = "ivf" if len(facts) >= IVF_THRESHOLD else "flat"

        if resolved_type == "flat":
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            used_nlist = used_nprobe = None
        elif resolved_type == "ivf":
            used_nlist = nlist or int(np.clip(4 * np.sqrt(len(facts)), 64, 65536))
            # FAISS requires at least ~39 training points per centroid.
            used_nlist = int(min(used_nlist, max(1, len(facts) // 39)))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(
                quantizer, dimension, used_nlist, faiss.METRIC_INNER_PRODUCT
            )
            logger.info("Training IVF index (nlist=%d) ...", used_nlist)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = nprobe
            used_nprobe = nprobe
        else:
            raise IndexError_(f"index_type must be auto/flat/ivf, got {index_type!r}")

        sources: dict[str, int] = {}
        for fact in facts:
            sources[fact.source] = sources.get(fact.source, 0) + 1

        metadata = IndexMetadata(
            n_facts=len(facts),
            dimension=dimension,
            index_type=resolved_type,
            encoder_backend=encoder.backend,
            encoder_name=encoder.name,
            encoder_is_fallback=bool(encoder.is_fallback),
            sources=sources,
            nlist=used_nlist,
            nprobe=used_nprobe,
            is_sample_index=is_sample_index,
            build_notes=build_notes,
        )
        logger.info(
            "Built %s index: %d facts, dim=%d, sources=%s",
            resolved_type,
            len(facts),
            dimension,
            sources,
        )
        return cls(
            index,
            facts,
            metadata,
            embeddings=embeddings if store_embeddings else None,
        )

    # -------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        """Write index, facts and metadata into ``directory``."""
        faiss = _import_faiss()
        target = ensure_dir(directory)
        faiss.write_index(self._index, str(target / INDEX_FILENAME))
        write_facts(target / FACTS_FILENAME, self.facts)
        write_json(target / META_FILENAME, self.metadata.as_dict())
        if self._embeddings is not None:
            np.save(target / EMBEDDINGS_FILENAME, self._embeddings)
        logger.info("Saved knowledge index -> %s", target)
        return target

    @classmethod
    def load(
        cls,
        directory: str | Path,
        expected_encoder_name: str | None = None,
        strict: bool = True,
    ) -> "KnowledgeIndex":
        """Load an index previously written by :meth:`save`.

        Args:
            directory: Directory containing ``knowledge.faiss`` etc.
            expected_encoder_name: When given, verify the index was built with
                this encoder.
            strict: Raise on encoder mismatch (``True``) or only warn.

        Raises:
            IndexError_: if files are missing or the encoder does not match.
        """
        faiss = _import_faiss()
        source_dir = resolve(directory)
        index_path = source_dir / INDEX_FILENAME
        facts_path = source_dir / FACTS_FILENAME
        meta_path = source_dir / META_FILENAME
        for path in (index_path, facts_path, meta_path):
            if not path.exists():
                raise IndexError_(
                    f"Knowledge index is incomplete: {path} not found. "
                    f"Build it with:  python scripts/build_knowledge_index.py"
                )

        index = faiss.read_index(str(index_path))
        facts = read_facts(facts_path)
        raw_meta = read_json(meta_path)
        metadata = IndexMetadata(**raw_meta)

        if metadata.n_facts != len(facts):
            raise IndexError_(
                f"Index metadata says {metadata.n_facts} facts but {facts_path} "
                f"holds {len(facts)}; the index directory is inconsistent."
            )
        if expected_encoder_name and metadata.encoder_name != expected_encoder_name:
            message = (
                f"Knowledge index was built with encoder {metadata.encoder_name!r} "
                f"but the current configuration uses {expected_encoder_name!r}. "
                f"Querying an index with a different encoder yields meaningless "
                f"neighbours. Rebuild the index or change retrieval.encoder_name."
            )
            if strict:
                raise IndexError_(message)
            logger.warning(message)

        if metadata.index_type == "ivf" and metadata.nprobe:
            index.nprobe = metadata.nprobe
        logger.info(
            "Loaded knowledge index from %s (%d facts, encoder=%s)",
            source_dir,
            len(facts),
            metadata.encoder_name,
        )
        return cls(index, facts, metadata)

    # ------------------------------------------------------------------ search
    def search(
        self, query_vectors: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, indices)`` for each query row.

        ``indices`` may contain ``-1`` when fewer than ``top_k`` neighbours
        exist; callers must filter those out.
        """
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        if query_vectors.shape[1] != self.dimension:
            raise IndexError_(
                f"query dimension {query_vectors.shape[1]} != index dimension "
                f"{self.dimension}"
            )
        if self.is_empty or top_k <= 0:
            n = query_vectors.shape[0]
            return (
                np.zeros((n, 0), dtype=np.float32),
                np.zeros((n, 0), dtype=np.int64),
            )
        k = min(top_k, len(self.facts))
        scores, indices = self._index.search(
            np.ascontiguousarray(query_vectors.astype(np.float32)), k
        )
        return scores, indices

    def reconstruct(self, ids: Sequence[int]) -> np.ndarray:
        """Return the stored embedding vectors for ``ids`` as ``[len(ids), D]``.

        Used by the model to obtain evidence vectors without re-encoding the
        fact text on every forward pass. IVF indices need an explicit direct
        map before individual vectors can be reconstructed; it is built lazily
        and cached.
        """
        if not len(ids):
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._embeddings is not None:
            return np.ascontiguousarray(
                self._embeddings[np.asarray(ids, dtype=np.int64)].astype(np.float32)
            )
        if self.metadata.index_type == "ivf" and not getattr(self, "_direct_map_ready", False):
            try:
                self._index.make_direct_map()
            except Exception as exc:  # noqa: BLE001 # pragma: no cover
                raise IndexError_(
                    f"Cannot reconstruct vectors from this IVF index: {exc}"
                ) from exc
            self._direct_map_ready = True
        out = np.zeros((len(ids), self.dimension), dtype=np.float32)
        for row, fact_id in enumerate(ids):
            if 0 <= int(fact_id) < len(self.facts):
                out[row] = self._index.reconstruct(int(fact_id))
        return out

    def fact_at(self, position: int) -> KnowledgeFact | None:
        """Look up a fact by FAISS id, tolerating the ``-1`` padding value."""
        if position < 0 or position >= len(self.facts):
            return None
        return self.facts[position]

    def describe(self) -> dict[str, object]:
        """Metadata dict surfaced by ``GET /model/info``."""
        return self.metadata.as_dict()


def index_exists(directory: str | Path) -> bool:
    """Whether ``directory`` contains a complete, loadable index."""
    base = resolve(directory)
    return all(
        (base / name).exists() for name in (INDEX_FILENAME, FACTS_FILENAME, META_FILENAME)
    )


def peek_metadata(directory: str | Path) -> IndexMetadata | None:
    """Read index metadata without loading the (potentially large) index."""
    base = resolve(directory)
    meta_path = base / META_FILENAME
    if not meta_path.exists():
        return None
    try:
        return IndexMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError) as exc:  # pragma: no cover
        logger.warning("Could not parse %s: %s", meta_path, exc)
        return None
