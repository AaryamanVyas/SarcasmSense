"""Sentence embedding backends for commonsense retrieval.

Primary backend: **Sentence-Transformers ``all-MiniLM-L6-v2``**.

Why that model:

* 384-dimensional, 6-layer, ~22M parameters -- it encodes a batch of short facts
  in milliseconds on a laptop CPU, which matters because the project targets
  ``<300 ms`` end-to-end CPU inference;
* trained with a contrastive objective on >1B sentence pairs, so cosine
  similarity in its space is directly meaningful (unlike raw BERT ``[CLS]``);
* it is the de-facto default in the retrieval literature and in the
  Sentence-Transformers documentation, so the choice is easy to defend and easy
  for a reader to reproduce.

Fallback backend: a deterministic **hashed character/word n-gram encoder**. It
requires no download and no network, so tests, CI and air-gapped machines still
exercise the *entire* retrieval pipeline. Its retrieval quality is lexical, not
semantic -- markedly weaker than MiniLM -- so it announces itself loudly through
:attr:`SentenceEncoder.is_fallback` and every downstream artefact records which
backend produced it. It is never silently substituted for research runs.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "sentence-transformers", "hashing")

_TOKEN_RE = re.compile(r"[a-z0-9']+")


class EncoderUnavailableError(RuntimeError):
    """Raised when a specifically requested backend cannot be constructed."""


#: Substring of the warning sentence-transformers logs when it falls back to
#: building an untrained mean-pooling model instead of loading a real checkpoint.
_DEGENERATE_WARNING = "creating a new one with mean pooling"


class _DegenerateModelDetector(logging.Handler):
    """Context manager that notices the silent untrained-model fallback.

    ``sentence-transformers`` reports this situation only as a WARNING on its
    own logger and then returns a perfectly usable-looking object. Capturing
    that record is the only reliable way to distinguish "loaded a real
    checkpoint" from "invented one", so we attach a handler for the duration of
    the load and inspect it afterwards.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.triggered = False
        self._logger = logging.getLogger("sentence_transformers.SentenceTransformer")

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            if _DEGENERATE_WARNING in record.getMessage().lower():
                self.triggered = True
        except Exception:  # noqa: BLE001 # pragma: no cover
            pass

    def __enter__(self) -> "_DegenerateModelDetector":
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._logger.removeHandler(self)


class SentenceEncoder(ABC):
    """Common interface for anything that maps text to a dense vector."""

    #: Backend key, one of :data:`BACKENDS` (never ``"auto"``).
    backend: str = "abstract"
    #: Human-readable model identifier recorded in checkpoints and API metadata.
    name: str = "abstract"
    #: True for backends whose quality is materially below the research default.
    is_fallback: bool = False

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimensionality."""

    @abstractmethod
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode ``texts`` into an L2-normalised ``float32`` array ``[N, D]``.

        Normalisation is done here (not by the caller) so that a FAISS inner
        product index computes exact cosine similarity.
        """

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string; returns a 1-D vector of length :attr:`dimension`."""
        return self.encode([text])[0]

    def describe(self) -> dict[str, object]:
        """Metadata recorded alongside every index and every API response."""
        return {
            "backend": self.backend,
            "name": self.name,
            "dimension": self.dimension,
            "is_fallback": self.is_fallback,
        }


class SentenceTransformerEncoder(SentenceEncoder):
    """Production backend built on ``sentence-transformers``."""

    backend = "sentence-transformers"
    is_fallback = False

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        cache_folder: str | None = None,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model id or a local directory.
            device: ``"cpu"`` / ``"cuda"``; ``None`` lets the library decide.
            cache_folder: Where to cache downloaded weights.

        Raises:
            EncoderUnavailableError: if the library or the weights cannot be
                loaded (no network, model not cached, ...).
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EncoderUnavailableError(
                "sentence-transformers is not installed. "
                "pip install -r requirements.txt"
            ) from exc
        with _DegenerateModelDetector() as detector:
            try:
                self._model = SentenceTransformer(
                    model_name, device=device, cache_folder=cache_folder
                )
            except Exception as exc:  # noqa: BLE001
                raise EncoderUnavailableError(
                    f"Could not load sentence-transformers model {model_name!r}: {exc}"
                ) from exc
        if detector.triggered:
            # sentence-transformers does NOT raise when a model id has no
            # ST configuration: it quietly wraps the id in a fresh Transformer
            # + mean-pooling stack, which for an unavailable checkpoint means
            # *randomly initialised weights*. Retrieval would then return
            # confident-looking nonsense. Refuse it explicitly.
            raise EncoderUnavailableError(
                f"sentence-transformers could not find a real checkpoint for "
                f"{model_name!r} and silently constructed an untrained "
                f"mean-pooling model. Refusing to use it. Check your network "
                f"access to huggingface.co, or pre-download the model with "
                f"`huggingface-cli download {model_name}`."
            )
        self.name = model_name
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        logger.info(
            "Loaded sentence encoder %s (dim=%d, device=%s)",
            model_name,
            self._dimension,
            self._model.device,
        )

    @property
    def dimension(self) -> int:  # noqa: D102
        return self._dimension

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:  # noqa: D102
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.ascontiguousarray(vectors.astype(np.float32))


class HashingEncoder(SentenceEncoder):
    """Deterministic, download-free lexical encoder (fallback / offline mode).

    Each text is projected into a fixed-width vector by hashing its word
    unigrams, word bigrams and character 4-grams into buckets with sub-linear
    term weighting. The result behaves like a hashed TF-IDF vector: cosine
    similarity captures lexical overlap.

    This is a *real* retrieval function -- it returns genuine facts from the
    knowledge base, ranked by a genuine similarity score -- but it has no
    semantic generalisation. Use it for tests, CI and offline demos only.
    """

    backend = "hashing"
    is_fallback = True

    def __init__(self, dimension: int = 384, char_ngram: int = 4) -> None:
        """
        Args:
            dimension: Output width. 384 matches MiniLM so that an index built
                with one backend has the same shape as the other.
            char_ngram: Character n-gram size, giving robustness to typos and
                morphology.
        """
        if dimension < 16:
            raise ValueError("HashingEncoder dimension must be >= 16")
        self._dimension = dimension
        self._char_ngram = char_ngram
        self.name = f"hashing-{dimension}d"

    @property
    def dimension(self) -> int:  # noqa: D102
        return self._dimension

    def _features(self, text: str) -> list[str]:
        """Extract the feature strings that get hashed into buckets."""
        lowered = text.lower()
        words = _TOKEN_RE.findall(lowered)
        features = list(words)
        features.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
        squashed = " ".join(words)
        n = self._char_ngram
        features.extend(squashed[i : i + n] for i in range(max(0, len(squashed) - n + 1)))
        return features

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:  # noqa: D102
        out = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for feature in self._features(text or ""):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                bucket = value % self._dimension
                # Signed hashing keeps unrelated collisions from always adding.
                sign = 1.0 if (value >> 63) & 1 else -1.0
                counts[bucket] = counts.get(bucket, 0.0) + sign
            for bucket, value in counts.items():
                # Sub-linear (TF-style) scaling stops long facts dominating.
                out[row, bucket] = np.sign(value) * np.log1p(abs(value))
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


def build_sentence_encoder(
    backend: str = "auto",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
    cache_folder: str | None = None,
    dimension: int = 384,
) -> SentenceEncoder:
    """Construct a sentence encoder.

    Args:
        backend: ``"auto"`` tries Sentence-Transformers and degrades to the
            hashing encoder with a warning. ``"sentence-transformers"`` and
            ``"hashing"`` force a specific backend.
        model_name: Model id for the Sentence-Transformers backend.
        device: Torch device string.
        cache_folder: Weight cache directory.
        dimension: Width for the hashing backend.

    Raises:
        ValueError: on an unknown backend name.
        EncoderUnavailableError: when a *specifically requested* backend fails.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEncoder(
                model_name=model_name, device=device, cache_folder=cache_folder
            )
        except EncoderUnavailableError:
            if backend == "sentence-transformers":
                raise
            logger.warning(
                "Sentence-Transformers backend unavailable (model %r could not be "
                "loaded). Falling back to the deterministic hashing encoder. "
                "Retrieval will be LEXICAL, not semantic -- do not report research "
                "results produced in this mode.",
                model_name,
            )
    return HashingEncoder(dimension=dimension)
