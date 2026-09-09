"""PyTorch dataset and collator.

The dataset itself is intentionally trivial -- it just hands back
:class:`~src.data.schema.SarcasmRecord` objects. All the work happens in
:class:`SarcasmCollator`, for one concrete reason: **batched tokenisation**.
Tokenising 16 texts in one ``tokenizer()`` call is several times faster than 16
separate calls because the fast tokenisers parallelise internally, and on a
2-core CPU that difference is the difference between a usable and an unusable
training loop.

Evidence vectors come from the pre-computed
:class:`~src.retrieval.cache.RetrievalCache` and are looked up from the FAISS
index by id, so no sentence encoding happens inside the training loop either.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.schema import SarcasmRecord
from src.models.tokenization import SarcasmTokenizer
from src.retrieval.cache import RetrievalCache
from src.retrieval.faiss_index import KnowledgeIndex

logger = logging.getLogger(__name__)


class SarcasmDataset(Dataset):
    """A thin, indexable view over a list of records."""

    def __init__(self, records: Sequence[SarcasmRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        """Number of examples."""
        return len(self.records)

    def __getitem__(self, index: int) -> SarcasmRecord:
        """Return the record at ``index`` (collation happens in the collator)."""
        return self.records[index]

    @property
    def labels(self) -> np.ndarray:
        """All labels as an ``int64`` array (used for class weighting)."""
        return np.array([record.label for record in self.records], dtype=np.int64)


class SarcasmCollator:
    """Turn a list of records into a model-ready batch of tensors."""

    def __init__(
        self,
        tokenizer: SarcasmTokenizer,
        knowledge_index: KnowledgeIndex | None = None,
        retrieval_cache: RetrievalCache | None = None,
        knowledge_slots: int = 5,
        knowledge_dim: int = 384,
        use_context: bool = True,
        use_knowledge: bool = True,
        max_context_turns: int = 3,
    ) -> None:
        """
        Args:
            tokenizer: Shared tokeniser for both branches.
            knowledge_index: Source of evidence vectors (via ``reconstruct``).
            retrieval_cache: Pre-computed top-k evidence ids per record.
            knowledge_slots: K -- number of evidence slots per example.
            knowledge_dim: Sentence-encoder dimensionality (for zero padding
                when no index is available).
            use_context: Emit context tensors.
            use_knowledge: Emit evidence tensors.
            max_context_turns: Truncate context to the most recent N turns. This
                is the knob the "context length" ablation sweeps.
        """
        self.tokenizer = tokenizer
        self.knowledge_index = knowledge_index
        self.retrieval_cache = retrieval_cache
        self.knowledge_slots = knowledge_slots
        self.knowledge_dim = (
            knowledge_index.dimension if knowledge_index is not None else knowledge_dim
        )
        self.use_context = use_context
        self.use_knowledge = use_knowledge
        self.max_context_turns = max_context_turns

    def __call__(self, records: Sequence[SarcasmRecord]) -> dict[str, Any]:
        """Collate ``records`` into a batch dict of tensors."""
        texts = [record.text for record in records]
        contexts = [
            list(record.context[-self.max_context_turns :]) if self.use_context else []
            for record in records
        ]
        batch = self.tokenizer.encode(texts, contexts)
        batch["labels"] = torch.tensor(
            [record.label for record in records], dtype=torch.long
        )
        batch["record_ids"] = [record.id for record in records]

        if not self.use_context:
            # Baselines 1 and 3 must not see context at all. Dropping the tensors
            # (rather than zeroing them) guarantees the branch is never executed.
            batch.pop("context_input_ids", None)
            batch.pop("context_attention_mask", None)
            batch["has_context"] = torch.zeros(len(records), dtype=torch.bool)

        if self.use_knowledge:
            vectors, mask = self._evidence(records)
            batch["knowledge_vectors"] = vectors
            batch["knowledge_mask"] = mask
        return batch

    def _evidence(
        self, records: Sequence[SarcasmRecord]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up ``[B, K, D]`` evidence vectors and their ``[B, K]`` mask."""
        batch_size = len(records)
        vectors = np.zeros(
            (batch_size, self.knowledge_slots, self.knowledge_dim), dtype=np.float32
        )
        mask = np.zeros((batch_size, self.knowledge_slots), dtype=bool)
        if self.retrieval_cache is None or self.knowledge_index is None:
            return torch.from_numpy(vectors), torch.from_numpy(mask)

        for row, record in enumerate(records):
            ids, _ = self.retrieval_cache.get(record.id)
            valid = [int(i) for i in ids[: self.knowledge_slots] if int(i) >= 0]
            if not valid:
                continue
            found = self.knowledge_index.reconstruct(valid)
            vectors[row, : len(valid)] = found
            mask[row, : len(valid)] = True
        return torch.from_numpy(vectors), torch.from_numpy(mask)


def model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """Strip bookkeeping keys so the dict can be splatted into ``model(**batch)``."""
    return {
        key: value
        for key, value in batch.items()
        if key
        in {
            "utterance_input_ids",
            "utterance_attention_mask",
            "context_input_ids",
            "context_attention_mask",
            "has_context",
            "knowledge_vectors",
            "knowledge_mask",
            "labels",
        }
    }


def compute_class_weights(labels: np.ndarray, num_labels: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1.

    ``w_c = N / (C * n_c)`` -- the standard "balanced" scheme. Normalising to
    mean 1 keeps the loss magnitude (and therefore a good learning rate)
    comparable to the unweighted case, so switching weighting on does not
    silently require a different learning rate.
    """
    counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = labels.shape[0] / (num_labels * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
