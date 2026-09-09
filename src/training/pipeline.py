"""Assemble everything an experiment needs from a single :class:`Config`.

``scripts/train.py``, ``scripts/evaluate.py`` and ``scripts/run_ablation.py``
all call :func:`build_experiment` so that the four baselines and the full model
are constructed by *identical* code differing only in configuration. That is
what makes the comparison in the results table a fair one.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from src.data.dataset import SarcasmCollator, SarcasmDataset, compute_class_weights
from src.data.schema import SarcasmRecord
from src.data.splitting import apply_split, corpus_fingerprint, filter_by_datasets, load_split
from src.models.sarcasm_model import SarcasmModel
from src.models.tokenization import SarcasmTokenizer
from src.retrieval.cache import RetrievalCache, ensure_retrieval_cache
from src.retrieval.encoder import SentenceEncoder, build_sentence_encoder
from src.retrieval.faiss_index import KnowledgeIndex, index_exists
from src.retrieval.retriever import CommonsenseRetriever
from src.utils.config import Config
from src.utils.io import read_jsonl
from src.utils.paths import resolve

logger = logging.getLogger(__name__)


@dataclass
class ExperimentContext:
    """Every object an experiment script needs, built once and shared."""

    config: Config
    records: list[SarcasmRecord]
    splits: dict[str, list[SarcasmRecord]]
    datasets: dict[str, SarcasmDataset]
    tokenizer: SarcasmTokenizer
    collator: SarcasmCollator
    model: SarcasmModel
    retriever: CommonsenseRetriever | None
    knowledge_index: KnowledgeIndex | None
    sentence_encoder: SentenceEncoder | None
    retrieval_cache: RetrievalCache | None
    class_weights: torch.Tensor | None
    warnings: list[str]


def load_corpus(config: Config) -> list[SarcasmRecord]:
    """Read the normalised corpus and filter it to the configured datasets."""
    path = resolve(config.data.processed_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed corpus not found at {path}. Run:\n"
            f"  python scripts/download_datasets.py\n"
            f"  python scripts/prepare_datasets.py"
        )
    records = [SarcasmRecord.from_dict(row) for row in read_jsonl(path)]
    return filter_by_datasets(records, config.data.datasets)


def load_splits(
    config: Config, records: Sequence[SarcasmRecord], verify_fingerprint: bool = True
) -> dict[str, list[SarcasmRecord]]:
    """Load the saved split and materialise it against ``records``."""
    fingerprint = corpus_fingerprint(records) if verify_fingerprint else None
    # Fingerprint verification only makes sense when no dataset filter is applied,
    # because filtering deliberately changes the corpus.
    if config.data.datasets:
        fingerprint = None
    assignment = load_split(config.data.splits_dir, expected_fingerprint=fingerprint)
    if config.data.datasets:
        wanted = {record.id for record in records}
        for name in ("train", "val", "test"):
            filtered = [i for i in assignment[name] if i in wanted]
            getattr(assignment, name).clear()
            getattr(assignment, name).extend(filtered)
    return apply_split(list(records), assignment)


def _subsample(
    records: list[SarcasmRecord], limit: int, seed: int
) -> list[SarcasmRecord]:
    """Deterministically keep ``limit`` records, preserving corpus proportions.

    Uses a seeded shuffle followed by a head slice, then restores the original
    ordering so that batches stay reproducible across runs.
    """
    if limit >= len(records):
        return records
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    keep = sorted(order[:limit])
    return [records[i] for i in keep]


def build_retrieval(
    config: Config, device: str | None = None
) -> tuple[CommonsenseRetriever | None, KnowledgeIndex | None, SentenceEncoder | None, list[str]]:
    """Construct the retrieval stack, degrading gracefully when it is absent.

    Returns ``(retriever, index, encoder, warnings)``. A missing index is *not*
    an error: the retriever is still constructed and reports
    ``RetrievalStatus.NO_INDEX`` for every query, so the classifier keeps working
    and the user is told why there is no evidence.
    """
    warnings: list[str] = []
    if not config.retrieval.enabled:
        return None, None, None, warnings

    encoder = build_sentence_encoder(
        backend=config.retrieval.encoder_backend,
        model_name=config.retrieval.encoder_name,
        device=device,
    )
    if encoder.is_fallback:
        warnings.append(
            f"Sentence encoder fell back to {encoder.name!r} (lexical similarity). "
            "Retrieval quality is reduced; results from this run are not "
            "comparable to runs using the configured encoder."
        )

    index: KnowledgeIndex | None = None
    if index_exists(config.retrieval.index_dir):
        try:
            index = KnowledgeIndex.load(
                config.retrieval.index_dir,
                expected_encoder_name=encoder.name,
                strict=True,
            )
        except Exception as exc:  # noqa: BLE001 - degrade, do not crash
            warnings.append(f"Knowledge index could not be loaded: {exc}")
            logger.error("Knowledge index load failed: %s", exc)
        else:
            if index.metadata.is_sample_index:
                warnings.append(
                    "The loaded knowledge index is the SAMPLE index "
                    "(development/smoke-test only). Build a full index with "
                    "scripts/build_knowledge_index.py before reporting results."
                )
    else:
        warnings.append(
            f"No knowledge index found at {resolve(config.retrieval.index_dir)}. "
            "Retrieval will report 'no_index' and the model will run without "
            "commonsense evidence. Build one with scripts/build_knowledge_index.py."
        )

    retriever = CommonsenseRetriever(
        index=index,
        encoder=encoder,
        top_k=config.retrieval.top_k,
        min_score=config.retrieval.min_score,
        query_mode=config.retrieval.query_mode,
        cache_size=config.retrieval.cache_size,
        enabled=config.retrieval.enabled,
    )
    return retriever, index, encoder, warnings


def build_experiment(
    config: Config,
    device: str | None = None,
    include_train: bool = True,
    rebuild_cache: bool = False,
    local_files_only: bool = False,
) -> ExperimentContext:
    """Build datasets, retrieval stack, tokeniser, collator and model.

    Args:
        config: Resolved experiment configuration.
        device: Torch device override (also used for the sentence encoder).
        include_train: Skip the training split when only evaluating.
        rebuild_cache: Force recomputation of the retrieval cache.
        local_files_only: Never hit the network when loading transformers.

    Returns:
        A fully-populated :class:`ExperimentContext`.
    """
    warnings: list[str] = []
    records = load_corpus(config)
    splits = load_splits(config, records)
    logger.info(
        "Corpus %d records | train=%d val=%d test=%d",
        len(records),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    # Subsampling uses a seeded shuffle rather than a head slice. Records are
    # stored sorted by id, so a head slice would silently keep a single corpus
    # and make every "per-dataset" number meaningless.
    if config.data.max_train_samples:
        splits["train"] = _subsample(
            splits["train"], config.data.max_train_samples, config.experiment.seed
        )
    if config.data.max_eval_samples:
        splits["val"] = _subsample(
            splits["val"], config.data.max_eval_samples, config.experiment.seed + 1
        )
        splits["test"] = _subsample(
            splits["test"], config.data.max_eval_samples, config.experiment.seed + 2
        )

    retriever, index, sentence_encoder, retrieval_warnings = build_retrieval(config, device)
    warnings.extend(retrieval_warnings)

    retrieval_cache: RetrievalCache | None = None
    if config.model.use_knowledge:
        if retriever is not None and retriever.available:
            cache_path = (
                Path(config.data.processed_path).with_suffix("")
                .as_posix()
                + f".retrieval_top{config.retrieval.top_k}.npz"
            )
            needed = list(records)
            if not include_train:
                needed = splits["val"] + splits["test"]
            retrieval_cache = ensure_retrieval_cache(
                needed, retriever, cache_path, rebuild=rebuild_cache
            )
        else:
            warnings.append(
                "model.use_knowledge=true but retrieval is unavailable; the "
                "knowledge branch will receive an all-zero, fully-masked evidence "
                "tensor. The model still runs, but this is NOT the proposed "
                "configuration -- fix the knowledge index before reporting results."
            )

    tokenizer = SarcasmTokenizer(
        model_name=config.model.encoder_name,
        max_utterance_length=config.model.max_utterance_length,
        max_context_length=config.model.max_context_length,
        local_files_only=local_files_only,
    )
    knowledge_dim = index.dimension if index is not None else 384
    collator = SarcasmCollator(
        tokenizer=tokenizer,
        knowledge_index=index,
        retrieval_cache=retrieval_cache,
        knowledge_slots=config.retrieval.top_k,
        knowledge_dim=knowledge_dim,
        use_context=config.model.use_context,
        use_knowledge=config.model.use_knowledge,
        max_context_turns=config.data.max_context_turns,
    )

    datasets = {name: SarcasmDataset(rows) for name, rows in splits.items()}

    class_weights = None
    if config.training.class_weights == "balanced" and include_train:
        class_weights = compute_class_weights(datasets["train"].labels)
        logger.info("Class weights (balanced): %s", class_weights.tolist())

    model = SarcasmModel(
        config=config.model,
        knowledge_dim=knowledge_dim,
        knowledge_slots=config.retrieval.top_k,
        class_weights=class_weights,
        local_files_only=local_files_only,
    )

    for warning in warnings:
        logger.warning(warning)

    return ExperimentContext(
        config=config,
        records=records,
        splits=splits,
        datasets=datasets,
        tokenizer=tokenizer,
        collator=collator,
        model=model,
        retriever=retriever,
        knowledge_index=index,
        sentence_encoder=sentence_encoder,
        retrieval_cache=retrieval_cache,
        class_weights=class_weights,
        warnings=warnings,
    )
