"""Builder for **DEVELOPMENT/TEST ONLY** model assets.

Purpose
-------
The unit tests, the smoke test and CI must be able to exercise *every* code path
-- tokenisation, encoding, cross-attention, training loop, checkpointing,
LIME/SHAP, the API, the UI -- on a machine with no GPU and no access to the
HuggingFace hub. Downloading ``roberta-base`` (500 MB) for that is neither fast
nor always possible.

So this module builds a genuinely functional but *tiny* RoBERTa: a real
:class:`transformers.RobertaModel` with a real byte-level BPE tokeniser trained
on the project's own corpus, at roughly 1/500th the size. Every layer type,
tensor shape and code path is identical to the research model; only the
capacity differs.

.. warning::

   **A model built here is randomly initialised and has learned nothing.** Any
   checkpoint derived from it is labelled ``is_development_asset: true`` and the
   API refuses to present its predictions as research results. Never report a
   metric obtained from a dev asset as a result of the proposed architecture.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

from src.utils.paths import ensure_dir, resolve

logger = logging.getLogger(__name__)

DEV_ASSET_MARKER = "DEVELOPMENT_TEST_ONLY.txt"
DEV_WARNING = (
    "DEVELOPMENT/TEST ONLY.\n"
    "This directory contains a randomly-initialised tiny RoBERTa and a BPE "
    "tokenizer trained on a small text sample. It exists so the software "
    "pipeline can be tested without downloading roberta-base.\n"
    "It has NOT learned anything about sarcasm. Any metric produced with it "
    "describes the plumbing, not the research model.\n"
)

DEFAULT_CORPUS = [
    "Oh great, another meeting that could have been an email.",
    "The weather today is pleasant and the sky is clear.",
    "Wow, what an amazing traffic jam. I only needed to get to work.",
    "Scientists published a new study on renewable energy adoption.",
    "My laptop crashed five minutes before the presentation. Perfect timing.",
    "The train arrived exactly on schedule this morning.",
    "Fantastic, the printer is out of toner again right before the deadline.",
    "Local council approves budget for the new public library.",
    "I just love waiting on hold for forty five minutes.",
    "She finished the marathon in under four hours.",
]


def build_dev_tokenizer(
    output_dir: str | Path,
    corpus: Iterable[str] | None = None,
    vocab_size: int = 2000,
    min_frequency: int = 1,
) -> Path:
    """Train and save a real byte-level BPE tokeniser with RoBERTa's interface.

    Args:
        output_dir: Where ``vocab.json`` / ``merges.txt`` / config are written.
        corpus: Text to learn merges from. Defaults to a small built-in sample.
        vocab_size: Target vocabulary size (kept small so the embedding matrix
            stays tiny).
        min_frequency: BPE merge frequency threshold.

    Returns:
        The output directory.
    """
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaTokenizerFast

    target = ensure_dir(output_dir)
    texts = list(corpus) if corpus is not None else list(DEFAULT_CORPUS)
    if not texts:
        raise ValueError("Cannot train a tokenizer on an empty corpus")

    trainer = ByteLevelBPETokenizer()
    trainer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    trainer.save_model(str(target))

    tokenizer = RobertaTokenizerFast(
        vocab_file=str(target / "vocab.json"),
        merges_file=str(target / "merges.txt"),
        model_max_length=512,
    )
    tokenizer.save_pretrained(str(target))
    logger.info("Dev tokenizer saved -> %s (vocab=%d)", target, tokenizer.vocab_size)
    return target


def build_dev_encoder(
    output_dir: str | Path,
    tokenizer_dir: str | Path,
    hidden_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    intermediate_size: int = 256,
    max_position_embeddings: int = 386,
) -> Path:
    """Create and save a tiny, randomly-initialised :class:`RobertaModel`.

    The saved directory is a drop-in replacement for ``roberta-base`` anywhere
    the project accepts an encoder name, and it also holds the tokeniser files
    so that a single path configures both.

    Args:
        output_dir: Destination directory.
        tokenizer_dir: Directory produced by :func:`build_dev_tokenizer`.
        hidden_size: Must be divisible by both ``num_attention_heads`` and the
            fusion head count (8 by default), hence the 128 default.
        num_hidden_layers: Transformer blocks.
        num_attention_heads: Encoder attention heads.
        intermediate_size: Feed-forward width.
        max_position_embeddings: RoBERTa offsets positions by ``pad_token_id+1``,
            so this must exceed the longest sequence by at least 2.

    Returns:
        The output directory.
    """
    import shutil

    from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

    target = ensure_dir(output_dir)
    source_tokenizer = resolve(tokenizer_dir)
    tokenizer = RobertaTokenizerFast.from_pretrained(str(source_tokenizer))

    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = RobertaModel(config)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))

    if source_tokenizer != target:
        for name in ("vocab.json", "merges.txt"):
            candidate = source_tokenizer / name
            if candidate.exists():
                shutil.copy(candidate, target / name)

    (target / DEV_ASSET_MARKER).write_text(DEV_WARNING, encoding="utf-8")
    (target / "dev_asset.json").write_text(
        json.dumps(
            {
                "is_development_asset": True,
                "purpose": "software pipeline testing only",
                "hidden_size": hidden_size,
                "num_hidden_layers": num_hidden_layers,
                "vocab_size": tokenizer.vocab_size,
                "n_parameters": sum(p.numel() for p in model.parameters()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Dev encoder saved -> %s (hidden=%d, layers=%d, params=%.2fM) [DEV ONLY]",
        target,
        hidden_size,
        num_hidden_layers,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return target


def build_dev_assets(
    output_dir: str | Path = "checkpoints/dev_encoder",
    corpus: Sequence[str] | None = None,
    vocab_size: int = 2000,
    hidden_size: int = 128,
) -> Path:
    """Build tokeniser + encoder into a single directory. Returns that directory."""
    target = ensure_dir(output_dir)
    build_dev_tokenizer(target, corpus=corpus, vocab_size=vocab_size)
    build_dev_encoder(target, target, hidden_size=hidden_size)
    return target


def is_development_asset(directory: str | Path) -> bool:
    """Whether ``directory`` was produced by this module."""
    return (resolve(directory) / DEV_ASSET_MARKER).exists()
