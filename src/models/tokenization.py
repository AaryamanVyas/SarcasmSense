"""Tokenisation for the two text branches.

RoBERTa's byte-level BPE tokeniser is used unchanged: no lowercasing, no
punctuation stripping, no emoji removal. Every byte sequence is representable,
so ``"WOW 😒 great!!!"`` survives intact into the model.

Context handling
----------------
The up-to-three preceding turns are flattened into **one** string with explicit
turn markers and the tokeniser's own separator token::

    turn 1: <oldest> </s> turn 2: <...> </s> turn 3: <immediately preceding>

Two deliberate decisions:

* ``turn N:`` prefixes tell the encoder *how far back* a turn is. Without them
  the model only knows that a boundary exists, not its recency, and recency
  matters -- the last turn is what the target utterance actually replies to.
* Truncation is **left-side** for context (``truncation_side="left"``). When the
  context exceeds the budget it is the *oldest* material that gets dropped,
  keeping the immediately preceding turn, which is the most informative one.
  The target utterance uses ordinary right-side truncation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch

from src.data.preprocessing import build_context_string
from src.utils.paths import ensure_dir, resolve

logger = logging.getLogger(__name__)

#: Placeholder encoded for examples with no conversational context. It is a real
#: string (so the tensor shapes stay uniform) but the ``has_context`` flag tells
#: the model to substitute its learned empty-context vector instead.
NO_CONTEXT_PLACEHOLDER = "no previous conversation turns"


class TokenizerLoadError(RuntimeError):
    """Raised when the tokeniser cannot be loaded."""


class SarcasmTokenizer:
    """Wraps a HuggingFace tokeniser and produces model-ready batches."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        max_utterance_length: int = 128,
        max_context_length: int = 128,
        local_files_only: bool = False,
    ) -> None:
        """
        Args:
            model_name: HuggingFace id or local directory.
            max_utterance_length: Token budget for the target (proposal: 128).
            max_context_length: Token budget for the flattened context.
            local_files_only: Never hit the network.

        Raises:
            TokenizerLoadError: if the tokeniser cannot be constructed.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise TokenizerLoadError("transformers is not installed") from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, local_files_only=local_files_only
            )
        except Exception as exc:  # noqa: BLE001
            raise TokenizerLoadError(
                f"Could not load tokenizer {model_name!r}: {exc}"
            ) from exc
        self.model_name = model_name
        self.max_utterance_length = max_utterance_length
        self.max_context_length = max_context_length
        self.separator = self.tokenizer.sep_token or "</s>"

    # ---------------------------------------------------------------- helpers
    @property
    def pad_token_id(self) -> int:
        """Padding token id."""
        return int(self.tokenizer.pad_token_id or 0)

    def flatten_context(self, turns: Sequence[str] | None) -> tuple[str, bool]:
        """Return ``(flattened_context_string, has_context)``."""
        turns = [turn for turn in (turns or []) if turn and turn.strip()]
        if not turns:
            return NO_CONTEXT_PLACEHOLDER, False
        return build_context_string(list(turns), separator=f" {self.separator} "), True

    def context_turn_spans(
        self, turns: Sequence[str] | None
    ) -> tuple[str, list[tuple[int, int]]]:
        """Return the flattened context and the character span of each turn.

        The spans let the explanation layer attribute cross-attention weight back
        to *individual turns* ("turn 3 mattered most"), rather than reporting one
        undifferentiated number for the whole context.
        """
        turns = [turn for turn in (turns or []) if turn and turn.strip()]
        if not turns:
            return NO_CONTEXT_PLACEHOLDER, []
        flat = ""
        spans: list[tuple[int, int]] = []
        for index, turn in enumerate(turns, start=1):
            if flat:
                flat += f" {self.separator} "
            prefix = f"turn {index}: "
            start = len(flat) + len(prefix)
            flat += prefix + turn
            spans.append((start, len(flat)))
        return flat, spans

    def encode_context_with_offsets(
        self, turns: Sequence[str] | None
    ) -> tuple[dict[str, Any], list[tuple[int, int]], list[tuple[int, int]]]:
        """Tokenise one context string, returning token offsets and turn spans."""
        flat, spans = self.context_turn_spans(turns)
        previous_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            encoded = self.tokenizer(
                [flat],
                padding="max_length",
                truncation=True,
                max_length=self.max_context_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
        finally:
            self.tokenizer.truncation_side = previous_side
        offsets = [tuple(pair) for pair in encoded["offset_mapping"][0].tolist()]
        return encoded, offsets, spans

    # ----------------------------------------------------------------- encode
    def encode(
        self,
        texts: Sequence[str],
        contexts: Sequence[Sequence[str]] | None = None,
        return_offsets: bool = False,
    ) -> dict[str, Any]:
        """Tokenise a batch of (utterance, context) pairs.

        Args:
            texts: Target utterances.
            contexts: Per-example lists of preceding turns (may be empty).
            return_offsets: Also return character offsets for the utterance,
                which the explanation layer uses to map token attributions back
                onto the original string.

        Returns:
            Dict of tensors: ``utterance_input_ids``, ``utterance_attention_mask``,
            ``context_input_ids``, ``context_attention_mask``, ``has_context``,
            and optionally ``utterance_offsets`` / ``utterance_tokens``.
        """
        texts = list(texts)
        contexts = list(contexts or [[] for _ in texts])
        flattened: list[str] = []
        present: list[bool] = []
        for turns in contexts:
            flat, has = self.flatten_context(turns)
            flattened.append(flat)
            present.append(has)

        utterance = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_utterance_length,
            return_tensors="pt",
            return_offsets_mapping=return_offsets,
        )

        # Left truncation keeps the most recent turns when context overflows.
        previous_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            context = self.tokenizer(
                flattened,
                padding="max_length",
                truncation=True,
                max_length=self.max_context_length,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.truncation_side = previous_side

        batch: dict[str, Any] = {
            "utterance_input_ids": utterance["input_ids"],
            "utterance_attention_mask": utterance["attention_mask"],
            "context_input_ids": context["input_ids"],
            "context_attention_mask": context["attention_mask"],
            "has_context": torch.tensor(present, dtype=torch.bool),
        }
        if return_offsets:
            batch["utterance_offsets"] = utterance["offset_mapping"]
            batch["utterance_tokens"] = [
                self.tokenizer.convert_ids_to_tokens(ids)
                for ids in utterance["input_ids"].tolist()
            ]
        return batch

    def decode_tokens(self, token_ids: Sequence[int]) -> list[str]:
        """Convert ids to readable token strings (``Ġ`` markers removed)."""
        tokens = self.tokenizer.convert_ids_to_tokens(list(token_ids))
        return [token.replace("Ġ", " ").replace("Ċ", "\n") for token in tokens]

    # ------------------------------------------------------------ persistence
    def save_pretrained(self, directory: str | Path) -> Path:
        """Save the tokeniser next to the model weights."""
        target = ensure_dir(directory)
        self.tokenizer.save_pretrained(str(target))
        return target

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        max_utterance_length: int = 128,
        max_context_length: int = 128,
    ) -> "SarcasmTokenizer":
        """Load a tokeniser saved inside a checkpoint directory."""
        return cls(
            model_name=str(resolve(directory)),
            max_utterance_length=max_utterance_length,
            max_context_length=max_context_length,
            local_files_only=True,
        )
