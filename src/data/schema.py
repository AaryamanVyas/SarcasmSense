"""The single normalised record schema shared by every dataset adapter.

Every corpus in this project (News-Headlines, MUStARD, SARC, FigLang2020) is
converted into a stream of :class:`SarcasmRecord` objects. Downstream code --
splitting, tokenisation, retrieval, training, evaluation, error analysis --
only ever sees this schema, which is what lets us add a new corpus by writing a
single adapter.

JSONL is used as the on-disk format because it streams, diffs and greps well,
and does not require the whole corpus to be resident in memory.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

#: Canonical label mapping used everywhere in the project.
LABEL_NOT_SARCASTIC = 0
LABEL_SARCASTIC = 1
LABEL_NAMES: dict[int, str] = {
    LABEL_NOT_SARCASTIC: "Not Sarcastic",
    LABEL_SARCASTIC: "Sarcastic",
}
LABEL_IDS: dict[str, int] = {v: k for k, v in LABEL_NAMES.items()}

#: Maximum number of preceding conversational turns kept (project spec).
MAX_CONTEXT_TURNS = 3

#: Domains we tag records with; used for per-domain reporting.
DOMAINS = ("news", "reddit", "twitter", "dialogue", "unknown")

_WS_RE = re.compile(r"\s+")


class SchemaError(ValueError):
    """Raised when a record violates the common schema."""


@dataclass(slots=True)
class SarcasmRecord:
    """One labelled example in the common schema.

    Attributes:
        id: Globally unique, stable identifier (``"<dataset>:<local id>"``).
        text: The *target utterance* to classify. Never empty.
        context: Preceding conversational turns in chronological order, oldest
            first, at most :data:`MAX_CONTEXT_TURNS` entries. Empty list for
            non-conversational corpora (headlines, standalone tweets).
        label: 0 = not sarcastic, 1 = sarcastic.
        dataset: Source corpus key, e.g. ``"figlang2020_twitter"``.
        domain: Coarse genre, one of :data:`DOMAINS`.
        language: ISO-639-1 code; every supported corpus is ``"en"``.
        conversation_id: Groups records that share a conversation thread. Used
            by the *conversation-level* splitter to prevent context leakage.
            Falls back to ``id`` for standalone items.
        metadata: Free-form provenance (original label string, subreddit,
            speaker, article link, ...). Never consumed by the model.
    """

    id: str
    text: str
    label: int
    dataset: str
    context: list[str] = field(default_factory=list)
    domain: str = "unknown"
    language: str = "en"
    conversation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ checks
    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise SchemaError(f"record {self.id!r}: 'text' must be a non-empty string")
        if self.label not in LABEL_NAMES:
            raise SchemaError(
                f"record {self.id!r}: label must be 0 or 1, got {self.label!r}"
            )
        if not isinstance(self.context, list) or any(
            not isinstance(turn, str) for turn in self.context
        ):
            raise SchemaError(f"record {self.id!r}: 'context' must be a list of strings")
        if len(self.context) > MAX_CONTEXT_TURNS:
            # Keep the *most recent* turns: they are the ones adjacent to the
            # target utterance and therefore the most informative.
            self.context = self.context[-MAX_CONTEXT_TURNS:]
        self.context = [turn for turn in self.context if turn and turn.strip()]
        if self.domain not in DOMAINS:
            self.domain = "unknown"
        if self.conversation_id is None:
            self.conversation_id = self.id

    # ------------------------------------------------------------- properties
    @property
    def has_context(self) -> bool:
        """Whether any conversational context is available for this record."""
        return len(self.context) > 0

    @property
    def label_name(self) -> str:
        """Human-readable label."""
        return LABEL_NAMES[self.label]

    @property
    def content_hash(self) -> str:
        """Stable hash of (context, text), used for duplicate detection.

        Whitespace and case are normalised so that trivially different copies of
        the same item collapse onto the same hash, but punctuation -- which
        carries sarcasm signal -- is preserved.
        """
        joined = " || ".join([*self.context, self.text])
        normalised = _WS_RE.sub(" ", joined).strip().lower()
        return hashlib.sha1(normalised.encode("utf-8")).hexdigest()

    # ------------------------------------------------------- (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dict in canonical key order."""
        return {
            "id": self.id,
            "text": self.text,
            "context": list(self.context),
            "label": int(self.label),
            "dataset": self.dataset,
            "domain": self.domain,
            "language": self.language,
            "conversation_id": self.conversation_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SarcasmRecord":
        """Rebuild a record from its dict form, validating as we go."""
        missing = {"id", "text", "label", "dataset"} - set(raw)
        if missing:
            raise SchemaError(f"record is missing required key(s): {sorted(missing)}")
        return cls(
            id=str(raw["id"]),
            text=str(raw["text"]),
            label=int(raw["label"]),
            dataset=str(raw["dataset"]),
            context=list(raw.get("context") or []),
            domain=str(raw.get("domain") or "unknown"),
            language=str(raw.get("language") or "en"),
            conversation_id=(
                str(raw["conversation_id"]) if raw.get("conversation_id") else None
            ),
            metadata=dict(raw.get("metadata") or {}),
        )


def records_to_dicts(records: Iterable[SarcasmRecord]) -> list[dict[str, Any]]:
    """Convenience helper for writing records with :func:`src.utils.io.write_jsonl`."""
    return [record.to_dict() for record in records]
