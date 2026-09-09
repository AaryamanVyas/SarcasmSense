"""Adapters for the FigLang 2020 sarcasm shared-task corpora (Twitter + Reddit).

Raw format (one JSON object per line)::

    {"label": "SARCASM",
     "response": "Did Kelly just call someone else messy? Baaaahaaahahahaha",
     "context": ["X is looking a First Lady should . #classact",
                 "didn't think it was tailored enough it looked messy"]}

``context`` is an ordered list: ``c1`` -> ``c2`` -> ... -> ``response``, so the
*last* element is the immediate parent of the target utterance. That is exactly
the ordering our schema wants, so no reversal is needed -- only truncation to
the most recent three turns.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from src.data.loaders.base import DatasetAdapter
from src.data.schema import SarcasmRecord
from src.data.sources import FIGLANG_REDDIT, FIGLANG_TWITTER, DatasetSource


class _FigLangAdapter(DatasetAdapter):
    """Shared implementation; the two subclasses only differ in their source."""

    source: DatasetSource

    def iter_raw(self) -> Iterator[tuple[str, Any]]:  # noqa: D102
        path = self._first_existing(
            *[f.relative_path for f in self.source.remote_files]
        )
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                yield str(line_no), json.loads(line)

    def to_record(self, local_id: str, payload: Any) -> SarcasmRecord | None:  # noqa: D102
        if not isinstance(payload, dict):
            self.stats.dropped_malformed += 1
            return None
        # The organisers use "id" in the test files and nothing in train files.
        row_id = str(payload.get("id") or local_id)
        raw_context = payload.get("context") or []
        return self._build(
            local_id=row_id,
            text=payload.get("response"),
            label=payload.get("label"),
            context=raw_context,
            # Every row in these files is an independent thread, so the
            # conversation id is the row id: there is no cross-row thread
            # structure to leak between splits.
            conversation_id=f"{self.key}:{row_id}",
            metadata={
                "original_label": payload.get("label"),
                "n_context_turns_available": len(raw_context),
                "split": "shared_task_train",
            },
        )


class FigLangTwitterAdapter(_FigLangAdapter):
    """FigLang 2020 Twitter subset (5,000 labelled responses)."""

    source = FIGLANG_TWITTER


class FigLangRedditAdapter(_FigLangAdapter):
    """FigLang 2020 Reddit subset (4,400 labelled responses)."""

    source = FIGLANG_REDDIT
