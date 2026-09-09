"""Adapter for the News Headlines Dataset for Sarcasm Detection.

Raw format: JSON Lines (despite the ``.json`` extension), one object per line::

    {"is_sarcastic": 1,
     "headline": "thirtysomething scientists unveil doomsday clock of hair loss",
     "article_link": "https://www.theonion.com/..."}

Headlines are *standalone*: there is no conversational context, so every record
is emitted with ``context=[]``. This corpus is therefore the main stress test
for the "handle missing context gracefully" requirement.

Both v1 (26,709 rows) and v2 (28,619 rows) files are accepted; v2 is preferred
and is what the download instructions ask for.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from src.data.loaders.base import DatasetAdapter
from src.data.schema import SarcasmRecord
from src.data.sources import NEWS_HEADLINES

_ACCEPTED_FILES = (
    "Sarcasm_Headlines_Dataset_v2.json",
    "Sarcasm_Headlines_Dataset.json",
)


class NewsHeadlinesAdapter(DatasetAdapter):
    """News-headline sarcasm adapter (no conversational context)."""

    source = NEWS_HEADLINES

    def expected_paths(self):  # noqa: D102 - v1 is an accepted alternative to v2
        return [self.raw_dir / name for name in _ACCEPTED_FILES]

    def iter_raw(self) -> Iterator[tuple[str, Any]]:  # noqa: D102
        path = self._first_existing(*_ACCEPTED_FILES)
        version = "v2" if path.name.endswith("_v2.json") else "v1"
        with path.open("r", encoding="utf-8") as handle:
            first = handle.read(1)
            handle.seek(0)
            if first == "[":
                # Some redistributions wrap the records in a JSON array.
                for idx, row in enumerate(json.load(handle)):
                    yield f"{version}-{idx}", row
                return
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                yield f"{version}-{idx}", json.loads(line)

    def to_record(self, local_id: str, payload: Any) -> SarcasmRecord | None:  # noqa: D102
        if not isinstance(payload, dict):
            self.stats.dropped_malformed += 1
            return None
        link = payload.get("article_link") or ""
        return self._build(
            local_id=local_id,
            text=payload.get("headline"),
            label=payload.get("is_sarcastic"),
            context=[],  # headlines are standalone by construction
            conversation_id=f"{self.key}:{local_id}",
            metadata={
                # The publisher is a strong (leaky) signal -- The Onion is
                # always sarcastic. We keep it for *error analysis only* and
                # never feed it to the model.
                "publisher": _publisher_from_link(link),
                "has_context": False,
            },
        )


def _publisher_from_link(link: str) -> str:
    """Extract a bare domain from an article link for error-analysis grouping."""
    if not link:
        return "unknown"
    without_scheme = link.split("://", 1)[-1]
    domain = without_scheme.split("/", 1)[0]
    return domain.removeprefix("www.") or "unknown"
