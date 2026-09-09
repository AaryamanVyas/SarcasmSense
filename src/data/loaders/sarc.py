"""Adapter for SARC -- the Self-Annotated Reddit Corpus.

Two on-disk layouts are supported, because the two commonly-distributed forms
of SARC have very different shapes.

**Layout A -- ``train-balanced-sarcasm.csv``** (the widely used flattened CSV)::

    label,comment,author,subreddit,score,ups,downs,date,created_utc,parent_comment

One row per comment, with the immediate parent comment inline. This gives at
most **one** context turn.

**Layout B -- official Princeton release** (``train-balanced.csv`` +
``comments.json``)::

    train-balanced.csv:  "<post_id> <ancestor ids...>|<response ids>|<labels>"
    comments.json:       {"<comment id>": {"text": ..., "author": ..., ...}}

Here the ancestor chain is available, so we can reconstruct up to three real
context turns -- which is what the architecture is designed for.

SARC is very large (~1M rows in Layout A). ``max_rows`` subsamples it by
**seeded reservoir sampling** (see :meth:`DatasetAdapter.load`), which is what
``--sarc-max-rows`` in ``scripts/prepare_datasets.py`` uses to keep the combined
corpus near the ~50k target instead of letting SARC dominate it.

Sampling rather than head-slicing matters here specifically: the distributed
CSV is roughly date-ordered, so taking the first N rows would quietly restrict
the corpus to the earliest slice of Reddit history -- a confound that would show
up as an unexplained domain gap in the error analysis rather than as an obvious
bug. The sample is reproducible from ``--seed``.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

from src.data.loaders.base import DatasetAdapter
from src.data.schema import SarcasmRecord
from src.data.sources import SARC

logger = logging.getLogger(__name__)

# Reddit comment bodies can be long; the default 128 KiB CSV field limit is
# occasionally exceeded by pathological rows.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_LAYOUT_A = "train-balanced-sarcasm.csv"
_LAYOUT_B_INDEX = "train-balanced.csv"
_LAYOUT_B_COMMENTS = "comments.json"


class SARCAdapter(DatasetAdapter):
    """SARC adapter supporting both the flattened CSV and the official release."""

    source = SARC

    def expected_paths(self) -> list[Path]:  # noqa: D102
        return [
            self.raw_dir / _LAYOUT_A,
            self.raw_dir / _LAYOUT_B_INDEX,
        ]

    # ------------------------------------------------------------------ raw
    def iter_raw(self) -> Iterator[tuple[str, Any]]:  # noqa: D102
        if (self.raw_dir / _LAYOUT_A).exists():
            yield from self._iter_layout_a()
        elif (self.raw_dir / _LAYOUT_B_INDEX).exists():
            yield from self._iter_layout_b()
        else:  # pragma: no cover - guarded by available()
            raise FileNotFoundError(
                f"sarc: expected {_LAYOUT_A} or {_LAYOUT_B_INDEX} in {self.raw_dir}"
            )

    def _iter_layout_a(self) -> Iterator[tuple[str, Any]]:
        path = self.raw_dir / _LAYOUT_A
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"label", "comment"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path}: missing required column(s) {sorted(missing)}; "
                    f"found {reader.fieldnames}"
                )
            for idx, row in enumerate(reader):
                row["_layout"] = "A"
                yield f"a{idx}", row

    def _iter_layout_b(self) -> Iterator[tuple[str, Any]]:
        index_path = self.raw_dir / _LAYOUT_B_INDEX
        comments_path = self.raw_dir / _LAYOUT_B_COMMENTS
        if not comments_path.exists():
            raise FileNotFoundError(
                f"sarc: {_LAYOUT_B_INDEX} found but {_LAYOUT_B_COMMENTS} is missing. "
                f"Both files are required for the official SARC layout."
            )
        logger.info("Loading SARC comment table from %s (this needs a few GB of RAM)", comments_path)
        with comments_path.open("r", encoding="utf-8") as handle:
            comments: dict[str, Any] = json.load(handle)

        with index_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) != 3:
                    continue
                ancestors = parts[0].split()
                responses = parts[1].split()
                labels = parts[2].split()
                for r_idx, (response_id, label) in enumerate(zip(responses, labels)):
                    yield (
                        f"b{idx}-{r_idx}",
                        {
                            "_layout": "B",
                            "response_id": response_id,
                            "label": label,
                            "ancestor_ids": ancestors,
                            "_comments": comments,
                            "thread_id": ancestors[0] if ancestors else response_id,
                        },
                    )

    # --------------------------------------------------------------- mapping
    def to_record(self, local_id: str, payload: Any) -> SarcasmRecord | None:  # noqa: D102
        if not isinstance(payload, dict):
            self.stats.dropped_malformed += 1
            return None
        if payload.get("_layout") == "A":
            return self._record_from_layout_a(local_id, payload)
        return self._record_from_layout_b(local_id, payload)

    def _record_from_layout_a(self, local_id: str, row: dict[str, Any]) -> SarcasmRecord | None:
        parent = row.get("parent_comment") or ""
        subreddit = row.get("subreddit") or "unknown"
        return self._build(
            local_id=local_id,
            text=row.get("comment"),
            label=row.get("label"),
            context=[parent] if parent else [],
            # Layout A has no thread identifier, so each row is its own
            # "conversation". Duplicate-parent detection in the splitter still
            # protects us from the same parent appearing in two splits.
            conversation_id=f"{self.key}:{_thread_key(parent) or local_id}",
            metadata={
                "subreddit": subreddit,
                "author": row.get("author"),
                "score": row.get("score"),
                "date": row.get("date"),
                "layout": "A",
                "n_context_turns_available": 1 if parent else 0,
            },
        )

    def _record_from_layout_b(self, local_id: str, payload: dict[str, Any]) -> SarcasmRecord | None:
        comments: dict[str, Any] = payload["_comments"]
        response = comments.get(payload["response_id"])
        if not response:
            self.stats.dropped_malformed += 1
            return None
        ancestor_texts: list[str] = []
        for ancestor_id in payload["ancestor_ids"]:
            ancestor = comments.get(ancestor_id)
            if ancestor and ancestor.get("text"):
                ancestor_texts.append(str(ancestor["text"]))
        return self._build(
            local_id=local_id,
            text=response.get("text"),
            label=payload.get("label"),
            context=ancestor_texts,
            conversation_id=f"{self.key}:{payload['thread_id']}",
            metadata={
                "subreddit": response.get("subreddit"),
                "author": response.get("author"),
                "score": response.get("score"),
                "layout": "B",
                "n_context_turns_available": len(ancestor_texts),
            },
        )


def _thread_key(parent_comment: str) -> str | None:
    """Derive a coarse thread key from the parent comment text.

    Layout A ships no thread id, so comments replying to the *same* parent are
    grouped by a hash of that parent. This is what stops the identical parent
    comment from appearing as context in both the train and the test split.
    """
    import hashlib

    parent = " ".join((parent_comment or "").split()).lower()
    if not parent:
        return None
    return "thread-" + hashlib.sha1(parent.encode("utf-8")).hexdigest()[:16]
