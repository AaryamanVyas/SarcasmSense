"""Adapter for the MUStARD corpus -- **text-only** subset.

Raw format (one big JSON object keyed by utterance id)::

    {"1_60": {"utterance": "It's just a privilege to watch your mind at work.",
              "speaker": "SHELDON",
              "context": ["I never would have identified the fingerprints ..."],
              "context_speakers": ["LEONARD"],
              "sarcasm": true,
              "show": "BBT"}}

This project is text-only, so the audio/video artefacts distributed alongside
``sarcasm_data.json`` are never touched. Speaker names *are* used, but only as
a textual prefix on each context turn ("SHELDON: ...") -- that is legitimate
textual dialogue information, not a multimodal feature.

Utterance ids look like ``"<scene>_<index>"``; the scene prefix is used as the
``conversation_id`` so that two utterances from the same scene can never be
split across train and test.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from src.data.loaders.base import DatasetAdapter
from src.data.schema import SarcasmRecord
from src.data.sources import MUSTARD


class MUStARDAdapter(DatasetAdapter):
    """MUStARD text-only adapter (690 annotated utterances)."""

    source = MUSTARD

    def iter_raw(self) -> Iterator[tuple[str, Any]]:  # noqa: D102
        path = self._first_existing("sarcasm_data.json")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(
                f"{path}: expected a JSON object keyed by utterance id, "
                f"got {type(payload).__name__}"
            )
        # ``sorted`` keeps the iteration order stable across runs/filesystems.
        for utterance_id in sorted(payload):
            yield utterance_id, payload[utterance_id]

    def to_record(self, local_id: str, payload: Any) -> SarcasmRecord | None:  # noqa: D102
        if not isinstance(payload, dict):
            self.stats.dropped_malformed += 1
            return None

        turns = list(payload.get("context") or [])
        speakers = list(payload.get("context_speakers") or [])
        # Prefix each context turn with its speaker when we have one; this is
        # the only reliable way for a text-only model to track who said what.
        labelled_turns: list[str] = []
        for idx, turn in enumerate(turns):
            speaker = speakers[idx] if idx < len(speakers) else None
            labelled_turns.append(f"{speaker}: {turn}" if speaker else str(turn))

        speaker = payload.get("speaker")
        utterance = payload.get("utterance")
        # MUStARD ids look like "1_60" / "2_410"; the prefix is an annotation
        # batch, *not* a scene, so it cannot be used for grouping (it would give
        # only two groups for 690 rows). Utterances from the same scene do,
        # however, share their opening context turn -- so we group on a hash of
        # that turn, which is the only scene signal the released JSON carries.
        scene_id = _scene_id(turns, local_id)

        # The *target* utterance is deliberately left unprefixed. Attaching the
        # speaker name would let the classifier exploit a speaker shortcut
        # (Sheldon is sarcastic far more often than Leonard), which would
        # inflate MUStARD scores without improving sarcasm understanding.
        return self._build(
            local_id=local_id,
            text=utterance,
            label=payload.get("sarcasm"),
            context=labelled_turns,
            conversation_id=f"{self.key}:{scene_id}",
            metadata={
                "speaker": speaker,
                "show": payload.get("show"),
                "scene_id": scene_id,
                "annotation_batch": local_id.split("_", 1)[0],
                "n_context_turns_available": len(turns),
                "modalities_used": "text_only",
            },
        )


def _scene_id(context_turns: list[str], fallback: str) -> str:
    """Derive a scene identifier from the first context turn.

    Utterances belonging to the same sitcom scene share their opening context
    line, so hashing it groups them. Utterances without context become their own
    single-item group, which is the safe default.
    """
    import hashlib

    if not context_turns:
        return fallback
    opener = " ".join(context_turns[0].split()).lower()
    return "scene-" + hashlib.sha1(opener.encode("utf-8")).hexdigest()[:16]
