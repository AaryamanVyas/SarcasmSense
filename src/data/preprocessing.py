"""Text preprocessing that is deliberately *conservative*.

Sarcasm lives in exactly the surface features that aggressive NLP cleaning
throws away: capitalisation ("oh GREAT"), repeated punctuation ("nice!!!"),
scare quotes, hashtags (``#blessed``), ellipses and emoji. So this module does
**not** lowercase, does not strip punctuation, and does not remove stop words.

What it *does* do is remove the things that are pure noise for a sarcasm
classifier and that would otherwise blow up the token budget:

* HTML entities left over from Reddit/Twitter dumps (``&amp;`` -> ``&``),
* URLs and @-mentions, replaced by stable placeholder tokens so the model can
  still learn "there was a link here" without memorising domains,
* Reddit/Twitter artefacts (``/u/name``, ``RT @x:``, zero-width characters),
* runaway character repetition (``!!!!!!!!!!!!!!!!`` -> ``!!!!``), which is
  capped rather than removed so the *emphasis signal survives* while the
  tokeniser does not waste 30 sub-word slots on it.

Every transformation is individually testable and individually switchable via
:class:`PreprocessingOptions`.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)

URL_TOKEN = "<url>"
USER_TOKEN = "@user"
SUBREDDIT_TOKEN = "<subreddit>"

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<![\w/])@(\w{1,30})\b")
_REDDIT_USER_RE = re.compile(r"(?<![\w])/?u/[A-Za-z0-9_\-]{1,30}\b")
_SUBREDDIT_RE = re.compile(r"(?<![\w])/?r/([A-Za-z0-9_]{1,30})\b")
_RT_RE = re.compile(r"^RT\s+@\w+\s*:\s*", re.IGNORECASE)
_ZERO_WIDTH_RE = re.compile(r"[​-‏‪-‮﻿]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t  - ]+")
_NEWLINES_RE = re.compile(r"(?:\r\n|\r|\n)+")
_REPEAT_PUNCT_RE = re.compile(r"([!?.,\-*~_=#])\1{2,}")
_REPEAT_CHAR_RE = re.compile(r"([A-Za-z])\1{3,}")
_MULTI_QUOTE_RE = re.compile(r'["“”]{2,}')
_EMPTY_MARKERS = frozenset({"[deleted]", "[removed]", "[deleted by user]", "nan", "none", ""})


@dataclass(frozen=True)
class PreprocessingOptions:
    """Switches for :func:`clean_text`.

    Attributes:
        replace_urls: Swap URLs for :data:`URL_TOKEN`.
        replace_mentions: Swap ``@handle`` / ``/u/handle`` for :data:`USER_TOKEN`.
        replace_subreddits: Swap ``/r/name`` for ``<subreddit>``. Kept separate
            because the subreddit name occasionally *is* the joke.
        unescape_html_entities: Decode ``&amp;`` etc.
        normalise_unicode: Apply NFKC so visually identical characters compare
            equal. Emoji are unaffected by NFKC.
        strip_retweet_prefix: Remove a leading ``RT @user:``.
        max_punct_repeat: Cap runs of the same punctuation mark at this length.
            ``0`` disables the cap.
        max_char_repeat: Cap runs of the same letter ("soooooo") at this length.
            ``0`` disables the cap.
        collapse_newlines: Replace newline runs with a single space.
        max_chars: Hard truncation guard against pathological inputs.
    """

    replace_urls: bool = True
    replace_mentions: bool = True
    replace_subreddits: bool = False
    unescape_html_entities: bool = True
    normalise_unicode: bool = True
    strip_retweet_prefix: bool = True
    max_punct_repeat: int = 4
    max_char_repeat: int = 3
    collapse_newlines: bool = True
    max_chars: int = 4000


DEFAULT_OPTIONS = PreprocessingOptions()


def clean_text(text: str | None, options: PreprocessingOptions = DEFAULT_OPTIONS) -> str:
    """Normalise one piece of text without destroying sarcasm cues.

    Args:
        text: Raw text. ``None`` and non-string inputs return ``""``.
        options: Which transformations to apply.

    Returns:
        The cleaned string, stripped of leading/trailing whitespace. May be
        empty, which callers must treat as "drop this record".

    Example:
        >>> clean_text("Oh GREAT!!!!!!! check https://x.co @bob &amp; co")
        'Oh GREAT!!!! check <url> @user & co'
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    if options.max_chars and len(text) > options.max_chars:
        text = text[: options.max_chars]

    if options.unescape_html_entities:
        # Twitter dumps are frequently double-escaped (``&amp;amp;``).
        for _ in range(2):
            unescaped = html.unescape(text)
            if unescaped == text:
                break
            text = unescaped

    if options.normalise_unicode:
        text = unicodedata.normalize("NFKC", text)

    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)

    if options.strip_retweet_prefix:
        text = _RT_RE.sub("", text)
    if options.replace_urls:
        text = _URL_RE.sub(URL_TOKEN, text)
    if options.replace_mentions:
        text = _REDDIT_USER_RE.sub(USER_TOKEN, text)
        text = _MENTION_RE.sub(USER_TOKEN, text)
    if options.replace_subreddits:
        text = _SUBREDDIT_RE.sub(SUBREDDIT_TOKEN, text)

    if options.collapse_newlines:
        text = _NEWLINES_RE.sub(" ", text)

    if options.max_punct_repeat > 0:
        keep = options.max_punct_repeat
        text = _REPEAT_PUNCT_RE.sub(lambda m: m.group(1) * keep, text)
    if options.max_char_repeat > 0:
        keep = options.max_char_repeat
        text = _REPEAT_CHAR_RE.sub(lambda m: m.group(1) * keep, text)
    text = _MULTI_QUOTE_RE.sub('"', text)

    text = _WS_RE.sub(" ", text)
    return text.strip()


def clean_context(
    turns: Iterable[str] | None,
    max_turns: int = 3,
    options: PreprocessingOptions = DEFAULT_OPTIONS,
) -> list[str]:
    """Clean a list of conversational turns, preserving chronological order.

    Empty/placeholder turns (Reddit's ``[deleted]``) are dropped *before*
    truncation so that a deleted turn does not consume one of the three slots.
    Only the most recent ``max_turns`` turns are kept, because they are the ones
    adjacent to the target utterance.
    """
    if not turns:
        return []
    cleaned: list[str] = []
    for turn in turns:
        value = clean_text(turn, options)
        if is_placeholder(value):
            continue
        cleaned.append(value)
    if max_turns >= 0:
        cleaned = cleaned[-max_turns:] if max_turns else []
    return cleaned


def is_placeholder(text: str) -> bool:
    """Whether a string is effectively empty (deleted comment, ``nan``, ...)."""
    return text.strip().lower() in _EMPTY_MARKERS


def is_valid_text(text: str, min_chars: int = 2, min_alnum: int = 1) -> bool:
    """Reject rows that cannot possibly carry a sarcasm judgement.

    A record must have at least ``min_chars`` characters and ``min_alnum``
    alphanumeric characters. Pure-emoji rows fail the alphanumeric test and are
    dropped, because neither RoBERTa nor a human annotator can reliably label
    them in a text-only setting.
    """
    if not text or is_placeholder(text):
        return False
    if len(text.strip()) < min_chars:
        return False
    return sum(ch.isalnum() for ch in text) >= min_alnum


def coerce_label(value: object) -> int | None:
    """Map heterogeneous label spellings onto {0, 1}; ``None`` when invalid.

    Handles the label vocabularies actually found in the four corpora:
    integers/floats (0/1), ``"SARCASM"``/``"NOT_SARCASM"`` (FigLang2020),
    ``"sarcastic"``/``"not sarcastic"`` and ``True``/``False``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        as_int = int(value)
        return as_int if as_int in (0, 1) else None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    positive = {"1", "sarcasm", "sarcastic", "true", "yes", "irony", "ironic"}
    negative = {"0", "not_sarcasm", "not_sarcastic", "non_sarcastic", "false", "no", "literal"}
    if text in positive:
        return 1
    if text in negative:
        return 0
    return None


def build_context_string(
    turns: list[str],
    separator: str = " </s> ",
    prefix_turns: bool = True,
) -> str:
    """Flatten conversational turns into one string for the shared encoder.

    Args:
        turns: Chronologically ordered turns (oldest first).
        separator: Inserted between turns. The default is RoBERTa's ``</s>``
            separator, which the tokeniser maps to a real special token, giving
            the encoder an explicit turn boundary signal.
        prefix_turns: Prepend ``turn 1:`` / ``turn 2:`` markers so the encoder
            can tell *how far back* a turn is, not merely that a boundary exists.

    Returns:
        The flattened context string; ``""`` when there are no turns.
    """
    if not turns:
        return ""
    if prefix_turns:
        pieces = [f"turn {i}: {turn}" for i, turn in enumerate(turns, start=1)]
    else:
        pieces = list(turns)
    return separator.join(pieces)
