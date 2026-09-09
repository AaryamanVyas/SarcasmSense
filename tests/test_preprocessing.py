"""Preprocessing must remove noise while preserving every sarcasm cue."""

from __future__ import annotations

import pytest

from src.data.preprocessing import (
    PreprocessingOptions,
    build_context_string,
    clean_context,
    clean_text,
    coerce_label,
    is_placeholder,
    is_valid_text,
)


class TestSarcasmCuePreservation:
    """The cues sarcasm actually lives in must survive cleaning."""

    def test_capitalisation_is_preserved(self):
        assert "GREAT" in clean_text("oh GREAT another delay")

    def test_emoji_are_preserved(self):
        assert "🙄" in clean_text("lovely 🙄")

    def test_hashtags_are_preserved(self):
        assert "#blessed" in clean_text("another 6am start #blessed")

    def test_repeated_punctuation_is_capped_not_removed(self):
        cleaned = clean_text("wow!!!!!!!!!!!!")
        assert "!!!!" in cleaned
        assert "!!!!!" not in cleaned

    def test_repeated_letters_are_capped_not_removed(self):
        cleaned = clean_text("sooooooooo good")
        assert "sooo good" == cleaned

    def test_scare_quotes_survive(self):
        assert '"help"' in clean_text('thanks for the "help"')

    def test_ellipsis_survives(self):
        assert "..." in clean_text("sure... whatever you say")


class TestNoiseRemoval:
    """URLs, mentions and encoding artefacts are normalised away."""

    def test_urls_become_a_placeholder(self):
        assert clean_text("see https://example.com/x?y=1 now") == "see <url> now"

    def test_www_urls_are_matched(self):
        assert "<url>" in clean_text("go to www.example.com")

    def test_mentions_become_a_placeholder(self):
        assert clean_text("@alice said hi") == "@user said hi"

    def test_reddit_usernames_are_normalised(self):
        assert "@user" in clean_text("/u/somebody disagreed")

    def test_html_entities_are_decoded(self):
        assert clean_text("fish &amp; chips") == "fish & chips"

    def test_double_escaped_entities_are_decoded(self):
        assert clean_text("fish &amp;amp; chips") == "fish & chips"

    def test_retweet_prefix_is_stripped(self):
        assert clean_text("RT @bob: nice one") == "nice one"

    def test_whitespace_is_collapsed(self):
        assert clean_text("  too    many \t spaces \n here ") == "too many spaces here"

    def test_zero_width_characters_are_removed(self):
        assert clean_text("a​b") == "ab"

    def test_subreddits_are_kept_by_default(self):
        assert "/r/science" in clean_text("posted in /r/science")

    def test_subreddits_can_be_replaced(self):
        options = PreprocessingOptions(replace_subreddits=True)
        assert "<subreddit>" in clean_text("posted in /r/science", options)


class TestEdgeCases:
    """Malformed input must never raise."""

    @pytest.mark.parametrize("value", [None, "", "   ", "\n\t", 42, 3.5])
    def test_never_raises(self, value):
        assert isinstance(clean_text(value), str)

    def test_very_long_text_is_truncated(self):
        options = PreprocessingOptions(max_chars=100)
        assert len(clean_text("x " * 5000, options)) <= 100

    def test_pure_emoji_is_not_valid_text(self):
        assert is_valid_text("😀😀") is False

    def test_single_character_is_not_valid_text(self):
        assert is_valid_text("a") is False

    def test_normal_text_is_valid(self):
        assert is_valid_text("this is fine") is True

    @pytest.mark.parametrize("marker", ["[deleted]", "[removed]", "nan", "", "  "])
    def test_placeholder_markers_are_detected(self, marker):
        assert is_placeholder(marker) is True


class TestLabelCoercion:
    """Every label vocabulary found in the four corpora maps onto {0, 1}."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (1, 1), (0, 0), (True, 1), (False, 0), ("1", 1), ("0", 0),
            ("SARCASM", 1), ("NOT_SARCASM", 0), ("sarcastic", 1),
            ("not sarcastic", 0), ("Not-Sarcasm", 0), ("true", 1), ("no", 0),
        ],
    )
    def test_known_spellings(self, value, expected):
        assert coerce_label(value) == expected

    @pytest.mark.parametrize("value", [None, "maybe", 7, -1, "", "unknown"])
    def test_unknown_spellings_return_none(self, value):
        assert coerce_label(value) is None


class TestContextHandling:
    """Context is cleaned, ordered and truncated to the most recent turns."""

    def test_empty_context_returns_empty_list(self):
        assert clean_context(None) == []
        assert clean_context([]) == []

    def test_deleted_turns_are_dropped_before_truncation(self):
        turns = ["[deleted]", "first real", "second real", "third real"]
        assert clean_context(turns, max_turns=3) == ["first real", "second real", "third real"]

    def test_only_the_most_recent_turns_are_kept(self):
        turns = ["oldest", "older", "old", "recent"]
        assert clean_context(turns, max_turns=2) == ["old", "recent"]

    def test_order_is_preserved(self):
        turns = ["one", "two", "three"]
        assert clean_context(turns, max_turns=3) == turns

    def test_zero_max_turns_drops_everything(self):
        assert clean_context(["a", "b"], max_turns=0) == []

    def test_context_string_marks_turn_positions(self):
        flattened = build_context_string(["a", "b"], separator=" | ")
        assert flattened == "turn 1: a | turn 2: b"

    def test_context_string_without_prefixes(self):
        assert build_context_string(["a", "b"], separator=" | ", prefix_turns=False) == "a | b"

    def test_empty_turns_give_empty_string(self):
        assert build_context_string([]) == ""
