"""Schema, dataset adapters, validation and splitting."""

from __future__ import annotations

import pytest

from src.data.loaders import ALL_DATASET_KEYS, build_adapter
from src.data.schema import LABEL_NAMES, SarcasmRecord, SchemaError
from src.data.splitting import (
    DEFAULT_RATIOS,
    SplitError,
    apply_split,
    build_conversation_groups,
    corpus_fingerprint,
    load_split,
    save_split,
    split_records,
)
from src.data.validation import check_leakage, deduplicate, validate_records


class TestSchema:
    """The common record schema enforces its own invariants."""

    def test_valid_record(self):
        record = SarcasmRecord(id="a", text="hello there", label=1, dataset="d")
        assert record.label_name == LABEL_NAMES[1]
        assert record.has_context is False
        assert record.conversation_id == "a"

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_text_is_rejected(self, text):
        with pytest.raises(SchemaError):
            SarcasmRecord(id="a", text=text, label=1, dataset="d")

    @pytest.mark.parametrize("label", [2, -1, "sarcastic", None])
    def test_invalid_label_is_rejected(self, label):
        with pytest.raises(SchemaError):
            SarcasmRecord(id="a", text="hi there", label=label, dataset="d")

    def test_context_is_truncated_to_three_turns(self):
        record = SarcasmRecord(
            id="a", text="hi there", label=1, dataset="d",
            context=["1", "2", "3", "4", "5"],
        )
        assert record.context == ["3", "4", "5"]

    def test_blank_context_turns_are_dropped(self):
        record = SarcasmRecord(
            id="a", text="hi there", label=0, dataset="d", context=["a", "  ", "b"]
        )
        assert record.context == ["a", "b"]

    def test_round_trip_through_dict(self):
        original = SarcasmRecord(
            id="a", text="hi there", label=1, dataset="d",
            context=["x"], domain="reddit", metadata={"k": "v"},
        )
        assert SarcasmRecord.from_dict(original.to_dict()).to_dict() == original.to_dict()

    def test_content_hash_ignores_case_and_whitespace(self):
        a = SarcasmRecord(id="1", text="Hello   There", label=1, dataset="d")
        b = SarcasmRecord(id="2", text="hello there", label=1, dataset="d")
        assert a.content_hash == b.content_hash

    def test_content_hash_respects_punctuation(self):
        a = SarcasmRecord(id="1", text="great", label=1, dataset="d")
        b = SarcasmRecord(id="2", text="great!!!", label=1, dataset="d")
        assert a.content_hash != b.content_hash

    def test_unknown_domain_falls_back(self):
        record = SarcasmRecord(id="a", text="hi there", label=1, dataset="d", domain="nonsense")
        assert record.domain == "unknown"


class TestAdapters:
    """Every registered adapter loads its own format into the common schema."""

    def test_all_adapters_are_registered(self):
        assert set(ALL_DATASET_KEYS) == {
            "figlang2020_twitter", "figlang2020_reddit", "mustard",
            "news_headlines", "sarc",
        }

    def test_figlang_adapter(self, raw_dir):
        adapter = build_adapter("figlang2020_twitter", raw_dir=raw_dir)
        assert adapter.available()
        records = list(adapter.load())
        assert len(records) == 2  # the empty and bad-label rows are dropped
        assert adapter.stats.dropped_empty_text == 1
        assert adapter.stats.dropped_invalid_label == 1
        assert records[0].label == 1
        assert records[0].context == ["forecast says sun", "it is pouring"]
        assert records[0].domain == "twitter"

    def test_mustard_adapter_prefixes_speakers_on_context_only(self, raw_dir):
        records = list(build_adapter("mustard", raw_dir=raw_dir).load())
        assert len(records) == 2
        assert records[0].context[0].startswith("LEONARD:")
        assert not records[0].text.startswith("SHELDON:")
        assert records[0].metadata["speaker"] == "SHELDON"

    def test_mustard_groups_by_shared_opening_context(self, raw_dir):
        records = list(build_adapter("mustard", raw_dir=raw_dir).load())
        assert records[0].conversation_id == records[1].conversation_id

    def test_headlines_adapter_emits_no_context(self, raw_dir):
        records = list(build_adapter("news_headlines", raw_dir=raw_dir).load())
        assert len(records) == 2
        assert all(record.context == [] for record in records)
        assert records[0].metadata["publisher"] == "theonion.com"

    def test_sarc_adapter_uses_parent_as_context(self, raw_dir):
        records = list(build_adapter("sarc", raw_dir=raw_dir).load())
        assert len(records) == 2
        assert records[0].context == ["we should just ignore the problem"]
        assert records[0].metadata["layout"] == "A"

    def test_missing_files_are_reported_not_crashed(self, tmp_path):
        adapter = build_adapter("mustard", raw_dir=tmp_path)
        assert adapter.available() is False
        assert adapter.missing_paths()
        with pytest.raises(FileNotFoundError):
            list(adapter.load())

    def test_unknown_dataset_key(self):
        with pytest.raises(KeyError):
            build_adapter("no_such_dataset")

    def test_max_rows_caps_output(self, raw_dir):
        adapter = build_adapter("figlang2020_twitter", raw_dir=raw_dir, max_rows=1)
        assert len(list(adapter.load())) == 1


class TestSubsampling:
    """``max_rows`` must be a seeded random sample, not a head slice.

    SARC's distributed CSV is roughly date-ordered, so a head slice would
    silently restrict the corpus to the earliest slice of Reddit history.
    """

    @pytest.fixture()
    def big_sarc(self, tmp_path):
        """A SARC-shaped CSV whose rows are ordered, so bias is detectable."""
        directory = tmp_path / "sarc"
        directory.mkdir(parents=True)
        rows = [
            "label,comment,author,subreddit,score,ups,downs,date,created_utc,parent_comment"
        ]
        for i in range(500):
            # The subreddit encodes position, so a head slice is obvious.
            bucket = "early" if i < 250 else "late"
            rows.append(
                f"{i % 2},comment number {i} with enough words,u{i},{bucket},"
                f"1,1,0,2016-01,2016-01-01 00:00:00,parent comment number {i}"
            )
        (directory / "train-balanced-sarcasm.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        return tmp_path

    def test_sample_size_is_respected(self, big_sarc):
        adapter = build_adapter("sarc", raw_dir=big_sarc, max_rows=100, seed=1)
        assert len(list(adapter.load())) == 100

    def test_sample_is_not_a_head_slice(self, big_sarc):
        records = list(build_adapter("sarc", raw_dir=big_sarc, max_rows=100, seed=1).load())
        buckets = {r.metadata["subreddit"] for r in records}
        assert buckets == {"early", "late"}, "sample should span the whole file"

    def test_sample_is_roughly_uniform(self, big_sarc):
        records = list(build_adapter("sarc", raw_dir=big_sarc, max_rows=200, seed=1).load())
        late = sum(1 for r in records if r.metadata["subreddit"] == "late")
        # A uniform sample of 200 from 500 should give ~100 from each half.
        assert 70 <= late <= 130

    def test_sample_is_reproducible(self, big_sarc):
        first = [r.id for r in build_adapter("sarc", raw_dir=big_sarc, max_rows=50, seed=7).load()]
        second = [r.id for r in build_adapter("sarc", raw_dir=big_sarc, max_rows=50, seed=7).load()]
        assert first == second

    def test_different_seeds_give_different_samples(self, big_sarc):
        a = [r.id for r in build_adapter("sarc", raw_dir=big_sarc, max_rows=50, seed=1).load()]
        b = [r.id for r in build_adapter("sarc", raw_dir=big_sarc, max_rows=50, seed=2).load()]
        assert a != b

    def test_file_order_is_preserved_within_the_sample(self, big_sarc):
        adapter = build_adapter("sarc", raw_dir=big_sarc, max_rows=50, seed=3)
        ids = [int(r.id.split(":a")[1]) for r in adapter.load()]
        assert ids == sorted(ids)

    def test_stats_report_the_population_size(self, big_sarc):
        adapter = build_adapter("sarc", raw_dir=big_sarc, max_rows=100, seed=1)
        list(adapter.load())
        stats = adapter.stats.as_dict()
        assert stats["emitted"] == 100
        assert stats["sampled_from"] == 500
        assert sum(stats["context_turn_histogram"].values()) == 100

    def test_cap_larger_than_the_corpus_keeps_everything(self, big_sarc):
        adapter = build_adapter("sarc", raw_dir=big_sarc, max_rows=10_000, seed=1)
        assert len(list(adapter.load())) == 500

    def test_no_cap_keeps_everything_and_reports_no_sampling(self, big_sarc):
        adapter = build_adapter("sarc", raw_dir=big_sarc, max_rows=None)
        assert len(list(adapter.load())) == 500
        assert adapter.stats.as_dict()["sampled_from"] == 0

    def test_max_context_turns_is_respected(self, raw_dir):
        adapter = build_adapter("figlang2020_twitter", raw_dir=raw_dir, max_context_turns=1)
        records = list(adapter.load())
        assert all(len(record.context) <= 1 for record in records)


class TestValidation:
    """The corpus report and deduplication behave as documented."""

    def test_report_counts(self, fixture_records):
        report = validate_records(fixture_records, name="fixture")
        assert report.n_rows == len(fixture_records)
        assert sum(report.class_distribution.values()) == report.n_rows
        assert 0.0 <= report.context_availability <= 1.0

    def test_duplicates_are_detected(self):
        records = [
            SarcasmRecord(id="1", text="the same thing", label=1, dataset="d"),
            SarcasmRecord(id="2", text="The Same   Thing", label=1, dataset="d"),
            SarcasmRecord(id="3", text="different", label=0, dataset="d"),
        ]
        assert validate_records(records).n_duplicate_content == 1

    def test_deduplicate_keeps_first(self):
        records = [
            SarcasmRecord(id="1", text="repeated line", label=1, dataset="d"),
            SarcasmRecord(id="2", text="repeated line", label=1, dataset="d"),
        ]
        kept, removed, conflicts = deduplicate(records)
        assert [r.id for r in kept] == ["1"]
        assert (removed, conflicts) == (1, 0)

    def test_conflicting_labels_are_dropped_entirely(self):
        records = [
            SarcasmRecord(id="1", text="ambiguous line", label=1, dataset="d"),
            SarcasmRecord(id="2", text="ambiguous line", label=0, dataset="d"),
        ]
        kept, removed, conflicts = deduplicate(records)
        assert kept == []
        assert (removed, conflicts) == (2, 1)

    def test_empty_corpus_warns(self):
        assert "corpus is empty" in validate_records([]).warnings


class TestSplitting:
    """Splits are stratified, conversation-level, reproducible and leak-free."""

    def test_ratios_are_respected(self, fixture_records):
        assignment = split_records(fixture_records, seed=1)
        total = len(fixture_records)
        assert sum(assignment.sizes.values()) == total
        assert abs(len(assignment.train) / total - DEFAULT_RATIOS[0]) < 0.12

    def test_split_is_deterministic(self, fixture_records):
        a = split_records(fixture_records, seed=7)
        b = split_records(fixture_records, seed=7)
        assert a.train == b.train and a.test == b.test

    def test_different_seeds_give_different_splits(self, fixture_records):
        a = split_records(fixture_records, seed=1)
        b = split_records(fixture_records, seed=2)
        assert a.train != b.train

    def test_no_record_appears_twice(self, fixture_records):
        assignment = split_records(fixture_records, seed=3)
        ids = assignment.train + assignment.val + assignment.test
        assert len(ids) == len(set(ids)) == len(fixture_records)

    def test_conversations_are_not_split(self, fixture_records):
        assignment = split_records(fixture_records, seed=5)
        splits = apply_split(fixture_records, assignment)
        seen: dict[str, str] = {}
        for name, rows in splits.items():
            for record in rows:
                previous = seen.setdefault(record.conversation_id, name)
                assert previous == name, f"conversation {record.conversation_id} spans splits"

    def test_leakage_check_passes_on_a_real_split(self, fixture_records):
        splits = apply_split(fixture_records, split_records(fixture_records, seed=11))
        assert check_leakage(splits).clean is True

    def test_leakage_check_detects_a_deliberate_leak(self, fixture_records):
        rows = list(fixture_records)
        leaked = check_leakage({"train": rows[:5], "test": rows[3:8]})
        assert leaked.clean is False
        assert leaked.shared_conversations

    def test_shared_utterances_are_merged_into_one_cluster(self):
        records = [
            SarcasmRecord(id="1", text="the model attends over the context turns",
                          label=1, dataset="d", conversation_id="c1"),
            SarcasmRecord(id="2", text="another response entirely here",
                          label=0, dataset="d", conversation_id="c2",
                          context=["the model attends over the context turns"]),
        ]
        groups = build_conversation_groups(records)
        assert len(groups) == 1

    def test_short_strings_do_not_chain_conversations(self):
        records = [
            SarcasmRecord(id="1", text="no.", label=1, dataset="d", conversation_id="c1"),
            SarcasmRecord(id="2", text="why", label=0, dataset="d",
                          conversation_id="c2", context=["no."]),
        ]
        assert len(build_conversation_groups(records)) == 2

    @pytest.mark.parametrize("ratios", [(0.5, 0.3, 0.3), (0.8, 0.1, 0.0, 0.1), (-0.1, 0.5, 0.6)])
    def test_invalid_ratios_are_rejected(self, fixture_records, ratios):
        with pytest.raises((SplitError, TypeError, ValueError)):
            split_records(fixture_records, ratios=ratios)  # type: ignore[arg-type]

    def test_empty_corpus_is_rejected(self):
        with pytest.raises(SplitError):
            split_records([])

    def test_save_and_load_round_trip(self, fixture_records, tmp_path):
        assignment = split_records(fixture_records, seed=13)
        save_split(assignment, tmp_path)
        loaded = load_split(tmp_path, expected_fingerprint=corpus_fingerprint(fixture_records))
        assert loaded.train == assignment.train
        assert loaded.seed == 13

    def test_stale_split_is_rejected(self, fixture_records, tmp_path):
        save_split(split_records(fixture_records, seed=1), tmp_path)
        with pytest.raises(SplitError, match="does not match the corpus"):
            load_split(tmp_path, expected_fingerprint="deadbeefdeadbeef")

    def test_apply_split_rejects_unknown_ids(self, fixture_records, tmp_path):
        assignment = split_records(fixture_records, seed=1)
        assignment.train.append("fixture:does-not-exist")
        with pytest.raises(SplitError):
            apply_split(fixture_records, assignment)
