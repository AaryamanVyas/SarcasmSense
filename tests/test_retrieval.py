"""Sentence encoders, the FAISS index and the commonsense retriever."""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.cache import (
    CacheMismatchError,
    RetrievalCache,
    ensure_retrieval_cache,
)
from src.retrieval.encoder import (
    BACKENDS,
    HashingEncoder,
    build_sentence_encoder,
)
from src.retrieval.faiss_index import (
    IndexError_,
    KnowledgeIndex,
    index_exists,
    peek_metadata,
)
from src.retrieval.knowledge_base import (
    CONCEPTNET_RELATIONS,
    KnowledgeFact,
    iter_conceptnet_facts,
    read_facts,
    write_facts,
)
from src.retrieval.retriever import (
    CommonsenseRetriever,
    RetrievalStatus,
    build_query,
)


class TestEncoders:
    """Encoders return L2-normalised float32 matrices of the right shape."""

    def test_hashing_encoder_shape_and_norm(self, sentence_encoder):
        vectors = sentence_encoder.encode(["hello world", "another sentence"])
        assert vectors.shape == (2, sentence_encoder.dimension)
        assert vectors.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_hashing_encoder_is_deterministic(self):
        a = HashingEncoder(32).encode(["repeatable text"])
        b = HashingEncoder(32).encode(["repeatable text"])
        np.testing.assert_array_equal(a, b)

    def test_similar_texts_score_higher_than_unrelated(self, sentence_encoder):
        vectors = sentence_encoder.encode(
            ["the laptop crashed during the presentation",
             "the laptop crashed in the presentation",
             "a recipe for lemon cheesecake"]
        )
        assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])

    def test_empty_input(self, sentence_encoder):
        assert sentence_encoder.encode([]).shape == (0, sentence_encoder.dimension)

    def test_encode_one(self, sentence_encoder):
        assert sentence_encoder.encode_one("x").shape == (sentence_encoder.dimension,)

    def test_describe_reports_fallback_status(self, sentence_encoder):
        described = sentence_encoder.describe()
        assert described["is_fallback"] is True
        assert described["backend"] == "hashing"

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError):
            build_sentence_encoder(backend="not-a-backend")

    def test_hashing_backend_can_be_forced(self):
        encoder = build_sentence_encoder(backend="hashing", dimension=48)
        assert encoder.backend == "hashing" and encoder.dimension == 48

    def test_backend_names(self):
        assert BACKENDS == ("auto", "sentence-transformers", "hashing")


class TestKnowledgeBase:
    """Fact verbalisation and persistence."""

    def test_conceptnet_templates_cover_sarcasm_relevant_relations(self):
        for relation in ("Causes", "HasProperty", "Antonym", "ObstructedBy", "NotDesires"):
            assert relation in CONCEPTNET_RELATIONS

    def test_conceptnet_parsing(self, tmp_path):
        dump = tmp_path / "assertions.csv"
        dump.write_text(
            "\n".join(
                [
                    "/a/x\t/r/Causes\t/c/en/rain\t/c/en/wetness\t"
                    '{"weight": 2.0}',
                    "/a/y\t/r/Causes\t/c/fr/pluie\t/c/fr/humidite\t"
                    '{"weight": 2.0}',  # dropped: not English
                    "/a/z\t/r/ExternalURL\t/c/en/a\t/c/en/b\t"
                    '{"weight": 5.0}',  # dropped: relation not whitelisted
                    "/a/w\t/r/Causes\t/c/en/a\t/c/en/b\t"
                    '{"weight": 0.2}',  # dropped: below min weight
                ]
            ),
            encoding="utf-8",
        )
        facts = list(iter_conceptnet_facts(dump, min_weight=1.0))
        assert len(facts) == 1
        assert facts[0].text == "rain causes wetness"
        assert facts[0].source == "ConceptNet"
        assert facts[0].weight == 2.0

    def test_facts_round_trip(self, tmp_path, fixture_facts):
        path = tmp_path / "facts.jsonl"
        assert write_facts(path, fixture_facts) == len(fixture_facts)
        restored = read_facts(path)
        assert [f.text for f in restored] == [f.text for f in fixture_facts]
        assert restored[0].relation == fixture_facts[0].relation


class TestIndex:
    """Building, saving, loading and querying the FAISS index."""

    def test_build_and_query(self, knowledge_index, sentence_encoder):
        assert len(knowledge_index) > 0
        query = sentence_encoder.encode(["rain causes wetness"])
        scores, ids = knowledge_index.search(query, top_k=3)
        assert scores.shape == (1, 3) and ids.shape == (1, 3)
        assert knowledge_index.fact_at(int(ids[0][0])) is not None

    def test_metadata_records_provenance(self, knowledge_index, sentence_encoder):
        metadata = knowledge_index.metadata
        assert metadata.encoder_name == sentence_encoder.name
        assert metadata.is_sample_index is True
        assert metadata.n_facts == len(knowledge_index)

    def test_index_exists_helper(self, knowledge_index, tmp_path):
        assert index_exists(tmp_path) is False
        knowledge_index.save(tmp_path)
        assert index_exists(tmp_path) is True
        assert peek_metadata(tmp_path).n_facts == len(knowledge_index)

    def test_loading_with_wrong_encoder_is_refused(self, knowledge_index, tmp_path):
        knowledge_index.save(tmp_path)
        with pytest.raises(IndexError_, match="built with encoder"):
            KnowledgeIndex.load(tmp_path, expected_encoder_name="some-other-model")

    def test_loading_incomplete_index_is_refused(self, tmp_path):
        with pytest.raises(IndexError_, match="incomplete"):
            KnowledgeIndex.load(tmp_path)

    def test_building_over_zero_facts_is_refused(self, sentence_encoder):
        with pytest.raises(IndexError_):
            KnowledgeIndex.build([], sentence_encoder)

    def test_wrong_query_dimension_is_refused(self, knowledge_index):
        with pytest.raises(IndexError_, match="dimension"):
            knowledge_index.search(np.zeros((1, 7), dtype=np.float32), top_k=1)

    def test_top_k_larger_than_index_is_clamped(self, knowledge_index, sentence_encoder):
        query = sentence_encoder.encode(["anything"])
        scores, _ids = knowledge_index.search(query, top_k=10_000)
        assert scores.shape[1] == len(knowledge_index)

    def test_reconstruct_returns_the_indexed_vectors(self, knowledge_index, sentence_encoder):
        vectors = knowledge_index.reconstruct([0, 1])
        assert vectors.shape == (2, knowledge_index.dimension)
        expected = sentence_encoder.encode([knowledge_index.facts[0].text])[0]
        np.testing.assert_allclose(vectors[0], expected, atol=1e-4)

    def test_reconstruct_with_no_ids(self, knowledge_index):
        assert knowledge_index.reconstruct([]).shape == (0, knowledge_index.dimension)


class TestRetriever:
    """Retrieval returns structured, cited evidence and degrades honestly."""

    def test_returns_structured_evidence(self, retriever):
        result = retriever.retrieve("my laptop crashed", [])
        assert result.status is RetrievalStatus.OK
        assert 1 <= len(result.evidence) <= 3
        fact = result.evidence[0]
        assert fact.text and fact.source and fact.fact_id
        assert -1.0 <= fact.score <= 1.0
        assert isinstance(fact.embedding_id, int)

    def test_query_uses_the_most_recent_turn(self):
        query = build_query("perfect", ["oldest", "newest"], "utterance_plus_context")
        assert query == "newest perfect"

    def test_query_mode_utterance_only(self):
        assert build_query("perfect", ["newest"], "utterance") == "perfect"

    def test_missing_index_reports_no_index(self, sentence_encoder):
        retriever = CommonsenseRetriever(None, sentence_encoder, top_k=3)
        result = retriever.retrieve("anything")
        assert result.status is RetrievalStatus.NO_INDEX
        assert result.evidence == []
        assert "unavailable" in (result.warning or "")
        assert retriever.available is False

    def test_disabled_retrieval(self, knowledge_index, sentence_encoder):
        retriever = CommonsenseRetriever(knowledge_index, sentence_encoder, enabled=False)
        assert retriever.retrieve("anything").status is RetrievalStatus.DISABLED

    def test_threshold_can_exclude_everything(self, knowledge_index, sentence_encoder):
        retriever = CommonsenseRetriever(
            knowledge_index, sentence_encoder, top_k=3, min_score=0.999
        )
        result = retriever.retrieve("completely unrelated gibberish zzz")
        assert result.status is RetrievalStatus.NO_RESULTS_ABOVE_THRESHOLD
        assert result.evidence == []
        assert retriever.n_failures >= 1

    def test_batch_matches_single(self, retriever):
        single = retriever.retrieve("rain causes problems")
        batch = retriever.retrieve_batch(["rain causes problems"], [[]])
        assert [e.fact_id for e in batch[0].evidence] == [e.fact_id for e in single.evidence]

    def test_cache_is_used(self, knowledge_index, sentence_encoder):
        retriever = CommonsenseRetriever(knowledge_index, sentence_encoder, top_k=2)
        retriever.retrieve("cached query text")
        before = retriever.n_cache_hits
        retriever.retrieve("cached query text")
        assert retriever.n_cache_hits == before + 1

    def test_evidence_matrix_shape_and_mask(self, retriever):
        results = retriever.retrieve_batch(["rain", "laptop crash"], [[], []])
        vectors, mask = retriever.evidence_matrix(results)
        assert vectors.shape == (2, retriever.top_k, retriever.dimension)
        assert mask.shape == (2, retriever.top_k)
        assert mask.sum() == sum(len(r.evidence) for r in results)

    def test_stats_are_reported(self, retriever):
        stats = retriever.stats()
        assert stats["available"] is True
        assert stats["top_k"] == retriever.top_k
        assert stats["index"]["n_facts"] == len(retriever.index)


class TestRetrievalCache:
    """The pre-computed cache is verified rather than blindly reused."""

    def test_build_and_lookup(self, fixture_records, retriever):
        cache = RetrievalCache.build(fixture_records[:8], retriever)
        assert len(cache) == 8
        ids, scores = cache.get(fixture_records[0].id)
        assert ids.shape == scores.shape == (retriever.top_k,)

    def test_unknown_record_returns_empty_slots(self, fixture_records, retriever):
        cache = RetrievalCache.build(fixture_records[:4], retriever)
        ids, _scores = cache.get("not-in-cache")
        assert (ids == -1).all()

    def test_save_and_load_round_trip(self, fixture_records, retriever, tmp_path):
        cache = RetrievalCache.build(fixture_records[:6], retriever)
        path = cache.save(tmp_path / "cache.npz")
        restored = RetrievalCache.load(path, retriever=retriever)
        np.testing.assert_array_equal(restored.embedding_ids, cache.embedding_ids)

    def test_mismatched_top_k_is_refused(self, fixture_records, retriever, tmp_path, knowledge_index, sentence_encoder):
        cache = RetrievalCache.build(fixture_records[:4], retriever)
        cache.save(tmp_path / "cache.npz")
        other = CommonsenseRetriever(knowledge_index, sentence_encoder, top_k=retriever.top_k + 1)
        with pytest.raises(CacheMismatchError):
            RetrievalCache.load(tmp_path / "cache.npz", retriever=other)

    def test_ensure_rebuilds_when_records_are_missing(
        self, fixture_records, retriever, tmp_path
    ):
        path = tmp_path / "cache.npz"
        RetrievalCache.build(fixture_records[:2], retriever).save(path)
        rebuilt = ensure_retrieval_cache(fixture_records[:8], retriever, path)
        assert len(rebuilt) == 8

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RetrievalCache.load(tmp_path / "absent.npz")
