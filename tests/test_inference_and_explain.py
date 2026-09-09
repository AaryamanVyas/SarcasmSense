"""Inference service, explainers, explanation generation and error analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.schema import SarcasmRecord
from src.evaluation.error_analysis import (
    ErrorAnalysisConfig,
    ErrorCategory,
    build_error_rows,
    categorise,
    summarise_errors,
)
from src.evaluation.metrics import compute_grouped_metrics, compute_metrics
from src.explainability.base import (
    EXPLANATION_METHODS,
    ExplanationResult,
    TokenAttribution,
    agreement,
    normalise_weights,
)
from src.explainability.explanation import SentimentLexicon, generate_explanation
from src.retrieval.retriever import RetrievalStatus


class TestPredictor:
    """Single-example prediction and the batched probability function."""

    def test_predict_returns_a_complete_result(self, predictor):
        result = predictor.predict(
            "Great, my laptop crashed before the presentation.",
            ["I have a presentation at three."],
        )
        assert result.prediction in ("Sarcastic", "Not Sarcastic")
        assert 0.0 <= result.confidence <= 1.0
        assert result.probabilities["sarcastic"] + result.probabilities["not_sarcastic"] == pytest.approx(1.0)
        assert result.context_used == ["I have a presentation at three."]
        assert result.timings_ms["total"] > 0

    def test_context_influence_is_reported_per_turn(self, predictor):
        result = predictor.predict("Perfect.", ["first turn here", "second turn here"])
        assert [c.turn_index for c in result.context_influence] == [1, 2]
        assert all(c.attention >= 0 for c in result.context_influence)

    def test_no_context_gives_no_context_influence(self, predictor):
        result = predictor.predict("A standalone headline about budgets.")
        assert result.context_used == []
        assert result.context_influence == []

    def test_evidence_attention_matches_evidence_count(self, predictor):
        result = predictor.predict("rain causes problems for the commute")
        assert len(result.evidence_attention) == len(result.retrieval.evidence)

    def test_empty_text_is_rejected(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict("   ")

    def test_very_long_text_does_not_crash(self, predictor):
        assert predictor.predict("word " * 3000).prediction in ("Sarcastic", "Not Sarcastic")

    def test_more_than_three_turns_are_truncated(self, predictor):
        result = predictor.predict("target", ["a", "b", "c", "d", "e"])
        assert len(result.context_used) <= 3
        assert result.context_used[-1] == "e"

    def test_preprocessing_is_applied_before_the_model(self, predictor):
        result = predictor.predict("check https://example.com now @bob")
        assert "<url>" in result.text and "@user" in result.text

    def test_predict_proba_shape_and_normalisation(self, predictor):
        probabilities = predictor.predict_proba(["one", "two", "three"], context=["ctx"])
        assert probabilities.shape == (3, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_handles_empty_strings(self, predictor):
        assert np.isfinite(predictor.predict_proba(["", "  "])).all()

    def test_describe(self, predictor):
        described = predictor.describe()
        assert described["model"]["fusion"] == "cross_attention"
        assert described["device"] == "cpu"
        assert described["retrieval"]["available"] is True


class TestExplainers:
    """Every explainer produces normalised, signed attributions."""

    @pytest.mark.parametrize("method", EXPLANATION_METHODS)
    def test_methods_produce_attributions(self, analysis_service, method):
        result = analysis_service.analyze(
            "Oh perfect, another unexpected delay.", ["the train is cancelled"],
            explanation_method=method,
        )
        assert result.explanation.tokens
        assert result.explanation.method in EXPLANATION_METHODS
        assert max(abs(t.weight) for t in result.explanation.tokens) == pytest.approx(1.0, abs=1e-6)

    def test_occlusion_is_exact_and_deterministic(self, analysis_service):
        a = analysis_service.analyze("stable input text", [], explanation_method="occlusion")
        b = analysis_service.analyze("stable input text", [], explanation_method="occlusion")
        assert [t.weight for t in a.explanation.tokens] == [t.weight for t in b.explanation.tokens]
        assert a.explanation.metadata["exact"] is True

    def test_unknown_method_is_rejected(self, analysis_service):
        with pytest.raises(ValueError, match="explanation_method"):
            analysis_service.analyze("text", [], explanation_method="telepathy")

    def test_explanation_can_be_skipped(self, analysis_service):
        result = analysis_service.analyze("text here", [], explain=False)
        assert result.explanation.method == "none"
        assert result.explanation.tokens == []

    def test_single_word_input(self, analysis_service):
        result = analysis_service.analyze("Wonderful.", [], explanation_method="occlusion")
        assert len(result.explanation.tokens) == 1

    def test_availability_is_reported(self, analysis_service):
        availability = analysis_service.explainers.availability()
        assert availability["occlusion"] is True
        assert set(availability) == set(EXPLANATION_METHODS)


class TestAttributionHelpers:
    """Normalisation and the SHAP/LIME agreement measure."""

    def test_normalisation_scales_the_maximum_to_one(self):
        normalised = normalise_weights(
            [TokenAttribution("a", 0.5), TokenAttribution("b", -2.0)]
        )
        assert max(abs(t.weight) for t in normalised) == pytest.approx(1.0)

    def test_normalisation_of_all_zeros_is_safe(self):
        normalised = normalise_weights([TokenAttribution("a", 0.0)])
        assert normalised[0].weight == 0.0

    def test_identical_explanations_agree_perfectly(self):
        tokens = [TokenAttribution(t, w) for t, w in [("a", 1.0), ("b", 0.5), ("c", -0.3)]]
        scores = agreement(
            ExplanationResult(method="lime", tokens=tokens),
            ExplanationResult(method="shap", tokens=list(tokens)),
        )
        assert scores["jaccard_topk"] == pytest.approx(1.0)
        assert scores["sign_agreement"] == pytest.approx(1.0)

    def test_disjoint_explanations_do_not_agree(self):
        scores = agreement(
            ExplanationResult(method="lime", tokens=[TokenAttribution("a", 1.0)]),
            ExplanationResult(method="shap", tokens=[TokenAttribution("z", 1.0)]),
        )
        assert scores["jaccard_topk"] == 0.0

    def test_top_filters_to_positive_by_default(self):
        result = ExplanationResult(
            method="lime",
            tokens=[TokenAttribution("a", 1.0), TokenAttribution("b", -1.0)],
        )
        assert [t.token for t in result.top(5)] == ["a"]
        assert len(result.top(5, positive_only=False)) == 2


class TestExplanationText:
    """The narrative is built from what happened, and cites nothing invented."""

    def test_mentions_prediction_and_confidence(self, analysis_service):
        result = analysis_service.analyze("Oh brilliant.", [], explanation_method="occlusion")
        text = result.explanation_text.lower()
        assert result.prediction.prediction.lower() in text
        assert "%" in result.explanation_text

    def test_only_retrieved_facts_are_cited(self, analysis_service):
        result = analysis_service.analyze(
            "rain causes wetness everywhere", [], explanation_method="occlusion"
        )
        retrieved = {fact.text for fact in result.prediction.retrieval.evidence}
        quoted = [
            segment for segment in result.explanation_text.split("“")[1:]
        ]
        for segment in quoted:
            snippet = segment.split("”")[0]
            # Every quoted fact must be one that was actually retrieved (context
            # turns are also quoted, so allow those too).
            assert (
                snippet in retrieved
                or snippet in result.prediction.context_used
                or any(snippet in turn for turn in result.prediction.context_used)
                or snippet.endswith("…")
            )

    def test_missing_context_is_stated_explicitly(self, analysis_service):
        result = analysis_service.analyze("Standalone line.", [], explanation_method="occlusion")
        assert "no conversational context" in result.explanation_text.lower()

    def test_retrieval_failure_is_stated_explicitly(self, predictor, experiment_config):
        from src.inference.service import SarcasmAnalysisService
        from src.retrieval.retriever import CommonsenseRetriever

        broken = SarcasmPredictorWithoutIndex(predictor)
        service = SarcasmAnalysisService(broken, experiment_config)
        result = service.analyze("anything at all", [], explanation_method="occlusion")
        assert "no commonsense evidence" in result.explanation_text.lower()
        assert result.prediction.retrieval.status is RetrievalStatus.NO_INDEX

    def test_low_confidence_is_hedged(self, analysis_service, monkeypatch):
        result = analysis_service.analyze("Something ambiguous.", [], explanation_method="occlusion")
        if result.prediction.confidence < 0.60:
            assert "not" in result.explanation_text.lower()

    def test_incongruity_is_reported_either_way(self, analysis_service):
        result = analysis_service.analyze("Great, a delay.", [], explanation_method="occlusion")
        assert result.incongruity.description
        assert isinstance(result.incongruity.detected, bool)

    def test_sentiment_lexicon_degrades_safely(self):
        lexicon = SentimentLexicon(nltk_data_dir=None)
        assert isinstance(lexicon.polarity("great"), float)


class SarcasmPredictorWithoutIndex:
    """Test double: a predictor whose retriever has no index."""

    def __init__(self, predictor):
        from src.retrieval.retriever import CommonsenseRetriever

        self._inner = predictor
        self.retriever = CommonsenseRetriever(None, predictor.retriever.encoder, top_k=3)
        self.config = predictor.config
        self.device = predictor.device
        self.model = predictor.model
        self.startup_warnings = []
        self.is_development_asset = True

    def prepare(self, text, context=None):
        return self._inner.prepare(text, context)

    def predict(self, text, context=None, top_k=None):
        original = self._inner.retriever
        self._inner.retriever = self.retriever
        try:
            return self._inner.predict(text, context, top_k)
        finally:
            self._inner.retriever = original

    def predict_proba(self, texts, context=None, retrieval=None, batch_size=32):
        return self._inner.predict_proba(texts, context, retrieval, batch_size)

    def describe(self):
        return self._inner.describe()


class TestMetrics:
    """The metric suite behaves on normal and degenerate inputs."""

    def test_perfect_prediction(self):
        result = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
        assert result.accuracy == 1.0
        assert result.macro_f1 == 1.0
        assert result.roc_auc == 1.0

    def test_confusion_matrix_orientation(self):
        result = compute_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        assert result.confusion_matrix == [[1, 1], [1, 1]]

    def test_single_class_disables_auc_with_a_note(self):
        result = compute_metrics([1, 1, 1], [1, 1, 1], [0.9, 0.8, 0.7])
        assert result.roc_auc is None
        assert any("ROC-AUC" in note for note in result.notes)

    def test_empty_input(self):
        result = compute_metrics([], [])
        assert result.n_examples == 0
        assert "no examples" in result.notes[0]

    def test_grouped_metrics_flag_small_groups(self):
        grouped = compute_grouped_metrics([0, 1, 0, 1], [0, 1, 1, 1], ["a", "a", "b", "b"])
        assert set(grouped) == {"a", "b"}
        assert any("indicative only" in note for note in grouped["a"].notes)


class TestErrorAnalysis:
    """Error categorisation is rule-based and checkable by hand."""

    def _record(self, context=None, text="a reasonably long utterance here"):
        return SarcasmRecord(
            id="x", text=text, label=1, dataset="d", domain="reddit", context=context or []
        )

    def test_correct_predictions_are_not_errors(self):
        category, matched = categorise(
            self._record(), 1, 0.9, "ok", 3, 0.5, 0.2, 0.8, 0.0
        )
        assert category == "correct" and matched == []

    def test_retrieval_failure_has_priority(self):
        category, matched = categorise(
            self._record(["ctx"]), 0, 0.9, "no_index", 0, 0.5, 0.0, 0.8, 0.0
        )
        assert category == ErrorCategory.RETRIEVAL_FAILURE.value
        assert ErrorCategory.RETRIEVAL_FAILURE.value in matched

    def test_context_unavailable(self):
        category, _ = categorise(self._record([]), 0, 0.9, "ok", 3, 0.0, 0.0, 0.8, 0.0)
        assert category == ErrorCategory.CONTEXT_UNAVAILABLE.value

    def test_insufficient_context(self):
        category, _ = categorise(
            self._record(["ctx"]), 0, 0.9, "ok", 3, 0.01, 0.0, 0.8, 0.0
        )
        assert category == ErrorCategory.INSUFFICIENT_CONTEXT.value

    def test_misleading_knowledge(self):
        category, _ = categorise(
            self._record(["ctx"]), 0, 0.9, "ok", 3, 0.5, 0.4, 0.8, 0.0
        )
        assert category == ErrorCategory.MISLEADING_KNOWLEDGE.value

    def test_lexical_ambiguity(self):
        category, _ = categorise(
            self._record(["ctx"], text="sure"), 0, 0.9, "ok", 3, 0.5, 0.0, 0.05, 0.0
        )
        assert category == ErrorCategory.LEXICAL_AMBIGUITY.value

    def test_domain_mismatch(self):
        category, _ = categorise(
            self._record(["ctx"]), 0, 0.9, "ok", 3, 0.5, 0.0, 0.8, -0.2
        )
        assert category == ErrorCategory.DOMAIN_MISMATCH.value

    def test_ambiguous_sarcasm(self):
        category, _ = categorise(
            self._record(["ctx"]), 0, 0.51, "ok", 3, 0.5, 0.0, 0.8, 0.0
        )
        assert category == ErrorCategory.AMBIGUOUS_SARCASM.value

    def test_uncategorised_residual(self):
        category, matched = categorise(
            self._record(["ctx"]), 0, 0.99, "ok", 3, 0.5, 0.0, 0.9, 0.0
        )
        assert category == ErrorCategory.UNCATEGORISED.value and matched == []

    def test_rows_and_summary(self, analysis_service, fixture_records):
        records = fixture_records[:4]
        analyses = [
            analysis_service.analyze(
                record.text, record.context, explanation_method="occlusion"
            ).to_dict()
            for record in records
        ]
        rows = build_error_rows(records, analyses, domain_f1={}, overall_f1=0.5)
        assert len(rows) == 4
        assert all(row.error_category for row in rows)
        summary = summarise_errors(rows)
        assert summary["n_examples"] == 4
        assert summary["n_errors"] + sum(1 for r in rows if r.correct) == 4

    def test_thresholds_are_configurable(self):
        config = ErrorAnalysisConfig(low_confidence=0.99)
        category, _ = categorise(
            self._record(["ctx"]), 0, 0.95, "ok", 3, 0.5, 0.0, 0.9, 0.0, config=config
        )
        assert category == ErrorCategory.AMBIGUOUS_SARCASM.value
