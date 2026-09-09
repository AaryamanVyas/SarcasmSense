"""End-to-end analysis service: prediction + attribution + explanation.

This is the object the FastAPI layer, the benchmark script and the error-analysis
script all share. Keeping the orchestration here (rather than in the API layer)
means the exact same code path is measured by the benchmark, exercised by the
tests and served to the UI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.explainability.base import EXPLANATION_METHODS, ExplanationResult
from src.explainability.explainers import ExplainerRegistry
from src.explainability.explanation import (
    IncongruitySignal,
    SentimentLexicon,
    generate_explanation,
)
from src.inference.predictor import PredictionResult, SarcasmPredictor
from src.utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """The full response payload for one analysed utterance."""

    prediction: PredictionResult
    explanation: ExplanationResult
    explanation_text: str
    incongruity: IncongruitySignal
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Assemble the JSON contract documented in the README."""
        prediction = self.prediction
        ordered = sorted(self.explanation.tokens, key=lambda t: -abs(t.weight))
        return {
            "prediction": prediction.prediction,
            "label": prediction.label,
            # The *cleaned* utterance the model actually saw. The UI highlights
            # this string, so it must be the same one the attributions index into
            # -- highlighting the raw input would misalign whenever preprocessing
            # changed anything (URLs, mentions, repeated punctuation).
            "analysed_text": prediction.text,
            "confidence": round(prediction.confidence, 4),
            "probabilities": {
                "sarcastic": round(prediction.probabilities["sarcastic"], 4),
                "not_sarcastic": round(prediction.probabilities["not_sarcastic"], 4),
            },
            "important_tokens": [t.token for t in ordered],
            "token_importance": [t.to_dict() for t in ordered],
            "context_used": prediction.context_used,
            "context_influence": [c.to_dict() for c in prediction.context_influence],
            "retrieved_evidence": [e.to_dict() for e in prediction.retrieval.evidence],
            "retrieval_status": prediction.retrieval.status.value,
            "evidence_attention": [round(v, 4) for v in prediction.evidence_attention],
            "explanation": self.explanation_text,
            "explanation_method": self.explanation.method,
            "incongruity": self.incongruity.to_dict(),
            "warnings": sorted({*prediction.warnings, *self.explanation.warnings}),
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }


class SarcasmAnalysisService:
    """Owns the predictor, the explainers and the explanation generator."""

    def __init__(self, predictor: SarcasmPredictor, config: Config) -> None:
        self.predictor = predictor
        self.config = config
        self.explainers = ExplainerRegistry(
            predictor,
            lime_num_samples=config.explain.lime_num_samples,
            shap_max_evals=config.explain.shap_num_samples,
        )
        self.lexicon = SentimentLexicon()
        # Touch the lexicon once here: NLTK loads SentiWordNet lazily and the
        # first lookup costs several seconds. Paying that at startup keeps it out
        # of the first user request's latency (and out of the benchmark).
        self.lexicon.polarity("great")
        logger.info(
            "Analysis service ready | explainers=%s | sentiment_lexicon=%s",
            self.explainers.availability(),
            self.lexicon.available,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: str | None = None,
        config: Config | None = None,
    ) -> "SarcasmAnalysisService":
        """Build the service from a trained checkpoint directory."""
        predictor = SarcasmPredictor.from_checkpoint(
            checkpoint_dir, config=config, device=device
        )
        return cls(predictor, predictor.config)

    # --------------------------------------------------------------- analyse
    def analyze(
        self,
        text: str,
        context: Sequence[str] | None = None,
        explanation_method: str | None = None,
        explain: bool = True,
        num_features: int | None = None,
    ) -> AnalysisResult:
        """Classify and explain one utterance.

        Args:
            text: The target utterance.
            context: Up to three preceding turns, oldest first.
            explanation_method: ``"lime"``, ``"shap"`` or ``"occlusion"``.
            explain: Set ``False`` to skip attribution entirely (classification
                and retrieval only) -- this is the fast path the ``<300 ms``
                target applies to.
            num_features: How many tokens the explainer should report.

        Raises:
            ValueError: on empty input or an unknown explanation method.
        """
        if not text or not text.strip():
            raise ValueError("Target utterance must not be empty")
        method = (explanation_method or self.config.explain.default_method).lower()
        if method not in EXPLANATION_METHODS:
            raise ValueError(
                f"explanation_method must be one of {EXPLANATION_METHODS}, got {method!r}"
            )

        started = time.perf_counter()
        prediction = self.predictor.predict(text, context)
        predict_ms = (time.perf_counter() - started) * 1000

        explain_started = time.perf_counter()
        if explain:
            explanation = self.explainers.explain(
                method,
                prediction.text,
                context=prediction.context_used,
                retrieval=prediction.retrieval,
                num_features=num_features or self.config.explain.lime_num_features,
            )
        else:
            explanation = ExplanationResult(
                method="none",
                predicted_label=prediction.label,
                warnings=["Token attribution was not requested for this call."],
            )
        explain_ms = (time.perf_counter() - explain_started) * 1000

        explanation_text, incongruity = generate_explanation(
            prediction,
            explanation,
            lexicon=self.lexicon,
            uses_knowledge=self.config.model.use_knowledge,
            max_tokens=3,
        )
        return AnalysisResult(
            prediction=prediction,
            explanation=explanation,
            explanation_text=explanation_text,
            incongruity=incongruity,
            timings_ms={
                "retrieval": prediction.timings_ms.get("retrieval", 0.0),
                "model": prediction.timings_ms.get("model", 0.0),
                "prediction_total": predict_ms,
                "explanation": explain_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        )

    # ------------------------------------------------------------------ info
    def describe(self) -> dict[str, Any]:
        """Metadata for ``GET /model/info`` and the UI's model panel."""
        info = self.predictor.describe()
        info["explanation"] = {
            "available_methods": self.explainers.availability(),
            "default_method": self.config.explain.default_method,
            "fast_modes": ["lime", "occlusion"],
            "audit_mode": "shap",
            "sentiment_lexicon": "SentiWordNet" if self.lexicon.available else None,
        }
        info["fusion"] = self.config.model.fusion
        return info
