"""Error analysis: categorise what the model got wrong, and why.

The categories are the seven required by the project specification. Each one is
decided by an **explicit, deterministic rule** over signals that the system
actually produces (retrieval status, cross-attention distribution, confidence,
context availability, per-domain performance). Nothing is guessed and nothing is
assigned by an LLM; a reader can check any single row by hand against the rule.

Rules are evaluated in priority order, so each error gets exactly one primary
category, with the full set of matching rules kept alongside it for
transparency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.schema import LABEL_NAMES, SarcasmRecord
from src.retrieval.retriever import RetrievalStatus

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """The seven required error categories, plus a catch-all."""

    RETRIEVAL_FAILURE = "retrieval_failure"
    CONTEXT_UNAVAILABLE = "context_unavailable"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    MISLEADING_KNOWLEDGE = "misleading_retrieved_knowledge"
    LEXICAL_AMBIGUITY = "lexical_ambiguity"
    DOMAIN_MISMATCH = "cultural_domain_mismatch"
    AMBIGUOUS_SARCASM = "genuinely_ambiguous_sarcasm"
    UNCATEGORISED = "uncategorised"


CATEGORY_DESCRIPTIONS: dict[str, str] = {
    ErrorCategory.RETRIEVAL_FAILURE.value: (
        "Commonsense retrieval returned nothing usable (no index, empty index, or "
        "no fact above the similarity threshold), so the knowledge branch was "
        "fully masked for this example."
    ),
    ErrorCategory.CONTEXT_UNAVAILABLE.value: (
        "The example carries no conversational context at all (news headlines and "
        "standalone tweets), so the context branch used the learned empty-context "
        "vector."
    ),
    ErrorCategory.INSUFFICIENT_CONTEXT.value: (
        "Context was present but the fusion layer gave it very little attention "
        "relative to the null slot, indicating the available turns did not help."
    ),
    ErrorCategory.MISLEADING_KNOWLEDGE.value: (
        "The fusion layer put substantial attention on retrieved evidence, yet the "
        "prediction was wrong -- the evidence plausibly pulled the model off course."
    ),
    ErrorCategory.LEXICAL_AMBIGUITY.value: (
        "A short utterance with no strongly-weighted token: there was too little "
        "lexical signal to decide either way."
    ),
    ErrorCategory.DOMAIN_MISMATCH.value: (
        "The example comes from a domain where this model performs materially "
        "worse than its overall average, suggesting a genre/register gap."
    ),
    ErrorCategory.AMBIGUOUS_SARCASM.value: (
        "The model was near its decision boundary (low confidence): the example is "
        "genuinely borderline rather than clearly mishandled."
    ),
    ErrorCategory.UNCATEGORISED.value: (
        "A confident error with context, evidence and clear lexical signal "
        "available -- none of the diagnostic rules fired."
    ),
}


@dataclass
class ErrorAnalysisConfig:
    """Thresholds for the categorisation rules (all explicit, all tunable)."""

    low_confidence: float = 0.60
    # Cross-attention mass is a softmax over (1 null + Lc context + K evidence)
    # slots, so these thresholds are shares of 1.0. With the default geometry
    # (64 context tokens, 5 evidence slots) uniform attention would put ~7% on
    # evidence, so 0.15 is roughly "twice what chance would give"; context would
    # get ~91% under uniform attention, so 0.10 is genuinely low.
    low_context_attention: float = 0.10
    high_evidence_attention: float = 0.15
    short_text_words: int = 6
    weak_token_weight: float = 0.35
    domain_gap: float = 0.08


@dataclass
class ErrorRow:
    """One analysed example, exported to CSV."""

    id: str
    text: str
    context: str
    dataset: str
    domain: str
    true_label: int
    predicted_label: int
    true_label_name: str
    predicted_label_name: str
    confidence: float
    prob_sarcastic: float
    correct: bool
    retrieval_status: str
    n_evidence: int
    retrieved_evidence: str
    top_tokens: str
    context_attention: float
    evidence_attention: float
    explanation: str
    error_category: str
    matched_rules: str

    def as_dict(self) -> dict[str, Any]:
        """CSV-ready row."""
        from dataclasses import asdict

        return asdict(self)


def categorise(
    record: SarcasmRecord,
    predicted_label: int,
    confidence: float,
    retrieval_status: str,
    n_evidence: int,
    context_attention: float,
    evidence_attention: float,
    top_token_weight: float,
    domain_gap: float,
    config: ErrorAnalysisConfig | None = None,
) -> tuple[str, list[str]]:
    """Assign a primary error category and list every rule that matched.

    Args:
        record: The example.
        predicted_label: What the model said.
        confidence: Probability of the predicted class.
        retrieval_status: Value of :class:`RetrievalStatus`.
        n_evidence: How many facts were retrieved.
        context_attention: Total cross-attention mass on context slots.
        evidence_attention: Total cross-attention mass on evidence slots.
        top_token_weight: Largest absolute normalised token attribution.
        domain_gap: This example's domain macro-F1 minus the overall macro-F1
            (negative means the domain is harder for this model).
        config: Thresholds.

    Returns:
        ``(primary_category, matched_rules)``. Correct predictions return
        ``("correct", [])``.
    """
    config = config or ErrorAnalysisConfig()
    if predicted_label == record.label:
        return "correct", []

    matched: list[str] = []
    if retrieval_status != RetrievalStatus.OK.value or n_evidence == 0:
        matched.append(ErrorCategory.RETRIEVAL_FAILURE.value)
    if not record.has_context:
        matched.append(ErrorCategory.CONTEXT_UNAVAILABLE.value)
    elif context_attention < config.low_context_attention:
        matched.append(ErrorCategory.INSUFFICIENT_CONTEXT.value)
    if n_evidence > 0 and evidence_attention >= config.high_evidence_attention:
        matched.append(ErrorCategory.MISLEADING_KNOWLEDGE.value)
    if (
        len(record.text.split()) <= config.short_text_words
        and top_token_weight < config.weak_token_weight
    ):
        matched.append(ErrorCategory.LEXICAL_AMBIGUITY.value)
    if domain_gap <= -config.domain_gap:
        matched.append(ErrorCategory.DOMAIN_MISMATCH.value)
    if confidence < config.low_confidence:
        matched.append(ErrorCategory.AMBIGUOUS_SARCASM.value)

    # Priority order: concrete, actionable system failures first; "the example is
    # just hard" last, so it never masks a fixable pipeline problem.
    priority = [
        ErrorCategory.RETRIEVAL_FAILURE.value,
        ErrorCategory.MISLEADING_KNOWLEDGE.value,
        ErrorCategory.INSUFFICIENT_CONTEXT.value,
        ErrorCategory.CONTEXT_UNAVAILABLE.value,
        ErrorCategory.DOMAIN_MISMATCH.value,
        ErrorCategory.LEXICAL_AMBIGUITY.value,
        ErrorCategory.AMBIGUOUS_SARCASM.value,
    ]
    for candidate in priority:
        if candidate in matched:
            return candidate, matched
    return ErrorCategory.UNCATEGORISED.value, matched


def build_error_rows(
    records: Sequence[SarcasmRecord],
    analyses: Sequence[Mapping[str, Any]],
    domain_f1: Mapping[str, float] | None = None,
    overall_f1: float = 0.0,
    config: ErrorAnalysisConfig | None = None,
) -> list[ErrorRow]:
    """Join records with their analysis payloads and categorise every example.

    Args:
        records: Examples, aligned with ``analyses``.
        analyses: The dicts produced by
            :meth:`~src.inference.service.AnalysisResult.to_dict`.
        domain_f1: Per-domain macro-F1 from the evaluation run.
        overall_f1: Overall macro-F1, used to compute the domain gap.
        config: Thresholds.
    """
    config = config or ErrorAnalysisConfig()
    domain_f1 = domain_f1 or {}
    rows: list[ErrorRow] = []

    for record, analysis in zip(records, analyses):
        context_attention = float(
            sum(c.get("attention", 0.0) for c in analysis.get("context_influence", []))
        )
        evidence_attention = float(sum(analysis.get("evidence_attention", []) or []))
        importances = analysis.get("token_importance", []) or []
        top_weight = max((abs(t.get("weight", 0.0)) for t in importances), default=0.0)
        evidence = analysis.get("retrieved_evidence", []) or []
        predicted = int(analysis.get("label", 0))
        gap = float(domain_f1.get(record.domain, overall_f1)) - float(overall_f1)

        category, matched = categorise(
            record=record,
            predicted_label=predicted,
            confidence=float(analysis.get("confidence", 0.0)),
            retrieval_status=str(analysis.get("retrieval_status", "")),
            n_evidence=len(evidence),
            context_attention=context_attention,
            evidence_attention=evidence_attention,
            top_token_weight=top_weight,
            domain_gap=gap,
            config=config,
        )
        rows.append(
            ErrorRow(
                id=record.id,
                text=record.text,
                context=" || ".join(record.context),
                dataset=record.dataset,
                domain=record.domain,
                true_label=record.label,
                predicted_label=predicted,
                true_label_name=LABEL_NAMES[record.label],
                predicted_label_name=LABEL_NAMES[predicted],
                confidence=round(float(analysis.get("confidence", 0.0)), 4),
                prob_sarcastic=round(
                    float(analysis.get("probabilities", {}).get("sarcastic", 0.0)), 4
                ),
                correct=predicted == record.label,
                retrieval_status=str(analysis.get("retrieval_status", "")),
                n_evidence=len(evidence),
                retrieved_evidence=" | ".join(
                    f"{e.get('text', '')} [{e.get('source', '')} {e.get('score', 0):.3f}]"
                    for e in evidence[:5]
                ),
                top_tokens=", ".join(
                    f"{t.get('token')}({t.get('weight'):+.2f})" for t in importances[:8]
                ),
                context_attention=round(context_attention, 4),
                evidence_attention=round(evidence_attention, 4),
                explanation=str(analysis.get("explanation", "")),
                error_category=category,
                matched_rules="|".join(matched),
            )
        )
    return rows


def summarise_errors(rows: Sequence[ErrorRow]) -> dict[str, Any]:
    """Aggregate counts, rates and per-dataset breakdown of the error categories."""
    errors = [row for row in rows if not row.correct]
    counts: dict[str, int] = {}
    for row in errors:
        counts[row.error_category] = counts.get(row.error_category, 0) + 1

    by_dataset: dict[str, dict[str, int]] = {}
    for row in errors:
        bucket = by_dataset.setdefault(row.dataset, {})
        bucket[row.error_category] = bucket.get(row.error_category, 0) + 1

    total = len(rows) or 1
    confidences = np.array([row.confidence for row in errors]) if errors else np.zeros(0)
    return {
        "n_examples": len(rows),
        "n_errors": len(errors),
        "error_rate": round(len(errors) / total, 4),
        "categories": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "category_share": {
            k: round(v / max(1, len(errors)), 4)
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        },
        "category_descriptions": {
            k: CATEGORY_DESCRIPTIONS[k] for k in counts if k in CATEGORY_DESCRIPTIONS
        },
        "by_dataset": by_dataset,
        "mean_error_confidence": round(float(confidences.mean()), 4) if errors else 0.0,
        "false_positives": sum(1 for r in errors if r.predicted_label == 1),
        "false_negatives": sum(1 for r in errors if r.predicted_label == 0),
    }
