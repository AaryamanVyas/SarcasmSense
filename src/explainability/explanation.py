"""Human-readable explanation generation.

The explanation is *composed from what actually happened* during the prediction:
the model's own confidence, the token attributions the explainer computed, the
cross-attention weights over context turns and evidence slots, and the facts the
retriever actually returned. Nothing is templated in advance, and no fact is
mentioned unless it was retrieved.

Incongruity detection
---------------------
Sarcasm is characterised by a clash between the *literal* evaluative wording of
an utterance and the *situation* it describes. The generator looks for that
clash explicitly:

* the polarity of the highest-attribution tokens in the utterance is scored with
  SentiWordNet (a real, published lexicon -- no hand-written word lists), and
* the polarity of the situation is scored from the retrieved evidence (WordNet
  ``HasSentiment`` facts carry an explicit connotation) and from the same lexicon
  applied to the most-attended context turn.

When the two disagree, the explanation names the clash. When they do not, it
says so rather than inventing one -- an honest "no incongruity signal was found"
is far more defensible than a fabricated narrative.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Sequence

from src.explainability.base import ExplanationResult, TokenAttribution
from src.inference.predictor import ContextInfluence, PredictionResult
from src.retrieval.retriever import RetrievalStatus, RetrievedEvidence

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z']+")

#: Confidence below this is reported as a hedge rather than a firm decision.
LOW_CONFIDENCE = 0.60
#: Attention share above this counts as "the model concentrated on" a slot.
NOTABLE_ATTENTION = 0.15


@dataclass
class IncongruitySignal:
    """The clash (or absence of one) that the explanation reports."""

    detected: bool
    utterance_polarity: float
    situation_polarity: float
    positive_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    situation_source: str = "none"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return asdict(self)


class SentimentLexicon:
    """Thin wrapper over SentiWordNet with a safe no-op fallback.

    SentiWordNet assigns positive/negative scores to WordNet synsets. Averaging
    the first few senses of a word is the standard lightweight way to get a
    lexical polarity without training anything. When the corpus is unavailable
    the lexicon reports ``0.0`` for everything and the generator degrades to
    describing attributions without a polarity claim.
    """

    def __init__(self, nltk_data_dir: str | None = "knowledge/raw/nltk_data") -> None:
        self._swn = None
        self.available = False
        try:
            import nltk

            if nltk_data_dir:
                from src.utils.paths import resolve

                directory = str(resolve(nltk_data_dir))
                if directory not in nltk.data.path:
                    nltk.data.path.insert(0, directory)
            from nltk.corpus import sentiwordnet as swn

            swn.ensure_loaded()
            self._swn = swn
            self.available = True
        except Exception as exc:  # noqa: BLE001 - optional dependency
            logger.info("SentiWordNet unavailable (%s); polarity claims disabled", exc)

    @lru_cache(maxsize=8192)
    def polarity(self, word: str) -> float:  # noqa: D102
        if not self.available or self._swn is None:
            return 0.0
        try:
            senses = list(self._swn.senti_synsets(word.lower()))[:3]
        except Exception:  # noqa: BLE001 # pragma: no cover
            return 0.0
        if not senses:
            return 0.0
        return float(
            sum(sense.pos_score() - sense.neg_score() for sense in senses) / len(senses)
        )

    def score_text(self, text: str) -> tuple[float, list[str], list[str]]:
        """Return ``(mean polarity, positive words, negative words)`` for ``text``."""
        positives: list[str] = []
        negatives: list[str] = []
        scores: list[float] = []
        for word in _WORD_RE.findall(text or ""):
            score = self.polarity(word)
            if abs(score) < 0.15:
                continue
            scores.append(score)
            (positives if score > 0 else negatives).append(word)
        mean = sum(scores) / len(scores) if scores else 0.0
        return mean, positives, negatives


def _evidence_polarity(evidence: Sequence[RetrievedEvidence]) -> tuple[float, str]:
    """Derive a situation polarity from the retrieved facts themselves.

    Only facts that *state* a connotation are used (WordNet ``HasSentiment``);
    other relations contribute nothing, because reading polarity into e.g. an
    ``IsA`` edge would be an invention.
    """
    votes: list[float] = []
    for fact in evidence:
        if fact.relation != "HasSentiment":
            continue
        if fact.tail == "negative":
            votes.append(-1.0)
        elif fact.tail == "positive":
            votes.append(1.0)
    if not votes:
        return 0.0, "none"
    return sum(votes) / len(votes), "retrieved evidence"


def detect_incongruity(
    prediction: PredictionResult,
    explanation: ExplanationResult,
    lexicon: SentimentLexicon,
    top_n: int = 5,
) -> IncongruitySignal:
    """Look for a literal-wording vs situation clash.

    Args:
        prediction: The classification result (context influence + evidence).
        explanation: Token attributions, used to focus on the words that mattered.
        lexicon: Polarity source.
        top_n: How many high-attribution tokens to consider.

    Returns:
        An :class:`IncongruitySignal`; ``detected`` is ``False`` when no clash
        was found, and the description says so plainly.
    """
    influential = [t.token for t in explanation.top(top_n, positive_only=False)]
    focus = " ".join(influential) if influential else prediction.text
    utterance_polarity, positives, negatives = lexicon.score_text(focus)

    situation_polarity, source = _evidence_polarity(prediction.retrieval.evidence)
    if source == "none" and prediction.context_influence:
        strongest = max(prediction.context_influence, key=lambda c: c.attention)
        if strongest.attention > 0:
            situation_polarity, _pos, _neg = lexicon.score_text(strongest.text)
            source = f"context turn {strongest.turn_index}"

    detected = (
        lexicon.available
        and abs(utterance_polarity) >= 0.15
        and abs(situation_polarity) >= 0.15
        and (utterance_polarity > 0) != (situation_polarity > 0)
    )
    if detected:
        wording = "positive" if utterance_polarity > 0 else "negative"
        situation = "negative" if situation_polarity < 0 else "positive"
        description = (
            f"the {wording}ly-worded terms the model relied on "
            f"({', '.join((positives if utterance_polarity > 0 else negatives)[:3])}) "
            f"clash with the {situation} situation indicated by the {source}"
        )
    elif not lexicon.available:
        description = (
            "polarity analysis was unavailable (SentiWordNet not installed), so no "
            "incongruity claim is made"
        )
    else:
        description = "no clear polarity clash was found between the wording and the situation"

    return IncongruitySignal(
        detected=detected,
        utterance_polarity=round(utterance_polarity, 3),
        situation_polarity=round(situation_polarity, 3),
        positive_terms=positives[:5],
        negative_terms=negatives[:5],
        situation_source=source,
        description=description,
    )


def _format_tokens(tokens: Sequence[TokenAttribution], limit: int = 3) -> str:
    """Render tokens as a readable quoted list."""
    chosen = [f"'{t.token}'" for t in tokens[:limit]]
    if not chosen:
        return ""
    if len(chosen) == 1:
        return chosen[0]
    return ", ".join(chosen[:-1]) + f" and {chosen[-1]}"


def _describe_context(influences: Sequence[ContextInfluence]) -> str:
    """One sentence about which conversational turn mattered."""
    if not influences:
        return ""
    total = sum(c.attention for c in influences)
    if total <= 0:
        return (
            "Conversational context was supplied but received almost no "
            "cross-attention weight, so it had little effect on this decision."
        )
    strongest = max(influences, key=lambda c: c.attention)
    share = strongest.attention / total
    snippet = strongest.text if len(strongest.text) <= 90 else strongest.text[:87] + "…"
    if share >= 0.5 and strongest.attention >= NOTABLE_ATTENTION * total:
        return (
            f"Of the {len(influences)} preceding turn(s), the model concentrated on "
            f"turn {strongest.turn_index} (“{snippet}”), which drew "
            f"{share:.0%} of the attention it paid to the conversation."
        )
    return (
        f"The model spread its attention fairly evenly across the "
        f"{len(influences)} preceding turn(s); turn {strongest.turn_index} was the "
        f"most influential at {share:.0%}."
    )


def _describe_evidence(
    prediction: PredictionResult, limit: int = 2, uses_knowledge: bool = True
) -> str:
    """One or two sentences citing the evidence that was actually retrieved."""
    status = prediction.retrieval.status
    if status is RetrievalStatus.DISABLED:
        return (
            "Commonsense retrieval is switched off in this model configuration, so "
            "no external evidence contributed to the decision."
        )
    if not prediction.retrieval.evidence:
        return (
            f"No commonsense evidence was available for this input "
            f"(retrieval status: {status.value}), so the decision rests on the "
            f"utterance and its conversational context alone."
        )

    ranked = list(prediction.retrieval.evidence)
    if prediction.evidence_attention and len(prediction.evidence_attention) == len(ranked):
        order = sorted(
            range(len(ranked)), key=lambda i: -prediction.evidence_attention[i]
        )
        ranked = [ranked[i] for i in order]

    cited = ranked[:limit]
    quoted = "; ".join(f"“{fact.text}” ({fact.source})" for fact in cited)
    lead = (
        "The retrieved commonsense evidence the fusion layer attended to most was"
        if prediction.evidence_attention
        else "The most similar commonsense evidence retrieved was"
    )
    tail = (
        ""
        if uses_knowledge
        else " This evidence is shown for transparency but was not consumed by "
        "this model configuration's classifier."
    )
    return f"{lead}: {quoted}.{tail}"


def generate_explanation(
    prediction: PredictionResult,
    explanation: ExplanationResult,
    lexicon: SentimentLexicon | None = None,
    uses_knowledge: bool = True,
    max_tokens: int = 3,
) -> tuple[str, IncongruitySignal]:
    """Compose the final human-readable explanation.

    Args:
        prediction: The classification result.
        explanation: Token attributions from LIME / SHAP / occlusion.
        lexicon: Polarity source; constructed on demand when ``None``.
        uses_knowledge: Whether the classifier actually consumed evidence.
        max_tokens: How many tokens to name in the explanation.

    Returns:
        ``(explanation_text, incongruity_signal)``.
    """
    lexicon = lexicon or SentimentLexicon()
    incongruity = detect_incongruity(prediction, explanation, lexicon)

    sentences: list[str] = []

    # 1. the decision itself, hedged when the model is unsure
    verdict = prediction.prediction.lower()
    if prediction.confidence < LOW_CONFIDENCE:
        # Note the plain quotes here: curly quotes are reserved for *quoted
        # source material* (retrieved facts and context turns), which keeps the
        # invariant "anything in curly quotes was actually retrieved or supplied".
        sentences.append(
            f"The model leans towards '{prediction.prediction}' but is not "
            f"confident ({prediction.confidence:.0%}); this input sits close to its "
            f"decision boundary and the explanation below should be read cautiously."
        )
    else:
        sentences.append(
            f"The statement was classified as {verdict} with "
            f"{prediction.confidence:.0%} confidence."
        )

    # 2. what in the wording drove it
    supporting = explanation.top(max_tokens, positive_only=True)
    opposing = [t for t in explanation.tokens if t.weight < 0]
    opposing.sort(key=lambda t: t.weight)
    if supporting:
        sentences.append(
            f"Using {explanation.method.upper()} attribution, the words that pushed "
            f"the decision hardest towards this label were "
            f"{_format_tokens(supporting, max_tokens)}."
        )
        if opposing:
            sentences.append(
                f"{_format_tokens(opposing, 2)} pulled in the opposite direction."
            )
    elif explanation.tokens:
        sentences.append(
            f"No single word dominated the {explanation.method.upper()} attribution; "
            f"the decision came from the utterance as a whole."
        )
    else:
        sentences.append(
            "Token-level attribution could not be computed for this input, so the "
            "explanation is based on the model's attention alone."
        )

    # 3. context
    context_sentence = _describe_context(prediction.context_influence)
    if context_sentence:
        sentences.append(context_sentence)
    elif not prediction.context_used:
        sentences.append(
            "No conversational context was supplied, so the model judged the "
            "utterance in isolation."
        )

    # 4. evidence
    sentences.append(_describe_evidence(prediction, uses_knowledge=uses_knowledge))

    # 5. incongruity
    if incongruity.detected:
        sentences.append(
            f"Taken together, {incongruity.description} — the literal-versus-"
            f"situational mismatch that typically signals sarcasm."
        )
    elif prediction.label == 1:
        sentences.append(
            f"On an explicit check for a literal-versus-situational polarity clash, "
            f"{incongruity.description}, so the sarcasm signal here came from the "
            f"model's learned representation rather than an overt wording mismatch."
        )
    else:
        sentences.append(
            f"On an explicit check for a literal-versus-situational polarity clash, "
            f"{incongruity.description}, which is consistent with a literal reading."
        )

    # 6. honest failure notes
    notes = [w for w in prediction.warnings if w]
    if notes:
        sentences.append("Caveats: " + " ".join(notes))

    return " ".join(sentences), incongruity
