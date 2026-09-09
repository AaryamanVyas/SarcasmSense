"""Shared types for the explanation layer.

Every explainer -- LIME, SHAP, occlusion -- returns the same
:class:`ExplanationResult`, so the API schema, the UI and the ablation that
compares SHAP against LIME all work against one shape.

Attribution sign convention (uniform across all methods):

* **positive** weight  -> the token pushed the prediction *towards the predicted
  class*;
* **negative** weight  -> the token pushed *away* from it.

Weights are additionally normalised so that the largest absolute value is 1.0.
Raw LIME coefficients and raw SHAP values are on different scales, and comparing
them without normalisation would make the SHAP-vs-LIME agreement study
meaningless.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

EXPLANATION_METHODS = ("lime", "shap", "occlusion")

#: Methods safe to run on the interactive request path.
FAST_METHODS = ("lime", "occlusion")


@dataclass(slots=True)
class TokenAttribution:
    """Signed importance of one token."""

    token: str
    weight: float
    position: int = -1

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {"token": self.token, "weight": round(self.weight, 4), "position": self.position}


@dataclass
class ExplanationResult:
    """Output of one explanation run."""

    method: str
    tokens: list[TokenAttribution] = field(default_factory=list)
    predicted_label: int = 0
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    available: bool = True

    @property
    def important_tokens(self) -> list[str]:
        """Tokens ordered by descending absolute importance."""
        return [t.token for t in sorted(self.tokens, key=lambda t: -abs(t.weight))]

    def top(self, n: int = 5, positive_only: bool = True) -> list[TokenAttribution]:
        """Return the ``n`` most influential tokens."""
        pool = [t for t in self.tokens if t.weight > 0] if positive_only else list(self.tokens)
        return sorted(pool, key=lambda t: -abs(t.weight))[:n]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        payload = asdict(self)
        payload["tokens"] = [t.to_dict() for t in self.tokens]
        payload["latency_ms"] = round(self.latency_ms, 2)
        return payload


def normalise_weights(
    attributions: Sequence[TokenAttribution],
) -> list[TokenAttribution]:
    """Scale weights so the largest absolute value is 1.0 (in place-safe copy)."""
    if not attributions:
        return []
    largest = max(abs(a.weight) for a in attributions)
    if largest <= 0:
        return list(attributions)
    return [
        TokenAttribution(token=a.token, weight=a.weight / largest, position=a.position)
        for a in attributions
    ]


def agreement(a: ExplanationResult, b: ExplanationResult, top_k: int = 5) -> dict[str, float]:
    """Quantify how much two explanations agree.

    Returns three complementary numbers:

    * ``jaccard_topk`` -- overlap of the top-k token *sets*;
    * ``rank_correlation`` -- Spearman correlation over tokens present in both,
      which captures whether the *ordering* matches;
    * ``sign_agreement`` -- fraction of shared tokens whose attributions point in
      the same direction.
    """
    top_a = {t.token for t in a.top(top_k, positive_only=False)}
    top_b = {t.token for t in b.top(top_k, positive_only=False)}
    union = top_a | top_b
    jaccard = len(top_a & top_b) / len(union) if union else 0.0

    weights_a = {t.token: t.weight for t in a.tokens}
    weights_b = {t.token: t.weight for t in b.tokens}
    shared = sorted(set(weights_a) & set(weights_b))
    if len(shared) < 3:
        return {
            "jaccard_topk": jaccard,
            "rank_correlation": float("nan"),
            "sign_agreement": float("nan"),
            "n_shared_tokens": float(len(shared)),
        }
    values_a = np.array([weights_a[t] for t in shared])
    values_b = np.array([weights_b[t] for t in shared])
    ranks_a = np.argsort(np.argsort(values_a))
    ranks_b = np.argsort(np.argsort(values_b))
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(ranks_a, ranks_b)[0, 1])
    signs = float(np.mean(np.sign(values_a) == np.sign(values_b)))
    return {
        "jaccard_topk": jaccard,
        "rank_correlation": correlation,
        "sign_agreement": signs,
        "n_shared_tokens": float(len(shared)),
    }


class BaseExplainer(ABC):
    """Interface every explainer implements."""

    #: Method key, one of :data:`EXPLANATION_METHODS`.
    method: str = "base"

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this explainer can actually run (library present, etc.)."""

    @abstractmethod
    def explain(
        self,
        text: str,
        context: Sequence[str] | None = None,
        retrieval: Any = None,
        num_features: int = 10,
    ) -> ExplanationResult:
        """Attribute the prediction for ``text`` to its tokens."""
