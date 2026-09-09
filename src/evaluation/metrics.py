"""Evaluation metrics.

**Macro-F1 is the primary metric.** Reasons, in the order they matter for this
project:

1. The corpora are only *approximately* balanced, and the merged corpus is not
   balanced at all once several sources are mixed. Accuracy would reward a model
   that leans on the majority class of whichever corpus dominates.
2. Sarcasm detection cares about both errors symmetrically: missing sarcasm and
   crying sarcasm are both failures, and macro-F1 weights the two classes
   equally regardless of their frequency.
3. It is the metric the FigLang 2020 shared task reported, so the numbers are
   comparable to published work.

Sarcastic-class precision and recall are reported separately, because the
macro average can hide an imbalance between them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.data.schema import LABEL_NAMES, LABEL_SARCASTIC

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "macro_f1"


@dataclass
class MetricResult:
    """A complete evaluation of one model on one set of examples."""

    n_examples: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    weighted_f1: float = 0.0
    sarcastic_precision: float = 0.0
    sarcastic_recall: float = 0.0
    sarcastic_f1: float = 0.0
    not_sarcastic_precision: float = 0.0
    not_sarcastic_recall: float = 0.0
    not_sarcastic_f1: float = 0.0
    roc_auc: float | None = None
    pr_auc: float | None = None
    confusion_matrix: list[list[int]] = field(default_factory=list)
    support: dict[str, int] = field(default_factory=dict)
    mean_confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return asdict(self)

    def summary(self) -> str:
        """One-line summary for progress logs."""
        return (
            f"acc={self.accuracy:.4f} macroF1={self.macro_f1:.4f} "
            f"P_sar={self.sarcastic_precision:.4f} R_sar={self.sarcastic_recall:.4f}"
            + (f" AUC={self.roc_auc:.4f}" if self.roc_auc is not None else "")
        )

    def table_row(self) -> dict[str, Any]:
        """Flat row for ``results/results.csv``."""
        return {
            "n": self.n_examples,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "sarcastic_precision": round(self.sarcastic_precision, 4),
            "sarcastic_recall": round(self.sarcastic_recall, 4),
            "sarcastic_f1": round(self.sarcastic_f1, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
        }


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray | None = None,
) -> MetricResult:
    """Compute the full metric suite.

    Args:
        y_true: Gold labels (0/1).
        y_pred: Predicted labels (0/1).
        y_prob: Probability of the *sarcastic* class, used for ROC-AUC/PR-AUC
            and mean confidence. Optional.

    Returns:
        A populated :class:`MetricResult`. Degenerate inputs (empty arrays, a
        single class present) are handled by returning ``None`` for the affected
        metric and appending an explanatory note, rather than raising.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    result = MetricResult(n_examples=int(y_true.shape[0]))
    if result.n_examples == 0:
        result.notes.append("no examples to evaluate")
        return result

    result.accuracy = float(accuracy_score(y_true, y_pred))
    result.macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    result.weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    result.macro_precision = float(np.mean(precision))
    result.macro_recall = float(np.mean(recall))
    result.not_sarcastic_precision = float(precision[0])
    result.not_sarcastic_recall = float(recall[0])
    result.not_sarcastic_f1 = float(f1[0])
    result.sarcastic_precision = float(precision[1])
    result.sarcastic_recall = float(recall[1])
    result.sarcastic_f1 = float(f1[1])
    result.support = {LABEL_NAMES[0]: int(support[0]), LABEL_NAMES[1]: int(support[1])}
    result.confusion_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        result.mean_confidence = float(
            np.mean(np.maximum(y_prob, 1.0 - y_prob))
        )
        if len(np.unique(y_true)) < 2:
            result.notes.append(
                "ROC-AUC/PR-AUC undefined: only one class present in y_true"
            )
        else:
            result.roc_auc = float(roc_auc_score(y_true, y_prob))
            result.pr_auc = float(
                average_precision_score(y_true, y_prob, pos_label=LABEL_SARCASTIC)
            )
    return result


def compute_grouped_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[str],
    y_prob: Sequence[float] | None = None,
    min_examples: int = 20,
) -> dict[str, MetricResult]:
    """Compute metrics per group (per dataset, per domain, per context length).

    Groups with fewer than ``min_examples`` members are still reported but carry
    a note, because F1 on a handful of examples is extremely noisy and should
    not be read as a finding.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probabilities = np.asarray(y_prob, dtype=float) if y_prob is not None else None
    out: dict[str, MetricResult] = {}
    for group in sorted(set(groups)):
        selector = np.array([g == group for g in groups], dtype=bool)
        result = compute_metrics(
            y_true[selector],
            y_pred[selector],
            probabilities[selector] if probabilities is not None else None,
        )
        if result.n_examples < min_examples:
            result.notes.append(
                f"only {result.n_examples} examples -- treat this row as indicative only"
            )
        out[group] = result
    return out


def roc_points(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return ``(fpr, tpr, auc)`` for plotting, or ``None`` if undefined."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr, float(roc_auc_score(y_true, y_prob))


def metrics_to_rows(results: Mapping[str, MetricResult]) -> list[dict[str, Any]]:
    """Flatten ``{name: MetricResult}`` into CSV-ready rows."""
    return [{"name": name, **result.table_row()} for name, result in results.items()]


def render_comparison_table(results: Mapping[str, MetricResult]) -> str:
    """Render a fixed-width comparison table for the console and the README."""
    headers = [
        ("model", 30),
        ("n", 7),
        ("acc", 8),
        ("macro-F1", 9),
        ("P(sar)", 8),
        ("R(sar)", 8),
        ("F1(sar)", 8),
        ("ROC-AUC", 8),
    ]
    lines = ["  ".join(name.ljust(width) for name, width in headers)]
    lines.append("-" * (sum(width for _, width in headers) + 2 * (len(headers) - 1)))
    for name, result in results.items():
        auc = f"{result.roc_auc:.4f}" if result.roc_auc is not None else "n/a"
        lines.append(
            "  ".join(
                [
                    name[:30].ljust(30),
                    f"{result.n_examples:,}".ljust(7),
                    f"{result.accuracy:.4f}".ljust(8),
                    f"{result.macro_f1:.4f}".ljust(9),
                    f"{result.sarcastic_precision:.4f}".ljust(8),
                    f"{result.sarcastic_recall:.4f}".ljust(8),
                    f"{result.sarcastic_f1:.4f}".ljust(8),
                    auc.ljust(8),
                ]
            )
        )
    return "\n".join(lines)
