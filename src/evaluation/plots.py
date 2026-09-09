"""Result plots.

Every figure here is drawn from data that was actually produced by a run: the
functions take arrays and metric objects, never literals. If a run has not
happened, the corresponding figure simply is not generated -- there are no
placeholder or illustrative plots anywhere in this project.

The ``Agg`` backend is selected explicitly so the module works headless (CI,
Docker, a remote box over SSH).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.schema import LABEL_NAMES  # noqa: E402
from src.evaluation.metrics import MetricResult, roc_points  # noqa: E402
from src.utils.paths import ensure_parent  # noqa: E402

logger = logging.getLogger(__name__)

PALETTE = {
    "primary": "#2b6cb0",
    "secondary": "#dd6b20",
    "accent": "#38a169",
    "muted": "#718096",
    "grid": "#e2e8f0",
}


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply a consistent, print-friendly style."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    """Save and close a figure."""
    target = ensure_parent(path)
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure %s", target)
    return target


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]], path: str | Path, title: str = "Confusion matrix"
) -> Path:
    """Draw a 2x2 confusion matrix with counts and row-normalised percentages."""
    data = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(data, cmap="Blues")
    row_totals = data.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1.0
    percentages = data / row_totals

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"{int(data[i, j]):,}\n{percentages[i, j]:.1%}",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if percentages[i, j] > 0.5 else "#1a202c",
                fontweight="bold",
            )
    labels = [LABEL_NAMES[0], LABEL_NAMES[1]]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def plot_training_curves(history: Mapping, path: str | Path, title: str = "Training") -> Path | None:
    """Plot train/val loss and validation macro-F1 across epochs."""
    epochs = history.get("epochs") or []
    if not epochs:
        logger.warning("No epochs in history; skipping training-curve plot")
        return None
    x = [e["epoch"] for e in epochs]
    train_loss = [e["train_loss"] for e in epochs]
    val_loss = [e["val_loss"] for e in epochs]
    val_f1 = [e["val_metrics"]["macro_f1"] for e in epochs]
    val_acc = [e["val_metrics"]["accuracy"] for e in epochs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(x, train_loss, "-o", color=PALETTE["primary"], label="train loss")
    axes[0].plot(x, val_loss, "-s", color=PALETTE["secondary"], label="validation loss")
    _style(axes[0], f"{title}: loss", "epoch", "cross-entropy")
    axes[0].legend(frameon=False)

    axes[1].plot(x, val_f1, "-o", color=PALETTE["accent"], label="validation macro-F1")
    axes[1].plot(x, val_acc, "--s", color=PALETTE["muted"], label="validation accuracy")
    best_epoch = history.get("best_epoch")
    if best_epoch and best_epoch in x:
        axes[1].axvline(best_epoch, color=PALETTE["secondary"], linestyle=":", linewidth=1.5)
        axes[1].annotate(
            f"best (epoch {best_epoch})",
            xy=(best_epoch, max(val_f1)),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=9,
            color=PALETTE["secondary"],
        )
    _style(axes[1], f"{title}: validation quality", "epoch", "score")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    return _save(fig, path)


def plot_roc(
    y_true: Sequence[int], y_prob: Sequence[float], path: str | Path, title: str = "ROC curve"
) -> Path | None:
    """Plot the ROC curve; returns ``None`` when AUC is undefined."""
    points = roc_points(y_true, y_prob)
    if points is None:
        logger.warning("ROC undefined (single class present); skipping plot")
        return None
    fpr, tpr, auc = points
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot(fpr, tpr, color=PALETTE["primary"], linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], linewidth=1, label="chance")
    _style(ax, title, "false positive rate", "true positive rate")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, path)


def plot_model_comparison(
    results: Mapping[str, MetricResult],
    path: str | Path,
    metrics: Sequence[str] = ("accuracy", "macro_f1", "sarcastic_precision", "sarcastic_recall"),
    title: str = "Baseline vs proposed model (test set)",
) -> Path | None:
    """Grouped bar chart comparing the systems on the requested metrics."""
    if not results:
        logger.warning("No results to compare; skipping comparison plot")
        return None
    names = list(results)
    n_groups = len(metrics)
    width = 0.8 / max(1, len(names))
    positions = np.arange(n_groups)
    colours = plt.cm.tab10(np.linspace(0, 1, max(3, len(names))))

    fig, ax = plt.subplots(figsize=(max(7.5, 2.0 * n_groups), 4.6))
    for i, name in enumerate(names):
        values = [getattr(results[name], metric, 0.0) or 0.0 for metric in metrics]
        offset = positions + i * width - 0.4 + width / 2
        bars = ax.bar(offset, values, width * 0.92, label=name, color=colours[i])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                fontsize=7.5,
            )
    ax.set_xticks(positions, [m.replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.08)
    _style(ax, title, "", "score")
    ax.legend(frameon=False, fontsize=9, ncol=min(3, len(names)))
    return _save(fig, path)


def plot_class_distribution(
    counts_per_split: Mapping[str, Mapping[str, int]],
    path: str | Path,
    title: str = "Class distribution per split",
) -> Path:
    """Stacked bars showing label balance in each split."""
    splits = list(counts_per_split)
    not_sarcastic = [counts_per_split[s].get(LABEL_NAMES[0], 0) for s in splits]
    sarcastic = [counts_per_split[s].get(LABEL_NAMES[1], 0) for s in splits]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(splits, not_sarcastic, label=LABEL_NAMES[0], color=PALETTE["muted"])
    ax.bar(splits, sarcastic, bottom=not_sarcastic, label=LABEL_NAMES[1], color=PALETTE["primary"])
    for i, split in enumerate(splits):
        total = not_sarcastic[i] + sarcastic[i]
        if total:
            ax.text(i, total * 1.01, f"{total:,}", ha="center", fontsize=9)
    _style(ax, title, "", "examples")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_metric_by_group(
    results: Mapping[str, MetricResult],
    path: str | Path,
    metric: str = "macro_f1",
    title: str = "Macro-F1 by dataset",
) -> Path | None:
    """Horizontal bars of one metric per group, annotated with support."""
    if not results:
        return None
    names = list(results)
    values = [getattr(results[name], metric, 0.0) or 0.0 for name in names]
    supports = [results[name].n_examples for name in names]

    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.55 * len(names) + 1.4)))
    bars = ax.barh(names, values, color=PALETTE["primary"])
    for bar, value, support in zip(bars, values, supports):
        ax.text(
            min(value + 0.015, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}  (n={support:,})",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 1.15)
    _style(ax, title, metric.replace("_", " "), "")
    return _save(fig, path)
