"""Lightweight, dependency-free experiment tracking.

Deliberately *not* Weights & Biases / MLflow: the project must be reproducible
by a student on a laptop with no account on any service. Every run appends one
JSON record to ``results/experiments.json`` and one row to
``results/experiments.csv``, which is enough to produce the comparison tables and
the dashboard, and is trivially diffable in git.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io import environment_snapshot, write_json
from src.utils.paths import ensure_parent, resolve

logger = logging.getLogger(__name__)

EXPERIMENTS_JSON = "results/experiments.json"
EXPERIMENTS_CSV = "results/experiments.csv"

CSV_COLUMNS = [
    "run_id",
    "timestamp",
    "experiment",
    "model_type",
    "datasets",
    "seed",
    "encoder",
    "fusion",
    "use_context",
    "use_knowledge",
    "context_turns",
    "retrieval_top_k",
    "retrieval_encoder",
    "batch_size",
    "grad_accum",
    "learning_rate",
    "epochs_run",
    "max_length",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_sarcastic_precision",
    "test_sarcastic_recall",
    "test_roc_auc",
    "train_seconds",
    "inference_ms_per_example",
    "is_development_asset",
    "git_commit",
]


@dataclass
class ExperimentRecord:
    """One tracked run."""

    run_id: str
    experiment: str
    model_type: str
    config: dict[str, Any]
    seed: int
    datasets: list[str] = field(default_factory=list)
    val_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    per_dataset_metrics: dict[str, Any] = field(default_factory=dict)
    per_domain_metrics: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    train_seconds: float = 0.0
    inference_ms_per_example: float | None = None
    warnings: list[str] = field(default_factory=list)
    is_development_asset: bool = False
    environment: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return asdict(self)

    def csv_row(self) -> dict[str, Any]:
        """Flatten to the fixed CSV schema."""
        model = self.config.get("model", {})
        data = self.config.get("data", {})
        retrieval = self.config.get("retrieval", {})
        training = self.config.get("training", {})
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "experiment": self.experiment,
            "model_type": self.model_type,
            "datasets": "|".join(self.datasets),
            "seed": self.seed,
            "encoder": model.get("encoder_name"),
            "fusion": model.get("fusion"),
            "use_context": model.get("use_context"),
            "use_knowledge": model.get("use_knowledge"),
            "context_turns": data.get("max_context_turns"),
            "retrieval_top_k": retrieval.get("top_k"),
            "retrieval_encoder": retrieval.get("encoder_name"),
            "batch_size": training.get("batch_size"),
            "grad_accum": training.get("gradient_accumulation_steps"),
            "learning_rate": training.get("learning_rate"),
            "epochs_run": len(self.history.get("epochs", [])),
            "max_length": model.get("max_utterance_length"),
            "val_macro_f1": _round(self.val_metrics.get("macro_f1")),
            "test_accuracy": _round(self.test_metrics.get("accuracy")),
            "test_macro_f1": _round(self.test_metrics.get("macro_f1")),
            "test_sarcastic_precision": _round(self.test_metrics.get("sarcastic_precision")),
            "test_sarcastic_recall": _round(self.test_metrics.get("sarcastic_recall")),
            "test_roc_auc": _round(self.test_metrics.get("roc_auc")),
            "train_seconds": round(self.train_seconds, 2),
            "inference_ms_per_example": _round(self.inference_ms_per_example, 3),
            "is_development_asset": self.is_development_asset,
            "git_commit": (self.environment or {}).get("git_commit"),
        }


def _round(value: Any, digits: int = 4) -> Any:
    """Round floats, pass everything else through."""
    return round(value, digits) if isinstance(value, (int, float)) else value


class ExperimentTracker:
    """Append-only store of experiment records."""

    def __init__(
        self,
        json_path: str | Path = EXPERIMENTS_JSON,
        csv_path: str | Path = EXPERIMENTS_CSV,
    ) -> None:
        self.json_path = resolve(json_path)
        self.csv_path = resolve(csv_path)

    def load(self) -> list[dict[str, Any]]:
        """Read all recorded runs (empty list when nothing has been run yet)."""
        if not self.json_path.exists():
            return []
        try:
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover
            logger.error("Corrupt experiment log at %s; starting a new one", self.json_path)
            return []
        return payload if isinstance(payload, list) else []

    def record(self, entry: ExperimentRecord) -> ExperimentRecord:
        """Append ``entry`` to both the JSON log and the CSV table."""
        entry.timestamp = entry.timestamp or datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        entry.environment = entry.environment or environment_snapshot()

        runs = self.load()
        runs.append(entry.as_dict())
        write_json(self.json_path, runs)

        ensure_parent(self.csv_path)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(entry.csv_row())

        logger.info("Recorded experiment %s -> %s", entry.run_id, self.json_path)
        return entry


def make_run_id(experiment_name: str, seed: int) -> str:
    """Human-sortable run id: ``<experiment>-s<seed>-<UTC timestamp>``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{experiment_name}-s{seed}-{stamp}"


def model_type_from_config(config: dict[str, Any]) -> str:
    """Name the system described by ``config`` using the study's terminology."""
    model = config.get("model", {})
    context = bool(model.get("use_context"))
    knowledge = bool(model.get("use_knowledge"))
    fusion = model.get("fusion", "none")
    if not context and not knowledge:
        return "baseline1_roberta"
    if context and not knowledge:
        return "baseline2_roberta_context"
    if knowledge and not context:
        return "baseline3_roberta_knowledge"
    return f"full_model_{fusion}"
