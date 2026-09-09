"""Training loop for every model variant.

Features required by the project specification, all implemented here:

* fixed seed and deterministic algorithms,
* configurable batch size and **gradient accumulation** (so an effective batch of
  32 fits on a T4 with a physical batch of 8),
* **mixed precision** when CUDA is available (``auto`` picks bf16 on Ampere+,
  fp16 otherwise; silently disabled on CPU where it would only slow things down),
* linear warmup + linear decay learning-rate schedule,
* **discriminative learning rates** -- the pre-trained encoder gets a small LR
  while the freshly-initialised fusion/head get a larger one, which is standard
  practice for transformer fine-tuning and matters a lot when the head is 2304
  dimensions wide,
* gradient clipping,
* **early stopping** and best-model selection on validation macro-F1,
* checkpoint saving and **resume-from-checkpoint**,
* per-epoch train/val logging written to disk as JSONL.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SarcasmCollator, SarcasmDataset, model_inputs
from src.evaluation.metrics import MetricResult, compute_metrics
from src.models.sarcasm_model import SarcasmModel
from src.utils.config import Config
from src.utils.io import write_json
from src.utils.paths import ensure_dir
from src.utils.seeding import seed_worker

logger = logging.getLogger(__name__)

TRAINER_STATE_FILENAME = "trainer_state.pt"
HISTORY_FILENAME = "history.jsonl"


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick the best available device, honouring an explicit override."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp(mode: str, device: torch.device) -> tuple[bool, torch.dtype | None]:
    """Decide whether to use autocast and with which dtype.

    Returns ``(enabled, dtype)``. Mixed precision on CPU is a pessimisation for
    this workload, so it is disabled there regardless of the requested mode.
    """
    if device.type != "cuda" or mode == "off":
        return False, None
    if mode == "fp16":
        return True, torch.float16
    if mode == "bf16":
        return True, torch.bfloat16
    supports_bf16 = torch.cuda.is_bf16_supported()
    return True, (torch.bfloat16 if supports_bf16 else torch.float16)


@dataclass
class EpochRecord:
    """One row of the training history."""

    epoch: int
    train_loss: float
    val_loss: float
    val_metrics: dict[str, Any]
    learning_rate: float
    seconds: float
    is_best: bool = False

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return asdict(self)


@dataclass
class TrainingHistory:
    """Full record of a training run."""

    epochs: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_metric: float = -1.0
    total_seconds: float = 0.0
    stopped_early: bool = False
    steps_per_epoch: int = 0

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "epochs": [e.as_dict() for e in self.epochs],
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "total_seconds": round(self.total_seconds, 2),
            "stopped_early": self.stopped_early,
            "steps_per_epoch": self.steps_per_epoch,
        }


@dataclass
class Predictions:
    """Raw model outputs for one split, reused by evaluation and error analysis."""

    record_ids: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob_sarcastic: np.ndarray
    loss: float


class Trainer:
    """Fine-tune and evaluate a :class:`SarcasmModel`."""

    def __init__(
        self,
        model: SarcasmModel,
        config: Config,
        collator: SarcasmCollator,
        train_dataset: SarcasmDataset | None = None,
        val_dataset: SarcasmDataset | None = None,
        device: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            model: The model to train.
            config: Full experiment configuration.
            collator: Batch builder (holds tokenizer + evidence lookup).
            train_dataset: Training split; may be ``None`` for eval-only use.
            val_dataset: Validation split used for model selection.
            device: Override for device selection.
            output_dir: Where checkpoints and logs go; defaults to
                ``config.experiment.output_dir``.
        """
        self.config = config
        self.collator = collator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.output_dir = ensure_dir(output_dir or config.experiment.output_dir)
        self.history = TrainingHistory()
        self.amp_enabled, self.amp_dtype = resolve_amp(
            config.training.mixed_precision, self.device
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype is torch.float16
        )
        self._start_epoch = 0
        logger.info(
            "Trainer ready | device=%s amp=%s(%s) output=%s",
            self.device,
            self.amp_enabled,
            self.amp_dtype,
            self.output_dir,
        )

    # ------------------------------------------------------------- dataloaders
    def _loader(
        self, dataset: SarcasmDataset, batch_size: int, shuffle: bool
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.config.experiment.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
            num_workers=self.config.data.num_workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
            persistent_workers=self.config.data.num_workers > 0,
        )

    # --------------------------------------------------------------- optimiser
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW with discriminative LRs and no weight decay on bias/LayerNorm."""
        encoder_params_decay, encoder_params_nodecay = [], []
        head_params_decay, head_params_nodecay = [], []
        no_decay_markers = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_encoder = name.startswith("encoder.") or name.startswith("context_encoder.")
            is_no_decay = any(marker in name for marker in no_decay_markers)
            if is_encoder:
                (encoder_params_nodecay if is_no_decay else encoder_params_decay).append(param)
            else:
                (head_params_nodecay if is_no_decay else head_params_decay).append(param)

        training = self.config.training
        groups = [
            {
                "params": encoder_params_decay,
                "lr": training.learning_rate,
                "weight_decay": training.weight_decay,
            },
            {"params": encoder_params_nodecay, "lr": training.learning_rate, "weight_decay": 0.0},
            {
                "params": head_params_decay,
                "lr": training.head_learning_rate,
                "weight_decay": training.weight_decay,
            },
            {
                "params": head_params_nodecay,
                "lr": training.head_learning_rate,
                "weight_decay": 0.0,
            },
        ]
        groups = [g for g in groups if g["params"]]
        return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Linear warmup then linear decay to zero."""
        warmup_steps = max(1, int(total_steps * self.config.training.warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------- train
    def train(
        self, on_epoch_end: Callable[[EpochRecord], None] | None = None
    ) -> TrainingHistory:
        """Run the full training loop and return its history.

        Raises:
            ValueError: if no training or validation dataset was supplied.
        """
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Trainer needs both a train and a validation dataset")

        training = self.config.training
        train_loader = self._loader(self.train_dataset, training.batch_size, shuffle=True)
        steps_per_epoch = math.ceil(
            len(train_loader) / training.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * training.epochs
        self.history.steps_per_epoch = steps_per_epoch

        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, total_steps)
        if training.resume_from:
            self._load_trainer_state(training.resume_from, optimizer, scheduler)

        logger.info(
            "Training %d epochs | %d examples | batch=%d x accum=%d (effective %d) "
            "| %d optimiser steps",
            training.epochs,
            len(self.train_dataset),
            training.batch_size,
            training.gradient_accumulation_steps,
            training.batch_size * training.gradient_accumulation_steps,
            total_steps,
        )

        history_path = self.output_dir / HISTORY_FILENAME
        best_metric = self.history.best_metric
        patience_left = training.early_stopping_patience
        run_started = time.perf_counter()

        for epoch in range(self._start_epoch, training.epochs):
            epoch_started = time.perf_counter()
            train_loss = self._train_one_epoch(
                train_loader, optimizer, scheduler, epoch, steps_per_epoch
            )
            val_predictions = self.predict(self.val_dataset)
            val_metrics = compute_metrics(
                val_predictions.y_true,
                val_predictions.y_pred,
                val_predictions.y_prob_sarcastic,
            )
            score = getattr(val_metrics, training.metric_for_best, val_metrics.macro_f1)
            is_best = score > best_metric

            record = EpochRecord(
                epoch=epoch + 1,
                train_loss=round(train_loss, 6),
                val_loss=round(val_predictions.loss, 6),
                val_metrics=val_metrics.as_dict(),
                learning_rate=float(scheduler.get_last_lr()[0]),
                seconds=round(time.perf_counter() - epoch_started, 2),
                is_best=is_best,
            )
            self.history.epochs.append(record)
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.as_dict()) + "\n")

            logger.info(
                "epoch %d/%d | train_loss=%.4f val_loss=%.4f | %s | %.1fs%s",
                epoch + 1,
                training.epochs,
                train_loss,
                val_predictions.loss,
                val_metrics.summary(),
                record.seconds,
                "  <- best" if is_best else "",
            )
            if on_epoch_end is not None:
                on_epoch_end(record)

            if is_best:
                best_metric = score
                self.history.best_metric = score
                self.history.best_epoch = epoch + 1
                self.save_checkpoint(self.output_dir / "best_model", epoch, optimizer, scheduler)
                patience_left = training.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info(
                        "Early stopping at epoch %d (no %s improvement for %d epochs)",
                        epoch + 1,
                        training.metric_for_best,
                        training.early_stopping_patience,
                    )
                    self.history.stopped_early = True
                    break
            if training.save_every_epoch:
                self.save_checkpoint(
                    self.output_dir / f"epoch_{epoch + 1}", epoch, optimizer, scheduler
                )
            self.save_checkpoint(self.output_dir / "last", epoch, optimizer, scheduler)

        self.history.total_seconds = time.perf_counter() - run_started
        write_json(self.output_dir / "history.json", self.history.as_dict())
        return self.history

    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch: int,
        steps_per_epoch: int,
    ) -> float:
        """Run one epoch; returns the mean loss per example."""
        self.model.train()
        training = self.config.training
        accumulated = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            inputs = self._to_device(model_inputs(batch))
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                output = self.model(**inputs)
                loss = output.loss
            if loss is None:  # pragma: no cover - labels are always present here
                raise RuntimeError("Model returned no loss; were labels supplied?")

            scaled = loss / training.gradient_accumulation_steps
            if self.scaler.is_enabled():
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            batch_size = inputs["utterance_input_ids"].shape[0]
            accumulated += float(loss.detach()) * batch_size
            seen += batch_size

            is_last = step == len(loader) - 1
            if (step + 1) % training.gradient_accumulation_steps == 0 or is_last:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), training.max_grad_norm
                )
                if self.scaler.is_enabled():
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if training.log_every and (step + 1) % training.log_every == 0:
                logger.info(
                    "  epoch %d | step %d/%d | loss=%.4f | lr=%.2e",
                    epoch + 1,
                    step + 1,
                    len(loader),
                    accumulated / max(1, seen),
                    scheduler.get_last_lr()[0],
                )
        return accumulated / max(1, seen)

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, dataset: SarcasmDataset, batch_size: int | None = None) -> Predictions:
        """Run the model over ``dataset`` and collect predictions."""
        self.model.eval()
        loader = self._loader(
            dataset, batch_size or self.config.training.eval_batch_size, shuffle=False
        )
        record_ids: list[str] = []
        trues: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        probs: list[np.ndarray] = []
        total_loss = 0.0
        seen = 0

        for batch in loader:
            inputs = self._to_device(model_inputs(batch))
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                output = self.model(**inputs)
            probabilities = output.probabilities.float().cpu().numpy()
            labels = inputs["labels"].cpu().numpy()
            record_ids.extend(batch["record_ids"])
            trues.append(labels)
            preds.append(probabilities.argmax(axis=1))
            probs.append(probabilities[:, 1])
            if output.loss is not None:
                total_loss += float(output.loss) * labels.shape[0]
            seen += labels.shape[0]

        return Predictions(
            record_ids=record_ids,
            y_true=np.concatenate(trues) if trues else np.zeros(0, dtype=int),
            y_pred=np.concatenate(preds) if preds else np.zeros(0, dtype=int),
            y_prob_sarcastic=np.concatenate(probs) if probs else np.zeros(0, dtype=float),
            loss=total_loss / max(1, seen),
        )

    def evaluate(self, dataset: SarcasmDataset) -> tuple[MetricResult, Predictions]:
        """Predict on ``dataset`` and compute the metric suite."""
        predictions = self.predict(dataset)
        metrics = compute_metrics(
            predictions.y_true, predictions.y_pred, predictions.y_prob_sarcastic
        )
        return metrics, predictions

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move every tensor in ``batch`` to the training device."""
        return {
            key: (value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(
        self,
        directory: str | Path,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    ) -> Path:
        """Save model weights, tokenizer, config and optimiser state."""
        target = ensure_dir(directory)
        self.model.save_pretrained(
            target,
            extra={
                "experiment": self.config.experiment.name,
                "epoch": epoch + 1,
                "best_metric": self.history.best_metric,
                "metric_for_best": self.config.training.metric_for_best,
            },
        )
        self.collator.tokenizer.save_pretrained(target)
        self.config.save(target / "training_config.yaml")
        state: dict[str, Any] = {
            "epoch": epoch + 1,
            "history": self.history.as_dict(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        if self.scaler.is_enabled():
            state["scaler"] = self.scaler.state_dict()
        torch.save(state, target / TRAINER_STATE_FILENAME)
        return target

    def _load_trainer_state(
        self,
        directory: str | Path,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
    ) -> None:
        """Restore optimiser/scheduler/epoch from a checkpoint directory."""
        path = Path(directory) / TRAINER_STATE_FILENAME
        if not path.exists():
            logger.warning("resume_from set but %s does not exist; starting fresh", path)
            return
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state and self.scaler.is_enabled():
            self.scaler.load_state_dict(state["scaler"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu().to(torch.uint8))
        self._start_epoch = int(state.get("epoch", 0))
        raw_history = state.get("history") or {}
        self.history.best_metric = float(raw_history.get("best_metric", -1.0))
        self.history.best_epoch = int(raw_history.get("best_epoch", -1))
        logger.info("Resumed from %s at epoch %d", path, self._start_epoch)


def build_predictions_frame(
    predictions: Predictions, records: Sequence[Any]
) -> list[dict[str, Any]]:
    """Join predictions back onto their records for error analysis / export."""
    by_id = {record.id: record for record in records}
    rows: list[dict[str, Any]] = []
    for position, record_id in enumerate(predictions.record_ids):
        record = by_id.get(record_id)
        if record is None:  # pragma: no cover
            continue
        probability = float(predictions.y_prob_sarcastic[position])
        predicted = int(predictions.y_pred[position])
        rows.append(
            {
                "id": record_id,
                "text": record.text,
                "context": record.context,
                "dataset": record.dataset,
                "domain": record.domain,
                "true_label": int(predictions.y_true[position]),
                "predicted_label": predicted,
                "prob_sarcastic": probability,
                "confidence": probability if predicted == 1 else 1.0 - probability,
                "correct": bool(int(predictions.y_true[position]) == predicted),
            }
        )
    return rows
