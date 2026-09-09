"""The inference service: one object that owns the whole prediction path.

Loaded **once** at process start (FastAPI lifespan, benchmark script, error
analysis) and reused for every request. Loading a 500 MB encoder per request
would dominate the latency budget by two orders of magnitude.

What a single call does::

    text + context
        -> preprocessing
        -> commonsense retrieval (FAISS, offline)
        -> tokenisation (utterance branch + context branch)
        -> SarcasmModel forward  (shared RoBERTa -> cross-attention fusion -> head)
        -> label, probabilities, confidence
        -> attention split back into per-turn context influence and per-fact
           evidence influence

Explanation (LIME/SHAP) is deliberately a *separate*, optional step: it costs
hundreds of extra forward passes and must not be on the critical path of a plain
classification request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.data.preprocessing import PreprocessingOptions, clean_context, clean_text
from src.data.schema import LABEL_NAMES, MAX_CONTEXT_TURNS
from src.models.dev_assets import is_development_asset
from src.models.sarcasm_model import SarcasmModel
from src.models.tokenization import SarcasmTokenizer
from src.retrieval.retriever import CommonsenseRetriever, RetrievalResult, RetrievalStatus
from src.utils.config import Config
from src.utils.paths import resolve

logger = logging.getLogger(__name__)


@dataclass
class ContextInfluence:
    """How much one conversational turn contributed, per cross-attention."""

    turn_index: int
    text: str
    attention: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return asdict(self)


@dataclass
class PredictionResult:
    """Everything one classification produces (before explanation)."""

    text: str
    context_used: list[str]
    label: int
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    retrieval: RetrievalResult
    context_influence: list[ContextInfluence] = field(default_factory=list)
    evidence_attention: list[float] = field(default_factory=list)
    unattributed_attention: float = 0.0
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "text": self.text,
            "context_used": self.context_used,
            "label": self.label,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "retrieved_evidence": [e.to_dict() for e in self.retrieval.evidence],
            "retrieval_status": self.retrieval.status.value,
            "context_influence": [c.to_dict() for c in self.context_influence],
            "evidence_attention": [round(v, 4) for v in self.evidence_attention],
            "unattributed_attention": round(self.unattributed_attention, 4),
            "warnings": self.warnings,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
        }


class SarcasmPredictor:
    """Stateful, thread-safe-enough inference service (single process, eval mode)."""

    def __init__(
        self,
        model: SarcasmModel,
        tokenizer: SarcasmTokenizer,
        retriever: CommonsenseRetriever | None,
        config: Config,
        device: torch.device | str = "cpu",
        checkpoint_dir: str | Path | None = None,
        warnings: Sequence[str] = (),
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.preprocessing = PreprocessingOptions()
        self.startup_warnings = list(warnings)
        self.is_development_asset = is_development_asset(config.model.encoder_name) or (
            bool(checkpoint_dir) and is_development_asset(checkpoint_dir)  # type: ignore[arg-type]
        )
        if self.is_development_asset:
            self.startup_warnings.append(
                "This server is running a DEVELOPMENT/TEST checkpoint built on a tiny "
                "randomly-initialised encoder. Predictions are structurally valid but "
                "carry no research meaning."
            )
        # Disable dropout etc. once, here, rather than per request.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    # -------------------------------------------------------------- factory
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        config: Config | None = None,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> "SarcasmPredictor":
        """Load a trained checkpoint plus its retrieval stack.

        The checkpoint directory is expected to contain what
        :meth:`SarcasmModel.save_pretrained` and the trainer write:
        ``pytorch_model.bin``, ``model_config.json``, the tokenizer files and
        ``training_config.yaml``.

        Raises:
            FileNotFoundError: if the checkpoint is missing or incomplete.
        """
        from src.training.pipeline import build_retrieval
        from src.training.trainer import resolve_device

        source = resolve(checkpoint_dir)
        if not source.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {source}. Set MODEL_PATH to a "
                f"directory produced by scripts/train.py (e.g. checkpoints/full_model/best_model)."
            )
        if config is None:
            config_path = source / "training_config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"{config_path} not found; cannot reconstruct the runtime "
                    f"configuration for this checkpoint."
                )
            config = Config.from_yaml(config_path)

        torch_device = resolve_device(device)
        model = SarcasmModel.from_pretrained(
            source, map_location=str(torch_device), local_files_only=local_files_only
        )
        tokenizer = SarcasmTokenizer(
            model_name=str(source),
            max_utterance_length=config.model.max_utterance_length,
            max_context_length=config.model.max_context_length,
            local_files_only=True,
        )
        retriever, _index, _encoder, warnings = build_retrieval(config, device=str(torch_device))
        return cls(
            model=model,
            tokenizer=tokenizer,
            retriever=retriever,
            config=config,
            device=torch_device,
            checkpoint_dir=source,
            warnings=warnings,
        )

    # ------------------------------------------------------------ preprocess
    def prepare(
        self, text: str, context: Sequence[str] | None = None
    ) -> tuple[str, list[str]]:
        """Clean the raw request and truncate the context to the last 3 turns."""
        cleaned = clean_text(text, self.preprocessing)
        turns = clean_context(
            context, max_turns=min(self.config.data.max_context_turns, MAX_CONTEXT_TURNS),
            options=self.preprocessing,
        )
        return cleaned, turns

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(
        self,
        text: str,
        context: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> PredictionResult:
        """Classify one utterance and report what influenced the decision."""
        started = time.perf_counter()
        warnings: list[str] = list(self.startup_warnings)
        cleaned, turns = self.prepare(text, context)
        if not cleaned:
            raise ValueError("Target utterance is empty after preprocessing")

        # 1. retrieval -----------------------------------------------------
        retrieval_started = time.perf_counter()
        if self.retriever is not None:
            retrieval = self.retriever.retrieve(cleaned, turns, top_k=top_k)
        else:
            retrieval = RetrievalResult(
                status=RetrievalStatus.DISABLED,
                query=cleaned,
                warning="Commonsense retrieval is disabled for this model configuration.",
            )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        if retrieval.warning:
            warnings.append(retrieval.warning)

        # 2. model ---------------------------------------------------------
        model_started = time.perf_counter()
        batch = self._build_batch([cleaned], [turns], [retrieval])
        output = self.model(**batch)
        probabilities = output.probabilities[0].cpu().numpy()
        label = int(probabilities.argmax())
        model_ms = (time.perf_counter() - model_started) * 1000

        # 3. attention attribution ----------------------------------------
        context_influence, evidence_attention, null_attention = self._split_attention(
            output, turns, len(retrieval.evidence)
        )

        return PredictionResult(
            text=cleaned,
            context_used=turns,
            label=label,
            prediction=LABEL_NAMES[label],
            confidence=float(probabilities[label]),
            probabilities={
                "not_sarcastic": float(probabilities[0]),
                "sarcastic": float(probabilities[1]),
            },
            retrieval=retrieval,
            context_influence=context_influence,
            evidence_attention=evidence_attention,
            unattributed_attention=null_attention,
            warnings=warnings,
            timings_ms={
                "retrieval": retrieval_ms,
                "model": model_ms,
                "total": (time.perf_counter() - started) * 1000,
            },
        )

    @torch.no_grad()
    def predict_proba(
        self,
        texts: Sequence[str],
        context: Sequence[str] | None = None,
        retrieval: RetrievalResult | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return ``[N, 2]`` probabilities for many utterances sharing one context.

        This is the callable LIME and SHAP perturb against. Context and retrieved
        evidence are **held fixed** across perturbations, so the attributions
        answer "which words of the utterance drove *this* decision, given the
        evidence that was actually retrieved" -- rather than mixing in the
        second-order effect of perturbations changing what gets retrieved.
        """
        turns = list(context or [])
        results: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = [t if t.strip() else "." for t in texts[start : start + batch_size]]
            retrievals = [retrieval or RetrievalResult()] * len(chunk)
            batch = self._build_batch(chunk, [turns] * len(chunk), retrievals)
            output = self.model(**batch)
            results.append(output.probabilities.cpu().numpy())
        if not results:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(results, axis=0)

    # ---------------------------------------------------------------- batch
    def _build_batch(
        self,
        texts: Sequence[str],
        contexts: Sequence[Sequence[str]],
        retrievals: Sequence[RetrievalResult],
    ) -> dict[str, Any]:
        """Tokenise and attach evidence vectors, moving everything to the device."""
        use_context = self.config.model.use_context
        batch = self.tokenizer.encode(
            list(texts), [list(c) for c in contexts] if use_context else [[] for _ in texts]
        )
        if not use_context:
            batch.pop("context_input_ids", None)
            batch.pop("context_attention_mask", None)
            batch["has_context"] = torch.zeros(len(texts), dtype=torch.bool)

        if self.config.model.use_knowledge:
            k = self.config.retrieval.top_k
            dim = self.model.knowledge_dim
            vectors = np.zeros((len(texts), k, dim), dtype=np.float32)
            mask = np.zeros((len(texts), k), dtype=bool)
            if self.retriever is not None and self.retriever.index is not None:
                for row, result in enumerate(retrievals):
                    ids = result.embedding_ids[:k]
                    if ids:
                        vectors[row, : len(ids)] = self.retriever.index.reconstruct(ids)
                        mask[row, : len(ids)] = True
            batch["knowledge_vectors"] = torch.from_numpy(vectors)
            batch["knowledge_mask"] = torch.from_numpy(mask)

        batch.pop("record_ids", None)
        return {
            key: (value.to(self.device) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }

    # ------------------------------------------------------------ attention
    def _split_attention(
        self, output: Any, turns: Sequence[str], n_evidence: int
    ) -> tuple[list[ContextInfluence], list[float], float]:
        """Split fusion attention into per-turn and per-evidence contributions.

        Context attention is over *tokens*, so token offsets are mapped back onto
        the character span of each turn and the weights are summed per turn.
        """
        if output.fusion_attention is None or output.memory_spans is None:
            return [], [], 0.0
        attention = output.fusion_attention[0].detach().float().cpu().numpy()
        spans = output.memory_spans

        null_start, null_end = spans.get("null", (0, 0))
        null_attention = float(attention[null_start:null_end].sum())

        evidence_attention: list[float] = []
        if "knowledge" in spans and n_evidence:
            start, end = spans["knowledge"]
            slice_ = attention[start:end]
            evidence_attention = [float(v) for v in slice_[:n_evidence]]

        context_influence: list[ContextInfluence] = []
        if "context" in spans and turns:
            start, end = spans["context"]
            context_weights = attention[start:end]
            _encoded, offsets, turn_spans = self.tokenizer.encode_context_with_offsets(turns)
            per_turn = np.zeros(len(turn_spans), dtype=np.float64)
            for position, (char_start, char_end) in enumerate(offsets):
                if position >= len(context_weights) or char_end <= char_start:
                    continue
                for turn_index, (span_start, span_end) in enumerate(turn_spans):
                    if char_start >= span_start and char_end <= span_end:
                        per_turn[turn_index] += float(context_weights[position])
                        break
            for turn_index, turn in enumerate(turns):
                context_influence.append(
                    ContextInfluence(
                        turn_index=turn_index + 1,
                        text=turn,
                        attention=float(per_turn[turn_index])
                        if turn_index < len(per_turn)
                        else 0.0,
                    )
                )
        return context_influence, evidence_attention, null_attention

    # ------------------------------------------------------------------ info
    def describe(self) -> dict[str, Any]:
        """Model + retrieval metadata for ``GET /model/info``."""
        info: dict[str, Any] = {
            "model": self.model.describe(),
            "device": str(self.device),
            "checkpoint": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "experiment": self.config.experiment.name,
            "is_development_asset": self.is_development_asset,
            "max_context_turns": self.config.data.max_context_turns,
            "startup_warnings": self.startup_warnings,
            "task": "Binary sarcasm classification",
            "input_modality": "text only",
        }
        info["retrieval"] = (
            self.retriever.stats() if self.retriever is not None else {"available": False}
        )
        return info
