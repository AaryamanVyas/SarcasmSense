"""The full context-aware, knowledge-grounded sarcasm classifier.

This single class implements all four systems required by the study; they differ
only in configuration, which is what makes the comparison fair:

======================  ============  ==============  ==================
System                  use_context   use_knowledge   fusion
======================  ============  ==============  ==================
Baseline 1 (RoBERTa)    False         False           ``none``
Baseline 2 (+context)   True          False           ``cross_attention``
Baseline 3 (+knowledge) False         True            ``cross_attention``
Proposed full model     True          True            ``cross_attention``
Ablation (concat)       True          True            ``concat``
======================  ============  ==============  ==================

Forward pass::

    utterance ids ─► shared RoBERTa ─► utterance states [B,Lu,H] ─► pooled [B,H]
    context ids   ─► shared RoBERTa ─► context  states [B,Lc,H] ─► pooled [B,H]
    evidence vecs ─► Linear(D_sent→H) + LayerNorm ─────────────► [B,K,H]
                                     │
                                     ▼
                     cross-attention fusion  (Q = utterance, KV = context ⊕ evidence)
                                     │
                                     ▼
        concat([utterance, context, attended]) ─► head ─► 2 logits ─► softmax

The concatenation at the end is what gives the ``2304 → 256`` dense layer in the
architecture diagram when the encoder is RoBERTa-base (3 x 768). The head input
size is nevertheless *computed*, never hard-coded, so smaller encoders work too.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.classifier import ClassificationHead, probabilities_from_logits
from src.models.encoder import TextEncoder
from src.models.fusion import FusionOutput, build_fusion
from src.utils.config import ModelConfig
from src.utils.paths import ensure_dir, resolve

logger = logging.getLogger(__name__)

WEIGHTS_FILENAME = "pytorch_model.bin"
MODEL_CONFIG_FILENAME = "model_config.json"


@dataclass
class SarcasmModelOutput:
    """Everything one forward pass produces.

    Attributes:
        logits: ``[B, 2]`` raw scores.
        probabilities: ``[B, 2]`` softmax over ``logits``.
        loss: Scalar training loss when labels were supplied.
        fusion_attention: ``[B, M]`` attention over memory slots, or ``None``.
        memory_spans: Slot ranges per source, used by the explanation layer.
        utterance_vector / context_vector: Pooled branch representations, kept
            for analysis and for the "context influence" panel in the UI.
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    loss: torch.Tensor | None = None
    fusion_attention: torch.Tensor | None = None
    memory_spans: dict[str, tuple[int, int]] | None = None
    utterance_vector: torch.Tensor | None = None
    context_vector: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class SarcasmModel(nn.Module):
    """RoBERTa + context + retrieved commonsense + cross-attention fusion."""

    def __init__(
        self,
        config: ModelConfig,
        knowledge_dim: int = 384,
        knowledge_slots: int = 5,
        num_labels: int = 2,
        class_weights: torch.Tensor | None = None,
        local_files_only: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        """
        Args:
            config: Architecture configuration.
            knowledge_dim: Dimensionality of the sentence-transformer evidence
                vectors (384 for MiniLM). Projected to the encoder hidden size.
            knowledge_slots: Number of evidence slots per example (top-k).
            num_labels: Always 2 for this task.
            class_weights: Optional ``[2]`` tensor for a weighted loss.
            local_files_only: Do not hit the network when loading the backbone.
            gradient_checkpointing: Memory-saving mode for small GPUs.
        """
        super().__init__()
        self.config = config
        self.knowledge_dim = knowledge_dim
        self.knowledge_slots = knowledge_slots
        self.num_labels = num_labels

        self.encoder = TextEncoder(
            model_name=config.encoder_name,
            pooling=config.pooling,
            freeze_layers=config.freeze_encoder_layers,
            gradient_checkpointing=gradient_checkpointing,
            local_files_only=local_files_only,
        )
        hidden = self.encoder.hidden_size
        self.hidden_size = hidden

        self.context_encoder: TextEncoder | None = None
        if config.use_context and not config.shared_encoder:
            self.context_encoder = TextEncoder(
                model_name=config.encoder_name,
                pooling=config.pooling,
                freeze_layers=config.freeze_encoder_layers,
                gradient_checkpointing=gradient_checkpointing,
                local_files_only=local_files_only,
            )

        # Learned stand-in for "this example has no conversational context".
        # A zero vector would be indistinguishable from a genuinely neutral
        # context; a learned parameter lets the model represent absence itself.
        self.empty_context = nn.Parameter(torch.randn(hidden) * 0.02)

        self.knowledge_projection: nn.Module | None = None
        if config.use_knowledge:
            if config.knowledge_projection or knowledge_dim != hidden:
                self.knowledge_projection = nn.Sequential(
                    nn.Linear(knowledge_dim, hidden),
                    nn.LayerNorm(hidden),
                )
            else:  # pragma: no cover - only when dims coincide and projection off
                self.knowledge_projection = nn.Identity()

        self.fusion = build_fusion(
            fusion_type=config.fusion,
            hidden_size=hidden,
            num_heads=config.num_attention_heads,
            num_layers=config.fusion_layers,
            dropout=config.fusion_dropout,
            use_context=config.use_context,
            use_knowledge=config.use_knowledge,
        )

        head_input = self.classifier_input_size
        self.head = ClassificationHead(
            input_size=head_input,
            hidden_size=config.classifier_hidden,
            num_labels=num_labels,
            dropout=config.classifier_dropout,
        )

        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_labels),
            persistent=True,
        )
        self.use_class_weights = class_weights is not None

        logger.info(
            "SarcasmModel ready | encoder=%s hidden=%d fusion=%s context=%s "
            "knowledge=%s head_input=%d params=%.1fM",
            config.encoder_name,
            hidden,
            config.fusion,
            config.use_context,
            config.use_knowledge,
            head_input,
            self.num_parameters() / 1e6,
        )

    # ----------------------------------------------------------------- shapes
    @property
    def classifier_input_size(self) -> int:
        """Width of the vector fed to the classification head.

        Full model with RoBERTa-base: ``768 (utterance) + 768 (context) +
        768 (attended) = 2304``.
        """
        size = self.hidden_size  # utterance vector, always present
        if self.config.use_context:
            size += self.hidden_size  # context vector
        if self.fusion is not None:
            size += self.fusion.output_size  # attended / concat-projected vector
        return size

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        utterance_input_ids: torch.Tensor,
        utterance_attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor | None = None,
        context_attention_mask: torch.Tensor | None = None,
        has_context: torch.Tensor | None = None,
        knowledge_vectors: torch.Tensor | None = None,
        knowledge_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SarcasmModelOutput:
        """Run the full pipeline.

        Args:
            utterance_input_ids: ``[B, Lu]``.
            utterance_attention_mask: ``[B, Lu]``.
            context_input_ids: ``[B, Lc]``; ignored when ``use_context`` is off.
            context_attention_mask: ``[B, Lc]``.
            has_context: ``[B]`` boolean/float flag marking examples that
                actually have conversational context. Examples without it get
                the learned :attr:`empty_context` vector and a fully masked
                context memory, so the model neither crashes nor hallucinates
                context that was not there.
            knowledge_vectors: ``[B, K, knowledge_dim]`` evidence embeddings.
            knowledge_mask: ``[B, K]`` 1 where a real fact was retrieved.
            labels: ``[B]`` gold labels; when given, ``loss`` is populated.

        Returns:
            A :class:`SarcasmModelOutput`.
        """
        batch = utterance_input_ids.shape[0]
        device = utterance_input_ids.device

        want_context = self.config.use_context and context_input_ids is not None
        if want_context and context_attention_mask is None:
            context_attention_mask = torch.ones_like(context_input_ids)

        # ------------------------------------------------------------ encode
        if want_context and self.context_encoder is None:
            utterance_out, context_out = self.encoder.encode_pair(
                utterance_input_ids,
                utterance_attention_mask,
                context_input_ids,
                context_attention_mask,
            )
        else:
            utterance_out = self.encoder(utterance_input_ids, utterance_attention_mask)
            context_out = None
            if want_context and self.context_encoder is not None:
                context_out = self.context_encoder(
                    context_input_ids, context_attention_mask
                )

        # ------------------------------------------------- context presence
        context_sequence = None
        context_mask = None
        context_vector = None
        if context_out is not None:
            if has_context is None:
                present = context_out.attention_mask.sum(dim=1) > 0
            else:
                present = has_context.bool().to(device)
            present_f = present.to(utterance_out.pooled.dtype).unsqueeze(-1)
            context_vector = (
                present_f * context_out.pooled
                + (1.0 - present_f) * self.empty_context.to(utterance_out.pooled.dtype)
            )
            context_sequence = context_out.sequence
            # Mask out every context token for examples that have no context,
            # so cross-attention cannot attend to the padding of an empty turn.
            context_mask = context_out.attention_mask * present.unsqueeze(-1).to(
                context_out.attention_mask.dtype
            )
        elif self.config.use_context:
            context_vector = self.empty_context.to(utterance_out.pooled.dtype).expand(
                batch, self.hidden_size
            )

        # ---------------------------------------------------------- evidence
        projected_knowledge = None
        if self.config.use_knowledge and self.knowledge_projection is not None:
            if knowledge_vectors is None:
                knowledge_vectors = torch.zeros(
                    batch,
                    self.knowledge_slots,
                    self.knowledge_dim,
                    device=device,
                    dtype=utterance_out.pooled.dtype,
                )
                knowledge_mask = torch.zeros(
                    batch, self.knowledge_slots, device=device, dtype=torch.bool
                )
            projected_knowledge = self.knowledge_projection(
                knowledge_vectors.to(utterance_out.pooled.dtype)
            )

        # ------------------------------------------------------------- fuse
        fusion_out: FusionOutput | None = None
        parts = [utterance_out.pooled]
        if self.config.use_context and context_vector is not None:
            parts.append(context_vector)
        if self.fusion is not None:
            fusion_out = self.fusion(
                utterance_sequence=utterance_out.sequence,
                utterance_mask=utterance_out.attention_mask,
                context_sequence=context_sequence,
                context_mask=context_mask,
                knowledge_vectors=projected_knowledge,
                knowledge_mask=knowledge_mask,
            )
            parts.append(fusion_out.fused)

        fused = torch.cat(parts, dim=-1)
        logits = self.head(fused)

        loss = None
        if labels is not None:
            weights = (
                self.class_weights.to(device=device, dtype=logits.dtype)
                if self.use_class_weights
                else None
            )
            loss = nn.functional.cross_entropy(logits, labels.long(), weight=weights)

        return SarcasmModelOutput(
            logits=logits,
            probabilities=probabilities_from_logits(logits),
            loss=loss,
            fusion_attention=fusion_out.attention if fusion_out else None,
            memory_spans=fusion_out.memory_spans if fusion_out else None,
            utterance_vector=utterance_out.pooled,
            context_vector=context_vector,
        )

    # ------------------------------------------------------------ persistence
    def save_pretrained(self, directory: str | Path, extra: dict[str, Any] | None = None) -> Path:
        """Save weights plus the architecture config needed to rebuild the model."""
        target = ensure_dir(directory)
        torch.save(self.state_dict(), target / WEIGHTS_FILENAME)
        payload = {
            "model_config": {
                key: getattr(self.config, key)
                for key in self.config.__dataclass_fields__  # type: ignore[attr-defined]
            },
            "knowledge_dim": self.knowledge_dim,
            "knowledge_slots": self.knowledge_slots,
            "num_labels": self.num_labels,
            "hidden_size": self.hidden_size,
            "classifier_input_size": self.classifier_input_size,
            "use_class_weights": self.use_class_weights,
            "class_weights": self.class_weights.detach().cpu().tolist(),
            "n_parameters": self.num_parameters(),
        }
        if extra:
            payload["extra"] = extra
        (target / MODEL_CONFIG_FILENAME).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.info("Saved model -> %s", target)
        return target

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        map_location: str | torch.device = "cpu",
        local_files_only: bool = False,
        encoder_name_override: str | None = None,
    ) -> "SarcasmModel":
        """Rebuild a model saved by :meth:`save_pretrained`.

        Args:
            directory: Checkpoint directory.
            map_location: Device to load the weights onto.
            local_files_only: Do not hit the network for the backbone.
            encoder_name_override: Load the backbone architecture from a
                different path (used when the checkpoint bundles its own copy of
                the encoder, or when a dev checkpoint is relocated).

        Raises:
            FileNotFoundError: if the checkpoint is incomplete.
        """
        source = resolve(directory)
        config_path = source / MODEL_CONFIG_FILENAME
        weights_path = source / WEIGHTS_FILENAME
        for path in (config_path, weights_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Checkpoint is incomplete: {path} not found. "
                    f"Train a model with scripts/train.py first."
                )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        model_config = ModelConfig(**payload["model_config"])
        if encoder_name_override:
            model_config.encoder_name = encoder_name_override

        model = cls(
            config=model_config,
            knowledge_dim=int(payload.get("knowledge_dim", 384)),
            knowledge_slots=int(payload.get("knowledge_slots", 5)),
            num_labels=int(payload.get("num_labels", 2)),
            local_files_only=local_files_only,
        )
        state = torch.load(weights_path, map_location=map_location, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Checkpoint is missing %d parameter(s): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning(
                "Checkpoint has %d unexpected parameter(s): %s", len(unexpected), unexpected[:5]
            )
        model.use_class_weights = bool(payload.get("use_class_weights", False))
        model.eval()
        return model

    # ----------------------------------------------------------------- info
    def describe(self) -> dict[str, Any]:
        """Architecture summary surfaced by ``GET /model/info`` and the UI."""
        return {
            "encoder_name": self.config.encoder_name,
            "hidden_size": self.hidden_size,
            "fusion": self.config.fusion,
            "fusion_layers": self.config.fusion_layers,
            "attention_heads": self.config.num_attention_heads,
            "uses_context": self.config.use_context,
            "uses_knowledge": self.config.use_knowledge,
            "shared_encoder": self.config.shared_encoder,
            "max_utterance_length": self.config.max_utterance_length,
            "max_context_length": self.config.max_context_length,
            "knowledge_dim": self.knowledge_dim,
            "knowledge_slots": self.knowledge_slots,
            "classifier_input_size": self.classifier_input_size,
            "classifier_hidden": self.config.classifier_hidden,
            "num_labels": self.num_labels,
            "n_parameters": self.num_parameters(),
            "n_trainable_parameters": self.num_parameters(trainable_only=True),
        }
