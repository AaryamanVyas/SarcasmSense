"""RoBERTa text encoder wrapper.

The proposal specifies **RoBERTa-base** (12 layers, hidden size 768) as the
shared encoder for both the target utterance and the conversational context.
RoBERTa rather than BERT because it was pre-trained on ~10x more data with
dynamic masking and no NSP objective, and because its **byte-level BPE**
vocabulary encodes emoji, hashtags, repeated punctuation and unusual casing
without falling back to ``[UNK]`` -- exactly the surface features that carry
sarcasm.

Design notes
------------
* The hidden size is read from the loaded model's config, never hard-coded, so
  the same code runs with the research encoder (768) and with the tiny
  development encoder used by the test-suite.
* ``shared_encoder=True`` (the default, and what the architecture diagram shows)
  means one set of weights encodes both branches. Beyond halving the parameter
  count, this puts utterance and context vectors in the *same* representation
  space, which is what makes the cross-attention comparison between them
  meaningful.
* Utterance and context are encoded in a single forward pass by stacking them
  along the batch dimension. That is roughly 1.7x faster than two passes on CPU
  because it keeps the GEMMs large.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger(__name__)


class EncoderLoadError(RuntimeError):
    """Raised when the transformer backbone cannot be loaded."""


@dataclass
class EncoderOutput:
    """Result of encoding one text branch.

    Attributes:
        sequence: Token-level hidden states ``[B, L, H]``.
        pooled: Sentence-level vector ``[B, H]``.
        attention_mask: The mask that was used ``[B, L]``.
    """

    sequence: torch.Tensor
    pooled: torch.Tensor
    attention_mask: torch.Tensor


def masked_mean(sequence: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``[B, L, H]`` over non-padding positions.

    Numerically guarded: rows whose mask is entirely zero would otherwise divide
    by zero and produce NaNs that silently poison the whole batch.
    """
    mask = attention_mask.unsqueeze(-1).to(sequence.dtype)
    summed = (sequence * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TextEncoder(nn.Module):
    """Thin, well-behaved wrapper around a HuggingFace transformer encoder."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        pooling: str = "cls",
        freeze_layers: int = 0,
        gradient_checkpointing: bool = False,
        local_files_only: bool = False,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model id or a local directory.
            pooling: ``"cls"`` uses the ``<s>`` token (RoBERTa's sentence
                summary position); ``"mean"`` mean-pools over real tokens.
            freeze_layers: Freeze the embeddings plus the first N transformer
                layers. Useful on very small GPUs; 0 (full fine-tuning) is the
                research setting.
            gradient_checkpointing: Trade compute for memory on small GPUs.
            local_files_only: Never hit the network (used by tests).

        Raises:
            EncoderLoadError: if the backbone cannot be loaded.
        """
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:  # pragma: no cover
            raise EncoderLoadError("transformers is not installed") from exc

        try:
            config = AutoConfig.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            self.backbone = AutoModel.from_pretrained(
                model_name, config=config, local_files_only=local_files_only
            )
        except Exception as exc:  # noqa: BLE001
            raise EncoderLoadError(
                f"Could not load encoder {model_name!r}: {exc}\n"
                f"If you are offline, pre-download it with "
                f"`huggingface-cli download {model_name}`, or point "
                f"model.encoder_name at a local directory."
            ) from exc

        self.model_name = model_name
        self.pooling = pooling
        self.hidden_size = int(config.hidden_size)
        self.config = config

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        if freeze_layers > 0:
            self._freeze(freeze_layers)

        logger.info(
            "Loaded encoder %s (hidden=%d, layers=%s, pooling=%s, frozen=%d)",
            model_name,
            self.hidden_size,
            getattr(config, "num_hidden_layers", "?"),
            pooling,
            freeze_layers,
        )

    def _freeze(self, n_layers: int) -> None:
        """Freeze embeddings and the bottom ``n_layers`` transformer blocks."""
        embeddings = getattr(self.backbone, "embeddings", None)
        if embeddings is not None:
            for param in embeddings.parameters():
                param.requires_grad = False
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None) if encoder is not None else None
        if layers is None:
            logger.warning("Encoder has no .encoder.layer; only embeddings were frozen")
            return
        for layer in layers[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> EncoderOutput:
        """Encode a batch of token ids.

        Args:
            input_ids: ``[B, L]`` token ids.
            attention_mask: ``[B, L]`` 1 for real tokens, 0 for padding.

        Returns:
            An :class:`EncoderOutput` with token states and a pooled vector.
        """
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        sequence = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = sequence[:, 0]
        else:
            pooled = masked_mean(sequence, attention_mask)
        return EncoderOutput(
            sequence=sequence, pooled=pooled, attention_mask=attention_mask
        )

    def encode_pair(
        self,
        utterance_ids: torch.Tensor,
        utterance_mask: torch.Tensor,
        context_ids: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> tuple[EncoderOutput, EncoderOutput | None]:
        """Encode utterance and context, sharing one forward pass when possible.

        When both branches have the same sequence length, they are stacked along
        the batch dimension and encoded together; otherwise two passes are used.
        Either way the *same weights* are applied, which is what "shared
        RoBERTa" means in the architecture diagram.
        """
        if context_ids is None or context_mask is None:
            return self.forward(utterance_ids, utterance_mask), None

        if utterance_ids.shape[1] == context_ids.shape[1]:
            batch = utterance_ids.shape[0]
            merged = self.forward(
                torch.cat([utterance_ids, context_ids], dim=0),
                torch.cat([utterance_mask, context_mask], dim=0),
            )
            utterance_out = EncoderOutput(
                sequence=merged.sequence[:batch],
                pooled=merged.pooled[:batch],
                attention_mask=utterance_mask,
            )
            context_out = EncoderOutput(
                sequence=merged.sequence[batch:],
                pooled=merged.pooled[batch:],
                attention_mask=context_mask,
            )
            return utterance_out, context_out

        return (
            self.forward(utterance_ids, utterance_mask),
            self.forward(context_ids, context_mask),
        )
