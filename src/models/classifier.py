"""Classification head.

Exactly the stack specified in the proposal::

    fused -> Linear -> ReLU -> Dropout -> Linear -> 2-class logits -> Softmax

The head returns **logits**; softmax is applied only where a probability is
actually needed. That is not a stylistic choice: ``CrossEntropyLoss`` expects
logits and applies ``log_softmax`` internally with a numerically stable
implementation, so feeding it already-softmaxed values would both double-apply
the non-linearity and lose precision.
"""

from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """Two-layer MLP producing binary sarcasm logits."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_labels: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Args:
            input_size: Width of the fused representation (2304 for the full
                model with RoBERTa-base: utterance + context + attended).
            hidden_size: Bottleneck width (proposal: 256).
            num_labels: 2 (not sarcastic / sarcastic).
            dropout: Applied between the two linear layers.
        """
        super().__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.input_norm = nn.LayerNorm(input_size)
        self.dense = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self._init_weights()

    def _init_weights(self) -> None:
        """Small-normal init, matching HuggingFace head conventions."""
        for module in (self.dense, self.out_proj):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Map ``[B, input_size]`` to ``[B, num_labels]`` logits."""
        hidden = self.input_norm(fused)
        hidden = self.activation(self.dense(hidden))
        hidden = self.dropout(hidden)
        return self.out_proj(hidden)


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over the label dimension."""
    return torch.softmax(logits.float(), dim=-1)


def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Confidence = probability assigned to the predicted class."""
    return probabilities_from_logits(logits).max(dim=-1).values
