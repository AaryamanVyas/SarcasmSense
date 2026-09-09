"""Fusion layers: the trainable core of the proposed architecture.

Two fusion strategies are implemented so the ablation study can compare them
under otherwise identical conditions.

:class:`CrossAttentionFusion` (proposed model)
    The target utterance forms the **queries**; the conversational context
    tokens and the retrieved commonsense evidence vectors form a single joint
    **key/value memory**. The utterance therefore attends *selectively*: it can
    concentrate on the one context turn that establishes the situation and on
    the two evidence facts that make the situation undesirable, and ignore the
    rest. The attention weights are returned, which gives the explanation layer
    a direct, non-post-hoc account of which context turn and which fact
    mattered.

:class:`ConcatFusion` (ablation baseline)
    Everything is mean-pooled and concatenated. Each source contributes a fixed
    share regardless of relevance; there is no way for the model to *select*.
    This is the control that shows whether cross-attention actually buys
    anything.

Robustness
----------
Both modules must survive an example with no context, no evidence, or neither.
Cross-attention over a fully-masked memory produces NaNs (softmax over ``-inf``
everywhere), so a learned **null memory slot** is always appended and always
unmasked. It gives attention somewhere safe to go and doubles as a measurable
"nothing relevant was available" signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from src.models.encoder import masked_mean

logger = logging.getLogger(__name__)


@dataclass
class FusionOutput:
    """Result of a fusion step.

    Attributes:
        fused: The representation handed to the classification head ``[B, F]``.
        attention: Mean attention distribution over memory slots ``[B, M]``,
            averaged across heads and query positions. ``None`` for concat
            fusion, which has no attention.
        memory_spans: Which memory slots belong to which source, as
            ``{"null": (0, 1), "context": (1, 1+Lc), "knowledge": (...)}``.
            The explanation layer uses this to split the attention vector back
            into "context influence" and "evidence influence".
    """

    fused: torch.Tensor
    attention: torch.Tensor | None = None
    memory_spans: dict[str, tuple[int, int]] | None = None


class CrossAttentionBlock(nn.Module):
    """One pre-norm transformer-decoder-style cross-attention block."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: ``[B, Lq, H]``.
            memory: ``[B, M, H]``.
            memory_padding_mask: ``[B, M]`` boolean, ``True`` marks a slot to
                **ignore** (PyTorch's convention).

        Returns:
            ``(updated_queries [B, Lq, H], attention [B, Lq, M])``.
        """
        normed_q = self.norm_q(queries)
        normed_kv = self.norm_kv(memory)
        attended, weights = self.attention(
            query=normed_q,
            key=normed_kv,
            value=normed_kv,
            key_padding_mask=memory_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        queries = queries + self.dropout(attended)
        queries = queries + self.feed_forward(self.norm_ff(queries))
        return queries, weights


class CrossAttentionFusion(nn.Module):
    """Utterance-as-query cross-attention over context + commonsense memory."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_context: bool = True,
        use_knowledge: bool = True,
    ) -> None:
        """
        Args:
            hidden_size: Encoder hidden size (768 for RoBERTa-base).
            num_heads: Attention heads. Must divide ``hidden_size``.
            num_layers: Number of stacked cross-attention blocks.
            dropout: Dropout inside attention and the feed-forward sublayer.
            use_context: Include context tokens in the memory.
            use_knowledge: Include evidence vectors in the memory.

        Raises:
            ValueError: if neither source is enabled, or heads do not divide
                the hidden size.
        """
        super().__init__()
        if not (use_context or use_knowledge):
            raise ValueError(
                "CrossAttentionFusion needs at least one of use_context/use_knowledge"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must divide hidden_size={hidden_size}"
            )
        self.hidden_size = hidden_size
        self.use_context = use_context
        self.use_knowledge = use_knowledge
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        # Learned slot that is always present and never masked; guarantees the
        # softmax always has at least one valid key even when an example has no
        # context and no retrieved evidence.
        self.null_memory = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    @property
    def output_size(self) -> int:
        """Size of the attended vector this module contributes."""
        return self.hidden_size

    def forward(
        self,
        utterance_sequence: torch.Tensor,
        utterance_mask: torch.Tensor,
        context_sequence: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        knowledge_vectors: torch.Tensor | None = None,
        knowledge_mask: torch.Tensor | None = None,
    ) -> FusionOutput:
        """Fuse the branches.

        Args:
            utterance_sequence: ``[B, Lu, H]`` target-utterance token states.
            utterance_mask: ``[B, Lu]`` 1 for real tokens.
            context_sequence: ``[B, Lc, H]`` context token states, or ``None``.
            context_mask: ``[B, Lc]`` 1 for real tokens, or ``None``.
            knowledge_vectors: ``[B, K, H]`` projected evidence vectors.
            knowledge_mask: ``[B, K]`` 1 for real evidence slots.

        Returns:
            A :class:`FusionOutput` whose ``fused`` field is the attended,
            mean-pooled utterance representation ``[B, H]``.
        """
        batch = utterance_sequence.shape[0]
        device = utterance_sequence.device
        dtype = utterance_sequence.dtype

        memory_parts: list[torch.Tensor] = [
            self.null_memory.to(dtype).expand(batch, 1, self.hidden_size)
        ]
        valid_parts: list[torch.Tensor] = [
            torch.ones(batch, 1, dtype=torch.bool, device=device)
        ]
        spans: dict[str, tuple[int, int]] = {"null": (0, 1)}
        cursor = 1

        if self.use_context and context_sequence is not None and context_mask is not None:
            memory_parts.append(context_sequence)
            valid_parts.append(context_mask.bool())
            spans["context"] = (cursor, cursor + context_sequence.shape[1])
            cursor += context_sequence.shape[1]

        if self.use_knowledge and knowledge_vectors is not None:
            k_mask = (
                knowledge_mask.bool()
                if knowledge_mask is not None
                else torch.ones(
                    batch, knowledge_vectors.shape[1], dtype=torch.bool, device=device
                )
            )
            memory_parts.append(knowledge_vectors.to(dtype))
            valid_parts.append(k_mask)
            spans["knowledge"] = (cursor, cursor + knowledge_vectors.shape[1])
            cursor += knowledge_vectors.shape[1]

        memory = torch.cat(memory_parts, dim=1)
        valid = torch.cat(valid_parts, dim=1)
        # PyTorch expects True == "ignore this key".
        key_padding_mask = ~valid

        queries = utterance_sequence
        attention: torch.Tensor | None = None
        for block in self.blocks:
            queries, attention = block(queries, memory, key_padding_mask)

        fused = self.output_norm(masked_mean(queries, utterance_mask))

        pooled_attention = None
        if attention is not None:
            # Average the [B, Lq, M] map over real query tokens -> [B, M].
            weights = utterance_mask.unsqueeze(-1).to(attention.dtype)
            pooled_attention = (attention * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp(min=1e-9)
        return FusionOutput(fused=fused, attention=pooled_attention, memory_spans=spans)


class ConcatFusion(nn.Module):
    """Ablation baseline: mean-pool every source and concatenate.

    Deliberately parameter-light and *non-selective*. It sees the same inputs as
    :class:`CrossAttentionFusion` but cannot weight them by relevance, which is
    exactly the capability the ablation is meant to isolate.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        use_context: bool = True,
        use_knowledge: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_context = use_context
        self.use_knowledge = use_knowledge
        n_sources = 1 + int(use_knowledge)
        # A single projection back to H keeps the head input dimension identical
        # to the cross-attention variant, so the ablation isolates the fusion
        # mechanism rather than head capacity.
        self.project = nn.Sequential(
            nn.Linear(n_sources * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    @property
    def output_size(self) -> int:
        """Size of the vector this module contributes."""
        return self.hidden_size

    def forward(
        self,
        utterance_sequence: torch.Tensor,
        utterance_mask: torch.Tensor,
        context_sequence: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        knowledge_vectors: torch.Tensor | None = None,
        knowledge_mask: torch.Tensor | None = None,
    ) -> FusionOutput:
        """Mean-pool and concatenate the available sources."""
        parts = [masked_mean(utterance_sequence, utterance_mask)]
        if self.use_knowledge:
            if knowledge_vectors is not None:
                mask = (
                    knowledge_mask
                    if knowledge_mask is not None
                    else torch.ones(
                        knowledge_vectors.shape[:2],
                        device=knowledge_vectors.device,
                    )
                )
                parts.append(masked_mean(knowledge_vectors, mask))
            else:
                parts.append(torch.zeros_like(parts[0]))
        fused = self.output_norm(self.project(torch.cat(parts, dim=-1)))
        return FusionOutput(fused=fused, attention=None, memory_spans=None)


def build_fusion(
    fusion_type: str,
    hidden_size: int,
    num_heads: int = 8,
    num_layers: int = 1,
    dropout: float = 0.1,
    use_context: bool = True,
    use_knowledge: bool = True,
) -> nn.Module | None:
    """Factory used by :class:`~src.models.sarcasm_model.SarcasmModel`.

    Returns ``None`` for ``fusion_type == "none"`` (the utterance-only baseline).
    """
    if fusion_type == "none":
        return None
    if fusion_type == "cross_attention":
        return CrossAttentionFusion(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_context=use_context,
            use_knowledge=use_knowledge,
        )
    if fusion_type == "concat":
        return ConcatFusion(
            hidden_size=hidden_size,
            dropout=dropout,
            use_context=use_context,
            use_knowledge=use_knowledge,
        )
    raise ValueError(f"Unknown fusion type {fusion_type!r}")
