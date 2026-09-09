"""Tokenisation, encoder, fusion, classification head and the full model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.dataset import (
    SarcasmCollator,
    SarcasmDataset,
    compute_class_weights,
    model_inputs,
)
from src.models.classifier import (
    ClassificationHead,
    confidence_from_logits,
    probabilities_from_logits,
)
from src.models.encoder import masked_mean
from src.models.fusion import ConcatFusion, CrossAttentionFusion, build_fusion
from src.models.sarcasm_model import SarcasmModel
from src.models.tokenization import NO_CONTEXT_PLACEHOLDER
from src.utils.config import ConfigError, ModelConfig


class TestTokenizer:
    """Batches have the right shapes and flag context presence correctly."""

    def test_shapes(self, tokenizer):
        batch = tokenizer.encode(["a b c", "d e"], [["ctx"], []])
        assert batch["utterance_input_ids"].shape == (2, 32)
        assert batch["context_input_ids"].shape == (2, 32)
        assert batch["utterance_attention_mask"].shape == (2, 32)

    def test_has_context_flag(self, tokenizer):
        batch = tokenizer.encode(["a", "b", "c"], [["x"], [], ["y", "z"]])
        assert batch["has_context"].tolist() == [True, False, True]

    def test_missing_context_uses_the_placeholder(self, tokenizer):
        flat, has = tokenizer.flatten_context([])
        assert has is False and flat == NO_CONTEXT_PLACEHOLDER

    def test_context_turns_are_numbered(self, tokenizer):
        flat, has = tokenizer.flatten_context(["first", "second"])
        assert has is True
        assert "turn 1: first" in flat and "turn 2: second" in flat

    def test_turn_spans_cover_each_turn(self, tokenizer):
        flat, spans = tokenizer.context_turn_spans(["alpha", "beta", "gamma"])
        assert len(spans) == 3
        for turn, (start, end) in zip(["alpha", "beta", "gamma"], spans):
            assert flat[start:end] == turn

    def test_long_text_is_truncated_not_rejected(self, tokenizer):
        batch = tokenizer.encode(["word " * 500], [[]])
        assert batch["utterance_input_ids"].shape == (1, 32)

    def test_context_truncation_keeps_the_most_recent_turn(self, tokenizer):
        long_turn = "filler " * 200
        batch = tokenizer.encode(["target"], [[long_turn, "the crucial final turn"]])
        decoded = tokenizer.tokenizer.decode(
            batch["context_input_ids"][0], skip_special_tokens=True
        )
        assert "final turn" in decoded

    def test_empty_string_still_produces_a_batch(self, tokenizer):
        assert tokenizer.encode([""], [[]])["utterance_input_ids"].shape == (1, 32)

    def test_offsets_are_returned_on_request(self, tokenizer):
        batch = tokenizer.encode(["hello there"], [[]], return_offsets=True)
        assert "utterance_offsets" in batch and "utterance_tokens" in batch


class TestPooling:
    """Masked mean pooling is numerically safe."""

    def test_ignores_padding(self):
        sequence = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        torch.testing.assert_close(masked_mean(sequence, mask), torch.tensor([[2.0, 2.0]]))

    def test_all_padding_does_not_produce_nan(self):
        result = masked_mean(torch.randn(1, 4, 3), torch.zeros(1, 4))
        assert torch.isfinite(result).all()


class TestFusion:
    """Cross-attention is a real, trainable, mask-aware module."""

    @pytest.fixture()
    def fusion(self):
        # eval() matters: in training mode nn.MultiheadAttention applies dropout
        # to the attention weights it returns, so they no longer sum to 1. The
        # attention used for explanation is always read in eval mode.
        module = CrossAttentionFusion(hidden_size=32, num_heads=4, num_layers=1)
        module.eval()
        return module

    def test_output_shape_and_attention(self, fusion):
        output = fusion(
            utterance_sequence=torch.randn(2, 6, 32),
            utterance_mask=torch.ones(2, 6),
            context_sequence=torch.randn(2, 5, 32),
            context_mask=torch.ones(2, 5),
            knowledge_vectors=torch.randn(2, 3, 32),
            knowledge_mask=torch.ones(2, 3, dtype=torch.bool),
        )
        assert output.fused.shape == (2, 32)
        # memory = 1 null + 5 context + 3 evidence
        assert output.attention.shape == (2, 9)
        assert output.memory_spans == {"null": (0, 1), "context": (1, 6), "knowledge": (6, 9)}

    def test_attention_is_a_distribution(self, fusion):
        output = fusion(
            torch.randn(2, 4, 32), torch.ones(2, 4),
            torch.randn(2, 5, 32), torch.ones(2, 5),
            torch.randn(2, 3, 32), torch.ones(2, 3, dtype=torch.bool),
        )
        torch.testing.assert_close(
            output.attention.sum(dim=-1), torch.ones(2), atol=1e-4, rtol=1e-4
        )

    def test_fully_masked_memory_does_not_produce_nan(self, fusion):
        """The learned null slot is what makes this safe."""
        output = fusion(
            torch.randn(2, 4, 32), torch.ones(2, 4),
            torch.randn(2, 5, 32), torch.zeros(2, 5),
            torch.randn(2, 3, 32), torch.zeros(2, 3, dtype=torch.bool),
        )
        assert torch.isfinite(output.fused).all()
        assert torch.isfinite(output.attention).all()
        # All attention must land on the null slot.
        torch.testing.assert_close(
            output.attention[:, 0], torch.ones(2), atol=1e-4, rtol=1e-4
        )

    def test_masked_evidence_is_ignored(self, fusion):
        utterance = torch.randn(1, 4, 32)
        context = torch.randn(1, 5, 32)
        evidence = torch.randn(1, 3, 32)
        mask = torch.tensor([[True, False, False]])
        fusion.eval()
        with torch.no_grad():
            a = fusion(utterance, torch.ones(1, 4), context, torch.ones(1, 5), evidence, mask)
            evidence2 = evidence.clone()
            evidence2[:, 1:] = torch.randn(1, 2, 32) * 50  # perturb masked slots only
            b = fusion(utterance, torch.ones(1, 4), context, torch.ones(1, 5), evidence2, mask)
        torch.testing.assert_close(a.fused, b.fused, atol=1e-5, rtol=1e-5)

    def test_context_only(self):
        fusion = CrossAttentionFusion(32, 4, use_knowledge=False)
        output = fusion(torch.randn(1, 4, 32), torch.ones(1, 4),
                        torch.randn(1, 5, 32), torch.ones(1, 5))
        assert output.fused.shape == (1, 32)
        assert "knowledge" not in output.memory_spans

    def test_knowledge_only(self):
        fusion = CrossAttentionFusion(32, 4, use_context=False)
        output = fusion(
            torch.randn(1, 4, 32), torch.ones(1, 4),
            knowledge_vectors=torch.randn(1, 3, 32),
            knowledge_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        assert "context" not in output.memory_spans

    def test_gradients_flow(self, fusion):
        fusion.train()
        output = fusion(
            torch.randn(2, 4, 32, requires_grad=True), torch.ones(2, 4),
            torch.randn(2, 5, 32), torch.ones(2, 5),
            torch.randn(2, 3, 32), torch.ones(2, 3, dtype=torch.bool),
        )
        output.fused.sum().backward()
        assert fusion.null_memory.grad is not None
        assert torch.isfinite(fusion.null_memory.grad).all()

    def test_heads_must_divide_hidden_size(self):
        with pytest.raises(ValueError, match="must divide"):
            CrossAttentionFusion(hidden_size=32, num_heads=5)

    def test_at_least_one_source_is_required(self):
        with pytest.raises(ValueError):
            CrossAttentionFusion(32, 4, use_context=False, use_knowledge=False)

    def test_concat_fusion_produces_no_attention(self):
        fusion = ConcatFusion(32)
        output = fusion(
            torch.randn(2, 4, 32), torch.ones(2, 4),
            torch.randn(2, 5, 32), torch.ones(2, 5),
            torch.randn(2, 3, 32), torch.ones(2, 3, dtype=torch.bool),
        )
        assert output.fused.shape == (2, 32)
        assert output.attention is None

    def test_factory(self):
        assert build_fusion("none", 32) is None
        assert isinstance(build_fusion("cross_attention", 32, num_heads=4), CrossAttentionFusion)
        assert isinstance(build_fusion("concat", 32), ConcatFusion)
        with pytest.raises(ValueError):
            build_fusion("mystery", 32)


class TestClassificationHead:
    """Linear -> ReLU -> Dropout -> Linear -> 2 logits."""

    def test_shape(self):
        head = ClassificationHead(input_size=48, hidden_size=16)
        assert head(torch.randn(4, 48)).shape == (4, 2)

    def test_probabilities_sum_to_one(self):
        probabilities = probabilities_from_logits(torch.randn(5, 2))
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(5))

    def test_softmax_is_numerically_stable(self):
        probabilities = probabilities_from_logits(torch.tensor([[1000.0, -1000.0]]))
        assert torch.isfinite(probabilities).all()
        assert probabilities[0, 0] == pytest.approx(1.0)

    def test_confidence_is_the_max_probability(self):
        logits = torch.tensor([[2.0, 0.0]])
        assert confidence_from_logits(logits).item() == pytest.approx(
            float(probabilities_from_logits(logits).max())
        )


class TestSarcasmModel:
    """The assembled model: shapes, degradation, persistence."""

    def test_forward_shapes(self, model, tokenizer, knowledge_index):
        batch = tokenizer.encode(["great, another delay"], [["the train is late"]])
        with torch.no_grad():
            output = model(
                **batch,
                knowledge_vectors=torch.randn(1, 3, knowledge_index.dimension),
                knowledge_mask=torch.ones(1, 3, dtype=torch.bool),
            )
        assert output.logits.shape == (1, 2)
        assert output.probabilities.shape == (1, 2)
        assert output.utterance_vector.shape == (1, model.hidden_size)
        assert output.context_vector.shape == (1, model.hidden_size)

    def test_head_input_is_three_times_hidden_for_the_full_model(self, model):
        assert model.classifier_input_size == 3 * model.hidden_size

    def test_loss_is_produced_with_labels(self, model, tokenizer, knowledge_index):
        batch = tokenizer.encode(["a b", "c d"], [["x"], []])
        output = model(
            **batch,
            knowledge_vectors=torch.zeros(2, 3, knowledge_index.dimension),
            knowledge_mask=torch.zeros(2, 3, dtype=torch.bool),
            labels=torch.tensor([1, 0]),
        )
        assert output.loss.ndim == 0 and torch.isfinite(output.loss)

    def test_no_context_no_evidence_still_works(self, model, tokenizer):
        batch = tokenizer.encode(["standalone headline"], [[]])
        with torch.no_grad():
            output = model(**batch, knowledge_vectors=None, knowledge_mask=None)
        assert torch.isfinite(output.logits).all()

    @pytest.mark.parametrize("n_turns", [0, 1, 2, 3])
    def test_every_context_length(self, model, tokenizer, n_turns, knowledge_index):
        context = [f"turn {i}" for i in range(n_turns)]
        batch = tokenizer.encode(["target utterance"], [context])
        with torch.no_grad():
            output = model(
                **batch,
                knowledge_vectors=torch.randn(1, 3, knowledge_index.dimension),
                knowledge_mask=torch.ones(1, 3, dtype=torch.bool),
            )
        assert torch.isfinite(output.logits).all()

    def test_empty_context_uses_the_learned_vector(self, model, tokenizer):
        batch = tokenizer.encode(["x"], [[]])
        with torch.no_grad():
            output = model(**batch)
        torch.testing.assert_close(
            output.context_vector[0], model.empty_context, atol=1e-5, rtol=1e-5
        )

    def test_baseline1_has_a_smaller_head(self, dev_encoder_dir):
        config = ModelConfig(
            encoder_name=dev_encoder_dir, use_context=False,
            use_knowledge=False, fusion="none", max_utterance_length=16,
        )
        baseline = SarcasmModel(config, local_files_only=True)
        assert baseline.classifier_input_size == baseline.hidden_size
        assert baseline.fusion is None

    def test_save_and_load_round_trip(self, model, tmp_path, tokenizer, knowledge_index):
        model.save_pretrained(tmp_path)
        restored = SarcasmModel.from_pretrained(tmp_path, local_files_only=True)
        batch = tokenizer.encode(["consistency check"], [["context turn"]])
        evidence = torch.randn(1, 3, knowledge_index.dimension)
        mask = torch.ones(1, 3, dtype=torch.bool)
        with torch.no_grad():
            a = model(**batch, knowledge_vectors=evidence, knowledge_mask=mask).logits
            b = restored(**batch, knowledge_vectors=evidence, knowledge_mask=mask).logits
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_loading_an_incomplete_checkpoint_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SarcasmModel.from_pretrained(tmp_path)

    def test_describe_exposes_the_architecture(self, model):
        described = model.describe()
        assert described["fusion"] == "cross_attention"
        assert described["uses_context"] is True
        assert described["n_parameters"] > 0

    def test_class_weights_change_the_loss(self, dev_encoder_dir, tokenizer):
        config = ModelConfig(
            encoder_name=dev_encoder_dir, max_utterance_length=32,
            max_context_length=32, num_attention_heads=4,
        )
        weighted = SarcasmModel(
            config, knowledge_dim=8, class_weights=torch.tensor([0.1, 10.0]),
            local_files_only=True,
        )
        weighted.eval()
        batch = tokenizer.encode(["a", "b"], [[], []])
        with torch.no_grad():
            positive = weighted(
                **batch, knowledge_vectors=torch.zeros(2, 5, 8),
                knowledge_mask=torch.zeros(2, 5, dtype=torch.bool),
                labels=torch.tensor([1, 1]),
            ).loss
            negative = weighted(
                **batch, knowledge_vectors=torch.zeros(2, 5, 8),
                knowledge_mask=torch.zeros(2, 5, dtype=torch.bool),
                labels=torch.tensor([0, 0]),
            ).loss
        assert not torch.isclose(positive, negative)


class TestCollator:
    """Batching, evidence lookup and class weighting."""

    def test_batch_keys(self, tokenizer, knowledge_index, fixture_records, retriever):
        from src.retrieval.cache import RetrievalCache

        records = fixture_records[:4]
        collator = SarcasmCollator(
            tokenizer=tokenizer,
            knowledge_index=knowledge_index,
            retrieval_cache=RetrievalCache.build(records, retriever),
            knowledge_slots=3,
            use_context=True,
            use_knowledge=True,
        )
        batch = collator(records)
        assert batch["knowledge_vectors"].shape == (4, 3, knowledge_index.dimension)
        assert batch["knowledge_mask"].shape == (4, 3)
        assert batch["labels"].shape == (4,)
        assert len(batch["record_ids"]) == 4
        assert set(model_inputs(batch)) <= {
            "utterance_input_ids", "utterance_attention_mask", "context_input_ids",
            "context_attention_mask", "has_context", "knowledge_vectors",
            "knowledge_mask", "labels",
        }

    def test_context_is_omitted_when_disabled(self, tokenizer, fixture_records):
        collator = SarcasmCollator(tokenizer, use_context=False, use_knowledge=False)
        batch = collator(fixture_records[:3])
        assert "context_input_ids" not in batch
        assert batch["has_context"].tolist() == [False, False, False]

    def test_context_turn_limit_is_applied(self, tokenizer, fixture_records):
        collator = SarcasmCollator(tokenizer, use_knowledge=False, max_context_turns=1)
        batch = collator(fixture_records[:4])
        assert batch["context_input_ids"].shape[0] == 4

    def test_dataset_indexing(self, fixture_records):
        dataset = SarcasmDataset(fixture_records)
        assert len(dataset) == len(fixture_records)
        assert dataset[0] is fixture_records[0]
        assert dataset.labels.dtype == np.int64

    def test_class_weights_are_inverse_frequency(self):
        weights = compute_class_weights(np.array([0] * 90 + [1] * 10))
        assert weights[1] > weights[0]
        assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)

    def test_class_weights_handle_a_missing_class(self):
        weights = compute_class_weights(np.array([0, 0, 0]))
        assert torch.isfinite(weights).all()


class TestModelConfigValidation:
    """Impossible architectures are rejected at config time, not at runtime."""

    def test_unknown_fusion(self):
        with pytest.raises(ConfigError):
            ModelConfig(fusion="quantum")

    def test_cross_attention_needs_a_source(self):
        with pytest.raises(ConfigError, match="requires at least one"):
            ModelConfig(fusion="cross_attention", use_context=False, use_knowledge=False)

    def test_bad_pooling(self):
        with pytest.raises(ConfigError):
            ModelConfig(pooling="max")
