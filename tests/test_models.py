"""Tests for src/models.py — model architectures and ensemble."""
import sys
sys.path.insert(0, '..')

import numpy as np
import torch
import pytest


class TestEnsemblePredictor:
    def test_weighted_average(self):
        from src.models import EnsemblePredictor
        ens = EnsemblePredictor(weight=0.6)
        P_a = np.array([[0.8, 0.2], [0.3, 0.7]])
        P_c = np.array([[0.4, 0.6], [0.5, 0.5]])
        result = ens.predict(P_a, P_c)
        expected = 0.6 * P_a + 0.4 * P_c
        np.testing.assert_allclose(result, expected)

    def test_weight_zero_gives_model_c(self):
        from src.models import EnsemblePredictor
        ens = EnsemblePredictor(weight=0.0)
        P_a = np.ones((5, 3))
        P_c = np.zeros((5, 3))
        result = ens.predict(P_a, P_c)
        np.testing.assert_allclose(result, P_c)

    def test_weight_one_gives_model_a(self):
        from src.models import EnsemblePredictor
        ens = EnsemblePredictor(weight=1.0)
        P_a = np.ones((5, 3))
        P_c = np.zeros((5, 3))
        result = ens.predict(P_a, P_c)
        np.testing.assert_allclose(result, P_a)


class TestLabelAttentionClassifierShape:
    """Test forward pass shapes (uses tiny random weights, no pretrained model)."""

    @pytest.fixture
    def model(self):
        from src.models import LabelAttentionClassifier
        # Use a tiny model for testing
        model = LabelAttentionClassifier(
            model_name='prajjwal1/bert-tiny',  # 4.4M params, fast to load
            num_labels=10,
            max_chunks=2,
            freeze_bert=True,
        )
        model.eval()
        return model

    def test_output_shape(self, model):
        batch_size = 2
        max_chunks = 2
        seq_len = 32  # short for testing
        ids  = torch.randint(0, 100, (batch_size, max_chunks, seq_len))
        mask = torch.ones(batch_size, max_chunks, seq_len, dtype=torch.long)
        logits = model(ids, mask)
        assert logits.shape == (batch_size, 10)

    def test_attention_output(self, model):
        batch_size = 1
        max_chunks = 2
        seq_len = 32
        ids  = torch.randint(0, 100, (batch_size, max_chunks, seq_len))
        mask = torch.ones(batch_size, max_chunks, seq_len, dtype=torch.long)
        logits, attn = model(ids, mask, return_attention=True)
        assert logits.shape == (batch_size, 10)
        assert attn.shape == (batch_size, 10, max_chunks * seq_len)

    def test_attention_sums_to_one(self, model):
        batch_size = 1
        max_chunks = 2
        seq_len = 32
        ids  = torch.randint(0, 100, (batch_size, max_chunks, seq_len))
        mask = torch.ones(batch_size, max_chunks, seq_len, dtype=torch.long)
        _, attn = model(ids, mask, return_attention=True)
        # Each label's attention should sum to ~1
        sums = attn.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)
