"""
Model architectures for ICD-10 code prediction.

- ICDClassifier:            Model B — ClinicalBERT + [CLS] linear head
- LabelAttentionClassifier: Model C — Chunk-based BERT + per-label attention
- EnsemblePredictor:        Weighted average of Model A + Model C probabilities
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from .config import TRANSFORMER_MODEL, HIDDEN_SIZE, MODEL_C_MAX_CHUNKS


# ═══════════════════════════════════════════════════════════════════════
#  Model B — Standard ClinicalBERT + Linear Head
# ═══════════════════════════════════════════════════════════════════════

class ICDClassifier(nn.Module):
    """
    Bio_ClinicalBERT encoder with a [CLS]-token linear classification head.
    Input is a single 512-token sequence (uses smart truncation upstream).
    """
    def __init__(self, model_name: str = TRANSFORMER_MODEL,
                 num_labels: int = 50, dropout: float = 0.1):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        hidden_size  = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]        # [CLS] token
        cls = self.dropout(cls)
        return self.head(cls)                        # raw logits


# ═══════════════════════════════════════════════════════════════════════
#  Model C — Chunk-Based BERT + Label-Wise Attention
# ═══════════════════════════════════════════════════════════════════════

class LabelAttentionClassifier(nn.Module):
    """
    PLM-ICD-inspired architecture:
      1. Split document into overlapping 512-token chunks
      2. Encode each chunk with Bio_ClinicalBERT
      3. Concatenate all token embeddings across chunks
      4. Apply per-label attention: one learnable query vector per ICD code
      5. Classify each label independently via sigmoid

    The label attention gives built-in explainability — attention weights
    show which tokens are most relevant for each predicted code.

    Args:
        model_name:   HuggingFace model identifier
        num_labels:   number of ICD-10 codes to predict
        max_chunks:   maximum number of 512-token chunks per document
        freeze_bert:  if True, freeze the BERT backbone (train only attention + head)
        dropout:      dropout rate before classification
    """
    def __init__(
        self,
        model_name: str = TRANSFORMER_MODEL,
        num_labels: int = 50,
        max_chunks: int = MODEL_C_MAX_CHUNKS,
        freeze_bert: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.max_chunks = max_chunks

        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Per-label attention query vectors (num_labels, hidden_size)
        self.label_queries = nn.Parameter(torch.empty(num_labels, hidden_size))
        nn.init.xavier_uniform_(self.label_queries)

        # Classification: dot product of attended repr with label query
        # Equivalent to per-label Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

        # Optional projection for richer classification
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, chunk_counts=None,
                return_attention=False):
        """
        Args:
            input_ids:      (batch, max_chunks, seq_len)
            attention_mask: (batch, max_chunks, seq_len)
            chunk_counts:   (batch,) — number of real chunks per sample
            return_attention: if True, also return attention weights per label

        Returns:
            logits: (batch, num_labels)
            attn_weights: (batch, num_labels, total_tokens) — only if return_attention=True
        """
        batch_size, max_chunks, seq_len = input_ids.shape

        # Flatten chunks: (batch * max_chunks, seq_len)
        flat_ids  = input_ids.view(batch_size * max_chunks, seq_len)
        flat_mask = attention_mask.view(batch_size * max_chunks, seq_len)

        # Encode all chunks through BERT
        outputs = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        hidden  = outputs.last_hidden_state  # (batch * max_chunks, seq_len, hidden)

        # Reshape back: (batch, max_chunks * seq_len, hidden)
        hidden = hidden.view(batch_size, max_chunks * seq_len, -1)

        # Build token-level mask: (batch, max_chunks * seq_len)
        token_mask = attention_mask.view(batch_size, max_chunks * seq_len).float()

        # ── Label-wise attention ──────────────────────────────────────
        # label_queries: (num_labels, hidden)
        # Attention scores: (batch, num_labels, total_tokens)
        #   = hidden @ label_queries^T → (batch, total_tokens, num_labels) → transpose
        scores = torch.matmul(hidden, self.label_queries.T)  # (B, T, L)
        scores = scores.permute(0, 2, 1)                     # (B, L, T)

        # Mask padding tokens with -inf before softmax
        mask_expanded = token_mask.unsqueeze(1).expand_as(scores)  # (B, L, T)
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # (B, L, T)
        # Replace NaN from all-masked rows (shouldn't happen, but safety)
        attn_weights = attn_weights.nan_to_num(0.0)

        # Weighted sum: (B, L, T) × (B, T, H) → (B, L, H)
        attended = torch.bmm(attn_weights, hidden)    # (B, L, H)

        # ── Classification ────────────────────────────────────────────
        attended = self.dropout(attended)

        # Dot product with label queries for per-label logit
        # Element-wise multiply + sum: (B, L, H) * (L, H) → (B, L)
        logits = (attended * self.label_queries.unsqueeze(0)).sum(dim=-1)

        if return_attention:
            return logits, attn_weights
        return logits

    def unfreeze_bert_layers(self, num_layers: int = 2):
        """Unfreeze the last N transformer layers + pooler for fine-tuning."""
        # Unfreeze embeddings? No — keep frozen for stability
        total_layers = len(self.bert.encoder.layer)
        for i in range(total_layers - num_layers, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        # Also unfreeze the pooler if it exists
        if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze last {num_layers} BERT layers. "
              f"Trainable params: {trainable:,}")


# ═══════════════════════════════════════════════════════════════════════
#  Ensemble Predictor — Weighted Average of Model A + Model C
# ═══════════════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """
    Combines probability outputs from Model A (TF-IDF) and Model C (BERT).
    P_ensemble = w * P_a + (1 - w) * P_c

    Tune `w` on the validation set via `tune_weight()`.
    """
    def __init__(self, weight: float = 0.5):
        self.weight = weight

    def predict(self, P_a: np.ndarray, P_c: np.ndarray) -> np.ndarray:
        """Weighted average of two probability matrices."""
        return self.weight * P_a + (1 - self.weight) * P_c

    def tune_weight(self, P_a: np.ndarray, P_c: np.ndarray,
                    Y: np.ndarray, metric_fn=None,
                    weights=np.arange(0.0, 1.05, 0.05)):
        """
        Grid search over mixing weights on validation set.

        Args:
            P_a, P_c: probability matrices (n_samples, n_labels)
            Y: true binary label matrix
            metric_fn: callable(Y_true, Y_pred) → float (higher is better)
                       defaults to micro-F1 at threshold 0.5
        Returns:
            best_weight, best_score
        """
        from sklearn.metrics import f1_score

        if metric_fn is None:
            def metric_fn(y, p):
                return f1_score(y, (p >= 0.5).astype(int),
                                average='micro', zero_division=0)

        best_w, best_score = 0.5, 0.0
        for w in weights:
            P_ens = w * P_a + (1 - w) * P_c
            score = metric_fn(Y, P_ens)
            if score > best_score:
                best_score = score
                best_w = w

        self.weight = best_w
        return best_w, best_score
