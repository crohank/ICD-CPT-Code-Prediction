"""
Model architectures for ICD-10 code prediction.

- ICDClassifier:            Model B — ClinicalBERT + [CLS] linear head
- LabelAttentionClassifier: Model C — Chunk-based BERT + per-label attention
- TemperatureScaler:        Post-hoc probability calibration
- EnsemblePredictor:        Weighted average of Model A + Model C probabilities

Implementation detail: sections below follow model families (B → C → D → calibration → ensemble).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .config import (
    TRANSFORMER_MODEL, HIDDEN_SIZE, MODEL_C_MAX_CHUNKS,
    MODEL_D_EMBED_DIM, MODEL_D_HIDDEN_DIM, MODEL_D_ATTN_DIM,
)


# --- Model B: vanilla [CLS] pooling + linear head ---

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


# --- Model C: multi-chunk BERT + per-label attention (v2 fixes noted in docstring) ---

class LabelAttentionClassifier(nn.Module):
    """
    PLM-ICD-inspired architecture:
      1. Split document into overlapping 512-token chunks
      2. Encode each chunk with Bio_ClinicalBERT
      3. Concatenate all token embeddings across chunks
      4. Apply per-label attention: one learnable query vector per ICD code
      5. Classify each label via a per-label linear projection

    v2 fixes over v1:
      - chunk_counts now properly used to mask padding chunks
      - Removed dead self.classifier layer
      - Added per-label linear projection (replaces dot-product logit)
      - Added init_label_queries_from_descriptions() for semantic initialization

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
        self.hidden_size = hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Per-label attention query vectors (num_labels, hidden_size)
        self.label_queries = nn.Parameter(torch.empty(num_labels, hidden_size))
        nn.init.xavier_uniform_(self.label_queries)

        self.dropout = nn.Dropout(dropout)

        # Per-label classifier: projects attended repr to a scalar logit
        # More expressive than simple dot-product
        self.label_classifier = nn.Linear(hidden_size, num_labels)

    def init_label_queries_from_descriptions(
        self, descriptions: list, tokenizer=None, device: str = 'cpu',
    ):
        """
        Initialize label_queries with ClinicalBERT embeddings of ICD code descriptions.
        E.g., descriptions = ["Heart failure, unspecified", "Type 2 diabetes mellitus", ...]

        This gives the attention mechanism semantic understanding of each code
        instead of random initialization.

        Args:
            descriptions: list of ICD code description strings (len = num_labels)
            tokenizer: HuggingFace tokenizer (auto-loaded if None)
            device: device to run encoding on
        """
        assert len(descriptions) == self.num_labels, \
            f"Expected {self.num_labels} descriptions, got {len(descriptions)}"

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

        self.bert.eval()
        embeddings = []
        with torch.no_grad():
            for desc in descriptions:
                enc = tokenizer(
                    desc, max_length=64, padding='max_length',
                    truncation=True, return_tensors='pt',
                )
                ids  = enc['input_ids'].to(device)
                mask = enc['attention_mask'].to(device)
                out  = self.bert(ids, attention_mask=mask)
                cls  = out.last_hidden_state[:, 0, :]  # [CLS] embedding
                embeddings.append(cls.squeeze(0).cpu())

        desc_embeddings = torch.stack(embeddings)  # (num_labels, hidden_size)

        # Copy into label_queries parameter
        with torch.no_grad():
            self.label_queries.copy_(desc_embeddings)

        print(f"Initialized label_queries from {len(descriptions)} code descriptions")

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

        # Mask padded chunk positions using per-sample chunk counts (v2 fix).
        # Start with token-level mask from attention_mask
        token_mask = attention_mask.view(batch_size, max_chunks * seq_len).float()

        # Additionally mask out entire padding chunks using chunk_counts
        if chunk_counts is not None:
            # chunk_mask: (batch, max_chunks) — 1 for real chunks, 0 for padding
            chunk_indices = torch.arange(max_chunks, device=input_ids.device)
            chunk_mask = (chunk_indices.unsqueeze(0) < chunk_counts.unsqueeze(1)).float()
            # Expand to token level: (batch, max_chunks, seq_len) → (batch, max_chunks * seq_len)
            chunk_token_mask = chunk_mask.unsqueeze(2).expand(-1, -1, seq_len)
            chunk_token_mask = chunk_token_mask.reshape(batch_size, max_chunks * seq_len)
            # Combine: token is valid only if BOTH its attention_mask=1 AND its chunk is real
            token_mask = token_mask * chunk_token_mask

        # Per-label attention over all real tokens across chunks.
        # label_queries: (num_labels, hidden)
        # Attention scores: (batch, num_labels, total_tokens)
        scores = torch.matmul(hidden, self.label_queries.T)  # (B, T, L)
        scores = scores.permute(0, 2, 1)                     # (B, L, T)

        # Mask padding tokens with -inf before softmax
        mask_expanded = token_mask.unsqueeze(1).expand_as(scores)  # (B, L, T)
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # (B, L, T)
        # Replace NaN from all-masked rows (safety)
        attn_weights = attn_weights.nan_to_num(0.0)

        # Weighted sum: (B, L, T) x (B, T, H) → (B, L, H)
        attended = torch.bmm(attn_weights, hidden)    # (B, L, H)

        # ── Classification (FIX #4: use per-label linear instead of dot-product) ──
        attended = self.dropout(attended)

        # Per-label projection: (B, L, H) → (B, L)
        # Use label_classifier which has shape (num_labels, hidden_size)
        # Apply as a batched operation: each label gets its own linear projection
        logits = (attended * self.label_classifier.weight.unsqueeze(0)).sum(dim=-1)
        logits = logits + self.label_classifier.bias.unsqueeze(0)

        if return_attention:
            return logits, attn_weights
        return logits

    def unfreeze_bert_layers(self, num_layers: int = 2):
        """Unfreeze the last N transformer layers for fine-tuning."""
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


# --- Model D: BiLSTM encoder + LAAT-style label attention (word-level) ---

class BiLSTMLAAT(nn.Module):
    """
    BiLSTM encoder + per-label attention from:
      "A Label Attention Model for ICD Coding from Clinical Text"
      (Vu, Nguyen & Nguyen, IJCAI 2020)

    Architecture:
      1. Word embedding layer
      2. BiLSTM encoder  → hidden states H ∈ R^{n × 2u}
      3. Label attention  → Z = tanh(W·H^T), A = softmax(U·Z)
      4. Per-label FFN    → logits = FFN_j(v_j) for each label j

    Unlike Model C (BERT-based), this processes full documents natively
    (up to 4000 tokens) without chunking.

    Args:
        vocab_size:   word vocabulary size (including <PAD> and <UNK>)
        num_labels:   number of ICD-10 codes to predict
        embed_dim:    word embedding dimension
        hidden_dim:   BiLSTM hidden size per direction (output = 2 * hidden_dim)
        num_layers:   number of stacked BiLSTM layers
        attn_dim:     label attention projection dimension (d_a in paper)
        dropout:      dropout rate
        pretrained_embeddings: optional (vocab_size, embed_dim) tensor for init
    """
    def __init__(
        self,
        vocab_size: int,
        num_labels: int = 50,
        embed_dim: int = MODEL_D_EMBED_DIM,
        hidden_dim: int = MODEL_D_HIDDEN_DIM,
        num_layers: int = 1,
        attn_dim: int = MODEL_D_ATTN_DIM,
        dropout: float = 0.3,
        pretrained_embeddings: torch.Tensor = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # 2. BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = 2 * hidden_dim  # bidirectional

        # 3. Label attention (paper equations 4-6)
        #    W ∈ R^{d_a × 2u}  — projects hidden states to attention space
        #    U ∈ R^{L × d_a}   — per-label attention queries
        self.W_attn = nn.Linear(lstm_out_dim, attn_dim, bias=False)
        self.U_attn = nn.Linear(attn_dim, num_labels, bias=False)

        # 4. Per-label classifier (paper: single-layer FFNN per label)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_labels)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids:      (batch, seq_len) word indices
            attention_mask: (batch, seq_len) 1=real token, 0=padding

        Returns:
            logits: (batch, num_labels)
        """
        batch_size, seq_len = input_ids.shape

        # Compute lengths from attention_mask for pack_padded_sequence
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()  # (batch,)
        else:
            lengths = torch.full((batch_size,), seq_len)

        # Clamp lengths to at least 1 to avoid empty sequences
        lengths = lengths.clamp(min=1)

        # 1. Embed
        embeds = self.embedding(input_ids)  # (B, T, embed_dim)
        embeds = self.dropout(embeds)

        # 2. BiLSTM with packed sequences for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=seq_len,
        )  # H: (B, T, 2*hidden_dim)

        # 3. Label attention (equations 4-6)
        # Z = tanh(H @ W^T)  → (B, T, d_a)
        Z = torch.tanh(self.W_attn(H))

        # A = softmax(Z @ U^T, dim=1)  → (B, T, L) → transpose to (B, L, T)
        A = self.U_attn(Z)  # (B, T, L)
        A = A.permute(0, 2, 1)  # (B, L, T)

        # Mask padding positions before softmax
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).float()  # (B, 1, T)
            A = A.masked_fill(mask_expanded == 0, float('-inf'))

        A = torch.softmax(A, dim=-1)  # (B, L, T)
        A = A.nan_to_num(0.0)  # safety for all-masked rows

        # V = A @ H  → (B, L, 2*hidden_dim)
        V = torch.bmm(A, H)

        # 4. Per-label classification
        V = self.dropout(V)
        logits = (V * self.classifier.weight.unsqueeze(0)).sum(dim=-1)
        logits = logits + self.classifier.bias.unsqueeze(0)

        return logits


# --- Single scalar temperature on logits (optional calibration after training) ---

class TemperatureScaler(nn.Module):
    """
    Learns a single temperature parameter T on validation logits to improve
    probability calibration:  calibrated_prob = sigmoid(logit / T)

    Optimizes T to minimize NLL on the validation set.

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)
        calibrated_probs = scaler.calibrate(test_logits)
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            lr: float = 0.01, max_iter: int = 100):
        """
        Optimize temperature on validation logits/labels.

        Args:
            logits: (n_samples, n_labels) raw model logits (before sigmoid)
            labels: (n_samples, n_labels) binary labels
            lr: learning rate for LBFGS optimizer
            max_iter: max optimization iterations
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / self.temperature
            loss = criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self.temperature.item()

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return calibrated probabilities."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        with torch.no_grad():
            calibrated = torch.sigmoid(logits_t / self.temperature)
        return calibrated.numpy()


# --- Blend sparse + neural probabilities (weight tuned on val) ---

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
            metric_fn: callable(Y_true, Y_pred) -> float (higher is better)
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
