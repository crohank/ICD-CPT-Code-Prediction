# Interview Context — ICD-10 Code Prediction Project

> Self-contained brief for an interview with Themis AI (Stewart Jamieson, Head of Technology). Covers the ICD-10 prediction project end-to-end: data, models, training, deployment, tests, and team ownership. Written for an interview reader who does not have access to the codebase.

---

## 0. TL;DR

- **Task**: Multi-label classification of ICD-10 diagnosis codes from MIMIC-IV discharge summaries.
- **Label space**: Top-50 most frequent ICD-10 codes (filtered down from 7,940 codes that appear ≥10 times in the cohort of 31,794 unique codes).
- **Models**: (A) TF-IDF + SGD baseline, (B) ClinicalBERT with smart truncation, (C) Chunk-based ClinicalBERT + per-label attention (PLM-ICD-inspired), (D) BiLSTM-LAAT (Vu et al. IJCAI 2020), plus 5 ensemble variants.
- **Best single model**: Model D (BiLSTM-LAAT) — Micro-F1 = **0.7176**.
- **Best ensemble**: Ensemble v4 = `0.15 × Model A + 0.85 × Model D`, threshold 0.35 → Micro-F1 = **0.7198**, Micro-AUROC = **0.9548**.
- **Stack**: PyTorch + HuggingFace; FastAPI + Streamlit + Docker; trained on GCP VM with NVIDIA L4 (23 GB VRAM).

⚠️ **Important discrepancy to be aware of**: `report.md` in the repo only documents Models A/B/C and Ensemble v1 (A+C). It was written before Model D and v4/v5 ensembles landed. The *current* state-of-the-codebase numbers (from `data/models/ensemble/final_comparison_all_models.csv`) supersede the report. Both are documented in §4 below.

---

## 1. Project Overview

### End-to-end flow
```
Raw MIMIC-IV CSVs
  → cohort construction (notebook 01)
  → text cleaning + label binarization (notebook 02)
  → 4 model families trained independently (notebooks 03/04/06/08)
  → ensemble weight + threshold tuning on val (notebook 07)
  → final test metrics, head/torso/tail analysis, plots (notebook 07)
  → artifacts (mlb.pkl, tfidf_vectorizer.pkl, *.pt) loaded by FastAPI ModelService
  → Streamlit calls POST /predict for an interactive UI
```

### Dataset (from `report.md` and `notebooks_local/01,02`)

- **Source**: MIMIC-IV (version not pinned in code, but the discharge / diagnoses_icd / procedures_icd CSVs are standard MIMIC-IV).
- **Raw**: 331,793 discharge notes, 6.3M diagnosis records, 860K procedure records.
- **Filtering**:
  - ICD-10 codes only → 3,824,904 unique (admission, code) pairs across **31,794 unique codes**.
  - Deduplicated to one discharge note per admission (kept latest).
  - Joined notes with ICD-10 codes → **122,304 records**.
  - Label vocabulary = codes appearing ≥10 times → **7,940 codes**.
  - **Top-50 most frequent** codes selected for training (`TOP_K_LABELS = 50` in `src/config.py`).
- **Split**: patient-level 70/15/15 to prevent leakage.
  - `cohort_train.parquet`: 85,081
  - `cohort_val.parquet`: 18,371
  - `cohort_test.parquet`: 18,852

### Problem framing
- **Multi-label** classification: a discharge summary maps to N codes (not exclusive).
- **Loss**: `BCEWithLogitsLoss` (Models B/C v1) and **focal loss** (Model C v2, Model D), with optional per-label `pos_weight`. Code in `src/train.py:sigmoid_focal_loss`.
- **Threshold**: tuned globally on val set via `tune_global_threshold` (`src/evaluate.py`) sweeping `np.arange(0.05, 0.65, 0.025)` to maximize micro-F1. Per-label thresholds also implemented in `tune_per_label_threshold` but the deployed config uses global thresholds.

---

## 2. Model Architectures (full detail)

### A. Model A — TF-IDF + One-vs-Rest SGD baseline
- `OneVsRestClassifier(SGDClassifier(loss='log_loss', penalty='l2', class_weight='balanced'))`
- TF-IDF config from `src/config.py`:
  ```python
  TFIDF_MAX_FEATURES = 50_000
  TFIDF_NGRAM_RANGE  = (1, 2)        # unigrams + bigrams
  TFIDF_SUBLINEAR_TF = True
  ```
- The earlier `saga` solver on 7,940 labels was >100h; switching to SGD on Top-50 made training ~2 min on CPU.
- Optimal global threshold tuned on val: **t = 0.525**.

### B. Model B — Bio_ClinicalBERT with smart truncation
- Checkpoint: `emilyalsentzer/Bio_ClinicalBERT` (~108M params).
- Single 512-token input via section-aware "smart truncation" (see §7).
- Architecture (`src/models.py:25-42`):
  ```python
  class ICDClassifier(nn.Module):
      def __init__(self, model_name=TRANSFORMER_MODEL, num_labels=50, dropout=0.1):
          super().__init__()
          self.bert    = AutoModel.from_pretrained(model_name)
          hidden_size  = self.bert.config.hidden_size   # 768
          self.dropout = nn.Dropout(dropout)
          self.head    = nn.Linear(hidden_size, num_labels)

      def forward(self, input_ids, attention_mask):
          out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
          cls = out.last_hidden_state[:, 0, :]   # [CLS]
          cls = self.dropout(cls)
          return self.head(cls)                  # raw logits
  ```
- Training: 3 epochs, batch 32, AdamW lr=2e-5, 10% linear warmup, fp16, `BCEWithLogitsLoss`. ~45 min on L4. Optimal t = 0.275.

### C. Model C — Chunk-Based ClinicalBERT + Label-Wise Attention (PLM-ICD-style)

**This is the most important model to discuss — it's the one Rohan personally architected, plus the v2 fixes.**

#### Chunking strategy (`src/data.py:ChunkedICDDataset`)
- Constants in `src/config.py`:
  ```python
  MAX_SEQ_LEN          = 512
  MODEL_C_MAX_CHUNKS   = 6      # up to 6 chunks per document
  MODEL_C_CHUNK_STRIDE = 256    # 50% overlap
  ```
- Tokenize the *full* document (no truncation), then create sliding 510-token windows (`content_len = max_seq_len - 2` to reserve room for [CLS]/[SEP]) with stride 256. Each chunk is wrapped as `[CLS] tokens [SEP]` and padded to 512. Padding chunks fill up to `max_chunks=6`. A `chunk_count` integer records how many chunks are real.
- Effective coverage: 6 × 510 with 50% overlap = ~1,800 unique tokens per document.

#### LabelAttentionClassifier (verbatim from `src/models.py:49-229`)

```python
class LabelAttentionClassifier(nn.Module):
    """
    PLM-ICD-inspired architecture:
      1. Split document into overlapping 512-token chunks
      2. Encode each chunk with Bio_ClinicalBERT
      3. Concatenate all token embeddings across chunks
      4. Apply per-label attention: one learnable query vector per ICD code
      5. Classify each label via a per-label linear projection
    """
    def __init__(self, model_name=TRANSFORMER_MODEL, num_labels=50,
                 max_chunks=MODEL_C_MAX_CHUNKS, freeze_bert=True, dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.max_chunks = max_chunks
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size   # 768
        self.hidden_size = hidden_size
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Per-label attention query vectors: (num_labels, hidden_size) = (50, 768)
        self.label_queries = nn.Parameter(torch.empty(num_labels, hidden_size))
        nn.init.xavier_uniform_(self.label_queries)

        self.dropout = nn.Dropout(dropout)

        # Per-label classifier: (num_labels, hidden_size) → scalar logit per label
        self.label_classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, chunk_counts=None,
                return_attention=False):
        # input_ids: (B, max_chunks, S)
        batch_size, max_chunks, seq_len = input_ids.shape

        # Flatten chunks to feed BERT in one batched call
        flat_ids  = input_ids.view(batch_size * max_chunks, seq_len)
        flat_mask = attention_mask.view(batch_size * max_chunks, seq_len)

        outputs = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        hidden  = outputs.last_hidden_state          # (B*C, S, H)
        hidden  = hidden.view(batch_size, max_chunks * seq_len, -1)  # (B, C*S, H)

        # Token mask: combines attention_mask AND chunk_counts (so padding chunks
        # are zeroed even though their attention_mask might be 0/1 mixed).
        token_mask = attention_mask.view(batch_size, max_chunks * seq_len).float()
        if chunk_counts is not None:
            chunk_indices = torch.arange(max_chunks, device=input_ids.device)
            chunk_mask = (chunk_indices.unsqueeze(0) < chunk_counts.unsqueeze(1)).float()
            chunk_token_mask = chunk_mask.unsqueeze(2).expand(-1, -1, seq_len)
            chunk_token_mask = chunk_token_mask.reshape(batch_size, max_chunks * seq_len)
            token_mask = token_mask * chunk_token_mask

        # Label-wise attention
        # hidden:        (B, T, H)  where T = max_chunks * seq_len = 6*512 = 3072
        # label_queries: (L, H)
        scores = torch.matmul(hidden, self.label_queries.T)  # (B, T, L)
        scores = scores.permute(0, 2, 1)                     # (B, L, T)

        # Mask padding tokens with -inf before softmax
        mask_expanded = token_mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)         # (B, L, T)
        attn_weights = attn_weights.nan_to_num(0.0)          # safety for all-masked rows

        # Weighted sum: (B, L, T) x (B, T, H) → (B, L, H)
        attended = torch.bmm(attn_weights, hidden)
        attended = self.dropout(attended)

        # Per-label linear: each row j of label_classifier.weight projects
        # the j-th attended representation to a scalar.
        logits = (attended * self.label_classifier.weight.unsqueeze(0)).sum(dim=-1)
        logits = logits + self.label_classifier.bias.unsqueeze(0)

        if return_attention:
            return logits, attn_weights
        return logits
```

**Key tensor shapes** (for whiteboarding the interview):
- `H = 768` (Bio_ClinicalBERT hidden size)
- `L = 50` (Top-50 labels)
- `S = 512` (tokens per chunk), `C = 6` (max chunks), so `T = C*S = 3072` tokens per doc.
- `label_queries`: `(L, H) = (50, 768)` — one learnable query per code.
- `attn_weights`: `(B, L, T) = (B, 50, 3072)` — for each (sample, label), a softmax over all 3072 tokens.
- `attended`: `(B, L, H) = (B, 50, 768)` — per-label pooled representation.
- `logits`: `(B, L) = (B, 50)`.

**Interpretability path**: `extract_attention_for_text` in `src/explain.py` calls `forward(return_attention=True)`, converts token IDs back to subword strings via `tokenizer.convert_ids_to_tokens`, filters out `[PAD]/[CLS]/[SEP]`, and returns a `{label_index: [(token, weight), ...]}` dict sorted by weight desc. This is what the API returns as `evidence` per code.

**v2 fixes vs v1** (called out in code comments):
1. Properly use `chunk_counts` to mask padding chunks (v1 leaked padding through token_mask).
2. Removed dead `self.classifier` layer.
3. Replaced simple dot-product logit with per-label `nn.Linear` projection (more expressive).
4. `init_label_queries_from_descriptions()` — semantic initialization: encode each ICD code's CMS description with ClinicalBERT and copy `[CLS]` embeddings into `label_queries` (instead of Xavier random).

**Two-phase training**:
- Phase 1: BERT frozen, lr=1e-3, 5 epochs (only attention + head train).
- Phase 2: `unfreeze_bert_layers(num_layers=2)` unfreezes the last 2 transformer layers + pooler, drops lr to 2e-5, trains 3–7 more epochs.

**Loss in v1**: `BCEWithLogitsLoss` with `pos_weight = clamp((N - n_j) / n_j, 1, 50)`.
**Loss in v2**: focal loss (γ=2.0, α=0.25) — `pos_weight` was clamped down from 50 to 10, but focal loss is now the preferred path. v2 also adds temperature scaling (`TemperatureScaler` in `src/models.py:367`) on validation logits for calibration.

### D. Model D — BiLSTM-LAAT (Vu et al., IJCAI 2020)

Word-level model, processes up to 4,000 tokens natively (no chunking). Verbatim from `src/models.py:236-360`:

```python
class BiLSTMLAAT(nn.Module):
    """
    BiLSTM encoder + per-label attention from:
      "A Label Attention Model for ICD Coding from Clinical Text"
      (Vu, Nguyen & Nguyen, IJCAI 2020)
    """
    def __init__(self, vocab_size, num_labels=50,
                 embed_dim=200, hidden_dim=256, num_layers=1,
                 attn_dim=256, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        # 1. Embedding (learned from scratch — pretrained_embeddings hook exists
        #    but is not used in the trained checkpoint).
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # 2. BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = 2 * hidden_dim   # 512

        # 3. Label attention (paper eqs 4-6)
        #    Z = tanh(H W^T) ; A = softmax(Z U^T, dim=time)
        self.W_attn = nn.Linear(lstm_out_dim, attn_dim, bias=False)  # (512 → 256)
        self.U_attn = nn.Linear(attn_dim, num_labels, bias=False)    # (256 → 50)

        # 4. Per-label classifier (single linear, applied per-label like Model C)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_labels)        # (512 → 50)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
        else:
            lengths = torch.full((batch_size,), seq_len)
        lengths = lengths.clamp(min=1)

        embeds = self.embedding(input_ids)
        embeds = self.dropout(embeds)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=seq_len,
        )                                       # (B, T, 512)

        # Label attention
        Z = torch.tanh(self.W_attn(H))          # (B, T, 256)
        A = self.U_attn(Z)                      # (B, T, 50)
        A = A.permute(0, 2, 1)                  # (B, 50, T)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).float()
            A = A.masked_fill(mask_expanded == 0, float('-inf'))
        A = torch.softmax(A, dim=-1)            # (B, 50, T)
        A = A.nan_to_num(0.0)

        V = torch.bmm(A, H)                     # (B, 50, 512)
        V = self.dropout(V)

        logits = (V * self.classifier.weight.unsqueeze(0)).sum(dim=-1)
        logits = logits + self.classifier.bias.unsqueeze(0)
        return logits
```

**Hyperparams** (`src/config.py:60-74`):
```python
MODEL_D_EMBED_DIM    = 200
MODEL_D_HIDDEN_DIM   = 256        # per direction → 512 BiLSTM output
MODEL_D_NUM_LAYERS   = 1
MODEL_D_ATTN_DIM     = 256        # d_a in paper
MODEL_D_DROPOUT      = 0.3
MODEL_D_LR           = 1e-3       # AdamW
MODEL_D_EPOCHS       = 10
MODEL_D_BATCH_SIZE   = 32
MODEL_D_GRAD_ACCUM   = 1
MODEL_D_MAX_TOKENS   = 4000       # whitespace-tokenized
MODEL_D_VOCAB_SIZE   = 50_000     # most-frequent words, 0=<PAD>, 1=<UNK>
MODEL_D_FOCAL_GAMMA  = 2.0
MODEL_D_FOCAL_ALPHA  = 0.25
```

**Tokenization**: word-level whitespace split (NOT BERT subwords). `build_word_vocab()` in `src/data.py` builds a 50K-word vocab. `BiLSTMDataset` truncates to 4,000 words and pads with `<PAD>=0`. Embeddings are **learned from scratch** — the trained checkpoint does not use Word2Vec/GloVe (the `pretrained_embeddings` arg exists but isn't wired up in the training notebook).

**Loss**: focal loss (γ=2, α=0.25). **Optimal threshold**: 0.3.

### E. Ensembles

`EnsemblePredictor` (in `src/models.py:426`) implements simple weighted-average of probability matrices. `tune_weight()` grid-searches `weights = np.arange(0.0, 1.05, 0.05)` on val set, picking the weight that maximizes micro-F1 at threshold 0.5; the threshold is re-tuned afterwards.

Ensembles actually evaluated (from `data/models/ensemble/ensemble_config.json`):

| Ensemble | Models | Weights | Threshold |
|---|---|---|---|
| v1 | A + C v1 | 0.65 A + 0.35 C v1 | 0.625 |
| v2 | A + C v2 | 0.35 A + 0.65 C v2 | 0.425 |
| v3 | A + C v1 + C v2 | 0.30 / 0.00 / 0.70 | 0.400 |
| **v4 (best)** | **A + D** | **0.15 A + 0.85 D** | **0.350** |
| v5 | A + C v2 + D | 0.10 / 0.30 / 0.60 | 0.350 |

The ensemble logic is just probability averaging — no stacking, no learned meta-classifier. Multi-model ensembles (v3, v5) generalize the 2-model formula to `Σ w_i · P_i` with weights summing to 1.

### F. TemperatureScaler (post-hoc calibration)

`src/models.py:367-419` — single scalar `T` fitted by LBFGS to minimize BCE on val logits; applied as `sigmoid(logit / T)`. Loaded by the API at startup for Model C v2 (`temperature.json`).

---

## 3. Training Pipeline

The unified loop is in `src/train.py:train_model`. Key features:

- **Optimizer**: `AdamW(weight_decay=0.01)` over only `p.requires_grad` params (matters for frozen-BERT Phase 1 of Model C).
- **Scheduler**: `get_linear_schedule_with_warmup(warmup_steps = 0.1 * total_steps)`.
- **fp16 mixed precision**:
  ```python
  use_amp = use_amp and device == 'cuda'
  scaler  = torch.amp.GradScaler(device, enabled=use_amp)
  ...
  with torch.amp.autocast(device, enabled=use_amp):
      logits = model(...)
      loss   = criterion(logits, labels) / grad_accum
  scaler.scale(loss).backward()
  if (step + 1) % grad_accum == 0:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
      scaler.step(optimizer); scaler.update()
      scheduler.step(); optimizer.zero_grad()
  ```
  Uses the new `torch.amp.autocast` / `torch.amp.GradScaler` API (not deprecated `torch.cuda.amp.*`).
- **Gradient accumulation**: per-step loss is scaled by `1 / grad_accum`, gradient step + scheduler step happen every `grad_accum` micro-batches. For Model C: `batch_size=4, grad_accum=8 → effective batch = 32` (fits the 6-chunk-per-sample VRAM budget on an L4).
- **Gradient clipping**: `max_grad_norm = 1.0` (after `scaler.unscale_`).
- **Loss**: either `BCEWithLogitsLoss(pos_weight=...)` OR `sigmoid_focal_loss` (defined in `src/train.py:35`).
- **Class imbalance** handled three ways:
  1. `compute_pos_weights` in `src/evaluate.py`: `pos_weight[j] = clip((N - n_j) / n_j, 1, clamp_max)`. clamp_max was 50 in v1, lowered to 10 in v2 for calibration.
  2. Focal loss (γ=2, α=0.25) — used in C v2 and Model D.
  3. Threshold tuning on val set.
- **Checkpointing**: `best_model.pt` saved whenever tuned val micro-F1 improves; mid-epoch checkpoints (full optimizer/scheduler/scaler state) every 2000 steps for resumability.
- **Early stopping**: configurable `early_stopping_patience`; Model C used patience=3.
- **Per-epoch eval**: `evaluate_predictions` runs the val loader, then `tune_global_threshold` sweeps thresholds; both `val_f1@0.5` and `val_f1_tuned` are logged.
- **Hardware**: GCP VM with NVIDIA L4 (23 GB VRAM), CUDA 12.8, accessed via VS Code Remote SSH. Training times:
  - Model A: ~2 min CPU
  - Model B: ~45 min (3 epochs × 2,659 batches)
  - Model C: ~8 hours (Phase 1 + Phase 2)
  - Model D: not pinned in the repo; expected ~30–60 min given lightweight model and 10 epochs.
- **Experiment tracking**: **Not implemented** — no MLflow / W&B. Results are saved as `training_history.csv` per model + `test_results.json`. Do not claim MLflow.

---

## 4. Evaluation & Results

### 4a. Full final results (from `data/models/ensemble/final_comparison_all_models.csv` — current source of truth)

| Model | Threshold | Micro-F1 | Macro-F1 | Micro-Prec | Micro-Rec | Macro-AUPRC | Micro-AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model A (TF-IDF + SGD) | 0.525 | 0.5952 | 0.5696 | 0.4941 | 0.7483 | 0.5741 | 0.9250 |
| Model B (ClinicalBERT) | 0.275 | 0.5242 | 0.4429 | 0.5247 | 0.5237 | 0.4541 | 0.8686 |
| Model C v1 (Chunk+Attn) | 0.625 | 0.5305 | 0.5000 | 0.4191 | 0.7225 | 0.5201 | 0.8938 |
| Model C v2 (Fixed+Focal) | 0.050 | 0.5508 | 0.4309 | 0.7895 | 0.4230 | 0.5567 | 0.9185 |
| **Model D (BiLSTM-LAAT)** | 0.300 | **0.7176** | **0.6635** | 0.7171 | 0.7181 | 0.6805 | 0.9523 |
| Ensemble v1 (A+Cv1, w=0.65) | 0.625 | 0.6249 | 0.5923 | 0.5712 | 0.6897 | 0.5921 | 0.9328 |
| Ensemble v2 (A+Cv2, w=0.35) | 0.425 | 0.5271 | 0.4005 | 0.8107 | 0.3905 | 0.6094 | 0.9358 |
| Ensemble v3 (A+Cv1+Cv2) | 0.400 | 0.5261 | 0.3992 | 0.8103 | 0.3895 | 0.6094 | 0.9359 |
| **Ensemble v4 (A+D, w=0.15)** | **0.350** | **0.7198** | **0.6682** | 0.7189 | 0.7207 | **0.6895** | **0.9548** |
| Ensemble v5 (A+Cv2+D) | 0.350 | 0.6943 | 0.6146 | 0.7952 | 0.6162 | 0.6847 | 0.9549 |

### 4b. Older `report.md` results (4-model story — what the README quotes)

This version was written before Model D existed and presents Ensemble v1 (A+C) as the winner. If asked about "the report," this is the narrative; if asked about the *current best* results, use the table above.

| Metric | A | B | C v1 | Ensemble v1 |
|---|---:|---:|---:|---:|
| Micro-F1 | 0.5952 | 0.5242 | 0.5305 | **0.6249** |
| Macro-F1 | 0.5696 | 0.4429 | 0.5000 | **0.5923** |

### 4c. Threshold selection
- Per-model global threshold tuned on val set with `tune_global_threshold` (sweep 0.05→0.65 step 0.025).
- Per-label thresholds available (`tune_per_label_threshold`) but the deployed API uses **global** thresholds (one per model).
- Ensemble threshold re-tuned after weight grid search (same sweep).

### 4d. Head/Torso/Tail analysis
- `head_tail_analysis` in `src/evaluate.py` buckets labels by training frequency: head ≥ 500, torso 100–499, tail < 100. It outputs per-bucket avg per-label F1 per model. Results saved in `data/head_tail_comparison.png` and CSVs under each model dir; the notebook commentary in `report.md` notes Model C achieved much higher recall on long-tail (0.72 vs Model B's 0.52) at the cost of precision.

### 4e. PLM-ICD comparison claim
- `report.md` claim: "PLM-ICD reports ~0.585 micro-F1 on MIMIC-IV ICD-10 full code set; our Top-50 ensemble exceeds this." The "within 2% of PLM-ICD" framing from the resume seems to refer to Ensemble v1's 0.6249 vs PLM-ICD's ~0.585 — a tighter framing than 2%. Be ready to clarify: PLM-ICD's published numbers are for the *full* code set, not Top-50, so the comparison is not apples-to-apples and our number being higher reflects easier label space, not strict SOTA-beating. **Do not over-claim SOTA**.

### 4f. Best overall
- Single best model: **Model D (BiLSTM-LAAT)**, Micro-F1 0.7176, beats every BERT-based model on this Top-50 task.
- Best ensemble: **Ensemble v4 (A+D)**, Micro-F1 0.7198 — only marginally above D alone (+0.0022), but Macro-AUPRC and AUROC both improve a bit.
- Counterintuitive finding to be ready to discuss: a 1-layer BiLSTM with learned word embeddings outperformed every Bio_ClinicalBERT variant. Hypothesis (from `report.md` reasoning and findings): BiLSTM sees the **full 4000-word document**, while even chunked BERT has to deal with discontinuities at chunk boundaries; medical terminology matters more than deep semantic context for Top-50 codes.

---

## 5. Deployment Stack

### 5a. FastAPI (`api/app.py`, `api/model_service.py`, `api/schemas.py`)

**Endpoints**:

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Predict ICD-10 codes from a discharge summary; optional attention evidence |
| GET | `/health` | Liveness; reports loaded models, device, num_labels |
| GET | `/model/info` | Lists models, thresholds, ensemble weight, full label vocab |

**Pydantic schemas** (`api/schemas.py`):
```python
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10)
    top_n: int = Field(10, ge=1, le=50)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    explain: bool = Field(False)

class EvidenceToken(BaseModel):
    token: str
    weight: float

class CodePrediction(BaseModel):
    icd_code: str
    description: str = ""
    probability: float
    predicted: bool
    evidence: Optional[list[EvidenceToken]] = None

class PredictionResponse(BaseModel):
    predictions: list[CodePrediction]
    model_version: str = "ensemble_v1"
    threshold_used: float
    processing_time_ms: float
```

**Model loading**: `ModelService` is a module-level singleton instantiated at import time; `lifespan(app)` (FastAPI's `asynccontextmanager`) calls `service.load()` once at startup. It loads:
1. `mlb.pkl` (MultiLabelBinarizer)
2. ClinicalBERT tokenizer
3. Model A: `tfidf_vectorizer.pkl` + `clf_sgd.pkl` + threshold from `results.json`
4. Model C: prefers `v2/best_model.pt`, falls back to `v1`; loads temperature for calibration
5. Ensemble config: `ensemble_config.json` → `ensemble_weight`, `threshold_ens`

⚠️ **Deployment caveat**: the current `ModelService` **only loads Models A and C and ensembles them as v1 (A + C, w from config)**. Model D is **not** loaded into the API. The Ensemble v4 (A+D) numbers in §4 are from offline notebook evaluation, not served. If asked, be honest: the API serves the A+C ensemble; productionizing the A+D ensemble would require adding `BiLSTMLAAT` + its vocab + word-level dataset to `model_service.py`.

**Prediction path** (`ModelService.predict`):
1. `clean_text` → run TF-IDF + SGD to get `probs_a`.
2. If `explain=True`: call `extract_attention_for_text` to get `(logits, probs_c, attention_dict)`.
3. Else: build a single-sample `ChunkedICDDataset`, run a forward pass under `torch.no_grad() + autocast`, apply `sigmoid(logits / temperature)` → `probs_c`.
4. Ensemble: `probs = w * probs_a + (1-w) * probs_c`.
5. Argsort top-N, attach evidence if `explain=True` (top-10 attention tokens per code).

**JSON evidence structure** (verbatim example):
```json
{
  "predictions": [
    {
      "icd_code": "I509",
      "description": "",
      "probability": 0.8732,
      "predicted": true,
      "evidence": [
        {"token": "heart", "weight": 0.0421},
        {"token": "failure", "weight": 0.0387},
        {"token": "edema", "weight": 0.0211}
      ]
    }
  ],
  "model_version": "ensemble_v1",
  "threshold_used": 0.625,
  "processing_time_ms": 412.3
}
```

**Performance**: no explicit batching (single-request inference); no async beyond FastAPI's coroutine wrapper around a synchronous `service.predict`. CORS is `allow_origins=["*"]` (permissive — fine for demo, not production). No queueing, no rate limiting.

### 5b. Streamlit demo (`demo/streamlit_app.py`)

- Two-column layout: left = text area for discharge summary, right = "How it works" info card.
- Sidebar: `top_n` (5–30), `threshold` (0.05–0.95), `explain` toggle, plus a live `GET /health` widget showing device, models loaded, num_labels.
- Three preset sample notes (Heart Failure + Diabetes; Pneumonia + COPD; Sepsis + UTI) populated via `selectbox`.
- "Predict" button POSTs JSON to `http://localhost:8000/predict` (60s timeout). Results render as:
  - A pandas DataFrame with a bar-styled probability column (`df.style.bar`).
  - If `explain=True`: per-code expandable panels containing up to 10 evidence tokens as `st.metric` cards with weight values.

### 5c. Docker

`Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt
COPY src/ src/
COPY api/ api/
# Model artifacts mounted as volumes (not baked in)
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- Single-stage build on the PyTorch CUDA 12.4 runtime image (GPU-ready).
- Model weights are **not** baked in — they're mounted at runtime via volumes (`-v ./data:/app/data -v ./datasets:/app/datasets`), keeping the image small and PHI artifacts out of the image.

`docker-compose.yml`:
- Two services: `icd-api` (built from Dockerfile, port 8000, requests 1 NVIDIA GPU via `deploy.resources.reservations.devices`) and `demo` (python:3.10-slim, pip-installs streamlit/requests/pandas at runtime, port 8501, `API_URL=http://icd-api:8000`).
- `demo` depends_on `icd-api`. GPU passthrough is via `NVIDIA_VISIBLE_DEVICES=all` + the nvidia driver block.

### 5d. GCP deployment

From `report.md`: trained on a **GCP Compute Engine VM** with NVIDIA L4 (23 GB VRAM), CUDA 12.8, accessed via VS Code Remote SSH. No Cloud Run / GKE / Vertex AI is wired up. No CI/CD — deployment is manual (SSH in, build container, run). The earlier GCP setup docs (`GCP_SETUP.md`) were removed in commit `ace0aa0` ("Remove ignored local and setup artifacts from tracking"), so most of the GCP setup detail now lives in commit history rather than the repo.

---

## 6. Testing (pytest)

Three test files (~175 lines total). All tests pass without needing real model weights — they use `prajjwal1/bert-tiny` (4.4M params) as a stand-in for Model C shape tests.

### `tests/test_data.py` — text + truncation
- `clean_text` removes `[**...**]` de-identification tokens; lowercases; strips non-alphanumeric; handles None/empty; preserves medical lab values like `wbc-4.8`.
- `smart_truncate` respects `max_chars`; extracts the "DISCHARGE DIAGNOSES" section when present; passes short text through unchanged.

### `tests/test_models.py` — model + ensemble shapes
- `EnsemblePredictor` weighted-average correctness (verified vs `0.6 * P_a + 0.4 * P_c`).
- `EnsemblePredictor(weight=0.0)` returns Model C only; `weight=1.0` returns Model A only.
- `LabelAttentionClassifier` output shape `(batch, num_labels)` and `(batch, num_labels, total_tokens)` when `return_attention=True`.
- **Attention sums to 1** test: `torch.testing.assert_close(attn.sum(dim=-1), torch.ones_like(...), atol=1e-5)` — verifies the softmax normalization invariant.

### `tests/test_api.py` — Pydantic schema validation
- `PredictionRequest` with valid fields parses; `min_length=10` constraint rejects short text; defaults are `top_n=10, explain=False`.
- `CodePrediction` constructs cleanly with `evidence=None`.

**Not covered**: real end-to-end API integration tests, training-loop tests, evaluation-metric tests. The API tests only exercise schemas, not endpoints — there's no `TestClient` wiring up the real `app`.

---

## 7. Section-Aware Truncation

This is the resume bullet about "section-aware truncation prioritizing diagnostically relevant content." Used by Model B only (Models C/D process full docs).

The patterns (`src/config.py:81`):
```python
SECTION_PATTERNS = [
    r'discharge diagnos[ei]s.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'discharge condition.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'hospital course.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'history of present illness.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'chief complaint.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
]
```

Each regex matches a header (case-insensitive, `re.DOTALL`) and greedily captures content until the next `\n` followed by another all-caps header.

`smart_truncate` (in `src/data.py:43`):
```python
def smart_truncate(text: str, max_chars: int = 2048) -> str:
    segments = []
    total = 0
    for pat in _COMPILED_SECTIONS:
        m = pat.search(text)
        if m:
            seg = m.group(0).strip()
            if total + len(seg) <= max_chars:
                segments.append(seg)
                total += len(seg)
    if total < max_chars:
        segments.insert(0, text[:max_chars - total])
    return ' '.join(segments)[:max_chars]
```

Behavior: walk the priority-ordered section list, greedily append sections that fit within the `max_chars` budget (default 2048 chars ≈ 512 BERT tokens). If budget remains, prepend the document head so we still get the patient context. Final string is the concatenation, truncated to `max_chars` as a safety net.

**Why this matters**: discharge summaries average ~1,500 words; naïve truncation to 512 tokens loses the "Discharge Diagnoses" section that explicitly lists the coded diagnoses. This is the cheapest way to get good Model B numbers without changing the architecture.

---

## 8. Data Pipeline

- **Loading**: `load_splits()` in `src/data.py` reads `cohort_{train,val,test}_clean.parquet`, splits `icd_codes_str` on `|`.
- **Cleaning** (`clean_text`): regex-strip `[**...**]` MIMIC de-id tokens, lowercase, keep `a-z0-9\s.,;:\-/`, collapse whitespace.
- **Label encoding**: `MultiLabelBinarizer` (sklearn) fitted on Top-50 codes, persisted as `mlb.pkl`. `build_label_matrix(df, mlb)` produces an `(n_samples, 50)` float32 matrix.
- **Split**: patient-level 70/15/15, done upstream in notebook 01.
- **Datasets**:
  - `ICDDataset` (Model B): tokenizes one 512-token chunk per doc.
  - `ChunkedICDDataset` (Model C): tokenizes full doc, splits into ≤6 stride-256 chunks, returns `(input_ids, attention_mask, chunk_count, labels)`. **Custom chunking logic in `_chunk_tokens`** — no HuggingFace `return_overflowing_tokens=True` because we need precise control over [CLS]/[SEP] insertion and chunk-count tracking.
  - `BiLSTMDataset` (Model D): whitespace tokenize, map words via `word2idx` (with `<UNK>=1`), pad to 4000.
- **DataLoader**: standard `torch.utils.data.DataLoader`; no custom `collate_fn` needed because each Dataset already returns fixed-shape tensors (pre-padded). Variable lengths are handled inside the model via attention masks + `pack_padded_sequence` (Model D).

---

## 9. Project Structure

```
.
├── api/
│   ├── __init__.py
│   ├── app.py             # FastAPI app: /predict, /health, /model/info
│   ├── model_service.py   # Singleton ModelService (loads A + C, serves ensemble)
│   └── schemas.py         # Pydantic request/response models
├── demo/
│   └── streamlit_app.py   # Streamlit UI calling http://localhost:8000
├── src/
│   ├── __init__.py
│   ├── config.py          # All hyperparams, paths, constants
│   ├── data.py            # clean_text, smart_truncate, MLB, 3 Dataset classes, vocab build
│   ├── models.py          # ICDClassifier (B), LabelAttentionClassifier (C),
│   │                      # BiLSTMLAAT (D), TemperatureScaler, EnsemblePredictor
│   ├── train.py           # train_model + focal_loss + evaluate_predictions
│   ├── evaluate.py        # full_metrics, threshold tuning, head/tail, ECE
│   └── explain.py         # extract_attention_for_text, highlight HTML, explain_prediction
├── tests/
│   ├── test_api.py        # Pydantic schema tests
│   ├── test_data.py       # text cleaning + truncation tests
│   └── test_models.py     # ensemble + attention shape/sum tests
├── notebooks_local/
│   ├── 01_data_extraction_local.ipynb        # MIMIC-IV cohort construction
│   ├── 02_preprocessing_local.ipynb          # Cleaning + TF-IDF + MLB
│   ├── 03_model_a_tfidf_baseline_local.ipynb
│   ├── 04_model_b_transformer_local.ipynb
│   ├── 05_evaluation_demo_local.ipynb        # A + B side-by-side
│   ├── 06_model_c_training.ipynb             # Model C v1
│   ├── 06_model_c_training_v2.ipynb          # Model C v2 (focal + fixes)
│   ├── 07_ensemble_evaluation.ipynb          # All ensembles + final tables/plots
│   └── 08_model_d_bilstm_local.ipynb         # Model D training
├── scripts/
│   └── snapshot_notebooks.py                 # Notebook version-control helper
├── data/
│   ├── final_comparison.csv                  # 2-model snapshot (older)
│   ├── head_tail_comparison.png
│   └── models/{model_a,model_b,model_c,model_d,ensemble}/   # all artifacts + JSON
├── Dockerfile
├── docker-compose.yml
├── requirements*.txt
├── README.md              # public-facing setup guide
└── report.md              # written before Model D existed — 4-model story
```

---

## 10. Design Decisions & Tradeoffs

**Why Bio_ClinicalBERT over BioBERT / PubMedBERT?**
- Bio_ClinicalBERT (Alsentzer et al.) is initialized from BioBERT and further pretrained on MIMIC-III clinical notes. For MIMIC-IV discharge summaries, the in-domain pretraining is the closest match — PubMedBERT/BioBERT see literature, not clinical narrative. The token distribution (e.g., "pt c/o", "s/p", "hx of") matches.

**Why chunk-based BERT over Longformer / Clinical-Longformer / hierarchical transformer?**
- Longformer is the principled answer but at the time of building, the available pretrained Clinical-Longformer checkpoints weren't as battle-tested for ICD coding as PLM-ICD's chunk approach. Chunking lets us reuse the strong Bio_ClinicalBERT checkpoint with no architectural change to the base encoder.
- Tradeoff: chunk boundaries can split a clinical phrase; the 50% stride helps but isn't free. Longformer was listed as a future improvement in `report.md`.

**Why BiLSTM as an additional model?**
- The LAAT paper (Vu et al., IJCAI 2020) is a known strong baseline for ICD coding; for this Top-50 task with documents up to several thousand words, a model that sees the full document end-to-end without chunking artifacts has a structural advantage. The result (D > all BERT variants) confirms that hypothesis for this label set. It also gives the ensemble a *different family* (recurrent vs. transformer) to combine.

**Ensemble strategy choice**: simple weighted probability averaging with a 1-D weight grid search. The hypothesis is that A and C/D make different *kinds* of errors (TF-IDF is precise keyword matching, neural models pick up semantic paraphrases) so averaging improves recall without trashing precision. The grid search keeps the ensemble interpretable; a learned stacking meta-classifier was out of scope.

**Why FastAPI over Flask?**
- Pydantic validation out of the box (auto-generated OpenAPI/Swagger), async support, type hints. The schemas in `api/schemas.py` double as both documentation and runtime contract; that's harder in Flask.

**Why Streamlit for the demo?**
- One-file UI with sliders/text areas; no React/JS build. The demo's purpose is to show the attention-evidence story to non-engineers (clinicians, course graders); Streamlit makes that 100 lines instead of 1000.

**Compromises / what I'd improve with more time**:
- Productionize Model D / Ensemble v4 in `ModelService` (currently it only serves A + C).
- Add proper integration tests with a `TestClient` and a tiny mock model.
- Implement MLflow / W&B for experiment tracking (currently just CSV files).
- Per-label threshold tuning (code exists, not deployed).
- Add request batching + async inference for the API.
- Scale to Top-500 or full code set; explore Clinical-Longformer.
- Add ICD code descriptions to the API response (`CodePrediction.description` is wired up but blank).

---

## 11. Team Ownership Map

Based on `git log --pretty=format:"%h | %an | %s"`, two authors contributed:
- **Rohan Kumar / crohank** (same person — different commit identities from different machines)
- **Deva Sai Sunder Tangella**

### What I (Rohan) wrote — strong evidence from git history

| Component | Evidence |
|---|---|
| `src/` package (config, data, models, train, evaluate, explain) | Commit `7997977` "Add Model C (chunk+label attention), src/ package, FastAPI, Streamlit demo, Docker, and tests" |
| `api/` FastAPI + ModelService + schemas | Same commit `7997977` |
| `demo/streamlit_app.py` (initial) | Same commit `7997977` |
| `Dockerfile` + `docker-compose.yml` | Same commit `7997977` |
| `tests/` pytest suite | Same commit `7997977` |
| Model C (Chunk+LabelAttn) v1 and v2 | `7997977` initial; `530756b` "Fix Model C bugs + add focal loss, semantic label init, temperature scaling" |
| Section-aware truncation (`smart_truncate` + SECTION_PATTERNS) | Part of `src/data.py` / `src/config.py` from `7997977` |
| fp16 + grad accumulation training loop | `src/train.py` from `7997977`; further optimizations in `50f51d0` "Optimize training pipeline: Top-K label filtering, faster solver, GPU acceleration" |
| Model D (BiLSTM-LAAT) — initial implementation | Commit `c661798` "Add Model D: BiLSTM-LAAT (Vu et al. IJCAI 2020)" by Rohan Kumar |
| Ensemble v1–v5 integration in notebook 07 | Commit `661086e` "Integrate Model D into ensemble evaluation (notebook 07)" by Rohan |
| GCP setup, notebooks 01/02/04, training infra | Earlier Rohan commits (`394e6e0`, `e87cc65`, `50f51d0`, `ac6d02b`) |

### What Deva contributed

| Component | Evidence |
|---|---|
| Re-running Model A | `3f2de39` "Re Ran model A" |
| Confusion matrix demo additions | `568bc86` "Added demo file and code for confusion matrix" |
| Code cleaning / formatting passes | `0bf9da5` "Done with code cleaning" |
| Comments | `b49bef2` "Added comments" |

### Interview talking-points on ownership

- **Bullets 1–7 on your resume** (chunk-based ClinicalBERT + label attention, FastAPI, Streamlit, Docker+GCP, pytest, fp16+grad accum, section-aware truncation) — git history strongly supports these as **Rohan's individual contributions**.
- **Model D (BiLSTM-LAAT)**: the resume frames "BiLSTM models" as a team effort, but **git shows Rohan authored the model class and training integration**. Deva's later commits touch the Model D notebook through general code cleaning. If asked directly, the honest framing is: *"I implemented BiLSTM-LAAT from the Vu et al. paper, including the training notebook and the ensemble integration. My teammate contributed code-cleanup and comments to that notebook later."*
- **Ensembles**: integration code (`src/models.py:EnsemblePredictor` and the v1–v5 logic in notebook 07) is all Rohan's. Tuning the weights was a single-author task.
- **Model A re-runs / final TF-IDF baseline numbers** quoted in the report come from Deva's re-run (`3f2de39`).
- **Demo file / confusion matrix**: Deva added confusion-matrix visualization helpers (the `568bc86` commit) on top of Rohan's initial Streamlit app.

In short: **the entire production stack and all four model architectures were written by Rohan**, with teammate contributions concentrated on re-running Model A, adding confusion-matrix code, and code/comment cleanup passes. The resume can be defended cleanly with git history if anyone asks.

---

## 12. Quick-reference numbers (memorize these)

| Thing | Value |
|---|---|
| MIMIC-IV records used | 122,304 (paired note + codes) |
| Train / val / test | 85,081 / 18,371 / 18,852 |
| Total ICD-10 codes seen | 31,794 unique |
| Codes with ≥10 occurrences | 7,940 |
| **Top-K labels trained** | **50** |
| ClinicalBERT checkpoint | `emilyalsentzer/Bio_ClinicalBERT`, 108M params, hidden=768 |
| Max seq len | 512 |
| Model C chunks × stride | 6 chunks × 256 stride (50% overlap) ≈ 3,072 tokens |
| Model C effective batch | 4 × 8 grad-accum = 32 |
| BiLSTM hidden / embed / vocab / max-tokens | 256 / 200 / 50,000 / 4,000 |
| GPU | NVIDIA L4, 23 GB VRAM, CUDA 12.8 |
| Best single model | **Model D, Micro-F1 0.7176** |
| Best ensemble (offline) | **v4 (A+D, 0.15/0.85), Micro-F1 0.7198** |
| Best ensemble (served by API) | **v1 (A+C v1, 0.65/0.35), Micro-F1 0.6249** |
| Optimal thresholds | A=0.525, B=0.275, C v1=0.625, C v2=0.05, D=0.3, Ens v1=0.625, Ens v4=0.35 |

---

## 13. Discrepancies / things to be honest about

1. **`report.md` is stale** — it documents the A/B/C 4-model story (Ensemble v1 best at 0.6249) and predates Model D entirely. The CSV in `data/models/ensemble/final_comparison_all_models.csv` has the current 10-row table including Model D and ensembles v4/v5. Have both versions of the story ready.
2. **API does NOT serve Model D / Ensemble v4** — `ModelService.load()` only loads Models A and C and serves Ensemble v1. The Model D win is from offline notebook evaluation.
3. **PLM-ICD benchmark comparison** — PLM-ICD's ~0.585 number is for the *full* code set; our 0.6249 / 0.7198 are for Top-50. The resume's "within 2% of PLM-ICD benchmarks" framing is closest to v1; with v4 we'd actually exceed PLM-ICD numerically, but on an *easier* label set. Don't claim SOTA; do claim "PLM-ICD-inspired architecture, competitive with reported numbers on a Top-50 subset."
4. **No MLflow / W&B** — experiment tracking is CSV files. Don't claim a tracking platform.
5. **No CI/CD** — deployment was manual SSH-and-run on a GCP VM. The `GCP_SETUP.md` was removed in cleanup before going public (`ace0aa0`).
6. **Pretrained word embeddings (Model D)** — the `pretrained_embeddings` arg in `BiLSTMLAAT.__init__` exists but isn't wired up in the training notebook. Word embeddings are learned from scratch.
7. **`CodePrediction.description` is empty** — the API schema has the field, but `ModelService.predict` always returns `""`. The `ICD10_DESCRIPTIONS` dict exists in `src/data.py` and is used for semantic label-query init, but isn't piped into the API response.
8. **Resume bullet "BiLSTM models" as team effort**: git shows Rohan as the actual author. Frame honestly per §11.
