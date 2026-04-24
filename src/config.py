"""
Single source of truth for paths and hyperparameters.

Notebooks and scripts import from here so a path or LR change does not require
hunting through copied literals. Nothing in this file reads the environment
except what you add explicitly — keep it boring and importable.
"""
import os
from pathlib import Path

# Repo root (parent of `src/`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Processed parquet/pickles and per-model checkpoint directories.
DATA_DIR     = PROJECT_ROOT / "datasets" / "processed"
MODEL_A_DIR  = PROJECT_ROOT / "data" / "models" / "model_a"
MODEL_B_DIR  = PROJECT_ROOT / "data" / "models" / "model_b"
MODEL_C_DIR  = PROJECT_ROOT / "data" / "models" / "model_c"
ENSEMBLE_DIR = PROJECT_ROOT / "data" / "models" / "ensemble"

MODEL_D_DIR  = PROJECT_ROOT / "data" / "models" / "model_d"

# Ensure model directories exist
for d in [MODEL_A_DIR, MODEL_B_DIR, MODEL_C_DIR, MODEL_D_DIR, ENSEMBLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# How many ICD labels to keep in the multilabel head (50 / 500 / None = all).
TOP_K_LABELS = 50           # Set to 50, 500, or None for full label set

# Model A — sparse linear baseline on character-like cleaned tokens.
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_SUBLINEAR_TF = True

# Shared ClinicalBERT backbone for Models B and C.
TRANSFORMER_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAX_SEQ_LEN       = 512
HIDDEN_SIZE       = 768     # Bio_ClinicalBERT hidden dimension

# Model B — single-sequence fine-tuning (512 tokens).
MODEL_B_LR         = 2e-5
MODEL_B_EPOCHS     = 3
MODEL_B_BATCH_SIZE = 32
MODEL_B_GRAD_ACCUM = 1
MODEL_B_WARMUP     = 0.1
MODEL_B_DROPOUT    = 0.1

# Model C — long documents via overlapping chunks + label-wise attention.
MODEL_C_MAX_CHUNKS      = 6       # max 512-token chunks per document
MODEL_C_CHUNK_STRIDE    = 256     # overlap between chunks (50%)
MODEL_C_FROZEN_LR       = 1e-3    # LR when BERT is frozen
MODEL_C_FINETUNE_LR     = 2e-5    # LR when fine-tuning BERT layers
MODEL_C_FROZEN_EPOCHS   = 5       # epochs with frozen BERT
MODEL_C_FINETUNE_EPOCHS = 7       # epochs with unfrozen last 2 layers (was 3, increased for convergence)
MODEL_C_BATCH_SIZE      = 4       # smaller batch due to multiple chunks
MODEL_C_GRAD_ACCUM      = 8       # effective batch = 4 * 8 = 32
MODEL_C_DROPOUT         = 0.1
MODEL_C_POS_WEIGHT_MAX  = 10.0    # clamp pos_weight (was 50, reduced to fix calibration)
MODEL_C_FOCAL_GAMMA     = 2.0     # focal loss gamma (0 = standard BCE)
MODEL_C_FOCAL_ALPHA     = 0.25    # focal loss alpha
MODEL_C_EARLY_STOP      = 3       # early stopping patience (epochs without improvement)

# Model D — word-level BiLSTM + LAAT (no subword tokenizer).
MODEL_D_EMBED_DIM    = 200       # Word embedding dimension
MODEL_D_HIDDEN_DIM   = 256       # BiLSTM hidden size (each direction)
MODEL_D_NUM_LAYERS   = 1         # BiLSTM layers
MODEL_D_ATTN_DIM     = 256       # Label attention projection dim (d_a in paper)
MODEL_D_DROPOUT      = 0.3       # Dropout (paper uses 0.3)
MODEL_D_LR           = 1e-3      # AdamW learning rate
MODEL_D_EPOCHS       = 10        # Training epochs
MODEL_D_BATCH_SIZE   = 32        # Batch size
MODEL_D_GRAD_ACCUM   = 1         # No accumulation needed (lightweight model)
MODEL_D_MAX_TOKENS   = 4000      # Max document length in word tokens
MODEL_D_VOCAB_SIZE   = 50_000    # Word vocabulary size
MODEL_D_EARLY_STOP   = 3         # Early stopping patience
MODEL_D_FOCAL_GAMMA  = 2.0
MODEL_D_FOCAL_ALPHA  = 0.25

# Reproducibility and mixed precision toggle.
SEED          = 42
USE_AMP       = True       # fp16 mixed precision (CUDA only)

# Regexes for "smart truncate": grab high-signal sections before stuffing BERT.
SECTION_PATTERNS = [
    r'discharge diagnos[ei]s.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'discharge condition.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'hospital course.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'history of present illness.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'chief complaint.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
]
