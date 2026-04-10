"""
Centralized configuration for the ICD-10 prediction pipeline.
All paths, hyperparameters, and constants in one place.
"""
import os
from pathlib import Path

# ── Project root (one level up from src/) ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ─────────────────────────────────────────────────────────
DATA_DIR     = PROJECT_ROOT / "datasets" / "processed"
MODEL_A_DIR  = PROJECT_ROOT / "data" / "models" / "model_a"
MODEL_B_DIR  = PROJECT_ROOT / "data" / "models" / "model_b"
MODEL_C_DIR  = PROJECT_ROOT / "data" / "models" / "model_c"
ENSEMBLE_DIR = PROJECT_ROOT / "data" / "models" / "ensemble"

# Ensure model directories exist
for d in [MODEL_A_DIR, MODEL_B_DIR, MODEL_C_DIR, ENSEMBLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Label filtering ────────────────────────────────────────────────────
TOP_K_LABELS = 50           # Set to 50, 500, or None for full label set

# ── TF-IDF config (Model A) ───────────────────────────────────────────
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_SUBLINEAR_TF = True

# ── Transformer config (Model B / Model C) ────────────────────────────
TRANSFORMER_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAX_SEQ_LEN       = 512
HIDDEN_SIZE       = 768     # Bio_ClinicalBERT hidden dimension

# ── Model B training hyperparams ───────────────────────────────────────
MODEL_B_LR         = 2e-5
MODEL_B_EPOCHS     = 3
MODEL_B_BATCH_SIZE = 32
MODEL_B_GRAD_ACCUM = 1
MODEL_B_WARMUP     = 0.1
MODEL_B_DROPOUT    = 0.1

# ── Model C training hyperparams ───────────────────────────────────────
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

# ── General ────────────────────────────────────────────────────────────
SEED          = 42
USE_AMP       = True       # fp16 mixed precision (CUDA only)

# ── Smart truncation section patterns (for Model B) ───────────────────
SECTION_PATTERNS = [
    r'discharge diagnos[ei]s.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'discharge condition.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'hospital course.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'history of present illness.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
    r'chief complaint.*?(?=\n[A-Z][A-Z ]{3,}:|$)',
]
