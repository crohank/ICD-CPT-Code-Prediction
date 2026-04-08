# GCP / Google Colab Setup Guide
## ICD-10 Code Prediction — MIMIC-IV Project

This guide walks through running the full training pipeline on **Google Colab Pro** using Google Drive for storage and **PhysioNet GCS** (`gs://physionet-data/`) for raw MIMIC-IV data.

---

## Overview

```
PhysioNet GCS (read-only)
  gs://physionet-data/mimiciv/3.1/hosp/
  gs://physionet-data/mimic-iv-note/2.2/note/
        │
        ▼  Notebook 01 extracts + joins
Google Drive: MyDrive/mimic_icd/datasets/
        │
        ▼  Notebook 02 preprocesses
        ├── X_train/val/test_tfidf.npz
        ├── Y_train/val/test.npy
        ├── mlb.pkl, tfidf_vectorizer.pkl
        │
        ├── Notebook 03 → models/model_a/  (TF-IDF + SGD)
        └── Notebook 04 → models/model_b/  (ClinicalBERT)
                               │
                        Notebook 05 → final_comparison.csv
```

---

## Part 1: Prerequisites

### 1.1 Accounts you need

| Account | Purpose | How to get it |
|---------|---------|--------------|
| **Google account (personal)** | Google Drive storage for outputs | Any Gmail account |
| **Google account (school/PhysioNet)** | Access `gs://physionet-data/` GCS bucket | Must be the email registered on PhysioNet |
| **PhysioNet credentialed access** | MIMIC-IV + MIMIC-IV-Note | physionet.org → complete CITI training + data use agreement |
| **Google Colab Pro** | T4 GPU, 25 GB RAM, 24h sessions | colab.google → subscribe ($9.99/mo) |

> **Important:** These can be the same Google account if your school email is also your PhysioNet account. If they differ, you will do two separate sign-ins in Notebook 01.

### 1.2 Verify PhysioNet GCS access

Before running any notebook, confirm you can read from the PhysioNet bucket. Go to:
```
https://physionet.org/settings/credentialing/
```
Under **Cloud Storage**, make sure MIMIC-IV (v3.1) and MIMIC-IV-Note (v2.2) are listed as accessible. If not, request access on each dataset's PhysioNet page.

### 1.3 Colab Pro setup

1. Go to [colab.google](https://colab.google)
2. Subscribe to **Colab Pro** ($9.99/mo) — needed for guaranteed T4 GPU and high-RAM mode
3. After subscribing, when you open a notebook go to: **Runtime → Change runtime type → T4 GPU + High RAM**

---

## Part 2: Google Drive Folder Structure

The notebooks automatically create directories when they first write to them, but it's good to understand the expected layout:

```
My Drive/
└── mimic_icd/
    ├── datasets/                    ← created by Notebook 01 & 02
    │   ├── cohort_train.parquet
    │   ├── cohort_val.parquet
    │   ├── cohort_test.parquet
    │   ├── cohort_full.parquet
    │   ├── cohort_train_clean.parquet
    │   ├── cohort_val_clean.parquet
    │   ├── cohort_test_clean.parquet
    │   ├── label_vocab.csv
    │   ├── label_freq.csv
    │   ├── mlb.pkl
    │   ├── tfidf_vectorizer.pkl
    │   ├── X_train_tfidf.npz        (~1.4 GB)
    │   ├── X_val_tfidf.npz
    │   ├── X_test_tfidf.npz
    │   ├── Y_train.npy
    │   ├── Y_val.npy
    │   └── Y_test.npy
    └── models/
        ├── model_a/                 ← created by Notebook 03
        │   ├── clf_sgd.pkl
        │   ├── results.json
        │   └── head_tail_f1.png
        └── model_b/                 ← created by Notebook 04
            ├── best_model.pt        (~450 MB)
            ├── checkpoint_e*.pt     (mid-epoch checkpoints)
            ├── P_val_best.npy
            ├── P_test.npy
            ├── Y_test.npy
            ├── test_results.json
            ├── training_history.csv
            └── head_tail_f1.png
```

**Required Drive space**: ~12–15 GB total (datasets + models). Make sure your Drive has enough free space.

---

## Part 3: How to Run — Step by Step

### Step 1: Upload notebooks to Colab

**Option A — GitHub (recommended):**
```
In Colab: File → Open notebook → GitHub tab
Enter repo URL → select branch: v2 → open notebooks/01_data_extraction.ipynb
```

**Option B — Direct upload:**
```
In Colab: File → Upload notebook → select file from your local machine
```

Run notebooks in this order: `01 → 02 → 03 → 04 → 05`

---

### Step 2: Run Notebook 01 — Data Extraction

**Runtime:** ~20–40 min (downloads ~5 GB from GCS)

**Before running:**
- Set runtime to **T4 GPU + High RAM** (Runtime → Change runtime type)

**Cell 2 — Two-account auth flow:**
```
1. Cell runs — Colab mounts your Drive (sign in with PERSONAL account)
2. A URL appears in the output — open it in a new tab
3. Sign in with your SCHOOL/PhysioNet account
4. Copy the verification code and paste it back into the Colab prompt
5. Press Enter — you should see "Authentication complete."
```

> If the gcloud auth fails, try: Runtime → Restart runtime → re-run from cell 1.

**Verify it worked** — Cell 6 should print:
```
GCS filesystem ready (authenticated via school account).
```

**What it does:**
- Reads `discharge.csv.gz`, `diagnoses_icd.csv.gz`, `procedures_icd.csv.gz` from GCS
- Filters to ICD-10 codes, joins notes ↔ codes
- Creates patient-level 70/15/15 train/val/test splits
- Saves cohort parquets + `label_vocab.csv` to Drive

**Expected output:**
```
Cohort size (note + ICD-10 codes): ~122,000
Vocab (freq>=10): 7940
train: 85437 rows
val:   18195 rows
test:  18672 rows
```

---

### Step 3: Run Notebook 02 — Preprocessing

**Runtime:** ~10–15 min

**Key config (Cell 4):**
```python
TOP_K_LABELS = 50       # Start with 50 (standard benchmark). Change to 500 for extended run.
TFIDF_MAX_FEATURES = 50_000
```

**What it does:**
- Cleans discharge note text (removes MIMIC de-id artifacts `[**...**]`)
- Selects top-50 ICD codes by training frequency
- Fits TF-IDF vectorizer (50k features, unigrams+bigrams)
- Saves TF-IDF matrices (X) and label matrices (Y) to Drive

**Expected output:**
```
Selected top-50 labels (coverage: ~32.5% of all assignments)
Y_train shape: (85437, 50)
X_train_tfidf shape: (85437, 50000)
All features and labels saved.
```

> To run Top-500 extended evaluation: Change `TOP_K_LABELS = 500` and re-run this notebook. It will overwrite the existing files. Run notebooks 03-05 again after.

---

### Step 4: Run Notebook 03 — Model A (TF-IDF + Linear)

**Runtime:** ~2–5 min (Top-50) | ~20–60 min (Top-500) | ~1 hour (full 7,940 labels)

**GPU use:** This notebook tries to use **cuML** (GPU-accelerated sklearn). On Colab with T4, cuML can be installed but may take 5–10 min to install. If it fails, it automatically falls back to CPU SGDClassifier which is still fast.

**What to expect in Cell 1 (install):**
```
# cuML install — may take several minutes
Successfully installed cuml-cu12 ...
  OR
cuML not available, falling back to CPU sklearn
```

**Cell 4 config:**
```python
CLASSIFIER = 'sgd'   # fastest CPU option; ignored if cuML installed
```

**Expected output after training:**
```
Training OvR on 85437 samples × 50 labels (GPU=True)...
Done.
=== Validation ===
  micro_f1    : 0.xxxx
  macro_f1    : 0.xxxx
  micro_prec  : 0.xxxx
  micro_rec   : 0.xxxx
  macro_auprc : 0.xxxx
  micro_auroc : 0.xxxx
Results saved.
```

---

### Step 5: Run Notebook 04 — Model B (ClinicalBERT)

**Runtime:** ~1–2 hours (Top-50 on T4) | ~3–5 hours (Top-500 on T4)

**GPU use:** This notebook requires GPU. Before running, confirm Cell 4 prints a GPU name:
```
GPU: Tesla T4, 14.7 GB VRAM
Device: cuda  Batch: 16  GradAccum: 2  Effective: 32
```

If it shows `Device: cpu`, stop and change the Colab runtime to T4 GPU.

**Model downloads:** ClinicalBERT (~450 MB) downloads from HuggingFace on first run. Subsequent runs use cache.

**Training progress** — you'll see per-step loss + epoch summaries:
```
  Step 500/5340  loss=0.0842
  Step 1000/5340  loss=0.0756
  → mid-epoch checkpoint saved     ← every 2000 steps (Colab disconnect safety)
Epoch 1/3  loss=0.0698  val_micro_F1=0.4821
  → saved best checkpoint
Epoch 2/3  loss=0.0534  val_micro_F1=0.5103
Epoch 3/3  loss=0.0471  val_micro_F1=0.5214
Training complete. Best val micro-F1: 0.5214
```

**If Colab disconnects mid-training:** The mid-epoch checkpoints are saved to Drive. You can resume from the latest checkpoint by loading it before re-running the training loop:
```python
# Add this before the training loop to resume
checkpoint = torch.load(f'{MODEL_DIR}/checkpoint_e2_s4000.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

### Step 6: Run Notebook 05 — Evaluation & Demo

**Runtime:** ~5–10 min

**What it produces:**
- Side-by-side comparison table: Model A vs Model B
- Top-K analysis (Top-50, Top-100, Top-500 codes)
- Head/Torso/Tail F1 breakdown
- Qualitative error analysis on 20 samples
- Interactive `predict_icd(note_text)` demo function

---

## Part 4: GCS Version Paths

The notebook uses these GCS paths (Cell 6 of Notebook 01):
```python
GCS_HOSPITAL = 'gs://physionet-data/mimiciv/3.1/hosp/'
GCS_NOTE     = 'gs://physionet-data/mimic-iv-note/2.2/note/'
```

If your PhysioNet access uses a different version, update these in Cell 6:
- MIMIC-IV versions on PhysioNet: `2.0`, `2.1`, `2.2`, `3.0`, `3.1`
- MIMIC-IV-Note versions: `2.2`

To verify what versions you have access to, run this in a Colab cell after auth:
```python
import gcsfs
fs = gcsfs.GCSFileSystem(token='google_default')
print(fs.ls('gs://physionet-data/mimiciv/'))
print(fs.ls('gs://physionet-data/mimic-iv-note/'))
```

---

## Part 5: Troubleshooting

### "403 Forbidden" when accessing GCS
- Your PhysioNet account doesn't have access to the requested MIMIC-IV version
- Go to physionet.org → find MIMIC-IV v3.1 → request access
- Or change `GCS_HOSPITAL` in Cell 6 to a version you do have access to

### "gcloud auth failed"
```python
# Try the alternative auth approach in Cell 2:
from google.colab import auth
auth.authenticate_user()
import gcsfs
fs = gcsfs.GCSFileSystem(token='google_default')
```
Replace the subprocess gcloud call with the above.

### "CUDA out of memory" in Notebook 04
The auto-detection may have set batch size too high. Override in Cell 4:
```python
BATCH_SIZE = 4
GRAD_ACCUM = 8
```

### Colab session disconnects during Notebook 04
Mid-epoch checkpoints are saved to Drive every 2000 steps. Find the latest:
```
MyDrive/mimic_icd/models/model_b/checkpoint_e{epoch}_s{step}.pt
```
Resume using the code snippet in Step 5 above.

### Drive runs out of space
The full pipeline uses ~12–15 GB. Free up Drive space or upgrade Google One storage if needed. The largest files are:
- `X_train_tfidf.npz`: ~1.4 GB
- `Y_train.npy`: ~17 MB (Top-50) / ~170 MB (Top-500)  
- `best_model.pt`: ~450 MB

### cuML fails to install
This is normal — cuML has specific CUDA version requirements. The notebooks automatically fall back to CPU SGDClassifier, which completes in under 5 minutes for Top-50 labels.

---

## Part 6: Run Order Summary

```
Notebook 01  →  Notebook 02  →  Notebook 03  (Model A, ~5 min)
                              →  Notebook 04  (Model B, ~2 hours)
                                           →  Notebook 05  (Eval, ~10 min)
```

After running 02, notebooks 03 and 04 are **independent** — your teammate can run 03 while you run 04, as long as both Colab sessions are connected to the same Drive.

---

## Part 7: Expected Training Time on Colab Pro (T4)

| Notebook | TOP_K=50 | TOP_K=500 |
|----------|----------|-----------|
| 01 — Data extraction | ~30 min | same |
| 02 — Preprocessing | ~10 min | ~12 min |
| 03 — Model A | ~2–5 min | ~20–60 min |
| 04 — Model B | ~1–2 hours | ~3–5 hours |
| 05 — Evaluation | ~5 min | ~5 min |
| **Total** | **~2–3 hours** | **~4–7 hours** |

---

## Quick Reference: Key Config Values

| Setting | File | Default | Change to |
|---------|------|---------|-----------|
| Label count | `02_preprocessing.ipynb` Cell 4 | `TOP_K_LABELS = 50` | `500` for extended run |
| TF-IDF features | `02_preprocessing.ipynb` Cell 4 | `TFIDF_MAX_FEATURES = 50_000` | keep |
| Classifier type | `03_model_a_tfidf_baseline.ipynb` Cell 4 | `CLASSIFIER = 'sgd'` | keep |
| BERT model | `04_model_b_transformer.ipynb` Cell 4 | `Bio_ClinicalBERT` | keep |
| Training epochs | `04_model_b_transformer.ipynb` Cell 4 | `EPOCHS = 3` | keep |
| GCS MIMIC-IV path | `01_data_extraction.ipynb` Cell 6 | `mimiciv/3.1/hosp/` | update if different version |
