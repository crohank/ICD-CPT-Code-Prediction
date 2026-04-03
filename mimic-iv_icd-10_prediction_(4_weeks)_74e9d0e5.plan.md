---
name: MIMIC-IV ICD-10 Prediction (4 weeks)
overview: Build an end-to-end, 2-person pipeline to predict ICD-10 codes from MIMIC-IV discharge summaries using shared preprocessing and two distinct modeling approaches, with strong evaluation and a final report/demo.
todos:
  - id: access-download
    content: Obtain PhysioNet access and download MIMIC-IV Hospital + MIMIC-IV-Note datasets
    status: pending
  - id: cohort-join-splits
    content: Extract discharge summaries, join ICD-10 labels (icd_version=10), and create patient-level train/val/test splits
    status: pending
  - id: baseline-model-a
    content: Implement shared preprocessing + train/tune TF-IDF + linear multi-label baseline (Model A) with threshold tuning and metrics
    status: pending
  - id: transformer-model-b
    content: Implement transformer multi-label model (Model B) with long-text handling (sectioning/windowing), tuning, and evaluation
    status: pending
  - id: final-eval-report-demo
    content: Run final experiments, perform error analysis/head-tail breakdown, and produce report + demo artifact
    status: pending
isProject: false
---

# Goal

Build an ICD‑10 multi‑label classifier from **MIMIC‑IV discharge summaries**. Shared data pipeline + two different models (one per team member).

# Data (exact sources)

- [MIMIC-IV Hospital](https://physionet.org/content/mimiciv/) (ICD tables: `diagnoses_icd`, `procedures_icd`; use `icd_version = 10`).
- [MIMIC-IV Note](https://physionet.org/content/mimic-iv-note/) (discharge summaries text).

# Prediction target (as requested)

- **Multi-label, full ICD‑10 code set** present in MIMIC‑IV.
- Practical note: the label space is extremely long‑tailed; plan includes:
  - **Primary**: “full-label” training/eval with frequency thresholds + calibrated output.
  - **Fallback/secondary** (kept in the same pipeline): Top‑K subset experiments (Top‑50/100/500) to ensure strong results if full‑label performance is weak.

# End-to-end pipeline design

## 1) Cohort construction + label join

- Extract discharge summaries from Notes.
- Join notes to admissions/encounters and to ICD‑10 codes:
  - Use patient/admission identifiers consistent across modules (commonly `subject_id`, `hadm_id`; some note tables also include `stay_id`).
  - For labels, pull codes from `diagnoses_icd` and/or `procedures_icd` filtered to ICD‑10 (`icd_version = 10`).
- Define one supervised example per discharge summary:
  - **Input**: discharge summary text.
  - **Output**: set of ICD‑10 codes assigned for that admission.

## 2) Shared preprocessing

- Text cleaning: de-identification artifacts handling, normalize whitespace, optional section filtering (e.g., keep “HOSPITAL COURSE”, “DISCHARGE DIAGNOSES”, etc. if you choose).
- Tokenization strategy depends on model:
  - For classical baseline: word/character n‑grams with TF‑IDF.
  - For neural: subword tokenizer (transformer).
- Label processing:
  - Build label vocabulary from all ICD‑10 codes in training set.
  - Handle rare codes with:
    - Minimum frequency threshold for training stability (keep full-label goal by reporting coverage and optionally a separate “rare-code bucket” analysis), and/or
    - Class weighting / focal loss.

## 3) Splits (to avoid leakage)

- Split by **patient** (`subject_id`) into train/val/test to prevent same patient appearing across splits.
- Keep a fixed random seed and export split files (CSV/JSON) for reproducibility.

## 4) Two-model requirement (two distinct algorithms)

- **Model A (Member 1)**: strong, interpretable baseline
  - One-vs-rest Logistic Regression or Linear SVM on TF‑IDF features.
  - Optional: Class weights, threshold tuning per label.
  - Strength: fast, transparent, strong baseline.
- **Model B (Member 2)**: transformer multi-label classifier
  - Fine-tune a clinical/biomedical transformer (e.g., ClinicalBERT/Bio+Clinical variants) with sigmoid output layer.
  - Long text handling options (pick one): truncation with smart sectioning, sliding window + max pooling, or long-context model if available.
  - Strength: better semantic understanding; likely higher recall.

## 5) Thresholding and prediction post-processing

- Convert probabilities to code sets using:
  - Global threshold tuned on validation for micro-F1, and/or
  - Per-label thresholds tuned on validation, and/or
  - Top‑N predicted labels per note (N tuned).

## 6) Evaluation (report both “full” and “Top‑K” views)

- Core metrics for multi-label ICD prediction:
  - **micro-F1**, **macro-F1**
  - **precision@k / recall@k**
  - AUROC and/or AUPRC (micro + macro; AUPRC often more informative with imbalance)
- Stratified analysis:
  - Performance vs label frequency (head vs tail).
  - Error analysis: common confusions; qualitative review of 20–50 samples.

## 7) Deliverables

- Reproducible pipeline (notebooks or scripts) that:
  - Extracts cohort, builds splits, preprocesses text/labels
  - Trains both models, evaluates, saves predictions
- Final report:
  - Data description, label distribution, modeling choices
  - Results table for (full-label + Top‑K)
  - Error analysis + limitations
- Demo:
  - CLI or notebook: input discharge summary → output top predicted ICD‑10 codes with probabilities.

# 4-week timeline (2 members)

## Week 1 — Data + pipeline foundation

- Get PhysioNet access; download MIMIC‑IV Hospital + Note.
- Build extraction scripts/notebooks:
  - Discharge summary selection
  - ICD‑10 label join
  - Patient-level splits
- Exploratory analysis:
  - Note length distribution
  - Label cardinality per note
  - Label frequency long-tail plots
- Freeze dataset artifacts (train/val/test files).

## Week 2 — Shared preprocessing + Baseline (Model A)

- Implement text preprocessing + TF‑IDF vectorization.
- Train and tune linear baseline:
  - Class weights, regularization
  - Threshold tuning
- Establish metric suite + evaluation harness.
- Produce initial results and sanity checks.

## Week 3 — Transformer model (Model B) + long-text strategy

- Implement transformer fine-tuning pipeline.
- Decide and implement long-text handling (section-based truncation or windowing).
- Tune training (batching, learning rate, epochs) and thresholding.
- Compare against baseline; run ablations (windowing vs truncation, etc.).

## Week 4 — Robustness, error analysis, write-up, demo

- Run final training with best settings for both models.
- Perform:
  - Head/tail analysis
  - Calibration/threshold sensitivity
  - Qualitative review of errors
- Package demo + reproducibility (requirements, run scripts).
- Write final report + slides.

# Suggested repo structure

- `data/` (ignored in git) for raw extracts
- `datasets/` for processed splits (CSV/Parquet)
- `src/` for preprocessing, modeling, evaluation
- `notebooks/` for EDA and results
- `reports/` for figures + final writeup

# Risks + mitigation

- Full ICD‑10 is very large and imbalanced → keep Top‑K experiments as a safety net, report coverage, and emphasize micro-F1/recall@k.
- Long discharge summaries → section filtering or sliding windows.
- Compute constraints → baseline always works; transformer can be run with smaller models and fewer epochs.
