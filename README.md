# ICD-10 Code Prediction from MIMIC-IV Discharge Summaries

End-to-end NLP pipeline for multi-label ICD-10 diagnosis code prediction from clinical discharge notes.

The project compares multiple model families and serves predictions through a FastAPI backend with a Streamlit demo UI.

## What This Project Includes

- **Model A**: TF-IDF + One-vs-Rest SGD baseline
- **Model B**: Bio_ClinicalBERT with smart truncation
- **Model C**: Chunk-based Bio_ClinicalBERT + label-wise attention
- **Model D**: BiLSTM-LAAT experimental model
- **Ensemble**: weighted combination of Model A + Model C
- **Serving stack**: FastAPI API, Streamlit demo, Docker support
- **Tests**: unit tests for data processing, schemas, and model components

## Key Results (Top-50 ICD-10 Labels)

From the final evaluation pipeline:

- Ensemble (A+C): **Micro-F1 = 0.6249** (best)
- Model A (TF-IDF): Micro-F1 = 0.5952
- Model C (Chunk+Attention): Micro-F1 = 0.5305
- Model B (ClinicalBERT): Micro-F1 = 0.5242

See `report.md` for full tables and analysis.

## Repository Structure

- `src/` - training, evaluation, data prep, configs, explainability utilities
- `api/` - FastAPI app, schemas, model service
- `demo/` - Streamlit application
- `notebooks_local/` - experiment notebooks
- `tests/` - pytest test suite
- `data/models/` - model artifacts and evaluation outputs
- `datasets/processed/` - processed splits and serialized preprocessing artifacts

## Data

This project is built on **MIMIC-IV** clinical data.

- Raw MIMIC-IV source files are not distributed in this repository
- You must have authorized access to MIMIC-IV and generate local processed datasets
- Keep all PHI-protected or institution-specific files out of version control

## Quick Start

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements-api.txt
pip install -r requirements-demo.txt
```

### 2) Run API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints:

- `POST /predict` - predict ICD-10 codes from input text
- `GET /health` - model/server health check
- `GET /model/info` - loaded models, thresholds, label vocab metadata

### 3) Run Streamlit Demo

In a second terminal:

```bash
streamlit run demo/streamlit_app.py
```

Open `http://localhost:8501`.

## Testing

```bash
pytest -q
```

## Docker

Build and run:

```bash
docker compose up --build
```

This brings up:

- API on `http://localhost:8000`
- Streamlit demo on `http://localhost:8501`

## Notes for Public Repo Hygiene

- `.gitignore` is configured to exclude local planning docs, local cloud setup notes, local assistant settings, and common secret/config patterns
- If you newly add ignore rules for already-tracked files, untrack them once with:

```bash
git ls-files -ci --exclude-standard -z | xargs -0 git rm --cached
```

## Citation / Course Context

Course project for CS6120 (NLP): ICD-10 coding from discharge summaries with baseline-to-production progression.
