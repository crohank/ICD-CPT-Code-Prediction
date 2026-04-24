"""
Streamlit UI for ICD-10 code prediction.

This app is intentionally minimal: pick a model, paste or choose sample text,
run predict, read the table. Thresholds are not user-tunable here — they are
read from the same JSON files the training notebooks write (`test_results.json`,
`results.json`, ensemble config, etc.) so what you see matches offline eval.

Run (from repo root):
    streamlit run demo/streamlit_app.py
Or use `python demo.py` → dashboard, which adds flags to avoid noisy optional
imports from Streamlit's file watcher.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import re
import sys
from pathlib import Path

import torch

# Repo root on `sys.path` so `import src...` works like the training scripts.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR   = PROJECT_ROOT / "data" / "models"
ALT_MODELS   = PROJECT_ROOT / "models"


def _resolve(primary: Path) -> Path:
    """Return primary if it exists, else try same relative path under models/."""
    if primary.exists():
        return primary
    try:
        rel = primary.relative_to(MODELS_DIR)
        alt = ALT_MODELS / rel
        if alt.exists():
            return alt
    except ValueError:
        pass
    return primary


# Friendly blurbs for the results table (codes themselves come from `mlb.classes_`).
ICD10_DESC = {
    "E119":  "Type 2 diabetes mellitus without complications",
    "I10":   "Essential primary hypertension",
    "E780":  "Pure hypercholesterolemia",
    "E785":  "Hyperlipidemia unspecified",
    "Z87891":"Personal history of nicotine dependence",
    "I2510": "Atherosclerotic heart disease of native coronary artery",
    "Z7901": "Long term use of anticoagulants",
    "N179":  "Acute kidney failure unspecified",
    "E1165": "Type 2 diabetes with hyperglycemia",
    "Z79899":"Other long term drug therapy",
    "I4891": "Unspecified atrial fibrillation",
    "Z7982": "Long term use of aspirin",
    "I509":  "Heart failure unspecified",
    "J449":  "COPD unspecified",
    "Z66":   "Do not resuscitate status",
    "E1122": "Type 2 diabetes with diabetic chronic kidney disease",
    "I2699": "Other pulmonary embolism",
    "N189":  "Chronic kidney disease unspecified",
    "D649":  "Anemia unspecified",
    "K219":  "GERD without esophagitis",
    "I480":  "Paroxysmal atrial fibrillation",
    "Z794":  "Long term use of insulin",
    "N390":  "Urinary tract infection",
    "G4733": "Obstructive sleep apnea",
    "J9601": "Acute respiratory failure with hypoxia",
    "E876":  "Hypokalemia",
    "I482":  "Chronic atrial fibrillation",
    "Z951":  "Presence of aortocoronary bypass graft",
    "F329":  "Major depressive disorder",
    "I350":  "Nonrheumatic aortic valve stenosis",
    "Z930":  "Tracheostomy status",
    "I5020": "Unspecified systolic heart failure",
    "J189":  "Pneumonia unspecified organism",
    "Z8546": "Personal history of prostate neoplasm",
    "G8929": "Other chronic pain",
    "E039":  "Hypothyroidism unspecified",
    "I4892": "Unspecified atrial flutter",
    "Z853":  "Personal history of breast neoplasm",
    "Z9811": "Acquired absence of right knee joint",
    "I252":  "Old myocardial infarction",
    "B9620": "E. coli infection",
    "J9600": "Acute respiratory failure with hypercapnia",
    "R6520": "Severe sepsis without septic shock",
    "K5900": "Constipation unspecified",
    "N183":  "Chronic kidney disease stage 3",
    "N184":  "Chronic kidney disease stage 4",
    "I5032": "Chronic diastolic heart failure",
    "E669":  "Obesity unspecified",
    "E8342": "Hyposmolality / hyponatremia",
}

st.set_page_config(
    page_title="ICD-10 Code Predictor",
    page_icon="🏥",
    layout="wide",
)

# Cached JSON / threshold helpers — keeps reruns fast when you change widgets.

@st.cache_data
def load_ensemble_config():
    cfg_path = MODELS_DIR / "ensemble" / "ensemble_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return None


def clean_text(text):
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,;:\-/]', ' ', text)
    text = re.sub(r'[\s\n\r\t]+', ' ', text).strip()
    return text


def get_fixed_threshold(selected_model: str) -> float:
    """
    Thresholds are fixed to the values saved by the training/eval notebooks.
    """
    if selected_model == "Model A (TF-IDF + SGD)":
        path = _resolve(MODELS_DIR / "model_a" / "results.json")
        if not path.exists():
            raise FileNotFoundError(str(path))
        with open(path) as f:
            d = json.load(f)
        return float(d["test"]["threshold"])

    if selected_model == "Model B (ClinicalBERT)":
        path = _resolve(MODELS_DIR / "model_b" / "test_results.json")
        if not path.exists():
            raise FileNotFoundError(str(path))
        with open(path) as f:
            d = json.load(f)
        return float(d["threshold"])

    if selected_model == "Model C v1 (Chunk+Attn)":
        path = _resolve(MODELS_DIR / "model_c" / "test_results.json")
        if not path.exists():
            raise FileNotFoundError(str(path))
        with open(path) as f:
            d = json.load(f)
        return float(d["Threshold"])

    if selected_model == "Model C v2 (Fixed+Focal)":
        path = _resolve(MODELS_DIR / "model_c" / "v2" / "test_results.json")
        if not path.exists():
            raise FileNotFoundError(str(path))
        with open(path) as f:
            d = json.load(f)
        # Notebook saves both global + per-label; Streamlit uses the global threshold.
        return float(d["global_threshold"]["Threshold"])

    if selected_model == "Model D (BiLSTM-LAAT)":
        path = _resolve(MODELS_DIR / "model_d" / "test_results.json")
        if not path.exists():
            raise FileNotFoundError(str(path))
        with open(path) as f:
            d = json.load(f)
        return float(d["Threshold"])

    if selected_model == "Ensemble v4 (A+D, best)":
        cfg = load_ensemble_config()
        if not cfg or "ensemble_v4" not in cfg:
            raise FileNotFoundError(str(MODELS_DIR / "ensemble" / "ensemble_config.json"))
        return float(cfg["ensemble_v4"]["threshold"])

    raise ValueError(f"Unknown model selection: {selected_model}")


# Each `try_load_*` mirrors the artifact layout from the notebooks; paths go through
# `_resolve` so weights can live under `data/models/...` or the legacy `models/...` tree.

@st.cache_resource
def try_load_model_a(data_path):
    """Try loading Model A artifacts. Returns (vec, clf, mlb) or Nones."""
    try:
        dp = Path(data_path)
        with open(dp / "tfidf_vectorizer.pkl", "rb") as f:
            vec = pickle.load(f)
        with open(dp / "mlb.pkl", "rb") as f:
            mlb = pickle.load(f)
        clf_path = _resolve(MODELS_DIR / "model_a" / "clf_sgd.pkl")
        if not clf_path.exists():
            return vec, None, mlb
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
        return vec, clf, mlb
    except Exception:
        return None, None, None


@st.cache_resource
def try_load_model_b(data_path):
    """Load Model B (ClinicalBERT). Returns (model, tokenizer, mlb, error_msg)."""
    try:
        from transformers import AutoTokenizer
        from src.models import ICDClassifier
        from src.config import TRANSFORMER_MODEL

        dp = Path(data_path)
        with open(dp / "mlb.pkl", "rb") as f:
            mlb = pickle.load(f)

        num_labels = len(mlb.classes_)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wt_path = _resolve(MODELS_DIR / "model_b" / "best_model.pt")
        if not wt_path.exists():
            return None, None, None, f"Weight file not found: `{wt_path}`"

        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
        model = ICDClassifier(TRANSFORMER_MODEL, num_labels=num_labels)
        model.load_state_dict(torch.load(wt_path, map_location=device, weights_only=True))
        model.to(device).eval()

        return model, tokenizer, mlb, None
    except Exception as e:
        return None, None, None, str(e)


@st.cache_resource
def try_load_model_c(data_path, version="v2"):
    """Load Model C (Chunk+Attn). Returns (model, tokenizer, mlb, temperature, error_msg)."""
    try:
        from transformers import AutoTokenizer
        from src.models import LabelAttentionClassifier
        from src.config import TRANSFORMER_MODEL, MODEL_C_MAX_CHUNKS

        dp = Path(data_path)
        with open(dp / "mlb.pkl", "rb") as f:
            mlb = pickle.load(f)

        num_labels = len(mlb.classes_)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if version == "v2":
            wt_path = _resolve(MODELS_DIR / "model_c" / "v2" / "best_model.pt")
            temp_path = _resolve(MODELS_DIR / "model_c" / "v2" / "temperature.json")
        else:
            wt_path = _resolve(MODELS_DIR / "model_c" / "best_model.pt")
            temp_path = None

        if not wt_path.exists():
            return None, None, None, 1.0, f"Weight file not found: `{wt_path}`"

        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
        model = LabelAttentionClassifier(
            TRANSFORMER_MODEL, num_labels=num_labels,
            max_chunks=MODEL_C_MAX_CHUNKS, freeze_bert=False,
        )
        model.load_state_dict(torch.load(wt_path, map_location=device, weights_only=True))
        model.to(device).eval()

        temperature = 1.0
        if temp_path and temp_path.exists():
            with open(temp_path) as f:
                temperature = json.load(f)["temperature"]

        return model, tokenizer, mlb, temperature, None
    except Exception as e:
        return None, None, None, 1.0, str(e)


@st.cache_resource
def try_load_model_d(data_path):
    """Load Model D (BiLSTM-LAAT). Returns (model, word2idx, mlb, error_msg)."""
    try:
        from src.models import BiLSTMLAAT

        dp = Path(data_path)
        with open(dp / "mlb.pkl", "rb") as f:
            mlb = pickle.load(f)
        with open(dp / "word_vocab.pkl", "rb") as f:
            word2idx = pickle.load(f)

        num_labels = len(mlb.classes_)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        wt_path = _resolve(MODELS_DIR / "model_d" / "best_model.pt")
        if not wt_path.exists():
            return None, None, None, f"Weight file not found: `{wt_path}`"

        model = BiLSTMLAAT(vocab_size=len(word2idx), num_labels=num_labels)
        model.load_state_dict(torch.load(wt_path, map_location=device, weights_only=True))
        model.to(device).eval()

        return model, word2idx, mlb, None
    except Exception as e:
        return None, None, None, str(e)


def predict_b(text, model, tokenizer):
    """Run Model B inference on a single text."""
    cleaned = clean_text(text)
    device = next(model.parameters()).device
    enc = tokenizer(
        cleaned, max_length=512, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(ids, mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


def predict_c(text, model, tokenizer, temperature=1.0):
    """Run Model C inference on a single text (chunked)."""
    from src.config import MODEL_C_MAX_CHUNKS, MAX_SEQ_LEN, MODEL_C_CHUNK_STRIDE

    cleaned = clean_text(text)
    device = next(model.parameters()).device

    full_enc = tokenizer(cleaned, add_special_tokens=False, return_attention_mask=False)
    token_ids = full_enc["input_ids"]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    content_len = MAX_SEQ_LEN - 2
    stride = min(MODEL_C_CHUNK_STRIDE, content_len)

    chunks_ids, chunks_mask = [], []
    start = 0
    while start < len(token_ids) and len(chunks_ids) < MODEL_C_MAX_CHUNKS:
        end = min(start + content_len, len(token_ids))
        chunk = token_ids[start:end]
        ids = [cls_id] + chunk + [sep_id]
        mask = [1] * len(ids)
        pad_len = MAX_SEQ_LEN - len(ids)
        ids += [pad_id] * pad_len
        mask += [0] * pad_len
        chunks_ids.append(ids)
        chunks_mask.append(mask)
        if end >= len(token_ids):
            break
        start += stride

    chunk_count = len(chunks_ids)
    while len(chunks_ids) < MODEL_C_MAX_CHUNKS:
        chunks_ids.append([pad_id] * MAX_SEQ_LEN)
        chunks_mask.append([0] * MAX_SEQ_LEN)

    ids_t = torch.tensor([chunks_ids], dtype=torch.long).to(device)
    mask_t = torch.tensor([chunks_mask], dtype=torch.long).to(device)
    cc_t = torch.tensor([chunk_count], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(ids_t, mask_t, chunk_counts=cc_t)
        probs = torch.sigmoid(logits / temperature).cpu().numpy()[0]
    return probs


def predict_d(text, model, word2idx):
    """Run Model D inference on a single text (word-level)."""
    from src.config import MODEL_D_MAX_TOKENS

    cleaned = clean_text(text)
    words = cleaned.split()[:MODEL_D_MAX_TOKENS]
    ids = [word2idx.get(w, 1) for w in words]
    mask = [1] * len(ids)

    pad_len = MODEL_D_MAX_TOKENS - len(ids)
    ids += [0] * pad_len
    mask += [0] * pad_len

    device = next(model.parameters()).device
    ids_t = torch.tensor([ids], dtype=torch.long).to(device)
    mask_t = torch.tensor([mask], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(ids_t, mask_t)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


SAMPLE_NOTES = {
    "Heart Failure + Diabetes": (
        "Patient is a 72-year-old male with a history of type 2 diabetes mellitus, "
        "chronic kidney disease stage 3, and hypertension who presented with shortness "
        "of breath and bilateral lower extremity edema. Hospital course was notable for "
        "decompensated heart failure treated with IV furosemide. Echocardiogram showed "
        "ejection fraction of 35%. Patient was started on lisinopril and carvedilol. "
        "Hemoglobin A1c was 8.2%, insulin regimen was adjusted. "
        "Discharge diagnoses: acute on chronic systolic heart failure, type 2 diabetes "
        "mellitus with hyperglycemia, hypertensive heart disease, CKD stage 3, "
        "hyperlipidemia, obesity."
    ),
    "Pneumonia + COPD": (
        "82-year-old female with history of COPD on home oxygen presented with productive "
        "cough, fever 101.2F, and worsening dyspnea over 3 days. Chest X-ray showed right "
        "lower lobe consolidation. WBC 14.2. Started on IV ceftriaxone and azithromycin for "
        "community-acquired pneumonia. Hospital course complicated by acute exacerbation of "
        "COPD requiring nebulizer treatments q4h. Potassium was 3.2 on admission, repleted. "
        "Discharge diagnoses: community-acquired pneumonia, acute exacerbation of COPD, "
        "chronic respiratory failure with hypoxia, hypokalemia, GERD, depression."
    ),
    "Sepsis + UTI": (
        "68-year-old male with diabetes, end-stage renal disease on hemodialysis presented "
        "with altered mental status, fever 102.4F, and hypotension to 82/50. Blood cultures "
        "grew E. coli. Urine culture positive for E. coli >100K colonies. Lactate 4.1. "
        "Started on IV piperacillin-tazobactam, given 3L normal saline bolus. Transferred "
        "to ICU for vasopressor support. Improved over 72 hours, transitioned to oral "
        "ciprofloxacin. Discharge diagnoses: severe sepsis secondary to urinary tract "
        "infection, acute kidney injury on CKD, type 2 diabetes mellitus with diabetic "
        "chronic kidney disease, essential hypertension, anemia."
    ),
    "Cardiac Surgery": (
        "65-year-old male with a history of coronary artery disease, prior myocardial "
        "infarction, aortic stenosis, and atrial fibrillation on warfarin. Admitted for "
        "elective aortic valve replacement and coronary artery bypass grafting x3. "
        "Postoperative course was uncomplicated. Patient was extubated on POD 1. "
        "Chest tubes removed POD 3. Started on aspirin, metoprolol, and atorvastatin. "
        "INR was subtherapeutic, warfarin restarted. Physical therapy consulted. "
        "Discharge diagnoses: aortic stenosis s/p AVR, CAD s/p CABG x3, old MI, "
        "chronic atrial fibrillation, hypertension, hyperlipidemia, history of nicotine "
        "dependence."
    ),
}


def render_predictions(probs, vocab, top_n, threshold):
    """Render a prediction results table given probability array."""
    ranked = np.argsort(probs)[::-1][:top_n]
    n_predicted = sum(1 for i in ranked if probs[i] >= threshold)
    st.subheader(f"Predictions — {n_predicted} codes above threshold ({threshold:.3f})")

    rows = []
    for idx in ranked:
        code = vocab[idx]
        prob = float(probs[idx])
        desc = ICD10_DESC.get(code, f"ICD-10 {code}")
        rows.append({
            "Predicted": "Yes" if prob >= threshold else "No",
            "ICD-10 Code": code,
            "Description": desc,
            "Probability": prob,
        })

    df_results = pd.DataFrame(rows)
    st.dataframe(
        df_results.style
            .bar(subset=["Probability"], color="#5fba7d", vmin=0, vmax=1)
            .format({"Probability": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
        height=min(40 + len(rows) * 35, 600),
    )

    predicted_codes = [vocab[i] for i in ranked if probs[i] >= threshold]
    if predicted_codes:
        st.markdown("**Predicted codes:** " + "  ".join(f"`{c}`" for c in predicted_codes))


# --- Main layout: model picker + threshold (read-only) + text in + predict ---

st.title("ICD-10 Code Prediction")

MODEL_OPTIONS = [
    "Model A (TF-IDF + SGD)",
    "Model B (ClinicalBERT)",
    "Model C v1 (Chunk+Attn)",
    "Model C v2 (Fixed+Focal)",
    "Model D (BiLSTM-LAAT)",
    "Ensemble v4 (A+D, best)",
]

col_input, col_settings = st.columns([2, 1])

with col_settings:
    st.markdown("**Model**")
    selected_model = st.selectbox("Choose model", MODEL_OPTIONS, label_visibility="collapsed")
    top_n = st.slider("Top N codes to show", 5, 50, 15)

    try:
        threshold = get_fixed_threshold(selected_model)
    except Exception as e:
        st.error(
            "Could not load the fixed decision threshold from your saved evaluation JSON files.\n\n"
            f"**Error:** `{e}`\n\n"
            "Re-run the model evaluation notebooks so `test_results.json` / `results.json` / "
            "`ensemble_config.json` exist under `data/models/`."
        )
        st.stop()
    st.metric("Decision threshold (fixed)", f"{threshold:.3f}")

    st.divider()
    st.markdown("**Datasets path**")
    data_path = st.text_input(
        "Path to datasets/processed/",
        value=str(PROJECT_ROOT / "datasets" / "processed"),
        help="Directory with mlb.pkl, tfidf_vectorizer.pkl, cohort_*_clean.parquet, etc.",
    )

with col_input:
    selected_sample = st.selectbox(
        "Load a sample note:", ["Custom"] + list(SAMPLE_NOTES.keys())
    )
    default_text = SAMPLE_NOTES.get(selected_sample, "")
    text_input = st.text_area(
        "Enter discharge summary:",
        value=default_text,
        height=280,
        placeholder="Paste a clinical discharge summary here...",
    )

if st.button("Predict ICD-10 Codes", type="primary", use_container_width=True):
    if not text_input or len(text_input.strip()) < 10:
        st.warning("Please enter a discharge summary (at least 10 characters).")
        st.stop()

    # Model A: sklearn TF-IDF → sparse probs (fast baseline).
    if selected_model == "Model A (TF-IDF + SGD)":
        vec, clf, mlb = try_load_model_a(data_path)
        if vec is None:
            st.error(
                f"Could not load Model A artifacts.\n\n"
                f"Make sure these files exist:\n"
                f"- `{data_path}/tfidf_vectorizer.pkl`\n"
                f"- `{data_path}/mlb.pkl`\n"
                f"- `data/models/model_a/clf_sgd.pkl`\n\n"
                f"These are created by notebook `03_model_a_tfidf_baseline_local.ipynb`."
            )
            st.stop()
        if clf is None:
            st.error(
                f"`clf_sgd.pkl` not found in `data/models/model_a/`.\n\n"
                f"Run notebook `03_model_a_tfidf_baseline_local.ipynb` to train and save the classifier."
            )
            st.stop()

        vocab = list(mlb.classes_)
        cleaned = clean_text(text_input)
        x = vec.transform([cleaned])
        probs = clf.predict_proba(x)[0]

        st.divider()
        render_predictions(probs, vocab, top_n, threshold)

    # Model B: single-window ClinicalBERT + linear head.
    elif selected_model == "Model B (ClinicalBERT)":
        with st.spinner("Loading Model B (ClinicalBERT)..."):
            model_b, tokenizer_b, mlb_b, err_b = try_load_model_b(data_path)
        if err_b:
            st.error(
                f"Could not load Model B.\n\n**Error:** {err_b}\n\n"
                f"Make sure `{data_path}/mlb.pkl` exists and "
                f"`data/models/model_b/best_model.pt` is present.\n\n"
                f"Created by notebook `04_model_b_transformer_local.ipynb`."
            )
            st.stop()

        vocab = list(mlb_b.classes_)
        with st.spinner("Running inference..."):
            probs = predict_b(text_input, model_b, tokenizer_b)

        st.divider()
        render_predictions(probs, vocab, top_n, threshold)

    # Model C v1: first chunk+attention checkpoint (root `model_c/`).
    elif selected_model == "Model C v1 (Chunk+Attn)":
        with st.spinner("Loading Model C v1 (Chunk+Attn)..."):
            model_c, tok_c, mlb_c, temp_c, err_c = try_load_model_c(data_path, version="v1")
        if err_c:
            st.error(
                f"Could not load Model C v1.\n\n**Error:** {err_c}\n\n"
                f"Make sure `{data_path}/mlb.pkl` exists and "
                f"`data/models/model_c/best_model.pt` is present.\n\n"
                f"Created by notebook `06_model_c_training.ipynb`."
            )
            st.stop()

        vocab = list(mlb_c.classes_)
        with st.spinner("Running inference (chunked BERT)..."):
            probs = predict_c(text_input, model_c, tok_c, temperature=temp_c)

        st.divider()
        render_predictions(probs, vocab, top_n, threshold)

    # Model C v2: focal loss + architecture fixes under `model_c/v2/`.
    elif selected_model == "Model C v2 (Fixed+Focal)":
        with st.spinner("Loading Model C v2 (Fixed+Focal)..."):
            model_c2, tok_c2, mlb_c2, temp_c2, err_c2 = try_load_model_c(data_path, version="v2")
        if err_c2:
            st.error(
                f"Could not load Model C v2.\n\n**Error:** {err_c2}\n\n"
                f"Make sure `{data_path}/mlb.pkl` exists and "
                f"`data/models/model_c/v2/best_model.pt` is present.\n\n"
                f"Created by notebook `06_model_c_training_v2.ipynb`."
            )
            st.stop()

        vocab = list(mlb_c2.classes_)
        with st.spinner("Running inference (chunked BERT v2)..."):
            probs = predict_c(text_input, model_c2, tok_c2, temperature=temp_c2)

        st.divider()
        render_predictions(probs, vocab, top_n, threshold)

    # Model D: word-level BiLSTM + LAAT (needs `word_vocab.pkl`).
    elif selected_model == "Model D (BiLSTM-LAAT)":
        with st.spinner("Loading Model D (BiLSTM-LAAT)..."):
            model_d, word2idx, mlb_d, err_d = try_load_model_d(data_path)
        if err_d:
            st.error(
                f"Could not load Model D.\n\n**Error:** {err_d}\n\n"
                f"Make sure these files exist:\n"
                f"- `{data_path}/mlb.pkl`\n"
                f"- `{data_path}/word_vocab.pkl`\n"
                f"- `data/models/model_d/best_model.pt`\n\n"
                f"Created by notebook `08_model_d_bilstm_local.ipynb`."
            )
            st.stop()

        vocab = list(mlb_d.classes_)
        with st.spinner("Running inference (BiLSTM)..."):
            probs = predict_d(text_input, model_d, word2idx)

        st.divider()
        render_predictions(probs, vocab, top_n, threshold)

    # Ensemble v4: convex blend of A and D probs; weights from ensemble JSON.
    elif selected_model == "Ensemble v4 (A+D, best)":
        ens_cfg = load_ensemble_config()
        if ens_cfg is None or "ensemble_v4" not in ens_cfg:
            st.error("Ensemble config not found at `data/models/ensemble/ensemble_config.json`.")
            st.stop()

        v4 = ens_cfg["ensemble_v4"]
        w_a, w_d = v4["weight_A"], v4["weight_D"]

        # Load Model A
        vec_a, clf_a, mlb_a = try_load_model_a(data_path)
        if vec_a is None or clf_a is None:
            st.error(
                f"Ensemble requires Model A.\n\n"
                f"Make sure `{data_path}/tfidf_vectorizer.pkl`, `{data_path}/mlb.pkl`, "
                f"and `data/models/model_a/clf_sgd.pkl` exist."
            )
            st.stop()

        # Load Model D
        with st.spinner("Loading Ensemble v4 (Model A + Model D)..."):
            model_d, word2idx, mlb_d, err_d = try_load_model_d(data_path)
        if err_d:
            st.error(
                f"Ensemble requires Model D.\n\n**Error:** {err_d}\n\n"
                f"Make sure `{data_path}/mlb.pkl`, `{data_path}/word_vocab.pkl`, "
                f"and `data/models/model_d/best_model.pt` exist."
            )
            st.stop()

        vocab = list(mlb_a.classes_)
        with st.spinner("Running ensemble inference (A + D)..."):
            cleaned = clean_text(text_input)
            x_a = vec_a.transform([cleaned])
            probs_a = clf_a.predict_proba(x_a)[0]
            probs_d_arr = predict_d(text_input, model_d, word2idx)
            probs = w_a * probs_a + w_d * probs_d_arr

        st.divider()
        st.info(f"Ensemble v4: **{w_a:.0%}** Model A + **{w_d:.0%}** Model D")
        render_predictions(probs, vocab, top_n, threshold)

st.divider()
st.caption(
    "CS6120 NLP — ICD-10 Code Prediction from MIMIC-IV Discharge Summaries | "
    "Models: TF-IDF + SGD, Bio_ClinicalBERT, Chunk-Based Label Attention, BiLSTM-LAAT"
)
