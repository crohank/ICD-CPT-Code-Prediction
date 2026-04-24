"""
Streamlit dashboard for ICD-10 Code Prediction project.

Run with:
    streamlit run demo/streamlit_app.py
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

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = DATA_DIR / "models"
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


MODEL_DIRS = {
    "Model A (TF-IDF + SGD)":     MODELS_DIR / "model_a",
    "Model B (ClinicalBERT)":     MODELS_DIR / "model_b",
    "Model C v1 (Chunk+Attn)":    MODELS_DIR / "model_c",
    "Model C v2 (Fixed+Focal)":   MODELS_DIR / "model_c" / "v2",
    "Model D (BiLSTM-LAAT)":      MODELS_DIR / "model_d",
}

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

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICD-10 Code Predictor",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar navigation ───────────────────────────────────────────────
st.sidebar.title("🏥 ICD-10 Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Predict", "Model Details", "Training Curves", "Confusion Matrices", "EDA"],
)
st.sidebar.divider()
st.sidebar.caption("CS6120 NLP Final Project\nICD-10 Code Prediction from MIMIC-IV")


# ── Helpers ───────────────────────────────────────────────────────────
@st.cache_data
def load_comparison():
    csv_path = MODELS_DIR / "ensemble" / "final_comparison_all_models.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_ensemble_config():
    cfg_path = MODELS_DIR / "ensemble" / "ensemble_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return None


def find_images(directory, pattern="*.png"):
    p = Path(directory)
    if not p.exists():
        return []
    return sorted(p.glob(pattern))


def clean_text(text):
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,;:\-/]', ' ', text)
    text = re.sub(r'[\s\n\r\t]+', ' ', text).strip()
    return text


# ── Model loading ────────────────────────────────────────────────────
# Looks for weight files that the notebooks produce.
# Model A: datasets/processed/tfidf_vectorizer.pkl + data/models/model_a/clf_sgd.pkl + datasets/processed/mlb.pkl
# Model D: datasets/processed/word_vocab.pkl + data/models/model_d/best_model.pt + datasets/processed/mlb.pkl
# Etc.

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
    st.subheader(f"Predictions — {n_predicted} codes above threshold ({threshold:.2f})")

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


# ══════════════════════════════════════════════════════════════════════
#  PAGE: Predict
# ══════════════════════════════════════════════════════════════════════
if page == "Predict":
    st.title("🏥 ICD-10 Code Prediction")

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
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.025)

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

        dp = Path(data_path)

        # ── Model A ──────────────────────────────────────────────
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

        # ── Model B (ClinicalBERT) ───────────────────────────────
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

        # ── Model C v1 (Chunk+Attn) ─────────────────────────────
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

        # ── Model C v2 (Fixed+Focal) ────────────────────────────
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

        # ── Model D (BiLSTM-LAAT) ───────────────────────────────
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

        # ── Ensemble v4 (A+D, best) ─────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════
#  PAGE: Overview
# ══════════════════════════════════════════════════════════════════════
elif page == "Overview":
    st.title("Model Comparison Overview")
    st.markdown(
        "Side-by-side performance of all models and ensembles on the **test set** "
        "(50-label ICD-10 multi-label classification on MIMIC-IV discharge summaries)."
    )

    df = load_comparison()
    if df is None:
        st.error("Comparison CSV not found. Run the ensemble evaluation notebook first.")
        st.stop()

    st.subheader("Performance Table")
    st.dataframe(
        df.style
          .highlight_max(subset=["Micro-F1", "Macro-F1", "Micro-Prec", "Micro-Rec", "Macro-AUPRC", "Micro-AUROC"], color="#c6efce")
          .format({
              "Threshold": "{:.3f}",
              "Micro-F1": "{:.4f}", "Macro-F1": "{:.4f}",
              "Micro-Prec": "{:.4f}", "Micro-Rec": "{:.4f}",
              "Macro-AUPRC": "{:.4f}", "Micro-AUROC": "{:.4f}",
          }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Micro-F1 Comparison")
    chart_df = df[["Model", "Micro-F1"]].set_index("Model").sort_values("Micro-F1", ascending=True)
    st.bar_chart(chart_df, horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Precision vs Recall")
        pr_df = df[["Model", "Micro-Prec", "Micro-Rec"]].set_index("Model")
        st.bar_chart(pr_df)
    with col2:
        st.subheader("AUROC Comparison")
        auc_df = df[["Model", "Micro-AUROC"]].set_index("Model").sort_values("Micro-AUROC", ascending=True)
        st.bar_chart(auc_df, horizontal=True)

    bars_png = MODELS_DIR / "ensemble" / "model_comparison_bars.png"
    if bars_png.exists():
        st.divider()
        st.subheader("Detailed Comparison (from Ensemble Notebook)")
        st.image(str(bars_png), use_container_width=True)

    ens_cfg = load_ensemble_config()
    if ens_cfg:
        st.divider()
        st.subheader("Ensemble Configurations")
        best_ens = ens_cfg.get("best_ensemble", "N/A")
        st.success(f"**Best ensemble:** {best_ens}")
        for key in ["ensemble_v1", "ensemble_v2", "ensemble_v3", "ensemble_v4", "ensemble_v5"]:
            if key in ens_cfg:
                with st.expander(key.replace("_", " ").title()):
                    st.json(ens_cfg[key])

# ══════════════════════════════════════════════════════════════════════
#  PAGE: Model Details
# ══════════════════════════════════════════════════════════════════════
elif page == "Model Details":
    st.title("Individual Model Details")

    selected = st.selectbox("Select model", list(MODEL_DIRS.keys()))
    model_dir = MODEL_DIRS[selected]

    if not model_dir.exists():
        st.warning(f"Directory not found: `{model_dir}`")
        st.stop()

    results_files = list(model_dir.glob("*results*.json"))
    if results_files:
        st.subheader("Test Results")
        for rf in results_files:
            with open(rf) as f:
                results = json.load(f)
            with st.expander(rf.name, expanded=True):
                if isinstance(results, dict):
                    flat = {}
                    for k, v in results.items():
                        if isinstance(v, dict):
                            for k2, v2 in v.items():
                                flat[f"{k} / {k2}"] = v2
                        else:
                            flat[k] = v
                    cols = st.columns(min(len(flat), 4))
                    for i, (k, v) in enumerate(flat.items()):
                        with cols[i % 4]:
                            if isinstance(v, float):
                                st.metric(k, f"{v:.4f}")
                            else:
                                st.metric(k, str(v))

    head_tail = list(model_dir.glob("head_tail_f1*.png"))
    if head_tail:
        st.subheader("Head vs Tail Label Performance")
        for img in head_tail:
            st.image(str(img), use_container_width=True)

    all_pngs = find_images(model_dir)
    other_pngs = [p for p in all_pngs if "head_tail" not in p.name and "training" not in p.name]
    if other_pngs:
        st.subheader("Other Plots")
        for img in other_pngs:
            st.image(str(img), caption=img.stem, use_container_width=True)

    # comparison CSVs if they exist
    csv_files = list(model_dir.glob("*comparison*.csv"))
    for cf in csv_files:
        st.subheader(cf.stem.replace("_", " ").title())
        st.dataframe(pd.read_csv(cf), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
#  PAGE: Training Curves
# ══════════════════════════════════════════════════════════════════════
elif page == "Training Curves":
    st.title("Training Curves")

    selected = st.selectbox("Select model", list(MODEL_DIRS.keys()))
    model_dir = MODEL_DIRS[selected]

    if not model_dir.exists():
        st.warning(f"Directory not found: `{model_dir}`")
        st.stop()

    history_files = list(model_dir.glob("*training_history*.csv"))
    if history_files:
        for hf in sorted(history_files):
            st.subheader(f"Training History ({hf.name})")
            hist = pd.read_csv(hf)
            st.dataframe(hist, use_container_width=True, hide_index=True)

            if "train_loss" in hist.columns:
                st.line_chart(hist.set_index("epoch")["train_loss"], y_label="Loss", x_label="Epoch")

            f1_cols = [c for c in hist.columns if "f1" in c.lower()]
            if f1_cols:
                st.line_chart(hist.set_index("epoch")[f1_cols], y_label="F1", x_label="Epoch")

    curve_pngs = [p for p in find_images(model_dir) if "training" in p.name]
    if curve_pngs:
        st.subheader("Saved Training Plots")
        for img in curve_pngs:
            st.image(str(img), caption=img.stem, use_container_width=True)
    elif not history_files:
        st.info("No training history found for this model.")

# ══════════════════════════════════════════════════════════════════════
#  PAGE: Confusion Matrices
# ══════════════════════════════════════════════════════════════════════
elif page == "Confusion Matrices":
    st.title("Confusion Matrices")
    st.markdown(
        "Per-label 2x2 confusion matrices and 50-label confusion heatmaps. "
        "Generated when you run the evaluation cells in the notebooks."
    )

    found_any = False
    for name, mdir in MODEL_DIRS.items():
        if not mdir.exists():
            continue
        cm_images = [p for p in find_images(mdir) if "confusion" in p.name.lower()]
        if cm_images:
            found_any = True
            st.subheader(name)
            for img in cm_images:
                st.image(str(img), caption=img.stem, use_container_width=True)

    ens_dir = MODELS_DIR / "ensemble"
    if ens_dir.exists():
        cm_ens = [p for p in find_images(ens_dir) if "confusion" in p.name.lower()]
        if cm_ens:
            found_any = True
            st.subheader("Ensemble")
            for img in cm_ens:
                st.image(str(img), caption=img.stem, use_container_width=True)

    if not found_any:
        st.warning(
            "No confusion matrix images found yet. Run the notebooks to generate them.\n\n"
            "The confusion matrix code is in each model notebook under sections like "
            "**'Confusion Matrix (Per-Label)'** and **'Full 50-Label Confusion Analysis'**."
        )

# ══════════════════════════════════════════════════════════════════════
#  PAGE: EDA
# ══════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("Plots from the preprocessing and EDA notebook.")

    eda_images = find_images(DATA_DIR)
    eda_images = [p for p in eda_images if p.parent == DATA_DIR]

    if not eda_images:
        st.info("No EDA plots found in `data/`.")
    else:
        for img in eda_images:
            label = img.stem.replace("_", " ").title()
            st.subheader(label)
            st.image(str(img), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "CS6120 NLP — ICD-10 Code Prediction from MIMIC-IV Discharge Summaries | "
    "Models: TF-IDF + SGD, Bio_ClinicalBERT, Chunk-Based Label Attention, BiLSTM-LAAT"
)
