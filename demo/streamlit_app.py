"""
Streamlit demo for ICD-10 Code Prediction.

Run with:
    streamlit run demo/streamlit_app.py

Requires the FastAPI server to be running:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
import streamlit as st
import requests
import pandas as pd
import time

# ── Config ─────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Page setup ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICD-10 Code Predictor",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 ICD-10 Code Prediction from Discharge Summaries")
st.markdown(
    "Paste a clinical discharge summary below to predict ICD-10 diagnosis codes. "
    "Powered by an ensemble of TF-IDF and Chunk-Based BERT with Label Attention."
)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Top N codes", 5, 30, 10)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.025)
    explain = st.checkbox("Show evidence (attention weights)", value=True)

    st.divider()
    st.header("API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"Connected — {health['device'].upper()}")
        st.write(f"Models: {', '.join(health['models_loaded'])}")
        st.write(f"Labels: {health['num_labels']}")
    except Exception:
        st.error("API not reachable. Start with:\n`uvicorn api.app:app --port 8000`")

# ── Sample notes ───────────────────────────────────────────────────────
SAMPLE_NOTES = {
    "Heart Failure + Diabetes": """
Patient is a 72-year-old male with a history of type 2 diabetes mellitus, chronic kidney disease
stage 3, and hypertension who presented with shortness of breath and bilateral lower extremity
edema. Hospital course was notable for decompensated heart failure treated with IV furosemide.
Discharge diagnoses: acute on chronic systolic heart failure, type 2 diabetes mellitus,
hypertensive heart disease, CKD stage 3.
""",
    "Pneumonia + COPD": """
82-year-old female with history of COPD and chronic obstructive pulmonary disease on home oxygen
presented with productive cough, fever 101.2F, and worsening dyspnea. Chest X-ray showed right
lower lobe consolidation. Treated with IV ceftriaxone and azithromycin for community-acquired
pneumonia. Hospital course complicated by acute exacerbation of COPD requiring nebulizer treatments.
Discharge diagnoses: community-acquired pneumonia, acute exacerbation of COPD, chronic respiratory
failure with hypoxia.
""",
    "Sepsis + UTI": """
68-year-old male with diabetes, end-stage renal disease on hemodialysis presented with altered
mental status, fever, and hypotension. Blood cultures grew E. coli. Urine culture positive for
E. coli >100K. Treated with IV piperacillin-tazobactam then transitioned to oral ciprofloxacin.
Discharge diagnoses: sepsis secondary to urinary tract infection, acute kidney injury, type 2
diabetes mellitus with diabetic chronic kidney disease, essential hypertension.
""",
}

# ── Input ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Discharge Summary")
    selected_sample = st.selectbox(
        "Load sample note:", ["Custom"] + list(SAMPLE_NOTES.keys())
    )
    default_text = SAMPLE_NOTES.get(selected_sample, "")
    text_input = st.text_area(
        "Enter discharge summary:",
        value=default_text,
        height=300,
        placeholder="Paste a clinical discharge summary here...",
    )

with col2:
    st.subheader("Quick Info")
    st.info(
        "**How it works:**\n"
        "1. Text is cleaned and tokenized\n"
        "2. Split into overlapping 512-token chunks\n"
        "3. Bio_ClinicalBERT encodes each chunk\n"
        "4. Label attention attends to relevant tokens per code\n"
        "5. Ensemble with TF-IDF for best accuracy"
    )

# ── Predict button ─────────────────────────────────────────────────────
if st.button("🔍 Predict ICD-10 Codes", type="primary", use_container_width=True):
    if not text_input or len(text_input.strip()) < 10:
        st.warning("Please enter a discharge summary (at least 10 characters).")
    else:
        with st.spinner("Predicting..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "text": text_input,
                        "top_n": top_n,
                        "threshold": threshold,
                        "explain": explain,
                    },
                    timeout=60,
                ).json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        # ── Results ────────────────────────────────────────────────
        st.divider()
        st.subheader(f"Predictions ({resp['processing_time_ms']:.0f}ms)")

        # Summary table
        rows = []
        for p in resp['predictions']:
            rows.append({
                'ICD-10 Code': p['icd_code'],
                'Probability': p['probability'],
                'Predicted': '✅' if p['predicted'] else '—',
            })
        df = pd.DataFrame(rows)

        # Color the probability column
        st.dataframe(
            df.style.bar(subset=['Probability'], color='#5fba7d', vmin=0, vmax=1),
            use_container_width=True,
            hide_index=True,
        )

        # Evidence details (expandable)
        if explain:
            st.subheader("Evidence (Attention Weights)")
            for p in resp['predictions']:
                if p['predicted'] and p.get('evidence'):
                    with st.expander(f"**{p['icd_code']}** (p={p['probability']:.3f})"):
                        evidence_tokens = p['evidence']
                        cols = st.columns(min(len(evidence_tokens), 5))
                        for i, ev in enumerate(evidence_tokens[:10]):
                            with cols[i % 5]:
                                intensity = min(ev['weight'] * 100, 100)
                                st.metric(
                                    label=ev['token'],
                                    value=f"{ev['weight']:.4f}",
                                )

# ── Footer ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "CS6120 NLP — ICD-10 Code Prediction from MIMIC-IV Discharge Summaries | "
    "Models: TF-IDF + SGD, Bio_ClinicalBERT, Chunk-Based Label Attention"
)
