#!/usr/bin/env python3
"""
ICD-10 Code Prediction — Interactive Demo
==========================================

Run:  python demo.py

Offers two modes:
  1. CLI  — pick a model + sample input → see predictions in terminal
  2. Dashboard — launch the Streamlit web app

Assumes the full project directory is available with weight files
in data/models/ and processed data in datasets/processed/.

CS6120 NLP — Final Project
"""

import importlib
import subprocess
import sys
import os
import re
import json
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Dependency bootstrap ──────────────────────────────────────────────

REQUIRED = [
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("sklearn",      "scikit-learn"),
    ("scipy",        "scipy"),
    ("matplotlib",   "matplotlib"),
    ("torch",        "torch"),
    ("transformers", "transformers"),
    ("streamlit",    "streamlit"),
]


def ensure_packages():
    missing = []
    for imp, pip_name in REQUIRED:
        try:
            importlib.import_module(imp)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"  Installing {len(missing)} package(s): {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + missing,
            stdout=subprocess.DEVNULL,
        )
        print("  Done.\n")


# ── Paths ─────────────────────────────────────────────────────────────

DATA_DIR     = PROJECT_ROOT / "datasets" / "processed"
MODELS_DIR   = PROJECT_ROOT / "data" / "models"
MODEL_A_DIR  = MODELS_DIR / "model_a"
MODEL_B_DIR  = MODELS_DIR / "model_b"
MODEL_C_DIR  = MODELS_DIR / "model_c"
MODEL_D_DIR  = MODELS_DIR / "model_d"
ENSEMBLE_DIR = MODELS_DIR / "ensemble"

ICD10_DESC = {
    "E119":  "Type 2 diabetes w/o complications",
    "I10":   "Essential primary hypertension",
    "E780":  "Pure hypercholesterolemia",
    "E785":  "Hyperlipidemia unspecified",
    "Z87891":"Hx nicotine dependence",
    "I2510": "Atherosclerotic heart disease",
    "Z7901": "Long-term anticoagulants",
    "N179":  "Acute kidney failure unspecified",
    "E1165": "Type 2 diabetes w/ hyperglycemia",
    "Z79899":"Other long-term drug therapy",
    "I4891": "Unspecified atrial fibrillation",
    "Z7982": "Long-term aspirin use",
    "I509":  "Heart failure unspecified",
    "J449":  "COPD unspecified",
    "Z66":   "Do not resuscitate",
    "E1122": "Type 2 DM w/ diabetic CKD",
    "I2699": "Other pulmonary embolism",
    "N189":  "CKD unspecified",
    "D649":  "Anemia unspecified",
    "K219":  "GERD w/o esophagitis",
    "I480":  "Paroxysmal atrial fibrillation",
    "Z794":  "Long-term insulin use",
    "N390":  "UTI site unspecified",
    "G4733": "Obstructive sleep apnea",
    "J9601": "Acute resp failure w/ hypoxia",
    "E876":  "Hypokalemia",
    "I482":  "Chronic atrial fibrillation",
    "Z951":  "Aortocoronary bypass graft",
    "F329":  "Major depressive disorder",
    "I350":  "Aortic valve stenosis",
    "Z930":  "Tracheostomy status",
    "I5020": "Systolic heart failure",
    "J189":  "Pneumonia unspecified",
    "Z8546": "Hx prostate neoplasm",
    "G8929": "Other chronic pain",
    "E039":  "Hypothyroidism unspecified",
    "I4892": "Unspecified atrial flutter",
    "Z853":  "Hx breast neoplasm",
    "Z9811": "Absence of right knee joint",
    "I252":  "Old myocardial infarction",
    "B9620": "E. coli infection",
    "J9600": "Acute resp failure w/ hypercapnia",
    "R6520": "Severe sepsis w/o shock",
    "K5900": "Constipation unspecified",
    "N183":  "CKD stage 3",
    "N184":  "CKD stage 4",
    "I5032": "Chronic diastolic HF",
    "E669":  "Obesity unspecified",
    "E8342": "Hyponatremia",
}

# ── Sample inputs ─────────────────────────────────────────────────────

SAMPLES = {
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

# ── Text cleaning (mirrors src/data.py) ──────────────────────────────

def clean_text(text):
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,;:\-/]', ' ', text)
    text = re.sub(r'[\s\n\r\t]+', ' ', text).strip()
    return text


# ── Model loaders ────────────────────────────────────────────────────

def load_model_a():
    import pickle
    import scipy.sparse as sp

    vec_path = DATA_DIR / "tfidf_vectorizer.pkl"
    clf_path = MODEL_A_DIR / "clf_sgd.pkl"
    mlb_path = DATA_DIR / "mlb.pkl"

    for p in [vec_path, clf_path, mlb_path]:
        if not p.exists():
            return None, None, None, f"Missing: {p}"

    with open(vec_path, "rb") as f:
        vec = pickle.load(f)
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)

    return vec, clf, mlb, None


def predict_model_a(text, vec, clf):
    cleaned = clean_text(text)
    x = vec.transform([cleaned])
    probs = clf.predict_proba(x)[0]
    return probs


def load_model_b():
    import pickle
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models import ICDClassifier
    from src.config import TRANSFORMER_MODEL

    mlb_path = DATA_DIR / "mlb.pkl"
    wt_path  = MODEL_B_DIR / "best_model.pt"

    for p in [mlb_path, wt_path]:
        if not p.exists():
            return None, None, None, f"Missing: {p}"

    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)

    num_labels = len(mlb.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    model = ICDClassifier(TRANSFORMER_MODEL, num_labels=num_labels)
    model.load_state_dict(torch.load(wt_path, map_location=device))
    model.to(device).eval()

    return model, tokenizer, mlb, None


def predict_model_b(text, model, tokenizer):
    import torch

    cleaned = clean_text(text)
    device = next(model.parameters()).device

    enc = tokenizer(
        cleaned, max_length=512, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(ids, mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


def load_model_c():
    import pickle
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models import LabelAttentionClassifier
    from src.config import TRANSFORMER_MODEL, MODEL_C_MAX_CHUNKS

    mlb_path = DATA_DIR / "mlb.pkl"
    wt_path  = MODEL_C_DIR / "best_model.pt"

    for p in [mlb_path, wt_path]:
        if not p.exists():
            return None, None, None, f"Missing: {p}"

    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)

    num_labels = len(mlb.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    model = LabelAttentionClassifier(
        TRANSFORMER_MODEL, num_labels=num_labels,
        max_chunks=MODEL_C_MAX_CHUNKS, freeze_bert=False,
    )
    model.load_state_dict(torch.load(wt_path, map_location=device))
    model.to(device).eval()

    return model, tokenizer, mlb, None


def predict_model_c(text, model, tokenizer):
    import torch
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
        ids  = [cls_id] + chunk + [sep_id]
        mask = [1] * len(ids)
        pad_len = MAX_SEQ_LEN - len(ids)
        ids  += [pad_id] * pad_len
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

    ids_t  = torch.tensor([chunks_ids], dtype=torch.long).to(device)
    mask_t = torch.tensor([chunks_mask], dtype=torch.long).to(device)
    cc_t   = torch.tensor([chunk_count], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(ids_t, mask_t, chunk_counts=cc_t)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


def load_model_d():
    import pickle
    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models import BiLSTMLAAT

    mlb_path   = DATA_DIR / "mlb.pkl"
    vocab_path = DATA_DIR / "word_vocab.pkl"
    wt_path    = MODEL_D_DIR / "best_model.pt"

    for p in [mlb_path, vocab_path, wt_path]:
        if not p.exists():
            return None, None, None, None, f"Missing: {p}"

    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)
    with open(vocab_path, "rb") as f:
        word2idx = pickle.load(f)

    num_labels = len(mlb.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMLAAT(
        vocab_size=len(word2idx), num_labels=num_labels,
    )
    model.load_state_dict(torch.load(wt_path, map_location=device))
    model.to(device).eval()

    return model, word2idx, mlb, None


def predict_model_d(text, model, word2idx):
    import torch
    from src.config import MODEL_D_MAX_TOKENS

    cleaned = clean_text(text)
    words = cleaned.split()[:MODEL_D_MAX_TOKENS]
    ids = [word2idx.get(w, 1) for w in words]
    mask = [1] * len(ids)

    pad_len = MODEL_D_MAX_TOKENS - len(ids)
    ids  += [0] * pad_len
    mask += [0] * pad_len

    device = next(model.parameters()).device
    ids_t  = torch.tensor([ids], dtype=torch.long).to(device)
    mask_t = torch.tensor([mask], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(ids_t, mask_t)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


# ── Display helpers ───────────────────────────────────────────────────

def get_threshold(model_key):
    """Load the tuned threshold from the saved results."""
    threshold_map = {
        "a": (MODEL_A_DIR / "results.json",     lambda d: d["test"]["threshold"]),
        "b": (MODEL_B_DIR / "test_results.json", lambda d: d["threshold"]),
        "c": (MODEL_C_DIR / "test_results.json", lambda d: d["Threshold"]),
        "d": (MODEL_D_DIR / "test_results.json", lambda d: d["Threshold"]),
    }
    if model_key in threshold_map:
        path, extractor = threshold_map[model_key]
        if path.exists():
            with open(path) as f:
                return extractor(json.load(f))
    return 0.5


def print_predictions(probs, vocab, threshold, model_name, top_n=20):
    predicted = [(i, probs[i]) for i in range(len(probs)) if probs[i] >= threshold]
    predicted.sort(key=lambda x: x[1], reverse=True)

    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_n]

    n_pos = len(predicted)
    w = 78

    print()
    print("=" * w)
    print(f"  {model_name}")
    print(f"  Threshold: {threshold:.3f}  |  {n_pos} code(s) predicted positive")
    print("=" * w)
    print(f"  {'#':>2s}   {'Code':<8s}  {'Description':<38s}  {'Prob':>6s}  {'Pred':>4s}")
    print("  " + "-" * (w - 4))

    for rank, idx in enumerate(ranked, 1):
        code = vocab[idx]
        prob = probs[idx]
        desc = ICD10_DESC.get(code, f"ICD-10 {code}")
        if len(desc) > 38:
            desc = desc[:35] + "..."
        flag = " YES" if prob >= threshold else "    "
        marker = "*" if prob >= threshold else " "
        print(f"  {rank:>2d} {marker} {code:<8s}  {desc:<38s}  {prob:6.4f} {flag}")

    print("=" * w)

    if predicted:
        codes = [vocab[i] for i, _ in predicted]
        print(f"\n  Predicted codes: {', '.join(codes)}")
    print()


# ── Main menu ────────────────────────────────────────────────────────

MODEL_CHOICES = [
    ("a", "Model A — TF-IDF + SGD",                       "(sklearn, fastest)"),
    ("b", "Model B — ClinicalBERT (512-token)",            "(torch + transformers)"),
    ("c", "Model C — Chunk-Based BERT + Label Attention",  "(torch + transformers)"),
    ("d", "Model D — BiLSTM-LAAT",                         "(torch)"),
    ("e", "Ensemble v4 — A + D blend",                     "(best overall)"),
]

SAMPLE_KEYS = list(SAMPLES.keys())


def prompt_choice(prompt, valid):
    while True:
        ans = input(prompt).strip().lower()
        if ans in valid:
            return ans
        print(f"  Invalid choice. Options: {', '.join(valid)}")


def choose_sample():
    print("\n  Sample discharge summaries:")
    for i, key in enumerate(SAMPLE_KEYS, 1):
        print(f"    [{i}] {key}")
    print(f"    [c] Enter custom text")

    valid = [str(i) for i in range(1, len(SAMPLE_KEYS) + 1)] + ["c"]
    ans = prompt_choice("\n  Select sample: ", valid)

    if ans == "c":
        print("  Paste your discharge summary (end with an empty line):")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        text = " ".join(lines)
        if len(text) < 10:
            print("  Text too short, using first sample instead.")
            text = SAMPLES[SAMPLE_KEYS[0]]
        return "Custom", text
    else:
        key = SAMPLE_KEYS[int(ans) - 1]
        return key, SAMPLES[key]


def run_single_model(model_key, text):
    """Load and run one model, return (probs, vocab, threshold) or None."""
    import numpy as np

    if model_key == "a":
        print("  Loading Model A (TF-IDF + SGD)...", end="", flush=True)
        vec, clf, mlb, err = load_model_a()
        if err:
            print(f" FAILED\n  {err}")
            return None
        print(" done")
        vocab = list(mlb.classes_)
        probs = predict_model_a(text, vec, clf)
        t = get_threshold("a")
        return probs, vocab, t, "Model A (TF-IDF + SGD)"

    elif model_key == "b":
        print("  Loading Model B (ClinicalBERT)...", end="", flush=True)
        model, tokenizer, mlb, err = load_model_b()
        if err:
            print(f" FAILED\n  {err}")
            return None
        print(" done")
        vocab = list(mlb.classes_)
        probs = predict_model_b(text, model, tokenizer)
        t = get_threshold("b")
        return probs, vocab, t, "Model B (ClinicalBERT)"

    elif model_key == "c":
        print("  Loading Model C (Chunk + Label Attention)...", end="", flush=True)
        model, tokenizer, mlb, err = load_model_c()
        if err:
            print(f" FAILED\n  {err}")
            return None
        print(" done")
        vocab = list(mlb.classes_)
        probs = predict_model_c(text, model, tokenizer)
        t = get_threshold("c")
        return probs, vocab, t, "Model C (Chunk + Label Attention)"

    elif model_key == "d":
        print("  Loading Model D (BiLSTM-LAAT)...", end="", flush=True)
        result = load_model_d()
        model, word2idx, mlb, err = result[0], result[1], result[2], result[3]
        if err:
            print(f" FAILED\n  {err}")
            return None
        print(" done")
        vocab = list(mlb.classes_)
        probs = predict_model_d(text, model, word2idx)
        t = get_threshold("d")
        return probs, vocab, t, "Model D (BiLSTM-LAAT)"

    return None


def run_ensemble(text):
    """Run Ensemble v4 (A + D blend)."""
    import numpy as np

    cfg_path = ENSEMBLE_DIR / "ensemble_config.json"
    if not cfg_path.exists():
        print(f"  Missing: {cfg_path}")
        return None

    with open(cfg_path) as f:
        ens_cfg = json.load(f)
    v4 = ens_cfg["ensemble_v4"]
    w_a, w_d, t_ens = v4["weight_A"], v4["weight_D"], v4["threshold"]

    res_a = run_single_model("a", text)
    if res_a is None:
        return None
    probs_a, vocab, _, _ = res_a

    res_d = run_single_model("d", text)
    if res_d is None:
        return None
    probs_d = res_d[0]

    probs_ens = w_a * probs_a + w_d * probs_d
    return probs_ens, vocab, t_ens, f"Ensemble v4 (A+D, w_A={w_a:.2f}, w_D={w_d:.2f})"


def cli_mode():
    print("\n  Available models:")
    for key, name, note in MODEL_CHOICES:
        print(f"    [{key}] {name}  {note}")
    print(f"    [*] Run ALL models on same input")

    valid = [k for k, _, _ in MODEL_CHOICES] + ["*"]
    model_choice = prompt_choice("\n  Choose model: ", valid)

    sample_name, text = choose_sample()

    print(f"\n  Input: {sample_name}")
    preview = text[:120] + ("..." if len(text) > 120 else "")
    print(f"  \"{preview}\"\n")

    if model_choice == "*":
        keys_to_run = ["a", "b", "c", "d", "e"]
    else:
        keys_to_run = [model_choice]

    for key in keys_to_run:
        if key == "e":
            result = run_ensemble(text)
        else:
            result = run_single_model(key, text)

        if result is not None:
            probs, vocab, threshold, name = result
            print_predictions(probs, vocab, threshold, name)
        elif key != "e":
            name = dict((k, n) for k, n, _ in MODEL_CHOICES).get(key, key)
            print(f"\n  Skipping {name} (weight file not found)\n")


def streamlit_mode():
    demo_script = PROJECT_ROOT / "demo" / "streamlit_app.py"
    if not demo_script.exists():
        print(f"  ERROR: {demo_script} not found")
        return

    print("  Launching Streamlit dashboard...")
    print("  (Press Ctrl+C to stop)\n")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(demo_script),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(PROJECT_ROOT),
    )


def main():
    print()
    print("=" * 60)
    print("  ICD-10 Code Prediction from Discharge Summaries")
    print("  CS6120 NLP — Final Project Demo")
    print("=" * 60)
    print()

    ensure_packages()

    while True:
        print("  Demo modes:")
        print("    [1] CLI  — Run model predictions in the terminal")
        print("    [2] Dashboard — Launch Streamlit web app")
        print("    [q] Quit")

        choice = prompt_choice("\n  Select mode: ", ["1", "2", "q"])

        if choice == "q":
            print("\n  Goodbye!\n")
            break
        elif choice == "1":
            cli_mode()
            print("\n  " + "-" * 56)
            cont = prompt_choice("  Run another prediction? [y/n]: ", ["y", "n"])
            if cont == "n":
                print()
                break
        elif choice == "2":
            streamlit_mode()
            break


if __name__ == "__main__":
    main()
