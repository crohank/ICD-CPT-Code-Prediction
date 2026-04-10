"""
Singleton model service — loads all models once at startup and provides
inference methods for the API.
"""
import pickle
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_DIR, MODEL_A_DIR, MODEL_B_DIR, MODEL_C_DIR, ENSEMBLE_DIR,
    TRANSFORMER_MODEL, MAX_SEQ_LEN,
)
from src.data import clean_text, ChunkedICDDataset
from src.models import LabelAttentionClassifier
from src.explain import extract_attention_for_text


class ModelService:
    """
    Loads and serves ICD-10 prediction models.
    Designed as a singleton — instantiate once at FastAPI startup.
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models_loaded = []
        self.tokenizer = None
        self.mlb = None
        self.vocab = []
        self.num_labels = 0

        # Model artifacts
        self.clf_a = None           # sklearn classifier (Model A)
        self.tfidf_vec = None       # TF-IDF vectorizer
        self.model_c = None         # LabelAttentionClassifier (Model C)
        self.threshold_a = 0.525
        self.threshold_c = 0.275
        self.threshold_ens = 0.5
        self.ensemble_weight = 0.5  # w * A + (1-w) * C

    def load(self):
        """Load all model artifacts from disk."""
        print(f"Loading models on device: {self.device}")

        # Label binarizer
        with open(DATA_DIR / 'mlb.pkl', 'rb') as f:
            self.mlb = pickle.load(f)
        self.vocab = list(self.mlb.classes_)
        self.num_labels = len(self.vocab)
        print(f"  Labels: {self.num_labels}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

        # Model A: TF-IDF + SGD
        try:
            with open(DATA_DIR / 'tfidf_vectorizer.pkl', 'rb') as f:
                self.tfidf_vec = pickle.load(f)
            with open(MODEL_A_DIR / 'clf_sgd.pkl', 'rb') as f:
                self.clf_a = pickle.load(f)
            with open(MODEL_A_DIR / 'results.json') as f:
                self.threshold_a = json.load(f)['test']['threshold']
            self.models_loaded.append('model_a')
            print(f"  Model A loaded (threshold={self.threshold_a})")
        except FileNotFoundError as e:
            print(f"  Model A not found: {e}")

        # Model C: Chunk + Label Attention
        try:
            self.model_c = LabelAttentionClassifier(
                model_name=TRANSFORMER_MODEL,
                num_labels=self.num_labels,
                freeze_bert=False,
            ).to(self.device)
            state = torch.load(MODEL_C_DIR / 'best_model.pt', map_location=self.device)
            self.model_c.load_state_dict(state)
            self.model_c.eval()
            with open(MODEL_C_DIR / 'test_results.json') as f:
                self.threshold_c = json.load(f)['Threshold']
            self.models_loaded.append('model_c')
            print(f"  Model C loaded (threshold={self.threshold_c})")
        except FileNotFoundError as e:
            print(f"  Model C not found: {e}")

        # Ensemble config
        try:
            with open(ENSEMBLE_DIR / 'ensemble_config.json') as f:
                ens_cfg = json.load(f)
            self.ensemble_weight = ens_cfg['weight_model_a']
            self.threshold_ens = ens_cfg['threshold']
            self.models_loaded.append('ensemble')
            print(f"  Ensemble loaded (w={self.ensemble_weight}, t={self.threshold_ens})")
        except FileNotFoundError:
            print("  Ensemble config not found — will use equal weights")

        print(f"Models loaded: {self.models_loaded}")

    def predict(self, text: str, top_n: int = 10,
                threshold: float = None, explain: bool = False):
        """
        Run prediction on a single discharge summary.

        Returns:
            list of dicts with keys: icd_code, probability, predicted, evidence
        """
        cleaned = clean_text(text)
        threshold = threshold or self.threshold_ens

        probs_a = None
        probs_c = None

        # Model A prediction
        if self.clf_a is not None and self.tfidf_vec is not None:
            xv = self.tfidf_vec.transform([cleaned])
            probs_a = self.clf_a.predict_proba(xv)[0]

        # Model C prediction
        attention_dict = None
        if self.model_c is not None:
            if explain:
                _, probs_c, attention_dict = extract_attention_for_text(
                    self.model_c, cleaned,
                    tokenizer=self.tokenizer, device=self.device,
                )
            else:
                # Quick inference without attention extraction
                ds = ChunkedICDDataset(
                    [cleaned], np.zeros((1, self.num_labels), dtype=np.float32),
                    tokenizer=self.tokenizer,
                )
                sample = ds[0]
                ids  = sample['input_ids'].unsqueeze(0).to(self.device)
                mask = sample['attention_mask'].unsqueeze(0).to(self.device)
                with torch.no_grad(), torch.amp.autocast(self.device, enabled=self.device == 'cuda'):
                    logits = self.model_c(ids, mask)
                probs_c = torch.sigmoid(logits).cpu().float().numpy()[0]

        # Ensemble
        if probs_a is not None and probs_c is not None:
            probs = self.ensemble_weight * probs_a + (1 - self.ensemble_weight) * probs_c
        elif probs_c is not None:
            probs = probs_c
        elif probs_a is not None:
            probs = probs_a
        else:
            raise RuntimeError("No models loaded!")

        # Rank and return top-N
        ranked_indices = np.argsort(probs)[::-1][:top_n]
        results = []
        for idx in ranked_indices:
            evidence = None
            if explain and attention_dict and idx in attention_dict:
                evidence = [
                    {'token': tok, 'weight': round(w, 4)}
                    for tok, w in attention_dict[idx][:10]
                ]
            results.append({
                'icd_code':    self.vocab[idx],
                'probability': round(float(probs[idx]), 4),
                'predicted':   float(probs[idx]) >= threshold,
                'evidence':    evidence,
            })

        return results
