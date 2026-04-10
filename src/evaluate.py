"""
Evaluation utilities: metrics computation, threshold tuning, head/tail analysis.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
)


def full_metrics(P: np.ndarray, Y: np.ndarray, threshold: float,
                 name: str = "Model") -> dict:
    """
    Compute comprehensive multi-label classification metrics.

    Args:
        P: probability matrix (n_samples, n_labels)
        Y: binary label matrix (n_samples, n_labels)
        threshold: decision threshold
        name: model name for the result dict

    Returns:
        dict with Model, Threshold, Micro-F1, Macro-F1, etc.
    """
    preds = (P >= threshold).astype(int)
    mask  = Y.sum(0) > 0  # labels with at least one positive in this split

    return {
        'Model':       name,
        'Threshold':   round(threshold, 3),
        'Micro-F1':    round(f1_score(Y, preds, average='micro',  zero_division=0), 4),
        'Macro-F1':    round(f1_score(Y, preds, average='macro',  zero_division=0), 4),
        'Micro-Prec':  round(precision_score(Y, preds, average='micro', zero_division=0), 4),
        'Micro-Rec':   round(recall_score(Y, preds, average='micro',    zero_division=0), 4),
        'Macro-AUPRC': round(average_precision_score(Y[:, mask], P[:, mask], average='macro'), 4),
        'Micro-AUROC': round(roc_auc_score(Y[:, mask], P[:, mask], average='micro'), 4),
    }


def tune_global_threshold(P: np.ndarray, Y: np.ndarray,
                          thresholds=None) -> tuple:
    """
    Sweep global thresholds on a validation set to maximize micro-F1.

    Returns:
        (best_threshold, best_f1)
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.65, 0.025)

    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (P >= t).astype(int)
        f1 = f1_score(Y, preds, average='micro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def tune_per_label_threshold(P: np.ndarray, Y: np.ndarray,
                             thresholds=None) -> np.ndarray:
    """
    Tune a separate threshold for each label on the validation set.

    Returns:
        array of per-label thresholds (n_labels,)
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.65, 0.025)

    n_labels = Y.shape[1]
    best_thresholds = np.full(n_labels, 0.5)

    for j in range(n_labels):
        best_f1 = 0.0
        for t in thresholds:
            preds = (P[:, j] >= t).astype(int)
            f1 = f1_score(Y[:, j], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[j] = t

    return best_thresholds


def head_tail_analysis(
    vocab: list,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    predictions: dict,
    buckets=None,
) -> pd.DataFrame:
    """
    Compute per-label F1 and group by frequency buckets (head/torso/tail).

    Args:
        vocab: list of ICD-10 code strings
        Y_train: training label matrix (for computing frequencies)
        Y_test: test label matrix
        predictions: dict of {model_name: (preds_binary, P_probs)}

    Returns:
        DataFrame with per-label F1 for each model + frequency bucket
    """
    if buckets is None:
        buckets = [
            ('head (>=500)',    500, 1e9),
            ('torso (100-499)', 100, 499),
            ('tail (<100)',      0,   99),
        ]

    freq = Y_train.sum(0)
    label_df = pd.DataFrame({'icd_code': vocab, 'train_freq': freq})

    for name, (preds, _) in predictions.items():
        per_label_f1 = f1_score(Y_test, preds, average=None, zero_division=0)
        label_df[f'f1_{name}'] = per_label_f1

    # Bucket summary
    rows = []
    for bucket_name, lo, hi in buckets:
        mask = (label_df['train_freq'] >= lo) & (label_df['train_freq'] <= hi)
        subset = label_df[mask]
        row = {'bucket': bucket_name, 'n_codes': len(subset)}
        for name in predictions:
            row[f'avg_f1_{name}'] = round(subset[f'f1_{name}'].mean(), 4) if len(subset) > 0 else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows)
    return label_df, summary


def compute_pos_weights(Y_train: np.ndarray, clamp_max: float = 50.0) -> np.ndarray:
    """
    Compute positive class weights for BCEWithLogitsLoss.
    pos_weight[j] = (N - n_j) / n_j, clamped to [1, clamp_max].

    This helps the model pay more attention to rare labels.
    """
    n_samples = Y_train.shape[0]
    pos_counts = Y_train.sum(axis=0)
    neg_counts = n_samples - pos_counts

    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1)

    weights = neg_counts / pos_counts
    weights = np.clip(weights, 1.0, clamp_max)
    return weights.astype(np.float32)
