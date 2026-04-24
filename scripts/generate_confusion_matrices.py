#!/usr/bin/env python3
"""
Generate per-label confusion matrices (with percentages) and 50-label
confusion heatmaps for all models, using saved prediction probabilities
and model weights. Standalone — does not require running full notebooks.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "datasets" / "processed"
MODELS_DIR   = PROJECT_ROOT / "data" / "models"
ALT_MODELS   = PROJECT_ROOT / "models"

TOP_N = 10
TOP_PAIRS = 20


def resolve(primary: Path) -> Path:
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


def load_vocab():
    mlb_path = DATA_DIR / "mlb.pkl"
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)
    return list(mlb.classes_)


def load_y_test():
    yt_path = DATA_DIR / "Y_test.npy"
    if not yt_path.exists():
        yt_path = resolve(ALT_MODELS / "model_b" / "Y_test.npy")
    return np.load(yt_path)


ICD10_DESC = {
    "B9620":"E. coli infection","D649":"Anemia unspecified","E039":"Hypothyroidism",
    "E1122":"DM w/ CKD","E1165":"DM w/ hyperglycemia","E119":"Type 2 DM",
    "E669":"Obesity","E780":"Hypercholesterolemia","E785":"Hyperlipidemia",
    "E8342":"Hyponatremia","E876":"Hypokalemia","F329":"Depression",
    "G4733":"Sleep apnea","G8929":"Chronic pain","I10":"Hypertension",
    "I2510":"CAD","I252":"Old MI","I2699":"Pulmonary embolism",
    "I350":"Aortic stenosis","I480":"Paroxysmal AFib","I482":"Chronic AFib",
    "I4891":"Atrial fibrillation","I4892":"Atrial flutter",
    "I5020":"Systolic HF","I5032":"Diastolic HF","I509":"Heart failure",
    "J189":"Pneumonia","J449":"COPD","J9600":"Resp fail hypercapnia",
    "J9601":"Resp fail hypoxia","K219":"GERD","K5900":"Constipation",
    "N179":"AKI","N183":"CKD3","N184":"CKD4","N189":"CKD unspecified",
    "N390":"UTI","R6520":"Severe sepsis","Z66":"DNR",
    "Z6841":"BMI 40-44.9","Z7901":"Anticoagulants","Z794":"Insulin",
    "Z7982":"Aspirin","Z79899":"Drug therapy","Z853":"Hx breast CA",
    "Z8546":"Hx prostate CA","Z87891":"Hx nicotine","Z930":"Tracheostomy",
    "Z951":"CABG graft","Z9811":"Absent R knee",
}


def per_label_cm(Y_test, preds, vocab, save_dir, model_name, top_n=TOP_N):
    from sklearn.metrics import multilabel_confusion_matrix

    mcm = multilabel_confusion_matrix(Y_test, preds)
    freq = Y_test.sum(axis=0)
    top_idx = np.argsort(freq)[::-1][:top_n]

    nrows, ncols = 2, top_n // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    for idx, ax in zip(top_idx, axes.flat):
        cm = mcm[idx]
        ax.imshow(cm, cmap='Blues')
        code = vocab[idx]
        desc = ICD10_DESC.get(code, code)
        ax.set_title(f"{code}\n({desc})", fontsize=9, fontweight='bold')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Neg', 'Pos']); ax.set_yticklabels(['Neg', 'Pos'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        for r in range(2):
            row_total = cm[r].sum()
            for c in range(2):
                pct = cm[r, c] / row_total * 100 if row_total > 0 else 0
                color = 'white' if cm[r, c] > cm.max() / 2 else 'black'
                ax.text(c, r, f'{cm[r, c]:,}\n({pct:.1f}%)',
                        ha='center', va='center', color=color, fontsize=10)

    plt.suptitle(
        f'Per-Label Confusion Matrices — {model_name}, Top {top_n} Labels\n(row percentages shown)',
        fontsize=14)
    plt.tight_layout()
    out_path = save_dir / f'confusion_matrix_top{top_n}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path.name}")

    agg = mcm.sum(axis=0)
    agg_total = agg.sum()
    print(f'  Aggregate CM (all {len(vocab)} labels):')
    print(f'    TN={agg[0,0]:>10,} ({agg[0,0]/agg_total*100:.1f}%)  '
          f'FP={agg[0,1]:>10,} ({agg[0,1]/agg_total*100:.1f}%)')
    print(f'    FN={agg[1,0]:>10,} ({agg[1,0]/agg_total*100:.1f}%)  '
          f'TP={agg[1,1]:>10,} ({agg[1,1]/agg_total*100:.1f}%)')

    return mcm


def full_label_heatmap(Y_test, preds, vocab, save_dir, model_name,
                       top_pairs=TOP_PAIRS):
    n_labels = len(vocab)
    fp_mask = (preds == 1) & (Y_test == 0)
    tp_mask = (Y_test == 1)
    label_confusion = tp_mask.astype(int).T @ fp_mask.astype(int)

    row_sums = tp_mask.sum(axis=0).reshape(-1, 1)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    label_confusion_pct = label_confusion / row_sums * 100

    fig, ax = plt.subplots(figsize=(18, 15))
    im = ax.imshow(label_confusion_pct, cmap='Reds', aspect='auto',
                   interpolation='nearest')
    ax.set_xticks(range(n_labels)); ax.set_yticks(range(n_labels))
    ax.set_xticklabels(vocab, rotation=90, fontsize=7)
    ax.set_yticklabels(vocab, fontsize=7)
    ax.set_xlabel('Falsely Predicted Label (FP)', fontsize=12)
    ax.set_ylabel('Truly Present Label (TP)', fontsize=12)
    ax.set_title(f'{model_name} — {n_labels}-Label Confusion Heatmap (%)',
                 fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8, label='% of true-label samples')
    plt.tight_layout()
    out_path = save_dir / f'label_confusion_{n_labels}x{n_labels}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path.name}")

    np.fill_diagonal(label_confusion, 0)
    flat_idx = np.argsort(label_confusion.ravel())[::-1]
    print(f'\n  Top {top_pairs} most confused label pairs ({model_name}):')
    print(f'    {"True":>12s}  →  {"FP Label":>18s}  {"Count":>7s}  {"% of TP":>8s}')
    print('    ' + '-' * 55)
    shown = 0
    for fi in flat_idx:
        if shown >= top_pairs:
            break
        i, j = divmod(fi, n_labels)
        count = label_confusion[i, j]
        if count == 0:
            break
        tp_total = tp_mask[:, i].sum()
        pct = count / tp_total * 100 if tp_total > 0 else 0
        print(f'    {vocab[i]:>12s}  →  {vocab[j]:>18s}  {count:>7,}  ({pct:.1f}%)')
        shown += 1


def process_model(model_name, save_dir, p_test_path, threshold,
                  Y_test, vocab):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    p_test_path = resolve(Path(p_test_path))
    if not p_test_path.exists():
        print(f"  SKIP: {p_test_path} not found")
        return

    P_test = np.load(p_test_path)
    if isinstance(threshold, np.ndarray):
        preds = (P_test >= threshold).astype(int)
    else:
        preds = (P_test >= threshold).astype(int)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  threshold={threshold if isinstance(threshold, float) else 'per-label'}")
    print(f"  P_test: {P_test.shape}, preds positive: {preds.sum()}")
    print(f"{'='*60}")

    per_label_cm(Y_test, preds, vocab, save_dir, model_name)
    full_label_heatmap(Y_test, preds, vocab, save_dir, model_name)


def main():
    print("Loading data...")
    vocab = load_vocab()
    Y_test = load_y_test()
    print(f"  vocab: {len(vocab)} labels, Y_test: {Y_test.shape}")

    # Model B (ClinicalBERT)
    b_results = resolve(MODELS_DIR / "model_b" / "test_results.json")
    if b_results.exists():
        with open(b_results) as f:
            t_b = json.load(f)["threshold"]
        process_model("Model B (ClinicalBERT)",
                      MODELS_DIR / "model_b",
                      MODELS_DIR / "model_b" / "P_test.npy",
                      t_b, Y_test, vocab)

    # Model C v1 (Chunk+Attn)
    c_results = resolve(MODELS_DIR / "model_c" / "test_results.json")
    if c_results.exists():
        with open(c_results) as f:
            t_c = json.load(f)["Threshold"]
        process_model("Model C v1 (Chunk+Attn)",
                      MODELS_DIR / "model_c",
                      MODELS_DIR / "model_c" / "P_test.npy",
                      t_c, Y_test, vocab)

    # Model C v2 (Fixed+Focal)
    cv2_results = resolve(MODELS_DIR / "model_c" / "v2" / "test_results.json")
    if cv2_results.exists():
        with open(cv2_results) as f:
            t_cv2 = json.load(f)["global_threshold"]["Threshold"]
        plt_path = MODELS_DIR / "model_c" / "v2" / "per_label_thresholds.npy"
        plt_path = resolve(plt_path)
        if plt_path.exists():
            per_label_t = np.load(plt_path)
            process_model("Model C v2 (Per-Label Threshold)",
                          MODELS_DIR / "model_c" / "v2",
                          MODELS_DIR / "model_c" / "v2" / "P_test_calibrated.npy",
                          per_label_t, Y_test, vocab)
        else:
            process_model("Model C v2 (Fixed+Focal)",
                          MODELS_DIR / "model_c" / "v2",
                          MODELS_DIR / "model_c" / "v2" / "P_test_calibrated.npy",
                          t_cv2, Y_test, vocab)

    # Model D (BiLSTM-LAAT)
    d_results = resolve(MODELS_DIR / "model_d" / "test_results.json")
    if d_results.exists():
        with open(d_results) as f:
            t_d = json.load(f)["Threshold"]
        process_model("Model D (BiLSTM-LAAT)",
                      MODELS_DIR / "model_d",
                      MODELS_DIR / "model_d" / "P_test_calibrated.npy",
                      t_d, Y_test, vocab)

    # Ensemble models
    ens_cfg_path = resolve(MODELS_DIR / "ensemble" / "ensemble_config.json")
    if ens_cfg_path.exists():
        with open(ens_cfg_path) as f:
            ens_cfg = json.load(f)

        ens_dir = MODELS_DIR / "ensemble"

        for ens_key, ens_name, p_file in [
            ("ensemble_v1", "Ensemble v1 (A+Cv1)", "P_ensemble_v1_test.npy"),
            ("ensemble_v2", "Ensemble v2 (A+Cv2)", "P_ensemble_v2_test.npy"),
            ("ensemble_v3", "Ensemble v3 (A+Cv1+Cv2)", "P_ensemble_v3_test.npy"),
            ("ensemble_v4", "Ensemble v4 (A+D, best)", "P_ensemble_test.npy"),
        ]:
            if ens_key in ens_cfg:
                t_ens = ens_cfg[ens_key]["threshold"]
                process_model(ens_name, ens_dir,
                              ens_dir / p_file,
                              t_ens, Y_test, vocab)

    print("\nDone! All confusion matrices saved.")


if __name__ == "__main__":
    main()
