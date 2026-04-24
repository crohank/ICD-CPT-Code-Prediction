#!/usr/bin/env python3
"""Regenerate ALL report figures with:
- xx.xx number format everywhere
- No ~ (tilde) symbols
- Proper aspect ratios for single-column A4 report
- High DPI (300)
- Consistent styling
"""
import os, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Paths ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "models")
FIG  = os.path.join(BASE, "report", "figures")
os.makedirs(FIG, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

def fmt2(x, pos=None):
    """Format number as x.xx"""
    return f"{x:.2f}"

def fmt0(x, pos=None):
    """Format integer"""
    return f"{int(x)}"

def save(name):
    path = os.path.join(FIG, name)
    plt.savefig(path)
    plt.close()
    print(f"  saved {name}")

# ══════════════════════════════════════════════════════════════════════
# 1. MODEL COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════════
print("1. Model comparison bars")
comp = pd.read_csv(os.path.join(DATA, "ensemble", "final_comparison_all_models.csv"))

models_short = [
    "Model A\n(TF-IDF+SGD)",
    "Model B\n(ClinicalBERT)",
    "Model C v1\n(Chunk+Attn)",
    "Model C v2\n(Focal)",
    "Model D\n(BiLSTM-LAAT)",
    "Ens. v1\n(A+Cv1)",
    "Ens. v2\n(A+Cv2)",
    "Ens. v3\n(A+Cv1+Cv2)",
    "Ens. v4\n(A+D)",
    "Ens. v5\n(A+Cv2+D)",
]

metrics = ["Micro-F1", "Macro-F1", "Micro-Prec", "Micro-Rec", "Micro-AUROC"]
colors = sns.color_palette("tab10", len(models_short))

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(metrics))
w = 0.08
for i, (_, row) in enumerate(comp.iterrows()):
    vals = [row[m] for m in metrics]
    ax.bar(x + i * w - (len(comp) - 1) * w / 2, vals, w, label=models_short[i], color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("Score")
ax.set_title("Model Comparison (Test Set -Top-50 ICD-10 Codes)")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
ax.legend(loc="upper left", ncol=5, fontsize=7, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
save("model_comparison_bars.png")

# ══════════════════════════════════════════════════════════════════════
# 2. TRAINING CURVES - Chunk-BERT v1
# ══════════════════════════════════════════════════════════════════════
print("2. Training curves - Chunk-BERT v1")
hist_c = pd.read_csv(os.path.join(DATA, "model_c", "full_training_history.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(hist_c["epoch"], hist_c["train_loss"], "b-o", markersize=5, linewidth=2)
ax1.axvline(x=5.5, color="red", linestyle="--", alpha=0.7, label="Unfreeze BERT")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax1.set_title("Chunk-BERT v1 -Training Loss")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2.plot(hist_c["epoch"], hist_c["val_micro_f1"], "g-o", markersize=5, linewidth=2)
ax2.axvline(x=5.5, color="red", linestyle="--", alpha=0.7, label="Unfreeze BERT")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val Micro-F1")
ax2.set_title("Chunk-BERT v1 -Validation Micro-F1")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
save("training_curves_c.png")

# ══════════════════════════════════════════════════════════════════════
# 3. TRAINING CURVES - Chunk-BERT v2
# ══════════════════════════════════════════════════════════════════════
print("3. Training curves - Chunk-BERT v2")
hist_c2 = pd.read_csv(os.path.join(DATA, "model_c", "v2", "full_training_history.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(hist_c2["epoch"], hist_c2["train_loss"], "b-o", markersize=5, linewidth=2)
ax1.axvline(x=5.5, color="red", linestyle="--", alpha=0.7, label="Unfreeze BERT")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax1.set_title("Chunk-BERT v2 (Focal) -Training Loss")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2.plot(hist_c2["epoch"], hist_c2["val_f1_tuned"], "g-o", markersize=5, linewidth=2, label="Tuned threshold")
ax2.plot(hist_c2["epoch"], hist_c2["val_f1_at_0.5"], "gray", linestyle="--", alpha=0.6, label="F1 @ t=0.50")
ax2.axvline(x=5.5, color="red", linestyle="--", alpha=0.7, label="Unfreeze BERT")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val Micro-F1")
ax2.set_title("Chunk-BERT v2 (Focal) -Validation F1")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
save("training_curves_c_v2.png")

# ══════════════════════════════════════════════════════════════════════
# 4. TRAINING CURVES - BiLSTM-LAAT
# ══════════════════════════════════════════════════════════════════════
print("4. Training curves - BiLSTM-LAAT")
hist_d = pd.read_csv(os.path.join(DATA, "model_d", "training_history.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(hist_d["epoch"], hist_d["train_loss"], "b-o", markersize=5, linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax1.set_title("BiLSTM-LAAT -Training Loss")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
ax1.grid(alpha=0.3)

ax2.plot(hist_d["epoch"], hist_d["val_f1_tuned"], "g-o", markersize=5, linewidth=2, label="Tuned threshold")
ax2.plot(hist_d["epoch"], hist_d["val_f1_at_0.5"], "gray", linestyle="--", alpha=0.6, label="F1 @ t=0.50")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Val Micro-F1")
ax2.set_title("BiLSTM-LAAT -Validation F1")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
save("training_curves_d.png")

# ── Also make individual loss/f1 plots ────────────────────────────
print("4b. Individual training loss/f1 plots")
for name, hist, title, has_phase in [
    ("training_loss_c.png", hist_c, "Chunk-BERT v1", True),
    ("training_loss_d.png", hist_d, "BiLSTM-LAAT", False),
]:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(hist["epoch"], hist["train_loss"], "b-o", markersize=5, linewidth=2)
    if has_phase:
        ax.axvline(x=5.5, color="red", linestyle="--", alpha=0.7, label="Unfreeze BERT")
        ax.legend(fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"{title} -Training Loss")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(name)

for name, hist, title, has_phase, f1_col in [
    ("training_f1_c.png", hist_c, "Chunk-BERT v1", True, "val_micro_f1"),
    ("training_f1_d.png", hist_d, "BiLSTM-LAAT", False, "val_f1_tuned"),
]:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(hist["epoch"], hist[f1_col], "g-o", markersize=5, linewidth=2)
    if has_phase:
        ax.axvline(x=5.5, color="red", linestyle="--", alpha=0.7, label="Unfreeze BERT")
        ax.legend(fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Micro-F1")
    ax.set_title(f"{title} -Validation F1")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(name)

# ══════════════════════════════════════════════════════════════════════
# 5-8. CONFUSION MATRICES - per model
# ══════════════════════════════════════════════════════════════════════
print("5. Confusion matrices")

# Load test labels
datasets_dir = os.path.join(BASE, "datasets", "processed")
Y_test = np.load(os.path.join(datasets_dir, "Y_test.npy"))

import pickle
with open(os.path.join(datasets_dir, "mlb.pkl"), "rb") as f:
    mlb = pickle.load(f)
code_names = list(mlb.classes_)

# Load predictions for each model
model_preds = {}
thresholds = {"model_a": 0.525, "model_b": 0.275, "model_c": 0.625, "model_c_v2": 0.05, "model_d": 0.30}

for mname, thr in thresholds.items():
    if mname == "model_c_v2":
        pfile = os.path.join(DATA, "model_c", "v2", "P_test_calibrated.npy")
        if not os.path.exists(pfile):
            pfile = os.path.join(DATA, "model_c", "v2", "P_test_uncalibrated.npy")
    elif mname == "model_d":
        pfile = os.path.join(DATA, mname, "P_test_calibrated.npy")
        if not os.path.exists(pfile):
            pfile = os.path.join(DATA, mname, "P_test_uncalibrated.npy")
    else:
        pfile = os.path.join(DATA, mname, "P_test.npy")
        if not os.path.exists(pfile):
            pfile = os.path.join(DATA, mname, "P_test_calibrated.npy")
            if not os.path.exists(pfile):
                pfile = os.path.join(DATA, mname, "P_test_uncalibrated.npy")
    if os.path.exists(pfile):
        model_preds[mname] = np.load(pfile)
        print(f"  loaded {mname}: {pfile} shape={model_preds[mname].shape}")
    else:
        print(f"  MISSING {mname} predictions")

# Also try to load ensemble predictions
for ens_name in ["P_ensemble_v1_test.npy", "P_ensemble_v4_test.npy"]:
    pfile = os.path.join(DATA, "ensemble", ens_name)
    if os.path.exists(pfile):
        key = ens_name.replace("P_", "").replace("_test.npy", "")
        model_preds[key] = np.load(pfile)
        print(f"  loaded {key}: shape={model_preds[key].shape}")

# Find top-10 codes by frequency
code_freq = Y_test.sum(axis=0)
top10_idx = np.argsort(code_freq)[::-1][:10]
top10_codes = [code_names[i] for i in top10_idx]

def plot_confusion_matrices(Y_true, P_pred, threshold, top_idx, top_codes, title, filename):
    Y_pred = (P_pred >= threshold).astype(int)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for k, (idx, code) in enumerate(zip(top_idx, top_codes)):
        ax = axes[k // 5, k % 5]
        cm = confusion_matrix(Y_true[:, idx], Y_pred[:, idx])
        total = cm.sum()
        cm_pct = cm / total * 100

        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        # Add annotations with count and percentage
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = cm_pct[i, j]
                ax.text(j + 0.5, i + 0.5, f"{val:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=7,
                        color="white" if val > total * 0.4 else "black")
        ax.set_title(code, fontsize=9, fontweight="bold")
        ax.set_ylabel("Actual" if k % 5 == 0 else "")
        ax.set_xlabel("Predicted")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(filename)

model_titles = {
    "model_a": ("Model A (TF-IDF + SGD)", "confusion_matrix_a.png"),
    "model_b": ("Model B (ClinicalBERT)", "confusion_matrix_b.png"),
    "model_c": ("Model C v1 (Chunk-BERT + Label Attn)", "confusion_matrix_c.png"),
    "model_c_v2": ("Model C v2 (Chunk-BERT + Focal Loss)", "confusion_matrix_c_v2.png"),
    "model_d": ("Model D (BiLSTM-LAAT)", "confusion_matrix_d.png"),
}

for mname, (title, fname) in model_titles.items():
    if mname in model_preds:
        P = model_preds[mname]
        if P.shape[0] == Y_test.shape[0] and P.shape[1] == Y_test.shape[1]:
            plot_confusion_matrices(Y_test, P, thresholds[mname], top10_idx, top10_codes,
                                    f"Per-Label Confusion -{title}, Top 10 Labels", fname)

# Ensemble confusion matrix
ens_thr = {"ensemble_v1": 0.625, "ensemble_v4": 0.35}
for ename, thr in ens_thr.items():
    if ename in model_preds:
        P = model_preds[ename]
        if P.shape[0] == Y_test.shape[0]:
            plot_confusion_matrices(Y_test, P, thr, top10_idx, top10_codes,
                                    f"Per-Label Confusion -Ensemble {ename[-2:]} (TF-IDF + BiLSTM-LAAT), Top 10",
                                    f"confusion_matrix_ensemble.png")

# ══════════════════════════════════════════════════════════════════════
# 9. 50x50 LABEL CONFUSION HEATMAPS
# ══════════════════════════════════════════════════════════════════════
print("6. 50x50 label confusion heatmaps")

def plot_label_confusion_heatmap(Y_true, P_pred, threshold, code_names, title, filename):
    Y_pred = (P_pred >= threshold).astype(int)
    n_labels = Y_true.shape[1]
    conf = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        true_mask = Y_true[:, i] == 1
        n_true = true_mask.sum()
        if n_true == 0:
            continue
        for j in range(n_labels):
            if i == j:
                continue
            fp_rate = Y_pred[true_mask, j].sum() / n_true * 100
            conf[i, j] = fp_rate

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(conf, cmap="Reds", ax=ax, square=True,
                xticklabels=code_names, yticklabels=code_names,
                cbar_kws={"label": "% of True-Label Samples", "format": mticker.FuncFormatter(fmt2)})
    ax.set_xlabel("Falsely Predicted Label (FP)", fontsize=10)
    ax.set_ylabel("Truly Present Label (TP)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=5, rotation=90)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    save(filename)

heatmap_configs = {
    "model_a": ("Model A -50-Label Confusion Heatmap (%)", "label_confusion_a.png", 0.525),
    "model_b": ("Model B -50-Label Confusion Heatmap (%)", "label_confusion_b.png", 0.275),
    "model_c": ("Model C v1 -50-Label Confusion Heatmap (%)", "label_confusion_c.png", 0.625),
    "model_c_v2": ("Model C v2 -50-Label Confusion Heatmap (%)", "label_confusion_c_v2.png", 0.05),
    "model_d": ("Model D -50-Label Confusion Heatmap (%)", "label_confusion_d.png", 0.30),
}

for mname, (title, fname, thr) in heatmap_configs.items():
    if mname in model_preds:
        P = model_preds[mname]
        if P.shape == Y_test.shape:
            plot_label_confusion_heatmap(Y_test, P, thr, code_names, title, fname)

for ename, thr in ens_thr.items():
    if ename in model_preds:
        P = model_preds[ename]
        if P.shape == Y_test.shape:
            plot_label_confusion_heatmap(Y_test, P, thr, code_names,
                                         f"Ensemble {ename[-2:]} -50-Label Confusion Heatmap (%)",
                                         f"label_confusion_ensemble.png")

# ══════════════════════════════════════════════════════════════════════
# 10. HEAD/TAIL per-label F1 scatter
# ══════════════════════════════════════════════════════════════════════
print("7. Head/tail scatter plots")

from sklearn.metrics import f1_score

train_labels_path = os.path.join(datasets_dir, "Y_train.npy")
if os.path.exists(train_labels_path):
    Y_train = np.load(train_labels_path)
    train_freq = Y_train.sum(axis=0)
else:
    print("  Y_train.npy not found, using Y_test frequencies as proxy")
    train_freq = Y_test.sum(axis=0)

def compute_per_label_f1(Y_true, P_pred, threshold):
    Y_pred = (P_pred >= threshold).astype(int)
    f1s = []
    for j in range(Y_true.shape[1]):
        f1s.append(f1_score(Y_true[:, j], Y_pred[:, j], zero_division=0.0))
    return np.array(f1s)

per_label_f1 = {}
for mname, thr in thresholds.items():
    if mname in model_preds and model_preds[mname].shape == Y_test.shape:
        per_label_f1[mname] = compute_per_label_f1(Y_test, model_preds[mname], thr)

# Head / Torso / Tail color coding by frequency terciles
sorted_idx = np.argsort(-train_freq)
n_labels = len(sorted_idx)
head_idx = set(sorted_idx[:n_labels // 3])
torso_idx = set(sorted_idx[n_labels // 3:2 * n_labels // 3])
colors_ht = []
for j in range(n_labels):
    if j in head_idx:
        colors_ht.append('#2196F3')
    elif j in torso_idx:
        colors_ht.append('#FF9800')
    else:
        colors_ht.append('#F44336')

from matplotlib.lines import Line2D
ht_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=8, label='Head'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', markersize=8, label='Torso'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=8, label='Tail'),
]

# Individual head/tail scatter
for mname, title, fname in [
    ("model_a", "Model A (TF-IDF + SGD)", "head_tail_a.png"),
    ("model_b", "Model B (ClinicalBERT)", "head_tail_b.png"),
    ("model_c", "Model C v1 (Chunk-BERT)", "head_tail_c.png"),
    ("model_c_v2", "Model C v2 (Focal)", "head_tail_c_v2.png"),
    ("model_d", "Model D (BiLSTM-LAAT)", "head_tail_d.png"),
]:
    if mname not in per_label_f1:
        continue
    f1s = per_label_f1[mname]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(train_freq, f1s, c=colors_ht, s=50, alpha=0.75, edgecolors='white', linewidths=0.5)
    ax.set_xlabel("Training Frequency (log scale)", fontsize=12)
    ax.set_ylabel("Test F1", fontsize=12)
    ax.set_title(f"{title} - Per-Label F1 vs. Frequency", fontsize=13)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(handles=ht_legend, loc='lower right', fontsize=10)
    plt.tight_layout()
    save(fname)

# Comparison head/tail (2x2 grid)
comp_models = [
    ("model_a", "Model A (TF-IDF + SGD)"),
    ("model_d", "Model D (BiLSTM-LAAT)"),
    ("model_c", "Model C v1 (Chunk-BERT)"),
    ("model_b", "Model B (ClinicalBERT)"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (mname, title) in zip(axes.flat, comp_models):
    if mname in per_label_f1:
        f1s = per_label_f1[mname]
        ax.scatter(train_freq, f1s, c=colors_ht, s=45, alpha=0.75, edgecolors='white', linewidths=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Training Frequency", fontsize=10)
    ax.set_ylabel("Test F1", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
fig.legend(handles=ht_legend, loc='lower center', ncol=3, fontsize=11, frameon=True)
plt.tight_layout(rect=[0, 0.04, 1, 1])
save("head_tail_comparison.png")

# ══════════════════════════════════════════════════════════════════════
# 11. PER-LABEL SCATTER - Model D vs others
# ══════════════════════════════════════════════════════════════════════
print("8. Per-label scatter comparison")

if "model_d" in per_label_f1:
    f1_d = per_label_f1["model_d"]
    scatter_pairs = [
        ("model_a", "TF-IDF + SGD", "Model D vs. Model A"),
        ("model_b", "ClinicalBERT", "Model D vs. Model B"),
        ("model_c", "Chunk-BERT", "Model D vs. Model C v1"),
    ]
    valid_pairs = [(m, xl, t) for m, xl, t in scatter_pairs if m in per_label_f1]
    n = len(valid_pairs)
    if n > 0:
        fig, axes = plt.subplots(1, n, figsize=(15, 5))
        if n == 1:
            axes = [axes]
        for ax, (mname, xlabel, title) in zip(axes, valid_pairs):
            f1_other = per_label_f1[mname]
            ax.scatter(f1_other, f1_d, s=50, alpha=0.7, edgecolors='white', linewidths=0.5, color='#4488CC')
            ax.plot([0, 1], [0, 1], '--', color='#E57373', lw=1.5, alpha=0.8)
            ax.set_xlabel(f"{xlabel} F1", fontsize=11)
            ax.set_ylabel("BiLSTM-LAAT F1", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=10)
        plt.tight_layout()
        save("model_d_scatter.png")

    # A vs C scatter
    if "model_a" in per_label_f1 and "model_c" in per_label_f1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(per_label_f1["model_c"], per_label_f1["model_a"], s=55, alpha=0.7,
                   edgecolors='white', linewidths=0.5, color='#4488CC')
        ax.plot([0, 1], [0, 1], '--', color='#E57373', lw=1.5, alpha=0.8)
        ax.set_xlabel("Model C v1 (Chunk-BERT) F1", fontsize=12)
        ax.set_ylabel("Model A (TF-IDF + SGD) F1", fontsize=12)
        ax.set_title("Model A vs. Model C v1 - Per-Label F1", fontsize=13)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt2))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save("a_vs_c_scatter.png")

# ══════════════════════════════════════════════════════════════════════
# 12. EDA plots -note length, label cardinality, code frequency, splits
# ══════════════════════════════════════════════════════════════════════
print("9. EDA plots")

# Try to load cohort data for note lengths
cohort_path = os.path.join(datasets_dir, "cohort_train.parquet")
if os.path.exists(cohort_path):
    try:
        df_train = pd.read_parquet(cohort_path)
        if "text" in df_train.columns:
            word_counts = df_train["text"].str.split().str.len()
        elif "note_text" in df_train.columns:
            word_counts = df_train["note_text"].str.split().str.len()
        else:
            word_counts = None

        if word_counts is not None:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(word_counts.clip(upper=4000), bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
            ax.set_xlabel("Word Count")
            ax.set_ylabel("Number of Notes")
            ax.set_title("Discharge Note Length Distribution (Train)")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt0))
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            save("note_length_dist.png")
    except Exception as e:
        print(f"  Could not load cohort: {e}")

# Label cardinality
labels_per = Y_test.sum(axis=1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(labels_per, bins=range(0, int(labels_per.max()) + 2), color="steelblue",
        edgecolor="white", linewidth=0.3, align="left")
ax.set_xlabel("Number of Labels per Admission")
ax.set_ylabel("Number of Admissions")
ax.set_title("Label Cardinality Distribution (Test Set)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save("label_cardinality.png")

# Code frequency (top-50)
code_freq_test = Y_test.sum(axis=0)
sorted_idx = np.argsort(code_freq_test)[::-1]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(50), code_freq_test[sorted_idx], color="steelblue", edgecolor="navy", linewidth=0.3)
ax.set_xticks(range(50))
ax.set_xticklabels([code_names[i] for i in sorted_idx], rotation=90, fontsize=6)
ax.set_xlabel("ICD-10 Code")
ax.set_ylabel("Frequency (Test Set)")
ax.set_title("Top-50 ICD-10 Code Frequencies (Long-Tail)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save("code_freq_tail.png")

# Split counts
splits = {"Train": 85081, "Validation": 18371, "Test": 18852}
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(splits.keys(), splits.values(), color=["steelblue", "coral", "seagreen"],
              edgecolor="black", linewidth=0.5)
for bar, v in zip(bars, splits.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Number of Admissions")
ax.set_title("Data Split Sizes")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save("split_counts.png")

print("\nDone! All figures regenerated.")
