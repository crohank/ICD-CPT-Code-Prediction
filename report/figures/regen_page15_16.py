import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path

BASE = Path('/Users/devasaisundertangella/Documents/Spring2025/NLP/Final Project/ICD-CPT-Code-Prediction/data')
OUT = Path('/Users/devasaisundertangella/Documents/Spring2025/NLP/Final Project/ICD-CPT-Code-Prediction/report/figures')

# Load ground truth
Y_test = np.load(BASE / 'Y_test.npy')
train_freq = np.load(BASE / 'Y_test.npy').sum(axis=0)  # fallback: use test freq as proxy
# Try loading Y_train for actual training frequencies
try:
    Y_train = np.load(BASE / 'Y_train.npy')
    train_freq = Y_train.sum(axis=0)
except:
    pass

n_labels = Y_test.shape[1]

# Load predictions and thresholds
models = {}

# Model A
P_a = np.load(BASE / 'models/model_a/P_test.npy')
models['Model A (TF-IDF + SGD)'] = {'P': P_a, 'thr': 0.525}

# Model B
P_b = np.load(BASE / 'models/model_b/P_test.npy')
models['Model B (ClinicalBERT)'] = {'P': P_b, 'thr': 0.275}

# Model C v1
P_c = np.load(BASE / 'models/model_c/P_test.npy')
models['Model C v1 (Chunk-BERT)'] = {'P': P_c, 'thr': 0.625}

# Model C v2
try:
    P_c2 = np.load(BASE / 'models/model_c/v2/P_test_calibrated.npy')
    models['Model C v2 (Focal)'] = {'P': P_c2, 'thr': 0.05}
except:
    pass

# Model D
try:
    P_d = np.load(BASE / 'models/model_d/P_test_calibrated.npy')
    models['Model D (BiLSTM-LAAT)'] = {'P': P_d, 'thr': 0.30}
except:
    P_d = np.load(BASE / 'models/model_d/P_test_uncalibrated.npy')
    models['Model D (BiLSTM-LAAT)'] = {'P': P_d, 'thr': 0.30}

# Compute per-label F1 for each model
per_label_f1 = {}
for name, m in models.items():
    Y_pred = (m['P'] >= m['thr']).astype(int)
    f1s = []
    for j in range(n_labels):
        f1s.append(f1_score(Y_test[:, j], Y_pred[:, j], zero_division=0))
    per_label_f1[name] = np.array(f1s)

# Head / Torso / Tail buckets
sorted_idx = np.argsort(-train_freq)
n = len(sorted_idx)
head_idx = set(sorted_idx[:n//3])
torso_idx = set(sorted_idx[n//3:2*n//3])
tail_idx = set(sorted_idx[2*n//3:])

colors_ht = []
for j in range(n_labels):
    if j in head_idx:
        colors_ht.append('#2196F3')  # blue
    elif j in torso_idx:
        colors_ht.append('#FF9800')  # orange
    else:
        colors_ht.append('#F44336')  # red

# ── Formatting helpers ──
def fmt_ax(ax):
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

FIGSIZE = (7, 5)  # consistent size for all single plots
DPI = 150

# ════════════════════════════════════════════
# 1. Individual head/tail scatter plots
# ════════════════════════════════════════════
plot_map = {
    'Model A (TF-IDF + SGD)': ('head_tail_a.png', 'Model A (TF-IDF + SGD)'),
    'Model B (ClinicalBERT)': ('head_tail_b.png', 'Model B (ClinicalBERT)'),
    'Model C v1 (Chunk-BERT)': ('head_tail_c.png', 'Model C v1 (Chunk-BERT)'),
    'Model D (BiLSTM-LAAT)': ('head_tail_d.png', 'Model D (BiLSTM-LAAT)'),
}
if 'Model C v2 (Focal)' in per_label_f1:
    plot_map['Model C v2 (Focal)'] = ('head_tail_c_v2.png', 'Model C v2 (Focal)')

for name, (fname, title) in plot_map.items():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    f1s = per_label_f1[name]
    ax.scatter(train_freq, f1s, c=colors_ht, s=50, alpha=0.75, edgecolors='white', linewidths=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Training Frequency (log scale)', fontsize=12)
    ax.set_ylabel('Test F1', fontsize=12)
    ax.set_title(f'{title} - Per-Label F1 vs. Frequency', fontsize=13)
    fmt_ax(ax)
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=8, label='Head'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', markersize=8, label='Torso'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=8, label='Tail'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {fname}')

# ════════════════════════════════════════════
# 2. Head/tail comparison (all models, 2x2 grid)
# ════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
comp_models = [
    ('Model A (TF-IDF + SGD)', 'Model A (TF-IDF + SGD)'),
    ('Model D (BiLSTM-LAAT)', 'Model D (BiLSTM-LAAT)'),
    ('Model C v1 (Chunk-BERT)', 'Model C v1 (Chunk-BERT)'),
    ('Model B (ClinicalBERT)', 'Model B (ClinicalBERT)'),
]
from matplotlib.lines import Line2D
for ax, (name, title) in zip(axes.flat, comp_models):
    if name in per_label_f1:
        f1s = per_label_f1[name]
        ax.scatter(train_freq, f1s, c=colors_ht, s=45, alpha=0.75, edgecolors='white', linewidths=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Training Frequency', fontsize=10)
    ax.set_ylabel('Test F1', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    fmt_ax(ax)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=8, label='Head'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', markersize=8, label='Torso'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=8, label='Tail'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, frameon=True)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT / 'head_tail_comparison.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved head_tail_comparison.png')

# ════════════════════════════════════════════
# 3. Model D scatter comparison (2x1 layout)
# ════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
scatter_pairs = [
    ('Model A (TF-IDF + SGD)', 'Model D (BiLSTM-LAAT)', 'Model D vs. Model A'),
    ('Model B (ClinicalBERT)', 'Model D (BiLSTM-LAAT)', 'Model D vs. Model B'),
    ('Model C v1 (Chunk-BERT)', 'Model D (BiLSTM-LAAT)', 'Model D vs. Model C v1'),
]

for ax, (xname, yname, title) in zip(axes, scatter_pairs):
    xf1 = per_label_f1[xname]
    yf1 = per_label_f1[yname]
    ax.scatter(xf1, yf1, s=50, alpha=0.7, edgecolors='white', linewidths=0.5, color='#4488CC')
    ax.plot([0, 1], [0, 1], '--', color='#E57373', lw=1.5, alpha=0.8)
    ax.set_xlabel(f'{xname.split("(")[1].rstrip(")")} F1', fontsize=11)
    ax.set_ylabel('BiLSTM-LAAT F1', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(OUT / 'model_d_scatter.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved model_d_scatter.png')

# ════════════════════════════════════════════
# 4. A vs C scatter (single plot)
# ════════════════════════════════════════════
fig, ax = plt.subplots(figsize=FIGSIZE)
xf1 = per_label_f1['Model C v1 (Chunk-BERT)']
yf1 = per_label_f1['Model A (TF-IDF + SGD)']
ax.scatter(xf1, yf1, s=55, alpha=0.7, edgecolors='white', linewidths=0.5, color='#4488CC')
ax.plot([0, 1], [0, 1], '--', color='#E57373', lw=1.5, alpha=0.8)
ax.set_xlabel('Model C v1 (Chunk-BERT) F1', fontsize=12)
ax.set_ylabel('Model A (TF-IDF + SGD) F1', fontsize=12)
ax.set_title('Model A vs. Model C v1 - Per-Label F1', fontsize=13)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'a_vs_c_scatter.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved a_vs_c_scatter.png')

print('\nAll plots regenerated successfully!')
