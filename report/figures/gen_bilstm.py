import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 8.5))
ax.set_xlim(-1.5, 16.0)
ax.set_ylim(-0.2, 9.5)
ax.set_aspect('equal')
ax.axis('off')

# === Colors matching reference ===
c_emb_circle = '#E8B84B'
c_emb_box = '#C89A30'
c_lstm_circle = '#1E3FA0'
c_lstm_box = '#4477CC'
c_attn_bg = '#D4E8C8'
c_attn_border = '#6A9A50'
c_attn_text = '#2A4A14'
c_lattn_circle = '#1A6020'
c_lattn_box = '#389038'
c_ffnn_bg = '#CCD4E6'
c_ffnn_border = '#6080AA'
c_out_circle = '#3858A0'
c_arrow = '#3A3A3A'
c_bracket = '#8A6A1A'

circ_r = 0.26
circ_sep = 0.62  # center-to-center, 2 circles per box

# Row y-positions: w1, w2, dots, wn
y_rows = [7.8, 6.0, 4.2, 2.4]

# Column x-positions
x_emb = 0.8
x_lstm = 3.8
x_attn = 6.6
x_lattn = 8.8
x_ffnn = 11.8
x_out = 14.2

box_w = 1.7
box_h = 1.0

def hollow_circle(x, y, r, color, lw=2.2):
    ax.add_patch(plt.Circle((x, y), r, fill=False, edgecolor=color, linewidth=lw))

def filled_circle(x, y, r, color):
    ax.add_patch(plt.Circle((x, y), r, facecolor=color, edgecolor=color, linewidth=0.8))

def dashed_box(cx, cy, w, h, color, lw=2.0):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor='none', edgecolor=color, linewidth=lw, linestyle=(0, (5, 3))
    ))

def arr(x1, y1, x2, y2, lw=1.6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=lw, mutation_scale=15))

def darr(x, y1, y2, lw=1.3):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='<->', color=c_arrow, lw=lw))

def bracket_above(x1, x2, y, label):
    mid = (x1 + x2) / 2
    bh = 0.15
    ax.plot([x1, x1, x2, x2], [y, y+bh, y+bh, y], color=c_bracket, lw=1.8, clip_on=False)
    ax.text(mid, y + bh + 0.1, label, fontsize=14, ha='center', va='bottom',
            fontfamily='serif', style='italic')

def dots(x, y, sz=26):
    ax.text(x, y, r'$\vdots$', fontsize=sz, ha='center', va='center', fontfamily='serif')

font_hdr = dict(fontsize=16, fontweight='bold', ha='center', va='bottom', fontfamily='serif')

# ================================================================
# EMBEDDING LAYER
# ================================================================
emb_info = [
    (y_rows[0], '$w_1$', '$h_1$'),
    (y_rows[1], '$w_2$', '$h_2$'),
    (y_rows[2], None, None),
    (y_rows[3], '$w_n$', '$h_n$'),
]

for yy, wlbl, hlbl in emb_info:
    if wlbl is None:
        dots(x_emb, yy)
        dots(x_emb + box_w/2 + 0.6, yy)
        continue
    dashed_box(x_emb, yy, box_w, box_h, c_emb_box)
    hollow_circle(x_emb - circ_sep/2, yy, circ_r, c_emb_circle)
    hollow_circle(x_emb + circ_sep/2, yy, circ_r, c_emb_circle)
    ax.text(x_emb - box_w/2 - 0.3, yy, wlbl, fontsize=15, ha='right', va='center',
            fontfamily='serif', fontweight='bold')
    arr(x_emb + box_w/2 + 0.08, yy, x_emb + box_w/2 + 0.48, yy)
    ax.text(x_emb + box_w/2 + 0.55, yy, hlbl, fontsize=14, ha='left', va='center',
            fontfamily='serif')

ax.text(x_emb, 9.15, 'Embedding layer', **font_hdr)
bracket_above(x_emb - box_w/2 + 0.05, x_emb + box_w/2 - 0.05, 8.65, '$d_e$')

# ================================================================
# BiLSTM
# ================================================================
for i, yy in enumerate(y_rows):
    if i == 2:
        dots(x_lstm, yy)
        continue
    dashed_box(x_lstm, yy, box_w, box_h, c_lstm_box)
    filled_circle(x_lstm - circ_sep/2, yy, circ_r, c_lstm_circle)
    filled_circle(x_lstm + circ_sep/2, yy, circ_r, c_lstm_circle)

# Bidirectional arrows
for i in range(len(y_rows) - 1):
    ya, yb = y_rows[i], y_rows[i+1]
    if i == 1:
        darr(x_lstm, ya - box_h/2 - 0.08, yb + 0.35)
    elif i == 2:
        darr(x_lstm, ya - 0.35, yb + box_h/2 + 0.08)
    else:
        darr(x_lstm, ya - box_h/2 - 0.08, yb + box_h/2 + 0.08)

ax.text(x_lstm, 9.15, 'BiLSTM', **font_hdr)
bracket_above(x_lstm - box_w/2 + 0.05, x_lstm + box_w/2 - 0.05, 8.65, '$2u$')

# ================================================================
# ATTENTION BOX
# ================================================================
attn_cy = (y_rows[0] + y_rows[-1]) / 2
attn_h = y_rows[0] - y_rows[-1] + 1.2
attn_w = 0.9

ax.add_patch(mpatches.FancyBboxPatch(
    (x_attn - attn_w/2, attn_cy - attn_h/2), attn_w, attn_h,
    boxstyle="round,pad=0.15",
    facecolor=c_attn_bg, edgecolor=c_attn_border, linewidth=2.0
))
ax.text(x_attn, attn_cy, 'Attention', fontsize=15, fontweight='bold',
        ha='center', va='center', fontfamily='serif', color=c_attn_text, rotation=90)

# BiLSTM -> Attention
arr(x_lstm + box_w/2 + 0.12, attn_cy, x_attn - attn_w/2 - 0.08, attn_cy, lw=2.0)

# Attention -> fan out point
fan_x = x_attn + attn_w/2 + 0.5
arr(x_attn + attn_w/2 + 0.08, attn_cy, fan_x, attn_cy, lw=2.0)

# Dots column after attention
dots(fan_x + 0.2, y_rows[2])

# ================================================================
# LABEL ATTENTION LAYER
# ================================================================
lattn_info = [
    (y_rows[0], '$v_1$'),
    (y_rows[1], ''),
    (y_rows[2], None),
    (y_rows[3], '$v_L$'),
]

for yy, vlbl in lattn_info:
    if vlbl is None:
        dots(x_lattn, yy)
        continue
    dashed_box(x_lattn, yy, box_w, box_h, c_lattn_box)
    filled_circle(x_lattn - circ_sep/2, yy, circ_r, c_lattn_circle)
    filled_circle(x_lattn + circ_sep/2, yy, circ_r, c_lattn_circle)
    if vlbl:
        ax.text(x_lattn - box_w/2 - 0.3, yy, vlbl, fontsize=14, ha='right', va='center',
                fontfamily='serif')

ax.text(x_lattn, 9.15, 'Label Attention Layer', **font_hdr)
bracket_above(x_lattn - box_w/2 + 0.05, x_lattn + box_w/2 - 0.05, 8.65, '$2u$')

# ================================================================
# OUTPUT LAYER — FFNN + sigmoid circles
# ================================================================
ffnn_info = [
    (y_rows[0], '$\\mathrm{FFNN}_1$', '$y_1$'),
    (y_rows[1], '', ''),
    (y_rows[2], None, None),
    (y_rows[3], '$\\mathrm{FFNN}_L$', '$y_L$'),
]

ffnn_w, ffnn_h = 1.3, 0.7

for yy, flbl, olbl in ffnn_info:
    if flbl is None:
        dots(x_ffnn, yy)
        dots(x_out, yy)
        dots((x_ffnn + ffnn_w/2 + x_out) / 2, yy, 20)
        continue

    # label attn -> FFNN
    arr(x_lattn + box_w/2 + 0.1, yy, x_ffnn - ffnn_w/2 - 0.08, yy)

    # FFNN box
    ax.add_patch(mpatches.FancyBboxPatch(
        (x_ffnn - ffnn_w/2, yy - ffnn_h/2), ffnn_w, ffnn_h,
        boxstyle="round,pad=0.08",
        facecolor=c_ffnn_bg, edgecolor=c_ffnn_border, linewidth=1.8
    ))
    if flbl:
        ax.text(x_ffnn, yy, flbl, fontsize=12, ha='center', va='center', fontfamily='serif')

    # FFNN -> output circle
    arr(x_ffnn + ffnn_w/2 + 0.08, yy, x_out - circ_r - 0.18, yy)

    # Output circle
    hollow_circle(x_out, yy, circ_r + 0.02, c_out_circle, lw=2.2)

    # Label
    if olbl:
        ax.text(x_out + circ_r + 0.22, yy, olbl, fontsize=14, ha='left', va='center',
                fontfamily='serif')

ax.text((x_ffnn + x_out) / 2 + 0.5, 9.15, 'Output Layer', **font_hdr)
ax.text(x_out, 8.65, '$\\mathit{sigmoid}$', fontsize=14, ha='center', va='bottom',
        fontfamily='serif', fontweight='bold')

plt.tight_layout(pad=0.2)
plt.savefig('/Users/devasaisundertangella/Documents/Spring2025/NLP/Final Project/ICD-CPT-Code-Prediction/report/figures/bilstm_laat_arch.png',
            dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Done")
