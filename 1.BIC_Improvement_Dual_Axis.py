"""
BIC_Improvement_Dual_Axis_Broken.py
Purpose: Dual‑axis plot of BIC (left) and improvement rate (right) for K=2..10
         with a broken right y‑axis.  Only the raw curves are shown; no K=6
         highlights or threshold lines.  Axis titles are not bold.
         BIC value is shown in Nature‑style scientific notation (×10ⁿ).
Input:   2.Biological Statistics_5 Ecological Indices.csv
Output:  bic_improvement_broken_axis.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# ================== 0. Plot settings ==================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# ================== 1. Load data & compute BIC ==================
input_path = "2.Biological Statistics_5 Ecological Indices.csv"
df = pd.read_csv(input_path, sep=',')

count_cols = [
    'Anemone count', 'Thick sea cucumber count', 'Williams galatheid crab count',
    'Spherical sea cucumber count', 'Mimic snail count', 'Coral count',
    'Red shrimp count', 'Fish count', 'Tubeworm count',
    'Mussel count', 'Chiridota count', 'Dead mussel count'
]
size_cols = [
    'Anemone mean area', 'Thick sea cucumber mean area', 'Williams galatheid crab mean area',
    'Spherical sea cucumber mean area', 'Mimic snail mean area', 'Coral mean area',
    'Red shrimp mean area', 'Fish mean area', 'Tubeworm mean area',
    'Mussel mean area', 'Chiridota mean area', 'Dead mussel mean area'
]
diversity_cols = ['Total abundance', 'Species richness', 'Shannon index',
                  'Simpson index', 'Pielou evenness']

for col in count_cols + size_cols + diversity_cols:
    if col not in df.columns:
        df[col] = 0
df[count_cols] = df[count_cols].fillna(0)
df[size_cols] = df[size_cols].fillna(0)
df[diversity_cols] = df[diversity_cols].fillna(0)

total_ind = df[count_cols].sum(axis=1).values.reshape(-1, 1)
rel_abund = np.divide(df[count_cols].values, total_ind,
                      out=np.zeros_like(df[count_cols].values, dtype=float),
                      where=total_ind != 0)
X = np.hstack([rel_abund, df[size_cols].values, df[diversity_cols].values])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute BIC for K=2..10
K_range = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
bic = []
for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          random_state=42, n_init=10)
    gmm.fit(X_scaled)
    bic.append(gmm.bic(X_scaled))
bic = np.array(bic)

# Compute improvement rate (K=3..10) and prepend a dummy 0 for K=2
imp = []
for i in range(1, len(bic)):
    imp.append((bic[i-1] - bic[i]) / abs(bic[i-1]) * 100)
improvement = np.array([0.0] + imp)   # length 9, aligned with K_range

# ================== 2. Nature colours ==================
color_bic = '#4D4D4D'   # dark grey
color_imp = '#D55E00'   # dark orange

# ================== 3. Broken axis mapping ==================
H_low_nominal = 2.0 / 3.0
gap = 0.04
break_y_top_low = H_low_nominal - gap / 2
break_y_bottom_high = H_low_nominal + gap / 2
H_high = 1.0 - break_y_bottom_high

low_val_max = 20.0
up_val_min = 100.0
up_val_max = 120.0

def val_to_y(v):
    if v <= low_val_max:
        return v / low_val_max * break_y_top_low
    else:
        return break_y_bottom_high + (v - up_val_min) / (up_val_max - up_val_min) * H_high

y_mapped = np.array([val_to_y(v) for v in improvement])

# Helper to format numbers into Nature-style scientific notation (e.g. -3.8×10⁶)
def nature_sci(x, decimals=1):
    s = f'{x:.{decimals}e}'
    base, exp_str = s.split('e')
    exp = int(exp_str)
    superscript_map = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
    if exp == 0:
        return base
    else:
        exp_sup = str(exp).translate(superscript_map)
        return f'{base}×10{exp_sup}'

# ================== 4. Plotting ==================
fig, ax1 = plt.subplots(figsize=(7/2.54, 7/2.54))

# Left axis: BIC (only the curve)
ax1.plot(K_range, bic, 'o-', color=color_bic, markersize=4, linewidth=1, label='BIC')

ax1.set_xlabel('Number of clusters (K)', labelpad=0)
ax1.set_ylabel('BIC', color=color_bic, labelpad=-1)
ax1.tick_params(axis='y', labelcolor=color_bic, direction='in', length=3, labelsize=7)
ax1.tick_params(axis='x', direction='in', length=3, labelsize=7)

# Right axis: improvement (broken scale)
ax2 = ax1.twinx()
ax2.plot(K_range[1:], y_mapped[1:], 's-', color=color_imp, markersize=4, linewidth=1,
         label='Improvement (%)')

# ---- Right axis ticks and labels (orange) ----
tick_values = [0, 5, 10, 15, 100, 110]
tick_y = [val_to_y(v) for v in tick_values]
ax2.set_yticks(tick_y)
ax2.set_yticklabels([str(v) for v in tick_values], color=color_imp)
ax2.set_ylabel('Improvement (%)', color=color_imp, labelpad=-1)
ax2.tick_params(axis='y', colors=color_imp, direction='in', length=3, labelsize=7)

# ---- Break indicators (diagonal slashes) – orange ----
slash_x_left = 0.97
slash_x_right = 1.03
slash_kw = dict(color=color_imp, lw=0.8, linestyle='-', transform=ax2.transAxes, clip_on=False)
ax2.plot([slash_x_left, slash_x_right],
         [break_y_bottom_high - 0.015, break_y_bottom_high + 0.015], **slash_kw)
ax2.plot([slash_x_left, slash_x_right],
         [break_y_top_low - 0.015, break_y_top_low + 0.015], **slash_kw)

# White patch to hide spine between the break
rect = plt.Rectangle((0.98, break_y_top_low), 0.04, break_y_bottom_high - break_y_top_low,
                     transform=ax2.transAxes, facecolor='white', edgecolor='none',
                     zorder=10, clip_on=False)
ax2.add_patch(rect)

# Colour the right axis spine orange
ax2.spines['right'].set_color(color_imp)

# ================== 5. Legend (Nature style, upper right) ==================
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
handles = lines1 + lines2
labels = labels1 + labels2

leg = ax1.legend(handles, labels, loc='upper right', frameon=True,
                 fancybox=True, framealpha=0.85, edgecolor='#CCCCCC',
                 facecolor='white', borderpad=0.3, labelspacing=0.3,
                 handlelength=1.0, handletextpad=0.5,
                 fontsize=7)
leg.get_frame().set_linewidth(0.5)

# Final spine adjustments
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout(pad=0.3)
plt.savefig('bic_improvement_broken_axis.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
plt.close()
print("Broken-axis plot saved as bic_improvement_broken_axis.png")
