"""
Feature_Importance_Heatmap.py
Purpose: Load the trained Random Forest models (one per community type),
         extract feature importances, normalize, rescale to 0–1, and generate
         a heatmap with a custom warm colormap (from #FEDA78 to dark red).
         The x‑axis order follows the ecological strategies.
         All cell annotations are displayed in white.
Input:   Binary_SHAP_Plots/  (folder containing RF_C*.pkl models)
Output:  Binary_SHAP_Plots/Feature_importance_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

model_dir = "Binary_SHAP_Plots"

feature_labels = ['Conductivity', 'Temperature', 'Distance\nto vent',
                  'Water depth', 'Slope', 'Backscatter\nintensity']

# Cluster short labels (internal IDs 0–5)
cluster_short = {
    0: 'MD',   # Mussel-dominated
    1: 'SD',   # Snail-dominated
    2: 'AD',   # Anemone-dominated
    3: 'OM',   # Opportunistic mixed
    4: 'TM',   # Tubeworm–mussel
    5: 'MS'    # Mussel–shrimp
}

# New x‑axis order: Vent‑affiliated (SD, OM, MD) → Patchy (TM, MS) → Vent‑repelled (AD)
new_order = [1, 3, 0, 4, 5, 2]   # SD, OM, MD, TM, MS, AD
new_labels = [cluster_short[c] for c in new_order]

n_features = len(feature_labels)
n_clusters = 6
importance_matrix = np.zeros((n_features, n_clusters))

for cid in range(n_clusters):
    model_path = os.path.join(model_dir, f'RF_C{cid}.pkl')
    if os.path.exists(model_path):
        rf = joblib.load(model_path)
        importance_matrix[:, cid] = rf.feature_importances_
        print(f'Loaded: RF_C{cid}.pkl')
    else:
        print(f'Warning: RF_C{cid}.pkl not found, using zeros')
        importance_matrix[:, cid] = 0

# Normalize across features (each feature sums to 1 across clusters)
importance_norm = importance_matrix / importance_matrix.sum(axis=1, keepdims=True)

# Reorder columns according to the new x‑axis order
importance_norm = importance_norm[:, new_order]

# Rescale to 0–1 using the global min and max of the reordered matrix
vmin_actual = importance_norm.min()
vmax_actual = importance_norm.max()
importance_scaled = (importance_norm - vmin_actual) / (vmax_actual - vmin_actual)

# Create custom warm colormap from light yellow (#FEDA78) to dark red (#D73027)
cmap = LinearSegmentedColormap.from_list('custom_warm', ['#FFF5E0', '#FEDA78', '#E05952'])

fig_width = 9 / 2.54
fig_height = 7 / 2.54
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

im = ax.imshow(importance_scaled, aspect='auto', cmap=cmap, vmin=0, vmax=1)

ax.set_xticks(range(n_clusters))
ax.set_xticklabels(new_labels, rotation=0, ha='center', fontsize=7)
ax.set_yticks(range(n_features))
ax.set_yticklabels(feature_labels, fontsize=7)

# Annotate with the scaled values (0–1), all in white
for i in range(n_features):
    for j in range(n_clusters):
        val = importance_scaled[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7, color='black', fontweight='bold')

# Colorbar with ticks at 0.0, 0.25, 0.5, 0.75, 1.0
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                    ticks=[0, 0.25, 0.5, 0.75, 1.0])
cbar.set_label('Relative importance', fontsize=7, labelpad=2)
cbar.ax.tick_params(labelsize=7, pad=0.5)

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.tick_params(labelsize=6, pad=1)

plt.tight_layout(pad=0.3)

output_path = os.path.join(model_dir, 'Feature_importance_heatmap.png')
fig.savefig(output_path, format='png', dpi=500)
print(f'\nSaved: {output_path}')