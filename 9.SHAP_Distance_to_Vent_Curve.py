"""
SHAP_Distance_to_Vent_Curve.py
Purpose: For each community type, load the trained binary RF model,
         compute SHAP values for the "Distance to vent" feature, smooth
         and plot the SHAP dependence curve.  Vertical dashed lines mark
         the distance at which the SHAP value crosses zero for selected
         community types (downward from y=0 only), and a triangle is placed
         at the crossing point (upward for negative-to-positive, downward
         for positive-to-negative).  Crossing distances are labelled above
         the x‑axis with a white background to mask the dashed line.
         Legend order follows Cluster 0–5.
Input:   3.Biological Statistics_5 Ecological Indices_Clustering Results.csv
         Binary_SHAP_Plots/  (folder containing RF_C*.pkl models)
Output:  Fig4b_SHAP_distance_to_vent.png / .svg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.02

# ================== Input ==================
df = pd.read_csv("3.Biological Statistics_5 Ecological Indices_Clustering Results.csv")

feature_names = ['Conductivity', 'Temperature', 'Distance to vent',
                 'Water depth', 'Slope', 'Backscatter intensity']
df = df.dropna(subset=feature_names + ['Cluster_k6'])
df['Water depth'] = df['Water depth'].abs()

X_all = df[feature_names].values
X_dist = X_all[:, 2]

model_dir = "Binary_SHAP_Plots"   # folder with trained RF models

cluster_names = {
    0: 'MD',
    1: 'SD',
    2: 'AD',
    3: 'OM',
    4: 'TM',
    5: 'MS',
}

# Updated colors to match t-SNE/radar chart: blue, green, purple, pink, yellow, cyan
cluster_colors = {
    0: '#1f77b4',   # MD - blue
    1: '#2ca02c',   # SD - green
    2: '#9467bd',   # AD - purple
    3: '#e377c2',   # OM - pink
    4: '#BCBD22',   # TD - yellow
    5: '#17becf',   # MS - cyan
}

# Communities whose zero crossings will be marked (1 = positive→negative, -1 = negative→positive)
target_communities = {
    0: 1,   # Mussel dominated
    3: 1,   # Opportunistic mixed
    1: 1,   # Snail dominated
    2: -1   # Anemone dominated
}

# ================== Helper: find zero crossing ==================
def find_zero_crossing(x_sorted, y_smooth):
    """Return the x-coordinate where y_smooth crosses zero (linear interpolation)."""
    signs = np.sign(y_smooth)
    for i in range(len(signs)-1):
        if signs[i] != signs[i+1] and signs[i] != 0 and signs[i+1] != 0:
            x0, x1 = x_sorted[i], x_sorted[i+1]
            y0, y1 = y_smooth[i], y_smooth[i+1]
            return x0 - y0 * (x1 - x0) / (y1 - y0)
    return None   # no crossing found

# ================== Plot ==================
fig, ax = plt.subplots(figsize=(6.5/2.54, 7.1/2.54))

critical_distances = {}   # store (distance, direction) for each target community

# Plot in Cluster 0–5 order for legend consistency
for cid in range(6):
    model_path = os.path.join(model_dir, f'RF_C{cid}.pkl')
    if not os.path.exists(model_path):
        continue

    rf = joblib.load(model_path)
    explainer = shap.TreeExplainer(rf)
    shap_vals_all = explainer.shap_values(X_all)

    if len(shap_vals_all.shape) == 3:
        shap_feat = shap_vals_all[:, 2, 1]
    else:
        shap_feat = shap_vals_all[:, 2]

    color = cluster_colors[cid]
    label = cluster_names[cid]

    # Sort and smooth
    sorted_idx = np.argsort(X_dist)
    x_sorted = X_dist[sorted_idx]
    shap_sorted = shap_feat[sorted_idx]
    window = max(len(X_dist) // 40, 20)
    smooth = np.convolve(shap_sorted, np.ones(window)/window, mode='same')

    ax.plot(x_sorted, smooth, color=color, linewidth=1.2, label=label, alpha=0.9)

    if cid in target_communities:
        crossing = find_zero_crossing(x_sorted, smooth)
        if crossing is not None:
            critical_distances[cid] = (crossing, target_communities[cid])
            print(f"Cluster {cid} ({label}) crosses zero at x = {crossing:.1f} m")

# ================== Add vertical dashes, triangles, and labels ==================
ymin, ymax = ax.get_ylim()
for cid, (distance, direction) in critical_distances.items():
    color = cluster_colors[cid]
    # downward dashed line from y=0 to bottom of the plot
    ax.plot([distance, distance], [0, ymin], color=color, linestyle='--', linewidth=0.6, alpha=0.7)
    # directional triangle at the crossing point
    marker = 'v' if direction == 1 else '^'   # 1=positive→negative (downward), -1=negative→positive (upward)
    ax.plot(distance, 0, marker=marker, color=color, markersize=5, markeredgewidth=0.5, markeredgecolor=color)
    # label above the x-axis with white background to mask the dashed line
    ax.text(distance, 0.055, f'{distance:.1f}',
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=6, color=color,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
# reference line at 50 m (optional, kept from original)
ax.axvline(x=50, color='#888888', linestyle=':', linewidth=0.5, alpha=0.4)

ax.set_xlabel('Distance to vent (m)', fontsize=7, labelpad=1)
ax.set_ylabel('SHAP value', fontsize=7, labelpad=1)
ax.set_xlim(0, 100)
ax.set_ylim(ymin+0.01, 0.14)   # keep original y-limits set before
ax.legend(loc='upper left', bbox_to_anchor=(0.01, 1), frameon=True, fancybox=True, framealpha=0.85,
          borderaxespad=0.0, edgecolor='#CCCCCC', facecolor='white', fontsize=6, labelspacing=0.2,
          handlelength=1.2, handleheight=0.8, ncol=2, columnspacing=0.5, handletextpad=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=6, pad=1)
ax.grid(alpha=0.12, linewidth=0.3)

plt.tight_layout(pad=0.2)
fig.savefig('SHAP_distance_to_vent.png', dpi=600)
fig.savefig('SHAP_distance_to_vent.svg')
plt.close()
print("Saved: Fig4b")