"""
Cluster_and_Visualize.py
Purpose: K-Means clustering (K=6) on biological features (relative abundance,
         individual size, 5 ecological indices), then visualize clusters via
         t-SNE plot (with confidence ellipses) and radar chart of ecological indices.
         A global transparency parameter TRANSPARENCY is provided for adjusting
         the opacity of all visual elements.
Input:  2.Biological Statistics_5 Ecological Indices.csv
Output: 3.Biological Statistics_5 Ecological Indices_Clustering Results.csv
        tSNE_cluster_k6.tiff
        radar_diversity.tiff
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# ========== Global transparency parameter (adjust as needed) ==========
TRANSPARENCY = 0.8   # 1.0 = fully opaque, 0.5 = half transparent, etc.

# ========== Helper: confidence ellipse ==========
def plot_confidence_ellipse(ax, x, y, n_std=2.0, **kwargs):
    """Draw a confidence ellipse for a 2D dataset."""
    if len(x) < 2:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = (plt.matplotlib.transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(mean_x, mean_y))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

# ========== 0. Plot settings (Nature style) ==========
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

# ========== 1. Load data ==========
input_path = "2.Biological Statistics_5 Ecological Indices.csv"
df = pd.read_csv(input_path, sep=',')

# ========== 2. Define feature columns ==========
count_cols = [
    'Anemone count', 'Thick sea cucumber count', 'Williams galatheid crab count',
    'Spherical sea cucumber count', 'Mimic snail count', 'Coral count',
    'Red shrimp count', 'Fish count', 'Tubeworm count',
    'Haima mussel count', 'Chiridota count', 'Dead Haima mussel count'
]
size_cols = [
    'Anemone mean area', 'Thick sea cucumber mean area', 'Williams galatheid crab mean area',
    'Spherical sea cucumber mean area', 'Mimic snail mean area', 'Coral mean area',
    'Red shrimp mean area', 'Fish mean area', 'Tubeworm mean area',
    'Haima mussel mean area', 'Chiridota mean area', 'Dead Haima mussel mean area'
]
diversity_cols = ['Total abundance', 'Species richness', 'Shannon index',
                  'Simpson index', 'Pielou evenness']

for col in count_cols + size_cols + diversity_cols:
    if col not in df.columns:
        df[col] = 0
df[count_cols] = df[count_cols].fillna(0)
df[size_cols] = df[size_cols].fillna(0)
df[diversity_cols] = df[diversity_cols].fillna(0)

# ========== 3. Build feature matrix ==========
total_ind = df[count_cols].sum(axis=1).values.reshape(-1, 1)
rel_abund = np.divide(df[count_cols].values, total_ind,
                      out=np.zeros_like(df[count_cols].values, dtype=float),
                      where=total_ind != 0)
X = np.hstack([rel_abund, df[size_cols].values, df[diversity_cols].values])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ========== 4. Standardize and cluster (K=6) ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster_k6'] = cluster_labels

# ========== 5. Save clustering results ==========
output_csv = "3.Biological Statistics_5 Ecological Indices_Clustering Results.csv"
df.to_csv(output_csv, sep=',', index=False, encoding='utf-8-sig')
print(f"Clustering completed. Results saved to: {output_csv}")

# ========== 6. t-SNE dimension reduction ==========
print("Performing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# ========== 7. Plot t-SNE with confidence ellipses ==========
colors = plt.cm.tab10(np.linspace(0, 1, 6))
fig, ax = plt.subplots(figsize=(9.3/2.54, 6.5/2.54))

for i in range(6):
    mask = cluster_labels == i
    x = X_tsne[mask, 0]
    y = X_tsne[mask, 1]
    # scatter points (transparency applied)
    ax.scatter(x, y, c=[colors[i]], label=f'Cluster {i}',
               alpha=0.6 * TRANSPARENCY, s=8, edgecolors='none', rasterized=True)
    # 2σ confidence ellipse (transparency applied)
    plot_confidence_ellipse(ax, x, y, n_std=2.0,
                            edgecolor=colors[i], linewidth=1.0,
                            alpha=0.8 * TRANSPARENCY)

ax.set_xlabel('t-SNE 1', labelpad=0)
ax.set_ylabel('t-SNE 2', labelpad=-6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', direction='in', length=3)

leg = ax.legend(markerscale=1.5, fontsize=8, frameon=True, loc='upper right', ncol=2,
                bbox_to_anchor=(0.98, 0.96), bbox_transform=ax.transAxes,
                labelspacing=0.4, handlelength=0.8, columnspacing=0.4, handletextpad=0.02,
                fancybox=True, framealpha=0.85, borderaxespad=0.0, borderpad=0.2,
                edgecolor='#CCCCCC', facecolor='white')
leg.get_frame().set_linewidth(0.5)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('tSNE_cluster_k6.tiff', dpi=600, bbox_inches='tight',
            pil_kwargs={'compress': 'lzw'}, pad_inches=0.03)
plt.close()
print("t-SNE plot with ellipses saved as tSNE_cluster_k6.tiff")

# ========== 8. Radar chart of ecological indices ==========
radar_df = df.groupby('Cluster_k6')[diversity_cols].mean()
radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-10)

labels_en = ['    Total\n        abundance', 'Species richness', 'Shannon  \nindex  ',
             'Simpson       \nindex     ', 'Pielou evenness']

num_vars = len(labels_en)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7.5/2.54, 7.5/2.54), subplot_kw=dict(polar=True))
for i in range(6):
    values = radar_norm.iloc[i].tolist()
    values += values[:1]
    # Line and fill transparency: line remains slightly less than full, fill is very light
    ax.plot(angles, values, 'o-', linewidth=1.2, color=colors[i],
            label=f'Cluster {i}', markersize=3, alpha=0.9 * TRANSPARENCY)
    ax.fill(angles, values, alpha=0.08 * TRANSPARENCY, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_en, fontsize=8)
ax.tick_params(axis='x', pad=-1)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([])
radial_angle = 240
for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
    ax.text(np.deg2rad(radial_angle), r + 0.07, f'{r:.1f}',
            ha='center', va='center', fontsize=8, color='black')

ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.4, alpha=0.6)
ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.4)
ax.spines['polar'].set_edgecolor('grey')

leg = ax.legend(loc='lower right', bbox_to_anchor=(1.15, 0.7), ncol=1, frameon=True,
                fontsize=8, bbox_transform=ax.transAxes, labelspacing=0.4,
                handlelength=0.8, columnspacing=0.4, handletextpad=0.1,
                fancybox=True, framealpha=0.85, borderaxespad=0.0, borderpad=0.2,
                edgecolor='#CCCCCC', facecolor='white')
leg.get_frame().set_linewidth(0.5)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('radar_diversity.tiff', dpi=600, bbox_inches='tight',
            pil_kwargs={'compress': 'lzw'}, pad_inches=0.03)
plt.close()
print("Radar chart saved as radar_diversity.tiff")