"""
Supplementary_Boxplots.py
Purpose: Generate boxplots for each of the 5 ecological indices across the 6 clusters
         (K‑Means, K=6).  These figures accompany the radar chart to show the
         underlying distribution of each index.  Colors match the radar chart.
Input:   2.Biological Statistics_5 Ecological Indices.csv
Output:  boxplot_Total_abundance.tiff (y-max = 100)
         boxplot_Species_richness.tiff
         boxplot_Shannon_index.tiff
         boxplot_Simpson_index.tiff
         boxplot_Pielou_evenness.tiff
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ================== 0. Plot settings (Nature style) ==================
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

# ================== 1. Load data, build feature matrix, and cluster (K=6) ==================
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

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster_k6'] = cluster_labels

# ================== 2. Colors (matching the radar chart) ==================
colors = plt.cm.tab10(np.linspace(0, 1, 6))   # RGBA array, same as in radar chart

# ----- Transparency parameter -----
box_alpha = 0.6   # Adjust to change box fill transparency (0 = fully transparent, 1 = fully opaque)

# ================== 3. Generate boxplots for each diversity index ==================
for idx_name in diversity_cols:
    fig, ax = plt.subplots(figsize=(7.5/2.54, 5.5/2.54))

    # Collect data grouped by cluster
    data = [df.loc[df['Cluster_k6'] == c, idx_name].values for c in range(6)]

    # Boxplot
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={'color': 'black', 'linewidth': 0.8},
                    flierprops={'markersize': 2, 'markerfacecolor': 'gray',
                                'alpha': 0.3, 'markeredgecolor': 'none'})

    # Color the boxes with the same colors as the radar chart
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(box_alpha)   # Use the transparency parameter here

    # X-axis labels
    ax.set_xticklabels([f'Cluster {c}' for c in range(6)], fontsize=7)
    ax.set_ylabel(idx_name, fontsize=7, labelpad=2)

    # Special handling for Total abundance: set y-axis maximum to 100
    if idx_name == 'Total abundance':
        ax.set_ylim(0, 100)

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', direction='in', length=3)

    plt.tight_layout(pad=0.3)
    file_name = f"boxplot_{idx_name.replace(' ', '_')}.tiff"
    plt.savefig(file_name, dpi=600, bbox_inches='tight',
                pil_kwargs={'compress': 'lzw'}, pad_inches=0.03)
    plt.close()
    print(f"Saved {file_name}")

print("All boxplots generated.")