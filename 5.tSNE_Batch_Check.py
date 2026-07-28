"""
tSNE_Batch_Check.py
Purpose: Visualize t-SNE projection colored by sampling date and site,
         using existing cluster labels from the previous step.
Input:  3.Biological Statistics_5 Ecological Indices_Clustering Results.csv
Output: tSNE_Date.tiff, tSNE_Site.tiff
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# ================== 0. Plot settings (Nature style) ==================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# ================== 1. Load clustered data ==================
input_path = "3.Biological Statistics_5 Ecological Indices_Clustering Results.csv"
df = pd.read_csv(input_path, sep=',')

# ================== 2. Define feature columns (English names) ==================
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

# Ensure all required columns exist
for col in count_cols + size_cols + diversity_cols:
    if col not in df.columns:
        df[col] = 0
df[count_cols] = df[count_cols].fillna(0)
df[size_cols] = df[size_cols].fillna(0)
df[diversity_cols] = df[diversity_cols].fillna(0)

# Build feature matrix (relative abundance + sizes + diversity indices)
total_ind = df[count_cols].sum(axis=1).values.reshape(-1, 1)
rel_abund = np.divide(df[count_cols].values, total_ind,
                      out=np.zeros_like(df[count_cols].values, dtype=float),
                      where=total_ind != 0)
X = np.hstack([rel_abund, df[size_cols].values, df[diversity_cols].values])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================== 3. t-SNE projection ==================
print("Performing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# ================== 4. Prepare coloring variables ==================
# Convert Date to days since first day
df['Date_str'] = df['Date'].astype(str).str.strip()
df['Date_num'] = pd.to_datetime(df['Date_str'], format='%Y%m%d', errors='coerce')
if df['Date_num'].isna().all():
    df['Date_num'] = pd.to_datetime(df['Date_str'], errors='coerce')
df = df.dropna(subset=['Date_num'])
min_date = df['Date_num'].min()
df['Day'] = (df['Date_num'] - min_date).dt.days

# Top 10 sites for distinct colors, rest grey
top_sites = df['Site'].value_counts().nlargest(10).index.tolist()
site_color_dict = {}
for i, site in enumerate(top_sites):
    site_color_dict[site] = plt.cm.Set3(i % 12)
site_color_dict['Other'] = 'lightgrey'

# ================== 5. Plot t-SNE colored by Date ==================
fig, ax = plt.subplots(figsize=(7.5/2.54, 6.5/2.54))
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Day'], cmap='viridis',
                alpha=0.6, s=8, edgecolors='none', rasterized=True)

# Create an inset horizontal colorbar at upper right corner
box = ax.get_position()
cbar_width = 0.3          # length of colorbar (fraction of figure width)
cbar_height = 0.04        # thickness
cbar_left = box.x1 - cbar_width + 0.05
cbar_bottom = box.y1 - cbar_height + 0.06
cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
cbar.set_label('Days from start', fontsize=7, labelpad=1)
cbar.ax.tick_params(labelsize=6, pad=1)

# Remove default straight outline and background
cbar.outline.set_visible(False)
cax.patch.set_visible(False)

# Add a rounded frame with matching legend style
frame_patch = FancyBboxPatch(
    (-0.05, -2.6), 1.1, 4,
    boxstyle="round,pad=0.02",
    facecolor='white',          # same background as legend
    edgecolor='#CCCCCC',        # same border color
    linewidth=0.5,
    alpha=0.85,                 # same transparency
    transform=cax.transAxes,
    zorder=0,                   # behind the tick labels
    clip_on=False               # prevent clipping at the edges
)
cax.add_patch(frame_patch)

ax.set_xlabel('t-SNE 1', labelpad=0)
ax.set_ylabel('t-SNE 2',  labelpad=-6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', direction='in', length=3)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('tSNE_Date.tiff', dpi=600, bbox_inches='tight', pad_inches=0.03,
            pil_kwargs={'compress': 'lzw'})
plt.close()
print("Saved tSNE_Date.tiff")

# ================== 6. Plot t-SNE colored by Site ==================
fig, ax = plt.subplots(figsize=(7.5/2.54, 6.5/2.54))
# Other sites as grey background
other_mask = ~df['Site'].isin(top_sites)
if other_mask.sum() > 0:
    ax.scatter(X_tsne[other_mask, 0], X_tsne[other_mask, 1],
               c='lightgrey', alpha=0.3, s=8, edgecolors='none', rasterized=True)
# Top sites
for site in top_sites:
    mask = df['Site'] == site
    if mask.sum() == 0:
        continue
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=[site_color_dict[site]], label=site, alpha=0.6, s=8,
               edgecolors='none', rasterized=True)

# Legend with same style as reference
leg = ax.legend(markerscale=1.5, fontsize=8, frameon=True, loc='upper right',
                ncol=2, bbox_to_anchor=(0.98, 1), bbox_transform=ax.transAxes,
                labelspacing=0.4, handlelength=0.8, columnspacing=0.4,
                handletextpad=0.02, fancybox=True, framealpha=0.85,
                borderaxespad=0.0, borderpad=0.2, edgecolor='#CCCCCC',
                facecolor='white')
leg.get_frame().set_linewidth(0.5)

ax.set_xlabel('t-SNE 1',  labelpad=0)
ax.set_ylabel('t-SNE 2',  labelpad=-6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', direction='in', length=3)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('tSNE_Site.tiff', dpi=600, bbox_inches='tight', pad_inches=0.03,
            pil_kwargs={'compress': 'lzw'})
plt.close()
print("Saved tSNE_Site.tiff")

print("All figures saved.")