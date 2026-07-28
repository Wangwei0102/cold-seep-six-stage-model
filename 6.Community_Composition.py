"""
Community_Composition.py
Purpose: Stacked bar chart of community composition (relative abundance)
         for the six community types (Cluster_k6).
         Legend order arranged from warm to cool colours.
         A global transparency parameter is provided for fine-tuning.
Input:   3.Biological Statistics_5 Ecological Indices_Clustering Results.csv
Output:  community_composition.jpg / .svg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# ========== Global transparency parameter (adjust as needed) ==========
TRANSPARENCY = 0.6   # 1.0 = fully opaque, 0.5 = half transparent, etc.

# ================== 1. Load data ==================
input_path = "3.Biological Statistics_5 Ecological Indices_Clustering Results.csv"
df = pd.read_csv(input_path, sep=',')

# Organism columns (must match the translated column names)
species_cols = {
    'Mussel': 'Mussel count',
    'Dead mussel': 'Dead mussel count',
    'Tubeworm': 'Tubeworm count',
    'Anemone': 'Anemone count',
    'Chiridota sp.': 'Chiridota count',
    'Coral': 'Coral count',
    'Shrimp': 'Red shrimp count',
    'Mimic snail': 'Mimic snail count',
    'Fish': 'Fish count',
    'Thick sea cucumber': 'Thick sea cucumber count',
    'Spherical sea cucumber': 'Spherical sea cucumber count',
    'Williams galatheid crab': 'Williams galatheid crab count',
}

# Community ordering (sorted by cluster ID) and labels
plot_order = [0, 1, 2, 3, 4, 5]
community_labels = [
    'Cluster 0\nMD',       # Cluster 0 (Mussel-dominated)
    'Cluster 1\nSD',       # Cluster 1 (Snail-dominated)
    'Cluster 2\nAD',       # Cluster 2 (Anemone-dominated)
    'Cluster 3\nOM',       # Cluster 3 (Opportunistic mixed)
    'Cluster 4\nTM',       # Cluster 4 (Tubeworm-mussel)
    'Cluster 5\nMS'        # Cluster 5 (Mussel-shrimp)
]

# Stacking order and corresponding colours (each species' colour matches its dominant cluster)
stack_order = [
    'Mussel',               # blue (Cluster 0,5)
    'Dead mussel',          # red-brown
    'Tubeworm',             # yellow (Cluster 4)
    'Anemone',              # purple (Cluster 2)
    'Chiridota sp.',        # pink (Cluster 3)
    'Coral',                # orange
    'Shrimp',               # light blue (Cluster 5)
    'Mimic snail',          # green (Cluster 1)
    'Fish',                 # olive
    'Thick sea cucumber',   # grey-purple
    'Spherical sea cucumber', # deep blue
    'Williams galatheid crab' # grey-green
]

species_colors_stack = [
    '#1f77b4',   # Mussel – blue
    '#C44E52',   # Dead mussel – red-brown
    '#BCBD22',   # Tubeworm – yellow
    '#9467bd',   # Anemone – purple
    '#e377c2',   # Chiridota – pink
    '#DD8452',   # Coral – orange
    '#33C6D4',   # Shrimp – light blue
    '#2ca02c',   # Mimic snail – green
    '#6B9E7A',   # Fish – olive green
    '#8C86AA',   # Thick sea cucumber – grey-purple
    '#4A90B8',   # Spherical sea cucumber – deep blue
    '#8DA87B',   # Williams galatheid crab – grey-green
]

# Uniform transparency for all bars (controlled by TRANSPARENCY)
bar_alphas = [TRANSPARENCY] * len(stack_order)

# ================== 2. Stacked bar chart ==================
fig, ax = plt.subplots(figsize=(8/2.54, 7/2.54))

n_communities = len(plot_order)
n_species = len(stack_order)
data_sum = np.zeros((n_species, n_communities))

for i, sp_name in enumerate(stack_order):
    sp_col = species_cols[sp_name]
    for j, cid in enumerate(plot_order):
        data_sum[i, j] = df[df['Cluster_k6'] == cid][sp_col].sum()

data_pct = data_sum / data_sum.sum(axis=0) * 100

x = np.arange(n_communities)
bar_width = 0.55
bottom = np.zeros(n_communities)

for i in range(n_species):
    ax.bar(x, data_pct[i, :], bar_width, bottom=bottom,
           color=species_colors_stack[i], edgecolor='white', linewidth=0.2,
           alpha=bar_alphas[i])   # label removed to customise legend order
    bottom += data_pct[i, :]

# Reference line at 100%
ax.axhline(y=100, color='#AAAAAA', linewidth=0.5, linestyle='-', alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels(community_labels, fontsize=6, linespacing=1.2)
ax.set_ylabel('\nRelative abundance (%)', fontsize=7, labelpad=-1)
ax.set_ylim(0, 105)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', pad=1)

# ================== 3. Legend arranged from warm to cool colours ==================
warm_to_cool_order = [
    'Dead mussel',           # #C44E52 (red-brown)
    'Coral',                 # #DD8452 (orange)
    'Tubeworm',              # #BCBD22 (yellow)
    'Chiridota sp.',         # #e377c2 (pink)
    'Anemone',               # #9467bd (purple – cool side)
    'Mimic snail',           # #2ca02c (green)
    'Fish',                  # #6B9E7A (olive green)
    'Shrimp',                # #33C6D4 (light blue)
    'Mussel',                # #1f77b4 (blue)
    'Williams galatheid crab',# #8DA87B (grey-green)
    'Spherical sea cucumber',# #4A90B8 (deep blue)
    'Thick sea cucumber'     # #8C86AA (grey-purple)
]

# Create legend handles with the same transparency as bars
legend_patches = []
for sp in warm_to_cool_order:
    idx = stack_order.index(sp)
    color = species_colors_stack[idx]
    legend_patches.append(mpatches.Patch(color=color, alpha=TRANSPARENCY, label=sp))

ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.48, -0.33),
          ncol=4, frameon=True, fancybox=True, framealpha=0.9,
          edgecolor='#CCCCCC', facecolor='white',
          fontsize=6, columnspacing=0.8, handlelength=1.2, handleheight=0.7,
          handletextpad=0.5, labelspacing=0.25)

plt.subplots_adjust(bottom=0.25, top=0.98, left=0.1, right=1)
fig.savefig('community_composition.jpg', dpi=600, pad_inches=0.03)
# fig.savefig('community_composition.svg')
plt.close()
print("Saved")