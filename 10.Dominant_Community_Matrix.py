"""
Dominant_Community_Matrix.py
Purpose: Create a heatmap matrix showing the dominant community type and its
         probability for each site and distance bin, with developmental stages
         indicated on the left sidebar.
Input:   4.Dominant_Community_Statistics.csv
Output:  Fig5b_dominant_community_matrix.png / .svg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ================== 0. Plot settings (Nature style) ==================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 5
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.02

# ================== 1. Load data ==================
df = pd.read_csv("4.Dominant_Community_Statistics.csv", encoding='utf-8-sig')
df.columns = df.columns.str.strip()

# Fix distance format: "0 20" → "0-20" to match distance_bins
df['Distance (m)'] = df['Distance (m)'].astype(str).str.replace(' ', '-', regex=False)

# ================== 2. Configuration ==================
seep_order = ['WH4', 'WH2', 'WH3', 'WH6', 'Haima', 'Ice Valley', 'WH1', 'WH5']
stage_map = {
    'WH4': 'Incipient',
    'WH2': 'Early',
    'WH3': 'Early',
    'WH6': 'Developing',
    'Haima': 'Developing',
    'Ice Valley': 'Flourishing',
    'WH1': 'Flourishing',
    'WH5': 'Declining'
}

distance_bins = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-120',
                 '120-140', '140-160', '160-180', '180-200', '200-220',
                 '220-240', '240-260', '260-280', '280-300']

# Community colors and legend labels
community_colors = {
    'Mussel dominated': '#C44E52',
    'Anemone dominated': '#9DAE9B',
    'Opportunistic mixed': '#EABB60',
    'Barren': '#CCCCCC',
}

community_legend = {
    'Mussel dominated': 'Mussel dominated',
    'Anemone dominated': 'Anemone dominated',
    'Opportunistic mixed': 'Opportunistic mixed',
    'Barren': 'Barren',
}

# ================== 3. Build matrix ==================
n_rows = len(seep_order)
n_cols = len(distance_bins)
matrix_comm = np.empty((n_rows, n_cols), dtype=object)
matrix_prob = np.zeros((n_rows, n_cols))

for i, seep in enumerate(seep_order):
    for j, dist in enumerate(distance_bins):
        mask = (df['Site'] == seep) & (df['Distance (m)'] == dist)
        if mask.any():
            row = df[mask].iloc[0]
            comm = row['Dominant community']
            prob = row['Probability (%)']

            # Handle missing or empty values
            if pd.isna(comm) or comm == '':
                comm = None
            if pd.isna(prob) or prob == '':
                prob = 0
            else:
                prob = float(prob)

            matrix_comm[i, j] = comm
            matrix_prob[i, j] = prob
        else:
            matrix_comm[i, j] = None
            matrix_prob[i, j] = 0

# ================== 4. Plot heatmap ==================
cell_size = 10 / 2.54 / n_cols
fig_width = 12.5 / 2.54
fig_height = cell_size * (n_rows + 3.5)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for i in range(n_rows):
    for j in range(n_cols):
        comm = matrix_comm[i, j]
        prob = matrix_prob[i, j]
        if comm is not None and prob > 0:
            color = community_colors.get(comm, '#CCCCCC')
            alpha = 0.15 + (prob / 100) * 0.8
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor=color, alpha=alpha,
                                 edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(j, i, f'{prob:.0f}%', ha='center', va='center',
                    fontsize=5, color='white', fontweight='normal')
        else:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor='#F0F0F0', alpha=0.5,
                                 edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)

# ================== 5. Axis and labels ==================
ax.spines['left'].set_position(('data', -0.5))
ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels(seep_order, fontsize=5.5)

# Left sidebar: developmental stage groups
line_x = -2.15
label_x = -2.55
stage_groups = [
    ('Incipient', 0, 0),
    ('Early', 1, 2),
    ('Developing', 3, 4),
    ('Flourishing', 5, 6),
    ('Declining', 7, 7),
]

for stage_name, start_row, end_row in stage_groups:
    y_mid = (start_row + end_row) / 2
    if start_row == end_row:
        ax.plot([line_x, line_x], [y_mid - 0.35, y_mid + 0.35],
                color='#BBBBBB', linewidth=0.6, clip_on=False)
    else:
        ax.plot([line_x, line_x], [start_row - 0.4, end_row + 0.4],
                color='#BBBBBB', linewidth=0.6, clip_on=False)
    ax.text(label_x, y_mid, stage_name, ha='center', va='center',
            rotation=80, fontsize=6, color='#555555')

ax.set_xlim(-2.2, n_cols - 0.5)
ax.set_ylim(-0.5, n_rows - 0.5)

# X‑axis: distance bins (show every 20 m, add final 300)
x_tick_labels = [f'{int(d.split("-")[0])}' for d in distance_bins]
x_tick_labels.append('300')
ax.set_xticks(list(np.arange(n_cols) - 0.5) + [n_cols - 0.5])
ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=5.5)
ax.set_xlabel('Distance to vent (m)', fontsize=7, labelpad=1.2)

# ================== 6. Legend ==================
legend_patches = [mpatches.Patch(color=community_colors[name], alpha=0.7,
                                 label=community_legend[name])
                  for name in community_colors if name in community_legend]
ax.legend(handles=legend_patches, loc='lower center',
          bbox_to_anchor=(0.45, -0.23), ncol=4,
          frameon=True, fancybox=True, framealpha=0.85,
          edgecolor='#CCCCCC', facecolor='white',
          fontsize=5.5, columnspacing=0.6,
          handlelength=1.0, handleheight=0.8)

# ================== 7. Final adjustments and save ==================
ax.tick_params(labelsize=5.5, pad=1)
ax.set_aspect('equal')
plt.subplots_adjust(bottom=0.22, top=0.97, left=0.20, right=0.99)

fig.savefig('Fig5b_dominant_community_matrix.png', dpi=600)
fig.savefig('Fig5b_dominant_community_matrix.svg')
plt.close()
print("Saved: Fig5b_dominant_community_matrix.png")