"""
Fig5c_Tectonic_Stage_Comparison.py
Purpose: Grouped bar chart comparing the percentage of seeps in active and passive
         continental margins across five developmental stages.
Input:   Pre-defined count data.
Output:  Fig5c_tectonic_stage_comparison.png / .svg
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ================== 0. Plot settings (Nature style) ==================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.02

# ================== 1. Data ==================
stages = ['Incipient', 'Early', 'Developing', 'Flourishing', 'Declining']
passive_counts = [12, 5, 5, 85, 34]
active_counts  = [4,  0, 3, 27,  0]

# Calculate percentages within each margin type
active_pct  = [c / sum(active_counts) * 100 for c in active_counts]
passive_pct = [c / sum(passive_counts) * 100 for c in passive_counts]

# ================== 2. Color and transparency settings ==================
stage_colors = {
    'Incipient':  '#A7C5A7',
    'Early':      '#D3D189',
    'Developing': '#CDB58D',
    'Flourishing': '#F08C8C',
    'Declining':  '#B8B8B8',
}

bar_alpha = 0.7   # global transparency for all bars (1.0 = fully opaque)

# ================== 3. Create bar chart ==================
fig, ax = plt.subplots(figsize=(5.1/2.54, 5.7/2.54))

x = np.arange(len(stages))
width = 0.35

for i, stage in enumerate(stages):
    color = stage_colors[stage]

    # Active margins (solid fill)
    ax.bar(x[i] - width/2, active_pct[i], width,
           color=color, alpha=bar_alpha,
           edgecolor='white', linewidth=0.3,
           label='Active' if i == 0 else "")

    # Passive margins (hatched to distinguish)
    ax.bar(x[i] + width/2, passive_pct[i], width,
           color=color, alpha=bar_alpha,
           edgecolor='white', linewidth=0.3,
           hatch='//',
           label='Passive' if i == 0 else "")

# ================== 4. Axis labels and styling ==================
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=6.5, rotation=45, ha='right')
ax.set_ylabel('Percentage of seeps (%)', fontsize=7, labelpad=1)
ax.set_ylim(0, 80)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=6, pad=1)

# ================== 5. Legend ==================
legend_elements = [
    Patch(facecolor='gray', alpha=bar_alpha, label='Active'),
    Patch(facecolor='gray', alpha=bar_alpha, hatch='//', label='Passive')
]
ax.legend(handles=legend_elements, loc='upper left',
          frameon=True, fancybox=True, framealpha=0.85,
          edgecolor='#CCCCCC', facecolor='white',
          fontsize=6, labelspacing=0.3)

# ================== 6. Save ==================
plt.tight_layout(pad=0.3)
fig.savefig('Fig5c_tectonic_stage_comparison.png', dpi=600)
fig.savefig('Fig5c_tectonic_stage_comparison.svg')
plt.close()
print("Saved: Fig5c_tectonic_stage_comparison.png")