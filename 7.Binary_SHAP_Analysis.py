"""
Binary_SHAP_Analysis.py
Purpose: Train binary Random Forest classifiers (one-vs-rest) for each of the
         6 community types (Cluster_k6), compute SHAP values, and generate
         SHAP dependence plots (boxplots for discrete features, scatter for
         continuous) for the six environmental drivers.
Input:   3.Biological Statistics_5 Ecological Indices_Clustering Results.csv
Output:  Binary_SHAP_Plots/  (folder containing RF models and SHAP figures)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ============ Global plot settings ============
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

# ============ Load data ============
df = pd.read_csv("3.Biological Statistics_5 Ecological Indices_Clustering Results.csv")

# Keep only samples with complete environmental data
df = df.dropna(subset=['Conductivity', 'Temperature', 'Distance to vent',
                       'Water depth', 'Slope', 'Backscatter intensity', 'Cluster_k6'])

df['Water depth'] = df['Water depth'].abs()

feature_names = ['Conductivity', 'Temperature', 'Distance to vent',
                 'Water depth', 'Slope', 'Backscatter intensity']
feature_labels = ['Conductivity (S/m)', 'Temperature (°C)', 'Distance to vent (m)',
                  'Water depth (m)', 'Slope (°)', 'Backscatter intensity (dB)']

cluster_names = {
    0: 'Mussel-dominated',
    1: 'Snail-dominated',
    2: 'Anemone-dominated',
    3: 'Opportunistic-mixed',
    4: 'Tubeworm-mussel',
    5: 'Mussel-shrimp'
}

output_dir = "Binary_SHAP_Plots"
os.makedirs(output_dir, exist_ok=True)

X = df[feature_names].values
y_all = df['Cluster_k6'].values.astype(int)

# Features for which boxplots (rather than scatter) will be drawn
boxplot_features = ['Conductivity', 'Temperature', 'Water depth']

for target_cid in sorted(df['Cluster_k6'].unique()):
    target_cid = int(target_cid)
    name_en = cluster_names.get(target_cid, f'Type {target_cid}')

    y_binary = (y_all == target_cid).astype(int)
    n_pos = y_binary.sum()

    if n_pos < 50:
        print(f'Skip {name_en}: too few samples ({n_pos})')
        continue

    print(f'Training: {name_en} (n={n_pos}) vs Others (n={len(y_binary) - n_pos})')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    model_path = os.path.join(output_dir, f'RF_C{target_cid}.pkl')
    joblib.dump(rf, model_path)

    print(f'  Test accuracy: {rf.score(X_test, y_test):.4f}')

    explainer = shap.TreeExplainer(rf)
    shap_vals_raw = explainer.shap_values(X)

    if len(shap_vals_raw.shape) == 3:
        shap_vals = shap_vals_raw[:, :, 1]
    else:
        shap_vals = shap_vals_raw

    for feat_idx, feat_name in enumerate(feature_names):
        feat_label = feature_labels[feat_idx]
        feat_vals = X[:, feat_idx]
        shap_feat = shap_vals[:, feat_idx]

        fig_width = 7.5 / 2.54
        fig_height = fig_width * 0.7
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if feat_name in boxplot_features:
            unique_vals = np.unique(feat_vals)

            if len(unique_vals) > 20:
                n_bins = min(20, len(unique_vals))
                bins = np.linspace(feat_vals.min(), feat_vals.max(), n_bins + 1)

                if feat_name == 'Water depth':
                    bin_labels = [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(n_bins)]
                else:
                    bin_labels = [f'{bins[i]:.2f}-{bins[i + 1]:.2f}' for i in range(n_bins)]

                bin_indices = np.digitize(feat_vals, bins) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                data_to_plot = [shap_feat[bin_indices == i] for i in range(n_bins)
                                if len(shap_feat[bin_indices == i]) > 5]
                labels_to_plot = [bin_labels[i] for i in range(n_bins)
                                  if len(shap_feat[bin_indices == i]) > 5]

                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.6)
                bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6,
                                medianprops={'color': 'black', 'linewidth': 0.8},
                                flierprops={'markersize': 2})

                for box in bp['boxes']:
                    box.set_facecolor('#4A90B8')
                    box.set_alpha(0.5)

                ax.set_xticklabels(labels_to_plot, rotation=45, ha='right', fontsize=6)
                ax.set_xlabel(feat_label, fontsize=7, labelpad=1)
                ax.tick_params(axis='x', pad=0.5)

            else:
                unique_sorted = sorted(unique_vals)
                data_to_plot = [shap_feat[feat_vals == v] for v in unique_sorted
                                if len(shap_feat[feat_vals == v]) > 5]

                if feat_name == 'Water depth':
                    labels_to_plot = [f'{int(v)}' for v in unique_sorted
                                      if len(shap_feat[feat_vals == v]) > 5]
                else:
                    labels_to_plot = [f'{v:.4g}' for v in unique_sorted
                                      if len(shap_feat[feat_vals == v]) > 5]

                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.6)
                bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6,
                                medianprops={'color': 'black', 'linewidth': 0.8},
                                flierprops={'markersize': 2})

                for box in bp['boxes']:
                    box.set_facecolor('#4A90B8')
                    box.set_alpha(0.5)

                ax.set_xticklabels(labels_to_plot, rotation=45, ha='right', fontsize=6)
                ax.set_xlabel(feat_label, fontsize=7, labelpad=1)
                ax.tick_params(axis='x', pad=0.5)

            ax.set_ylabel('SHAP value', fontsize=7, labelpad=1)

        else:
            ax.scatter(feat_vals, shap_feat, c=feat_vals, cmap='coolwarm',
                       alpha=0.4, s=4, edgecolors='none', linewidth=0)

            sorted_idx = np.argsort(feat_vals)
            window = max(len(feat_vals) // 30, 15)
            smooth = np.convolve(shap_feat[sorted_idx], np.ones(window) / window, mode='same')
            ax.plot(feat_vals[sorted_idx], smooth, color='#D95F02', linewidth=1.2)

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.6)
            ax.set_xlabel(feat_label, fontsize=7, labelpad=1)
            ax.set_ylabel('SHAP value', fontsize=7, labelpad=1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2, linewidth=0.3)
        ax.tick_params(labelsize=6, pad=1)

        plt.tight_layout(pad=0.3)
        fig.savefig(os.path.join(output_dir, f'SHAP_C{target_cid}_{feat_name.replace(" ", "_")}.png'),
                    format='png', dpi=300)
        plt.close()

    print(f'  [Done] {name_en}')

print(f'\nFigures and models saved to: {output_dir}')