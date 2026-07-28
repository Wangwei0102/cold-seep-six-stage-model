markdown
# cold-seep-six-stage-model

Codes for the paper **"Irreversible succession in deep-sea chemosynthetic ecosystems"**.

## 📖 Overview

This repository contains the Python scripts used for:
- Calculating five ecological indices (total abundance, species richness, Shannon, Simpson, Pielou) from biological count data.
- K‑means clustering (K=6) on biological features to define macrofaunal community types.
- Visualisation: t‑SNE plots, radar charts, stacked bar charts, boxplots, and SHAP dependence curves.
- Binary Random Forest classification and SHAP analysis to identify environmental drivers.
- BIC‑based optimal cluster number evaluation.
- Global comparison of cold‑seep developmental stages across tectonic settings.

## 🛠️ Dependencies

Python 3.8+ and the following packages:

```bash
pip install numpy pandas matplotlib scikit-learn shap joblib umap-learn
```
Or install all at once: `pip install -r requirements.txt`

## 📁 File structure


| File | Description |
|------|-------------|
| `1.Biological Statistics.csv` | Raw biological count data (input) |
| `2.Biological Statistics_5 Ecological Indices.csv` | Data with calculated diversity indices |
| `3.Biological Statistics_5 Ecological Indices_Clustering Results.csv` | Data with cluster labels (Cluster_k6) |
| `Calculate_5_Ecological_Indices.py` | Compute ecological indices |
| `BIC_Improvement_Dual_Axis_Broken.py` | BIC plot for optimal K selection |
| `Cluster_and_Visualize.py` | K‑means clustering, t‑SNE and radar chart |
| `tSNE_Batch_Check.py` | t‑SNE colored by date and site |
| `Community_Composition.py` | Stacked bar chart of community composition |
| `Supplementary_Boxplots.py` | Boxplots of ecological indices per cluster |
| `Binary_SHAP_Analysis.py` | Train binary RF models and generate SHAP plots |
| `Feature_Importance_Heatmap.py` | Heatmap of normalised feature importance |
| `SHAP_Distance_to_Vent_Curve.py` | SHAP dependence curve for distance to vent |
| `Dominant_Community_Matrix.py` | Matrix of dominant community type probability |
| `Fig5c_Tectonic_Stage_Comparison.py` | Bar chart of seep stage percentages by tectonic margin |
| `Binary_SHAP_Plots/` | Trained RF models and SHAP figures |

## 🚀 How to run

1. Ensure the required data file `1.Biological Statistics.csv` is placed in the root directory.
2. Execute scripts in the following order:

```bash
python Calculate_5_Ecological_Indices.py
python BIC_Improvement_Dual_Axis_Broken.py   # optional, to confirm optimal K
python Cluster_and_Visualize.py
python tSNE_Batch_Check.py
python Community_Composition.py
python Supplementary_Boxplots.py
python Binary_SHAP_Analysis.py
python Feature_Importance_Heatmap.py
python SHAP_Distance_to_Vent_Curve.py
python Dominant_Community_Matrix.py
python Fig5c_Tectonic_Stage_Comparison.py
```
Each script will generate output files (CSV, PNG, TIFF, SVG) in the same directory.

## 📝 Citation
The associated paper is currently under review. If you use this code, please cite the paper (once published) and this repository.  
A temporary citation format:
> Li Y., Wang W., et al. Irreversible succession in deep‑sea chemosynthetic ecosystems. *Under review*.
The full author list and journal information will be updated after acceptance.

📧 Contact
For questions, contact 1406173009@qq.com.
