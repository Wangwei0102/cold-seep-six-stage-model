"""
Calculate_5_Ecological_Indices.py
Purpose: Compute per-sample ecological indices (total abundance, species richness,
         Shannon index, Simpson index, Pielou evenness) from biological count data.
Input:  1.Biological Statistics.csv (comma-separated, with translated column names)
Output: 2.Biological Statistics_5 Ecological Indices.csv
Note:   Only living individuals are used (dead Haima mussel shells excluded).
"""

import pandas as pd
import numpy as np

# File paths
input_path = r"1.Biological Statistics.csv"
output_path = r"2.Biological Statistics_5 Ecological Indices.csv"

# Read data (comma-separated as per previous convention)
df = pd.read_csv(input_path, sep=',')

# Define count columns for living organisms (excluding dead shells)
count_columns = [
    'Anemone count',
    'Thick sea cucumber count',
    'Williams galatheid crab count',
    'Spherical sea cucumber count',
    'Mimic snail count',
    'Coral count',
    'Red shrimp count',
    'Fish count',
    'Tubeworm count',
    'Haima mussel count',
    'Chiridota count'
]

# Ensure all required columns exist; fill missing with 0
for col in count_columns:
    if col not in df.columns:
        df[col] = 0

# Extract count matrix, fill NaN with 0
counts = df[count_columns].fillna(0).values

# Initialize result arrays
total_abundance = np.zeros(len(df))
species_richness = np.zeros(len(df))
shannon = np.zeros(len(df))
simpson = np.zeros(len(df))
pielou = np.zeros(len(df))

# Row‑wise calculation
for i, row in enumerate(counts):
    N = row.sum()                       # total abundance
    total_abundance[i] = N
    S = np.sum(row > 0)                 # species richness
    species_richness[i] = S

    if N > 0:
        p = row / N
        p_nonzero = p[p > 0]
        # Shannon index (natural log)
        shannon[i] = -np.sum(p_nonzero * np.log(p_nonzero))
        # Simpson diversity index (1 - sum(p^2))
        simpson[i] = 1 - np.sum(p ** 2)
        # Pielou evenness
        if S > 1:
            pielou[i] = shannon[i] / np.log(S)
        else:
            pielou[i] = 0.0
    else:
        shannon[i] = 0.0
        simpson[i] = 0.0
        pielou[i] = 0.0

# Append results to DataFrame with English column names
df['Total abundance'] = total_abundance
df['Species richness'] = species_richness
df['Shannon index'] = shannon
df['Simpson index'] = simpson
df['Pielou evenness'] = pielou

# Save to new CSV (comma-separated, UTF-8 with BOM for Excel compatibility)
df.to_csv(output_path, sep=',', index=False, encoding='utf-8-sig')

print(f"Calculation completed. Results saved to: {output_path}")