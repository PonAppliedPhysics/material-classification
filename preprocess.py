"""
preprocess.py — Raw ANSI alloy data → cleaned CSV with binary 'Use' label.

Reads data/raw_materials.csv, merges identifier columns, applies suitability
thresholds, and writes data/material.csv for downstream ML analysis.

Usage:
    python preprocess.py
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_PATH = Path("data/raw_materials.csv")
OUT_PATH = Path("data/material.csv")

# ---------------------------------------------------------------------------
# Suitability thresholds (MPa for strengths/moduli, kg/m³ for density)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "Su":  (336, 505),       # Ultimate tensile strength
    "Sy":  (251, 376),       # Yield strength
    "E":   (165_000, 248_000),  # Elastic modulus
    "G":   (63_000, 94_000),    # Shear modulus
    "mu":  (0.24, 0.36),     # Poisson's ratio
    "Ro":  (6_200, 9_400),   # Density
}

# ---------------------------------------------------------------------------
# Load and clean
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)

# Merge standard, material name, and heat treatment into one identifier
df["Material"] = (
    df[["Std", "Material", "Heat treatment"]]
    .fillna("")
    .agg(" ".join, axis=1)
    .str.strip()
)

# Clean yield strength (remove ' max' suffix in some rows)
df["Sy"] = df["Sy"].astype(str).str.replace(" max", "", regex=False).astype(int)

# Drop columns not needed for classification
drop_cols = ["Std", "ID", "Heat treatment", "Desc", "A5", "Bhn", "pH", "HV"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ---------------------------------------------------------------------------
# Generate binary target label
# ---------------------------------------------------------------------------
df["Use"] = True
for col, (lo, hi) in THRESHOLDS.items():
    df["Use"] &= df[col].between(lo, hi)

df["Use"] = df["Use"].astype(int)

# Reorder: Material first, then features, then label
feature_cols = ["Su", "Sy", "E", "G", "mu", "Ro"]
df = df[["Material"] + feature_cols + ["Use"]]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df.to_csv(OUT_PATH, index=False)

n_total = len(df)
n_pos = df["Use"].sum()
print(f"Saved {OUT_PATH}  ({n_total} alloys, {n_pos} suitable [{100*n_pos/n_total:.1f}%])")
