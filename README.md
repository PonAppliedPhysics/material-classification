# ML-Based Material Classification for Mechanical Design

An end-to-end supervised machine learning pipeline that classifies **1,552 ANSI-standard engineering alloys** as suitable or unsuitable for a target mechanical application, based on six physical properties. Five classifiers are trained, compared, and interpreted.

## Problem

Material selection in mechanical engineering traditionally relies on handbook filtering — checking whether candidate alloys fall within acceptable ranges for strength, stiffness, and density. This project reframes that process as a **binary classification task** and compares how well different ML models can learn the suitability boundary from data.

## Dataset

| Property | Symbol | Unit | Description |
|---|---|---|---|
| Ultimate Tensile Strength | S_u | MPa | Maximum stress before fracture |
| Yield Strength | S_y | MPa | Onset of plastic deformation |
| Elastic Modulus | E | MPa | Stiffness under axial load |
| Shear Modulus | G | MPa | Stiffness under shear load |
| Poisson's Ratio | μ | — | Lateral-to-axial strain ratio |
| Density | ρ | kg/m³ | Mass per unit volume |

- **1,552 alloys** from the ANSI standard (steels, aluminum, titanium, copper, cast iron)
- **Class imbalance**: 135 suitable (8.7%) vs. 1,417 unsuitable (91.3%)
- Raw data in `data/raw_materials.csv`, processed data in `data/material.csv`

## Engineered Features

| Feature | Formula | Interpretation |
|---|---|---|
| Strength-to-Weight | S_u / ρ | Specific strength — weight-sensitive design |
| E-to-Density Ratio | E / ρ | Specific stiffness — vibration and deflection |
| Yield-to-Density | S_y / ρ | Specific yield — elastic capacity per unit mass |
| Combined Modulus | E + G | Aggregate axial + shear resistance |

Feature relevance was validated using **ANOVA F-tests** and **Chi-squared tests** before model training.

## Models

| Model | Approach | Notes |
|---|---|---|
| Logistic Regression | Linear baseline | Class-weighted (~14:1) to handle imbalance |
| Support Vector Machine | RBF kernel | Default hyperparameters |
| Decision Tree | Unrestricted depth | Visualized with graphviz |
| Random Forest | 100-tree ensemble | Feature importance extraction |
| Neural Network | 64→32→1 dense, ReLU/sigmoid | 50 epochs, StandardScaler, train/val curves |

## Project Structure

```
material-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── preprocess.py              # Raw data → cleaned CSV with target label
├── material_classification.ipynb  # Full analysis notebook
└── data/
    ├── raw_materials.csv      # Original ANSI alloy data
    └── material.csv           # Cleaned data with 'Use' label
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/PonAppliedPhysics/material-classification.git
cd material-classification
pip install -r requirements.txt

# Regenerate processed data from raw (optional)
python preprocess.py

# Run the notebook
jupyter notebook material_classification.ipynb
```

## Key Takeaways

- **Ensemble methods** (Random Forest) and **class-weighted linear models** outperform default classifiers on this imbalanced dataset.
- **Physically motivated feature engineering** (specific strength, specific stiffness) improves classification beyond raw properties.
- The **Decision Tree visualization** provides a fully interpretable view of the classification logic.
- The **Neural Network** benefits from feature scaling, while tree-based models are scale-invariant.

## Tools

Python 3.10+ · scikit-learn · TensorFlow/Keras · Pandas · NumPy · Seaborn · Matplotlib · graphviz

## Author

**Pon Aung** — Physics, The Ohio State University
