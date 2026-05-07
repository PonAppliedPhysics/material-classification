# ML-Based Material Classification for Mechanical Design

An end-to-end supervised machine learning pipeline that classifies **1,552 ANSI-standard engineering alloys** as suitable or unsuitable for a target mechanical application, based on physical, strength, stiffness, and density properties. Multiple classifiers are trained, compared, and interpreted to show how machine learning can support early-stage material selection.

## Problem

Material selection in mechanical engineering traditionally relies on handbook filtering — checking whether candidate alloys fall within acceptable ranges for strength, stiffness, density, and other physical properties. This project reframes that process as a **binary classification task** and compares how well different machine learning models can learn the suitability boundary from material property data.

The engineering motivation is simple: instead of manually filtering hundreds or thousands of alloys, a trained model can quickly flag promising candidates for further review. This does not replace engineering judgment, but it can speed up the first-pass screening process.

## Dataset

| Property | Symbol | Unit | Description |
|---|---:|---:|---|
| Ultimate Tensile Strength | S_u | MPa | Maximum stress before fracture |
| Yield Strength | S_y | MPa | Onset of plastic deformation |
| Elastic Modulus | E | MPa | Stiffness under axial load |
| Shear Modulus | G | MPa | Stiffness under shear load |
| Poisson's Ratio | μ | — | Lateral-to-axial strain ratio |
| Density | ρ | kg/m³ | Mass per unit volume |

- **1,552 alloys** from ANSI-standard material data
- Material families include steels, aluminum, titanium, copper, and cast iron
- **Class imbalance:** 135 suitable materials (8.7%) vs. 1,417 unsuitable materials (91.3%)
- Raw data: `data/raw_materials.csv`
- Processed data: `data/material.csv`

## Engineered Features

| Feature | Formula | Physical Interpretation |
|---|---:|---|
| Strength-to-Weight | S_u / ρ | Specific strength — important for weight-sensitive design |
| E-to-Density Ratio | E / ρ | Specific stiffness — important for vibration and deflection behavior |
| Yield-to-Density | S_y / ρ | Specific yield — elastic load capacity per unit mass |
| Combined Modulus | E + G | Aggregate elastic resistance from axial and shear stiffness |

Feature relevance was checked using **ANOVA F-tests** and **Chi-squared tests** before model training.

## Models

| Model | Approach | Notes |
|---|---|---|
| Logistic Regression | Linear baseline | Tested both unweighted and class-weighted versions |
| Support Vector Machine | RBF kernel | Default hyperparameters |
| Decision Tree | Tree-based classifier | Interpretable threshold-based splits |
| Random Forest | 100-tree ensemble | Feature importance extraction |
| Neural Network | 64 → 32 → 1 dense network | ReLU hidden layers, sigmoid output, StandardScaler, train/validation curves |

## Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression (unweighted) | 89.2% | 25.0% | 15.8% | 19.4% |
| Logistic Regression (weighted ~14:1) | 88.4% | 42.6% | 96.3% | 59.1% |
| SVM (RBF, default) | 91.8% | 0.0% | 0.0% | 0.0% |
| Decision Tree | 99.6% | 95.0% | 100% | 97.4% |
| Random Forest (100 trees) | 99.6% | 95.0% | 100% | 97.4% |

**Dataset split:** 1,552 ANSI-standard alloys · 80/20 stratified split · 8.7% positive class rate

### Key Findings

#### SVM fails completely on the minority class

The default RBF-kernel SVM predicts every material as **Unsuitable** and scores **0% recall** on the suitable class. It still reaches 91.8% accuracy because the dataset is highly imbalanced. This is a clear example of why accuracy alone can be misleading in engineering classification problems with rare positive cases.

#### Class weighting rescues Logistic Regression

Without class weighting, Logistic Regression only catches about 16% of suitable materials. Applying an approximate **14:1 class weight** increases recall to 96%, meaning the model misses far fewer suitable materials. The tradeoff is lower precision, so the model flags more false positives. From an engineering standpoint, this can be acceptable during screening because missing a potentially good material may be worse than reviewing a few extra candidates.

#### Tree-based models dominate

Decision Tree and Random Forest models achieve near-perfect classification. This is expected because the suitability label is based on threshold-like rules using the same physical property features. Tree-based models naturally learn axis-aligned splits, which makes them well-suited for this type of material screening logic.

#### Feature engineering adds value

The engineered specific-strength and specific-stiffness features rank highly in both Random Forest importance and Logistic Regression coefficients. This validates the idea that physically motivated features can improve classification beyond raw material properties alone.

### Random Forest — Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Sy (Yield Strength) | 0.2624 |
| 2 | Strength-to-Weight (Su/ρ) | 0.1719 |
| 3 | Sy-to-Density (Sy/ρ) | 0.1698 |
| 4 | Su (Tensile Strength) | 0.1503 |
| 5 | E (Elastic Modulus) | 0.0681 |
| 6 | G (Shear Modulus) | 0.0634 |
| 7 | Combined Modulus (E+G) | 0.0610 |
| 8 | E-to-Density (E/ρ) | 0.0243 |
| 9 | Ro (Density) | 0.0189 |
| 10 | mu (Poisson's Ratio) | 0.0099 |

Yield strength and the two engineered density-ratio features account for roughly **60% of total feature importance**, showing that strength and strength-to-weight behavior dominate the material suitability decision.

### Detailed Classification Reports

#### Logistic Regression (Class-Weighted)

```text
              precision    recall  f1-score   support

  Unsuitable       1.00      0.88      0.93       283
    Suitable       0.43      0.96      0.59        27

    accuracy                           0.88       310
   macro avg       0.71      0.92      0.76       310
weighted avg       0.95      0.88      0.90       310
```

#### Random Forest

```text
              precision    recall  f1-score   support

  Unsuitable       1.00      1.00      1.00       283
    Suitable       1.00      0.96      0.98        27

    accuracy                           1.00       310
   macro avg       1.00      0.98      0.99       310
weighted avg       1.00      1.00      1.00       310
```

#### SVM (RBF)

```text
              precision    recall  f1-score   support

  Unsuitable       0.91      1.00      0.95       283
    Suitable       0.00      0.00      0.00        27

    accuracy                           0.91       310
   macro avg       0.46      0.50      0.48       310
weighted avg       0.83      0.91      0.87       310
```

## Project Structure

```text
material-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── preprocess.py
├── material_classification.ipynb
└── data/
    ├── raw_materials.csv
    └── material.csv
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/PonAppliedPhysics/material-classification.git
cd material-classification
pip install -r requirements.txt

# Regenerate processed data from raw data
python preprocess.py

# Run the notebook
jupyter notebook material_classification.ipynb
```

## Key Takeaways

- **Accuracy alone is not enough** for imbalanced engineering datasets.
- **Class weighting** can improve recall when the positive class is rare.
- **Tree-based models** perform especially well when the decision boundary is threshold-based.
- **Physically motivated feature engineering** improves interpretability and model performance.
- **Random Forest feature importance** highlights which material properties drive suitability.
- The project shows how machine learning can support early-stage material screening while still keeping the engineering logic interpretable.

## Tools

Python 3.10+ · scikit-learn · TensorFlow/Keras · Pandas · NumPy · Seaborn · Matplotlib · graphviz

## Author

**Pon Aung**  
Physics Graduate, The Ohio State University  
GitHub: [PonAppliedPhysics](https://github.com/PonAppliedPhysics)
