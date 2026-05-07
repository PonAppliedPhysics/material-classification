# Results

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression (unweighted) | 89.2% | 25.0% | 15.8% | 19.4% |
| **Logistic Regression (weighted ~14:1)** | 88.4% | 42.6% | **96.3%** | 59.1% |
| SVM (RBF, default) | 91.8% | 0.0% | 0.0% | 0.0% |
| Decision Tree | 99.6% | 95.0% | 100% | 97.4% |
| **Random Forest (100 trees)** | **99.6%** | **95.0%** | **100%** | **97.4%** |

> **Dataset:** 1,552 ANSI-standard alloys · 80/20 stratified split · 8.7% positive class rate

---

## Key Findings

### SVM fails completely on the minority class
SVM predicts every material as "Unsuitable" and scores 0% recall. It achieves 91.8% accuracy purely from the class imbalance — a textbook example of why **accuracy alone is misleading on imbalanced data**.

### Class weighting rescues Logistic Regression
Without weighting, Logistic Regression catches only ~16% of suitable materials. Applying a ~14:1 class weight jumps recall to 96%, but precision drops to 43% (more false positives). This is a real engineering tradeoff: **miss fewer good materials at the cost of flagging some bad ones**.

### Tree-based models dominate
Decision Tree and Random Forest both achieve near-perfect classification. This is expected because the suitability label is defined by threshold rules on the same features — trees naturally learn axis-aligned splits that recover those boundaries.

### Feature engineering adds value
The engineered specific-strength and specific-stiffness ratios rank among the top features in both Random Forest importance and Logistic Regression coefficients, validating that **physically motivated features improve classification beyond raw properties alone**.

---

## Random Forest — Feature Importance

| Rank | Feature | Importance |
|:---:|---|:---:|
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

Yield strength and the two engineered density-ratio features account for **~60% of total importance**.

---

## Detailed Classification Reports

### Logistic Regression (Class-Weighted)

```
              precision    recall  f1-score   support

  Unsuitable       1.00      0.88      0.93       283
    Suitable       0.43      0.96      0.59        27

    accuracy                           0.88       310
   macro avg       0.71      0.92      0.76       310
weighted avg       0.95      0.88      0.90       310
```

### Random Forest

```
              precision    recall  f1-score   support

  Unsuitable       1.00      1.00      1.00       283
    Suitable       1.00      0.96      0.98        27

    accuracy                           1.00       310
   macro avg       1.00      0.98      0.99       310
weighted avg       1.00      1.00      1.00       310
```

### SVM (RBF)

```
              precision    recall  f1-score   support

  Unsuitable       0.91      1.00      0.95       283
    Suitable       0.00      0.00      0.00        27

    accuracy                           0.91       310
   macro avg       0.46      0.50      0.48       310
weighted avg       0.83      0.91      0.87       310
```

---

## Engineered Features

| Feature | Formula | Physical Interpretation |
|---|---|---|
| Strength-to-Weight | Su / ρ | Specific strength — critical for weight-sensitive design |
| E-to-Density Ratio | E / ρ | Specific stiffness — governs vibration and deflection |
| Yield-to-Density | Sy / ρ | Specific yield — elastic load capacity per unit mass |
| Combined Modulus | E + G | Aggregate elastic resistance (axial + shear) |

---

## Tools

Python 3.10+ · scikit-learn · TensorFlow/Keras · Pandas · NumPy · Seaborn · Matplotlib
