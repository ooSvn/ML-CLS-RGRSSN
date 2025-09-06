
# Boston Housing – End-to-End Machine-Learning Mini-Project  

---

## 1. Goal
Predict the **median value of owner-occupied homes (MEDV)** and – after discretising this continuous target – classify each district into  
0 = *economical*, 1 = *normal*, 2 = *expensive*.

---

## 2. Data
| File | Shape | Description |
|------|-------|-------------|
| `DataSet.xlsx` | (506, 14) | Classic Boston Housing data + 1 extra sheet |

**Variables**  
CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV.

**Missingness**  
CHAS 5.1 %, DIS 5.3 %, B 4.0 %, MEDV 10.7 %.  
(Handled with 4-neighbour KNN-imputation – kept 506 rows.)

---

## 3. Pipeline Overview
1. EDA → correlation heat-map, uni-variate plots, scatter+hexbin vs MEDV.  
2. Clean → KNN-impute, drop NOX & DIS (multicollinearity), scale w/ StandardScaler + Normalizer (except CHAS, RAD, MEDV).  
3. Feature engineering → discretise MEDV into tercile-based `EXP` label.  
4. Modelling → 80 / 20 random split, all metrics on **hold-out test set**.  
5. Evaluation → Accuracy, precision / recall / F1, macro & weighted avg.

---

## 4. Algorithms Tried
| Model | Best Hyper-parameters | Test Accuracy | Notes |
|-------|-----------------------|---------------|-------|
| **Simple Linear Reg.** | – | R² ≈ –20 | Baseline, severe under-fit |
| **Decision Tree** | `entropy, max_depth=8, min_samples_split=2, max_features=4` | 0.88 | Grid-search 5-fold CV |
| **k-NN** | `n_neighbors=10` | 0.83 | 6-fold CV |
| **Random Forest** | `n_estimators=300, max_depth=None, min_samples_split=2` | 0.87 | Balanced precision |
| **SVM** | `linear, C=1` | 0.84 | Faster than RBF on this size |
| **XGBoost** | `max_depth=3, eta=0.1, binary logistic` | – | Switched to 2-class for speed |

---

## 5. Key Results
* **Best classifier**: Decision Tree (88 % accuracy, 0.87 macro F1).  
* **Most important features**: RM, LSTAT, PTRATIO, TAX.  
* **No evidence of over-fitting** after pruning – CV score ≈ test score.

---
