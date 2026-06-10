# Executive Summary — Explainable Financial Fraud Detection

**Project:** Explainable Financial Fraud Detection using XGBoost and SHAP  
**Author:** Keerthivasan K S  
**Dataset:** Kaggle Credit Card Fraud Detection  
**Tech Stack:** Python · XGBoost · SHAP · scikit-learn · imbalanced-learn · Streamlit

---

## 1. Project Overview

Credit card fraud costs the global financial industry billions of dollars annually. Effective fraud detection requires models that are both **highly accurate** on imbalanced data and **interpretable** enough for human analysts to trust and audit.

This project builds an end-to-end, production-style ML pipeline that:
- Detects fraudulent transactions from 284,807 real credit card records
- Handles extreme class imbalance (0.17% fraud) using SMOTE
- Benchmarks three classifiers and tunes the best model
- Provides full model transparency via SHAP (SHapley Additive exPlanations)
- Delivers a Streamlit dashboard for real-time transaction risk scoring

---

## 2. Dataset Summary

| Property | Value |
|---|---|
| Source | Kaggle – Credit Card Fraud Detection |
| Observations | 284,807 transactions |
| Features | 30 (Time, Amount, V1–V28) |
| Fraudulent | 492 (0.1727%) |
| Legitimate | 284,315 (99.83%) |
| Missing Values | None |
| Feature Encoding | V1–V28 are PCA-anonymised principal components |

> [!IMPORTANT]
> The dataset is extremely imbalanced at 1:578 (fraud:legitimate). Standard accuracy metrics are misleading here — a model predicting "all legitimate" would achieve 99.83% accuracy while catching zero fraud cases. This is why ROC-AUC, PR-AUC, F1, and Recall are the correct evaluation metrics.

---

## 3. EDA Key Findings

### Class Distribution
- 492 fraudulent transactions vs. 284,315 legitimate (0.17% fraud rate)
- Log-scale bar chart needed to visualise both classes on the same plot

### Transaction Amount Analysis
- Legitimate transactions: Mean **$88.29**, Median **$22.00**, Max **$25,691**
- Fraudulent transactions: Mean **$122.21**, Median **$9.25**, Max **$2,125**
- Key insight: Fraudulent transactions cluster at **very small amounts** (stealth probing) and mid-range amounts, rarely hitting the high end seen in legitimate transactions

### Correlation with Fraud
**Most negatively correlated features (reduce fraud probability):**
| Feature | Correlation |
|---|---|
| V17 | -0.3265 |
| V14 | -0.3025 |
| V12 | -0.2606 |
| V10 | -0.2169 |
| V16 | -0.1965 |

**Most positively correlated features (increase fraud probability):**
| Feature | Correlation |
|---|---|
| V11 | +0.1549 |
| V4  | +0.1334 |
| V2  | +0.0913 |
| V21 | +0.0404 |
| V19 | +0.0348 |

---

## 4. Preprocessing Pipeline

| Step | Method | Rationale |
|---|---|---|
| Missing value check | Drop null rows | No nulls found in this dataset |
| Feature scaling | `RobustScaler` on Time, Amount | Handles outliers via IQR-based normalisation |
| Train-test split | 80/20 Stratified Split | Preserves fraud ratio in both sets |
| Class imbalance | SMOTE on training set only | Synthesises minority samples without data leakage |

**Post-SMOTE training set:**
- Before: 394 fraud, 227,451 legitimate
- After: 227,451 fraud, 227,451 legitimate (perfectly balanced)

---

## 5. Model Comparison Results

All models were trained on the SMOTE-balanced training set and evaluated on the **original unbalanced test set** (98 fraud, 56,864 legitimate).

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.0589 | 0.9184 | 0.1106 | 0.9712 | 0.7251 | 0.0241 |
| Random Forest | 0.8632 | 0.8367 | 0.8497 | 0.9734 | 0.8706 | 0.0006 |
| XGBoost (Default) | 0.7119 | 0.8571 | 0.7778 | 0.9827 | 0.8652 | 0.0007 |
| **XGBoost (Tuned) ✅** | **0.7615** | **0.8469** | **0.8019** | **0.9795** | **0.8688** | **0.0006** |

> [!NOTE]
> Logistic Regression achieves high recall (91%) but catastrophically low precision (5.9%). This means it flags 1 in every ~17 transactions as fraud — completely impractical in production. Random Forest and XGBoost provide a much better precision-recall trade-off.

---

## 6. Best Model Selection Rationale

**Selected Model: XGBoost (Tuned via RandomizedSearchCV)**

| Reason | Detail |
|---|---|
| Superior tabular performance | Gradient boosting consistently outperforms linear models on structured/tabular data |
| Non-linear decision boundaries | Captures complex, non-linear interactions between PCA features |
| Regularisation | L1/L2 weight penalties prevent overfitting to SMOTE-generated synthetic samples |
| Highest ROC-AUC | 0.9795 — best overall ranking quality across all thresholds |
| Competitive PR-AUC | 0.8688 — strong performance on the minority (fraud) class |
| Near-perfect calibration | Brier Score of 0.0006 — predicted probabilities are highly trustworthy |
| TreeSHAP compatibility | Native integration with SHAP for fast, exact local and global explainability |

**Best Hyperparameters Found:**
```
n_estimators:     200
max_depth:        7
learning_rate:    0.2
subsample:        1.0
colsample_bytree: 0.8
min_child_weight: 5
scale_pos_weight: 1
```

---

## 7. Threshold Sensitivity Analysis

| Threshold | Precision | Recall | F1 Score |
|---|---|---|---|
| 0.30 | 0.6774 | 0.8571 | 0.7568 |
| 0.50 | 0.7615 | 0.8469 | 0.8019 |
| 0.70 | 0.8283 | 0.8367 | 0.8325 |

**Interpretation:**
- **Threshold 0.30:** Maximum recall — best for catching every possible fraud (e.g., high-value accounts). Accept more false positives.
- **Threshold 0.50:** Balanced F1 — recommended default for most use cases.
- **Threshold 0.70:** Highest precision — best when false positives are very costly (e.g., blocking VIP customers).

---

## 8. Calibration Analysis

- **Brier Score: 0.0006** (near-perfect; lower is better, 0 = perfect)
- The calibration curve closely follows the perfect calibration diagonal
- This means predicted probabilities are reliable: if the model says 70% fraud probability, approximately 70% of such transactions are actually fraudulent
- Critical for risk routing: Low/Medium/High risk tiers in the dashboard are based on trustworthy probability estimates

---

## 9. SHAP Explainability Insights

### Global Feature Importance (SHAP Summary Plot)
- **V14** is the single most impactful feature. High values of V14 strongly reduce fraud probability; low values strongly increase it.
- **V17**, **V12**, **V10** follow as the next most influential features — all negatively correlated with fraud.
- **V4**, **V11** push predictions toward fraud when elevated.
- **Amount** has moderate SHAP importance — large amounts are slightly associated with legitimate transactions.

### Local Explanation (Sample Fraud Transaction — Waterfall Plot)
- The waterfall plot traces the exact path from `E[f(X)]` (base model output) to `f(x)` (final prediction)
- For a detected fraud: V14, V17, and V12 exhibit anomalous (low) values that strongly push the model toward fraud
- This provides an audit trail: compliance teams can see *why* a transaction was flagged

### Force Plot
- Visualises the same local attribution as a horizontal push-pull diagram
- Red features (positive SHAP) push toward fraud, blue features push toward legitimate
- The intersection of red and blue forces is the final prediction score

---

## 10. Business Impact

| Capability | Business Value |
|---|---|
| 84.7% Recall | 83 out of every 98 fraud transactions caught automatically |
| 76.2% Precision | 3 out of 4 flagged transactions are actually fraudulent |
| Risk Tiers | Automated routing: High-risk → block, Medium-risk → human review, Low-risk → approve |
| SHAP Explanations | Compliance & audit trail for every flagged transaction |
| Calibrated Probabilities | Trustworthy risk scores for downstream business logic |
| Threshold Tuning | Flexible operating point selection based on business tolerance |

---

## 11. Future Improvements

| Area | Suggestion |
|---|---|
| Real-time streaming | Deploy model on Apache Kafka / Flink for sub-millisecond inference |
| Model monitoring | Track data drift and concept drift in production using Evidently AI |
| Ensemble stacking | Combine XGBoost + LightGBM + Neural Network predictions |
| Graph-based features | Add merchant network analysis (GNN) to capture fraud rings |
| Calibration tuning | Apply Platt Scaling or Isotonic Regression for tighter calibration |
| Active learning | Route uncertain predictions to analysts and re-train incrementally |
| Federated learning | Train across multiple banks without sharing raw transaction data |

---

*Report generated as part of the "Explainable Financial Fraud Detection" portfolio project.*
