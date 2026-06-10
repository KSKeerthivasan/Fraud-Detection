# 🛡️ Explainable Financial Fraud Detection using XGBoost and SHAP

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-EC6C00?logo=xgboost)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn)](https://scikit-learn.org)

> A production-grade machine learning pipeline for detecting fraudulent credit card transactions — featuring SMOTE for class imbalance, XGBoost with hyperparameter tuning, SHAP explainability, and a real-time Streamlit risk dashboard.

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Results](#-results)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Key Design Decisions (Interview Ready)](#-key-design-decisions-interview-ready)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)

---

## 🎯 Problem Statement

Credit card fraud causes **billions of dollars in annual losses** globally. Detecting fraud in real-time is a challenging machine learning problem because:

1. **Extreme class imbalance** — Fraud cases represent less than 0.2% of all transactions
2. **High cost of errors** — Missing actual fraud (false negative) is costly; wrongly blocking legitimate users (false positive) damages trust
3. **Black-box models** — Financial regulators (e.g., PSD2, GDPR) require explainability for automated credit decisions
4. **Concept drift** — Fraudsters continuously adapt, requiring models that generalise well

This project addresses all four challenges with a rigorous, explainable, production-ready pipeline.

---

## 🏗️ Architecture

```
Raw Transaction Data (CSV)
        │
        ▼
┌───────────────────────┐
│  1. Preprocessing     │  ← Missing value check, RobustScaler, Stratified Split
│     + SMOTE           │  ← Synthetic Minority Oversampling on training set only
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  2. EDA               │  ← Class distribution, Amount analysis, Correlation heatmap
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  3. Model Training    │  ← Logistic Regression / Random Forest / XGBoost (baseline)
│  + Tuning             │  ← RandomizedSearchCV + StratifiedKFold (5-fold)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  4. Evaluation        │  ← Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score
│  + Threshold Tuning   │  ← Compare thresholds: 0.30 / 0.50 / 0.70
│  + Calibration        │  ← Calibration Curve + Brier Score
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  5. SHAP Explainability│ ← TreeExplainer → Summary, Waterfall, Force plots
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  6. Streamlit Dashboard│ ← Real-time transaction risk scoring + SHAP explanations
└───────────────────────┘
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Fraud Cases | 492 (0.17%) |
| Features | 30 (Time, Amount, V1–V28 via PCA) |
| Missing Values | None |
| Time Period | September 2013, European cardholders |

> ⚠️ **Download the dataset** from Kaggle and place `creditcard.csv` in the `data/` directory before running the pipeline.

---

## 📁 Project Structure

```
Fraud-Detection/
├── data/
│   ├── creditcard.csv          # Raw Kaggle dataset (download separately)
│   └── test_samples.csv        # Auto-generated test set for dashboard
├── models/
│   ├── xgboost_model.joblib    # Trained & tuned XGBoost model
│   ├── scaler.joblib           # Fitted RobustScaler
│   └── training_summary.joblib # Metrics, baselines, thresholds
├── reports/
│   ├── figures/                # All generated plots (EDA + evaluation + SHAP)
│   └── executive_summary.md   # Full project report
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, seeds, hyperparameter search space
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── preprocess.py           # Scaling, SMOTE, dataset splitting
│   ├── train.py                # Training, tuning, evaluation orchestration
│   ├── evaluate.py             # Metrics, curves, calibration, thresholds
│   └── explain.py              # SHAP global and local explanations
├── app/
│   ├── app.py                  # Streamlit dashboard UI
│   └── utils.py                # Model loading, prediction, SHAP helpers
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.

### 5. Run EDA
```bash
python src/eda.py
```

### 6. Train models (runs preprocessing → training → evaluation → SHAP)
```bash
python src/train.py
```

### 7. Launch the Streamlit dashboard
```bash
streamlit run app/app.py
```

---

## 📈 Results

### Model Comparison

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.0589 | 0.9184 | 0.1106 | 0.9712 | 0.7251 | 0.0241 |
| Random Forest | 0.8632 | 0.8367 | 0.8497 | 0.9734 | 0.8706 | 0.0006 |
| XGBoost (Default) | 0.7119 | 0.8571 | 0.7778 | 0.9827 | 0.8652 | 0.0007 |
| **XGBoost (Tuned) ✅** | **0.7615** | **0.8469** | **0.8019** | **0.9795** | **0.8688** | **0.0006** |

### Threshold Analysis (Tuned XGBoost)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.6774 | 0.8571 | 0.7568 |
| **0.50** | **0.7615** | **0.8469** | **0.8019** |
| 0.70 | 0.8283 | 0.8367 | 0.8325 |

---

## 💻 Streamlit Dashboard

The dashboard provides three functional tabs:

| Tab | Features |
|---|---|
| 🔍 **Predict Transaction** | Test Dataset Explorer (select from real test set) or Manual Input (Advanced Mode). Real-time fraud probability + risk score (Low/Medium/High) + SHAP waterfall explanation |
| 📊 **Model Metrics** | Baseline comparison table, threshold sensitivity table, Confusion Matrix, ROC Curve, PR Curve, Calibration Curve |
| 🌐 **Global Explainability** | SHAP Summary (beeswarm) plot, XGBoost Feature Importance, Waterfall and Force plots for a sample fraud |

```bash
streamlit run app/app.py
```

---

## 🧠 Key Design Decisions (Interview Ready)

### ❓ Why is accuracy misleading for fraud detection?
With only 0.17% fraud, a classifier that labels everything as "legitimate" achieves **99.83% accuracy** while catching **zero fraud**. Accuracy rewards the majority class, making it completely uninformative. We instead use **Precision, Recall, F1, ROC-AUC, and PR-AUC** which focus on the minority class.

### ❓ Why is class imbalance a challenge?
- The model learns from 578× more legitimate transactions than fraudulent ones
- Without correction, models become biased toward the majority class (low recall for fraud)
- Decision boundaries are skewed far from the minority class cluster
- Standard metrics (accuracy) cannot reveal this failure mode

### ❓ Why SMOTE?
**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic minority class samples by interpolating between existing fraud transactions and their k-nearest neighbours. Compared to simple duplication (random over-sampling), SMOTE:
- Creates diverse synthetic samples that generalise better
- Expands the minority class decision region
- Reduces overfitting to the small set of original fraud samples

> ⚠️ **Critical:** SMOTE is applied **only to the training set** to prevent data leakage into the test set.

### ❓ Why Stratified K-Fold?
Standard K-Fold splitting can place all fraud cases in one fold by chance, creating folds with zero minority class examples. **Stratified K-Fold** preserves the original class ratio in every fold, ensuring consistent fraud representation across all cross-validation splits.

### ❓ Why XGBoost?
1. **Gradient boosting on trees** — superior tabular data performance vs. linear models and neural networks on structured features
2. **Handles non-linearity** — captures complex interactions between PCA features without manual feature engineering
3. **Built-in regularisation** — L1/L2 prevents overfitting on SMOTE-generated synthetic samples
4. **Best ROC-AUC and PR-AUC** — outperforms alternatives on the metric that matters most
5. **TreeSHAP** — native, exact SHAP computation in polynomial time

### ❓ Why ROC-AUC? Why also PR-AUC?
**ROC-AUC** measures the model's ability to rank fraud above legitimate transactions at every threshold. It is threshold-independent and robust, but can be optimistic on imbalanced datasets because it uses FPR (which is diluted by the massive TN count).

**PR-AUC (Average Precision)** measures performance exclusively on the positive (fraud) class — it quantifies the area under the Precision-Recall curve, ignoring True Negatives entirely. For imbalanced datasets, **PR-AUC provides a more realistic assessment** of how useful the model is in practice.

A model that achieves a high ROC-AUC but low PR-AUC is producing a lot of false positives that aren't visible in the ROC metric.

### ❓ How does SHAP improve model transparency?
SHAP assigns each feature a **Shapley value** — the average marginal contribution of that feature across all possible feature subsets (coalitions). For a given prediction:
- **Positive SHAP value** → feature pushes prediction toward fraud
- **Negative SHAP value** → feature pushes prediction toward legitimate
- The sum of all SHAP values + base value = final model output

**TreeSHAP** computes exact Shapley values for tree models in `O(TLD²)` time (where T=trees, L=leaves, D=depth) instead of exponential time, making it practical for production use.

SHAP enables:
- **Global explanations:** Which features matter most overall? (Summary Plot)
- **Local explanations:** Why was THIS transaction flagged? (Waterfall/Force Plot)
- **Regulatory compliance:** Audit trail for every automated decision

---

## 📸 Screenshots

> _Run `streamlit run app/app.py` to view the interactive dashboard._

<img width="1533" height="695" alt="image" src="https://github.com/user-attachments/assets/2890bce6-1c7e-4fa8-843d-afa6c58b22b3" />
<img width="1527" height="681" alt="image" src="https://github.com/user-attachments/assets/d328cfe0-b6ea-4682-a357-5406e61f1b07" />
<img width="1526" height="696" alt="image" src="https://github.com/user-attachments/assets/afacc36f-ad66-4405-ad96-2e4919dcc5b3" />
<img width="1527" height="691" alt="image" src="https://github.com/user-attachments/assets/f6e9360b-c3d4-4c7d-8c35-2cdc8da6daa5" />
<img width="1522" height="685" alt="image" src="https://github.com/user-attachments/assets/766e61fd-f1d4-4374-b4fd-eb8376366bd9" />
<img width="1530" height="697" alt="image" src="https://github.com/user-attachments/assets/afbd529f-4349-4634-adbb-3c68717754be" />
<img width="1523" height="688" alt="image" src="https://github.com/user-attachments/assets/ed9ab038-49ee-4f1a-9f11-e95467720237" />
<img width="1536" height="688" alt="image" src="https://github.com/user-attachments/assets/ad12c808-bfe0-4c45-8a27-8d721037450a" />
<img width="1527" height="682" alt="image" src="https://github.com/user-attachments/assets/8b08a86d-2cf3-4c5b-b598-169301ef3c4d" />




| EDA | Model Metrics |
|---|---|
| Class Distribution | Confusion Matrix |
| Amount Distribution | ROC Curve |
| Correlation Matrix | PR Curve + Calibration Curve |

| Explainability |  |
|---|---|
| SHAP Summary Plot | SHAP Waterfall Plot |
| XGBoost Feature Importance | SHAP Force Plot |

---

## 🔮 Future Improvements

| Area | Improvement |
|---|---|
| Real-time streaming | Apache Kafka + Flink for sub-millisecond inference |
| Monitoring | Evidently AI for data drift and model degradation alerts |
| Ensemble stacking | XGBoost + LightGBM + Neural Network meta-learner |
| Graph features | GNN-based merchant/cardholder network analysis for fraud ring detection |
| Calibration tuning | Platt Scaling or Isotonic Regression post-processing |
| Active learning | Route uncertain predictions to analysts and retrain incrementally |
| Federated learning | Multi-bank training without sharing raw transaction records |
| API deployment | FastAPI + Docker + Kubernetes for scalable production serving |

---

## 📜 License

MIT License — see `LICENSE` for details.

---

<div align="center">
  <b>Built by Keerthivasan K S</b><br>
  <sub>XGBoost · SHAP · scikit-learn · Streamlit · Python</sub>
</div>
