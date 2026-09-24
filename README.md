<div align="center">

# 🛡️ Explainable Financial Fraud Detection

**A production-grade, containerised machine learning system for real-time credit card fraud detection — featuring microservice architecture, SHAP explainability, and automated CI/CD.**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-EC6C00?style=for-the-badge&logo=xgboost)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Docker Deployment](#-docker-deployment)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Technical Design Decisions](#-technical-design-decisions)
- [Screenshots](#-screenshots)
- [Future Roadmap](#-future-roadmap)
- [License](#-license)

---

## 🎯 Problem Statement

Credit card fraud causes **billions of dollars in annual losses** globally. Detecting fraud in real-time is a challenging machine learning problem because:

| Challenge | Description |
|---|---|
| **Extreme Class Imbalance** | Fraud cases represent less than 0.2% of all transactions |
| **High Cost of Errors** | Missing actual fraud (false negative) is costly; wrongly blocking legitimate users (false positive) damages trust |
| **Regulatory Explainability** | Financial regulators (PSD2, GDPR) require transparent, auditable explanations for automated credit decisions |
| **Concept Drift** | Fraudsters continuously adapt their strategies, requiring models that generalise well |

This project addresses all four challenges with a rigorous, explainable, production-ready pipeline backed by a decoupled microservice architecture.

---

## 🏗️ System Architecture

```
                            ┌─────────────────────────────┐
                            │       Docker Compose         │
                            │       Orchestration          │
                            └─────────────┬───────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
        ┌───────────▼───────────┐               ┌──────────────▼──────────────┐
        │   Streamlit Frontend  │   HTTP/JSON   │     FastAPI Backend         │
        │   (Port 8501)         │◄─────────────►│     (Port 8000)             │
        │                       │               │                             │
        │  • Transaction Input  │               │  • Pydantic Validation      │
        │  • Risk Dashboard     │               │  • Feature Engineering      │
        │  • SHAP Visualisation │               │  • XGBoost Inference        │
        │  • Model Metrics      │               │  • Bayesian Recalibration   │
        │                       │               │  • SHAP Computation         │
        └───────────────────────┘               └─────────────────────────────┘
                                                          │
                                                ┌─────────▼─────────┐
                                                │  ML Artifacts      │
                                                │  (.joblib files)   │
                                                │  • XGBoost Model   │
                                                │  • RobustScaler    │
                                                │  • Training Summary│
                                                └────────────────────┘
```

### ML Training Pipeline

```
Raw CSV Data ──► Preprocessing ──► Stratified Split ──► SMOTE (Train Only) ──► Baseline Models
                  • Hour Feature     80/20              • Zero Data Leakage     • Logistic Reg.
                  • RobustScaler                        • imblearn Pipeline     • Random Forest
                                                                                • XGBoost
                                                                                     │
     SHAP Explainability ◄── Evaluation & Curves ◄── Hyperparameter Tuning ◄─────────┘
     • Summary Plot           • PR-AUC, ROC-AUC       • RandomizedSearchCV
     • Waterfall Plot         • Calibration Curve      • 5-Fold Stratified CV
     • Force Plot             • Threshold Analysis     • F1-Score Optimised
```

---

## ✨ Key Features

| Category | Feature | Details |
|---|---|---|
| **ML Pipeline** | Leak-Proof SMOTE | SMOTE confined inside `imblearn.Pipeline` within `RandomizedSearchCV` — zero data leakage across CV folds |
| | Hyperparameter Tuning | `RandomizedSearchCV` with 5-fold `StratifiedKFold`, optimised for F1-Score |
| | Bayesian Probability Recalibration | Corrects SMOTE-induced prior shift using `β = 0.00173` (true fraud prevalence) |
| | Cyclical Time Engineering | Raw `Time` (seconds) converted to `Hour` feature for better tree-model generalisation |
| **Explainability** | Real-Time SHAP | Per-transaction waterfall plots generated on-demand via `TreeExplainer` |
| | Global Explainability | SHAP summary (beeswarm), feature importance, and force plots |
| **Architecture** | REST API | Decoupled FastAPI backend with Pydantic schema validation |
| | Containerisation | Multi-service Docker Compose deployment |
| | CI/CD | GitHub Actions pipeline — automated testing + Docker image builds |
| **Dashboard** | Interactive UI | Test dataset explorer, manual input mode, risk badges, model metrics tab |

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Total Transactions** | 284,807 |
| **Fraud Cases** | 492 (0.17%) |
| **Features** | 30 — `Time`, `Amount`, `V1`–`V28` (PCA-transformed) |
| **Missing Values** | None |
| **Time Period** | September 2013, European cardholders |

> ⚠️ **Dataset not included.** Download `creditcard.csv` from Kaggle and place it in the `data/` directory before running the pipeline.

---

## 📁 Project Structure

```
Fraud-Detection/
│
├── api/                            # FastAPI REST Backend
│   └── main.py                     #   Prediction endpoint + SHAP response
│
├── app/                            # Streamlit Frontend
│   ├── app.py                      #   Dashboard UI (3-tab layout)
│   └── utils.py                    #   Model loading, scaling, prediction helpers
│
├── src/                            # Core ML Pipeline
│   ├── config.py                   #   Paths, seeds, hyperparameter search space
│   ├── eda.py                      #   Exploratory Data Analysis
│   ├── preprocess.py               #   Scaling, SMOTE, stratified splitting
│   ├── train.py                    #   Training, tuning, evaluation orchestration
│   ├── evaluate.py                 #   Metrics, curves, calibration, thresholds
│   └── explain.py                  #   SHAP global and local explanations
│
├── tests/                          # Automated Test Suite
│   └── test_api.py                 #   FastAPI endpoint tests (pytest)
│
├── models/                         # Serialised ML Artifacts
│   ├── xgboost_model.joblib        #   Trained & tuned XGBoost classifier
│   ├── scaler.joblib               #   Fitted RobustScaler
│   └── training_summary.joblib     #   Metrics, baselines, threshold data
│
├── data/                           # Data Directory
│   ├── creditcard.csv              #   Raw Kaggle dataset (download separately)
│   └── test_samples.csv            #   Auto-generated test set for dashboard
│
├── reports/
│   ├── figures/                    #   Generated plots (EDA + eval + SHAP)
│   └── executive_summary.md        #   Full project report
│
├── .github/workflows/ci.yml       # GitHub Actions CI/CD Pipeline
├── Dockerfile.api                  # Backend container definition
├── Dockerfile.frontend             # Frontend container definition
├── docker-compose.yml              # Multi-service orchestration
├── requirements.txt                # Pinned Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for containerised deployment)

### Option A — Local Development

```bash
# 1. Clone the repository
git clone https://github.com/KSKeerthivasan/Fraud-Detection.git
cd Fraud-Detection

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
#    Place creditcard.csv from Kaggle into data/

# 5. Run EDA (optional — generates exploratory plots)
python src/eda.py

# 6. Train the full pipeline (preprocessing → training → evaluation → SHAP)
python src/train.py

# 7. Start the FastAPI backend
uvicorn api.main:app --port 8000

# 8. Start the Streamlit dashboard (in a new terminal)
streamlit run app/app.py
```

### Option B — Docker Deployment

```bash
# Build and launch both services
docker-compose up --build

# Access the dashboard at http://localhost:8501
# API docs available at http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

The system is fully containerised as two independent microservices:

| Service | Container | Port | Description |
|---|---|---|---|
| **Backend** | `fraud-detection-api` | `8000` | FastAPI + Uvicorn serving XGBoost inference and SHAP |
| **Frontend** | `fraud-detection-frontend` | `8501` | Streamlit dashboard consuming the API |

Docker Compose automatically creates a shared bridge network. The frontend resolves the backend via the internal hostname `api` (i.e., `http://api:8000/predict`), while both services are externally accessible on `localhost`.

```bash
# Build images
docker-compose build

# Start services (detached)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 📡 API Reference

### `GET /health`

Health check endpoint.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Analyse a single transaction for fraud.

**Request Body** — All 30 features required:

```json
{
  "Time": 50000.0,
  "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
  "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
  "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
  "V13": -0.99, "V14": -5.34, "V15": 0.39, "V16": -1.14,
  "V17": -2.83, "V18": -0.02, "V19": 0.42, "V20": 0.14,
  "V21": -0.01, "V22": -0.01, "V23": -0.06, "V24": -0.33,
  "V25": 0.08, "V26": -0.06, "V27": 0.03, "V28": 0.01,
  "Amount": 500.00
}
```

**Response:**

```json
{
  "prediction_class": 1,
  "probability": 0.94,
  "risk_level": "High",
  "risk_color": "red",
  "shap_waterfall_base64": "<base64-encoded PNG>"
}
```

> Interactive API documentation is auto-generated at [`http://localhost:8000/docs`](http://localhost:8000/docs) (Swagger UI).

---

## 📈 Model Performance

### Baseline Comparison (Post-Recalibration)

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.056 | 0.918 | 0.105 | 0.971 | 0.724 | 0.0249 |
| Random Forest | 0.849 | 0.806 | 0.827 | 0.974 | 0.855 | 0.0006 |
| XGBoost (Default) | 0.705 | 0.878 | 0.782 | 0.977 | 0.876 | 0.0007 |
| **XGBoost (Tuned) ✅** | **0.962** | **0.766** | **0.852** | **0.982** | **0.870** | **0.0004** |

### Decision Threshold Sensitivity (Tuned XGBoost)

| Threshold | Precision | Recall | F1 Score |
|:---:|:---:|:---:|:---:|
| 0.30 | 0.962 | 0.776 | 0.859 |
| **0.50 (Default)** | **0.962** | **0.765** | **0.852** |
| 0.70 | 0.959 | 0.724 | 0.826 |

> **Why PR-AUC over ROC-AUC?** With 99.83% legitimate transactions, ROC-AUC is inflated by the massive True Negative count. PR-AUC evaluates performance exclusively on the fraud class, providing a more honest assessment.

---

## 💻 Streamlit Dashboard

The dashboard provides three functional tabs:

| Tab | Capabilities |
|---|---|
| 🔍 **Predict Transaction** | Browse the held-out test set via a built-in dataset explorer, or enter custom feature values in Advanced Mode. Each prediction returns a fraud probability, colour-coded risk badge (Low / Medium / High), and a per-transaction SHAP waterfall explanation. |
| 📊 **Model Metrics** | Baseline model comparison table, threshold sensitivity analysis, Confusion Matrix, ROC Curve, Precision-Recall Curve, and Calibration Curve. |
| 🌐 **Global Explainability** | SHAP summary (beeswarm) plot, XGBoost feature importance (Gain), and sample fraud waterfall and force plots. |

---

## ⚙️ CI/CD Pipeline

Every push to `main` or `master` triggers the [GitHub Actions workflow](.github/workflows/ci.yml):

```
Push to GitHub
     │
     ▼
┌──────────────────┐     ┌──────────────────────┐
│  1. Test Job      │────►│  2. Docker Build Job  │
│                   │     │                       │
│  • Checkout repo  │     │  • Build API image    │
│  • Setup Python   │     │  • Build Frontend     │
│    3.12           │     │    image              │
│  • Install deps   │     │                       │
│  • Run pytest     │     │  (Only runs if tests  │
│                   │     │   pass)               │
└──────────────────┘     └──────────────────────┘
```

---

## 🧠 Technical Design Decisions

<details>
<summary><strong>Why is accuracy misleading for fraud detection?</strong></summary>

With only 0.17% fraud, a classifier that labels **everything** as "legitimate" achieves 99.83% accuracy while catching **zero fraud**. Accuracy rewards the majority class, making it completely uninformative. We instead evaluate using **Precision, Recall, F1, ROC-AUC, and PR-AUC**, which focus on the minority class.
</details>

<details>
<summary><strong>Why SMOTE, and why is placement critical?</strong></summary>

**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic minority class samples by interpolating between existing fraud transactions and their k-nearest neighbours. Compared to simple duplication, SMOTE creates diverse synthetic samples that expand the minority class decision region.

**Critical implementation detail:** SMOTE is wrapped inside an `imblearn.Pipeline` within `RandomizedSearchCV`. This ensures synthetic samples are generated **only within each training fold**, preventing data leakage into the validation fold. Applying SMOTE before splitting would allow synthetic copies of the same original sample to appear in both training and validation sets, catastrophically inflating metrics.
</details>

<details>
<summary><strong>Why Bayesian probability recalibration?</strong></summary>

SMOTE artificially balances the training set to ~50/50, but the true fraud prevalence is 0.17%. This means the model's raw `predict_proba` outputs are calibrated for a 50% prior, not the real-world prior. We apply Bayes' theorem to correct for this prior shift:

```
P(fraud | x) = (β × P_smote) / (β × P_smote + 1 − P_smote)
```

Where `β = 0.00173` is the true fraud prevalence. This recalibration converts the SMOTE-biased probability into a mathematically correct real-world probability.
</details>

<details>
<summary><strong>Why XGBoost over deep learning?</strong></summary>

1. **Superior tabular performance** — Gradient-boosted trees consistently outperform neural networks on structured, tabular data with clear feature interactions.
2. **Built-in regularisation** — L1/L2 regularisation prevents overfitting on SMOTE-generated synthetic samples.
3. **TreeSHAP compatibility** — Exact Shapley values computed in polynomial time `O(TLD²)`, enabling real-time explanations.
4. **Best ROC-AUC and PR-AUC** — Outperforms Logistic Regression and Random Forest across all metrics.
</details>

<details>
<summary><strong>How does SHAP improve model transparency?</strong></summary>

SHAP assigns each feature a **Shapley value** — the average marginal contribution of that feature across all possible feature subsets. For a given prediction:

- **Positive SHAP value** → Feature pushes prediction toward fraud
- **Negative SHAP value** → Feature pushes prediction toward legitimate
- The sum of all SHAP values + base value = final model output

This enables **local explanations** (why was *this* transaction flagged?) and **global explanations** (which features matter most overall?), satisfying regulatory audit requirements.
</details>

<details>
<summary><strong>Why decouple the frontend from inference?</strong></summary>

The Streamlit dashboard originally called the XGBoost model directly. By extracting inference into a standalone FastAPI service:

- **Independent scaling** — The API can be scaled horizontally behind a load balancer without duplicating the frontend.
- **Language-agnostic consumption** — Any client (mobile app, internal tool, batch job) can call the REST API.
- **Separation of concerns** — UI changes never risk breaking the inference logic, and vice versa.
- **Cloud-ready** — Each service maps to its own container, ready for deployment on AWS ECS, Fargate, or Kubernetes.
</details>

---

## 📸 Screenshots & Dashboard Walkthrough

> Run `docker-compose up --build` or `streamlit run app/app.py` to view the interactive dashboard.

### 📊 Real-Time Model Performance KPI Dashboard
The landing page displays top-level evaluation metrics of the calibrated XGBoost classifier, providing instant transparency for executive stakeholders.

![Model Performance KPI Dashboard](output/Screenshot%202026-06-10%20211623.png)

### 🔍 Interactive Test Dataset Explorer & Transaction Inspector
Compliance analysts can browse transaction records from the held-out test split, with anomalous fraud cases highlighted in crimson red.

![Test Dataset Explorer](output/Screenshot%202026-06-10%20211658.png)

### ⚠️ Real-Time High-Risk Fraud Detection & Anomaly Indicators
When a fraudulent transaction is analysed, the system flags key risk factors (such as V14, V17, and V12 scoring significantly below normal thresholds) and alerts the analyst with a high-risk warning.

![High-Risk Fraud Detection](output/Screenshot%202026-06-10%20211804.png)

### 🟢 Real-Time Low-Risk Prediction & Parameter Analysis
Legitimate transactions are identified instantly. Visual indicator badges confirm parameters are within healthy thresholds, and predictions are cross-referenced with ground truth for live verification.

![Low-Risk Prediction](output/Screenshot%202026-06-10%20211743.png)

### ⚙️ Model Benchmarking & Decision Threshold Optimisation
Audit model performance against Logistic Regression and Random Forest baselines, and select the optimal decision boundary threshold depending on current fraud risk tolerance.

![Model Benchmarking](output/Screenshot%202026-06-10%20211903.png)

---

## 🔮 Future Roadmap

| Priority | Area | Planned Improvement |
|:---:|---|---|
| 🔴 | **AWS Cloud Storage** | Migrate datasets and `.joblib` model artifacts to S3 for dynamic loading at container startup |
| 🔴 | **AWS Cloud Deployment** | Deploy containerised services to AWS ECS / Fargate with Application Load Balancer |
| 🟡 | **Monitoring** | Evidently AI integration for data drift detection and model degradation alerts |
| 🟡 | **Ensemble Stacking** | XGBoost + LightGBM + Neural Network meta-learner |
| 🟢 | **Real-Time Streaming** | Apache Kafka + Flink for sub-millisecond inference on transaction streams |
| 🟢 | **Graph Features** | GNN-based merchant/cardholder network analysis for fraud ring detection |
| 🟢 | **Active Learning** | Route uncertain predictions to human analysts and retrain incrementally |
| 🟢 | **Federated Learning** | Multi-bank model training without sharing raw transaction records |

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **ML/AI** | XGBoost 3.2, scikit-learn 1.9, imbalanced-learn (SMOTE), SHAP 0.52 |
| **Backend** | FastAPI 0.141, Uvicorn, Pydantic v2 |
| **Frontend** | Streamlit 1.58, Matplotlib, Seaborn |
| **Infrastructure** | Docker, Docker Compose, GitHub Actions |
| **Data** | Pandas 3.0, NumPy 2.4, SciPy 1.17 |
| **Serialisation** | Joblib (model + scaler persistence) |

---

## 📜 License

MIT License — see `LICENSE` for details.

---

<div align="center">
  <b>Built by Keerthivasan K S</b><br>
  <sub>XGBoost · SHAP · FastAPI · Docker · scikit-learn · Streamlit · GitHub Actions</sub>
</div>
