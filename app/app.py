"""
app/app.py
Streamlit Dashboard for Explainable Financial Fraud Detection.

Tabs:
  1. 🔍 Predict Transaction  – Test Dataset Explorer + Manual (Advanced) mode
  2. 📊 Model Metrics        – Comparative baseline table, threshold table, evaluation curves
  3. 🌐 Global Explainability – SHAP Summary Plot + XGBoost Feature Importance
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Allow importing from src package and sibling utils module
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, APP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import config
from utils import (
    load_ml_assets,
    predict_transaction,
    get_local_shap_waterfall_fig,
    get_risk_badge_html,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #1e40af33;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .hero-header h1 {
        color: #f0f9ff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: #1d4ed8;
        color: #bfdbfe;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 0.75rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .metric-card .label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        color: #f1f5f9;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1;
    }
    .metric-card .sub {
        color: #475569;
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.6rem;
        margin-top: 1rem;
    }
    .result-box h3 {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 0.75rem 0;
    }

    /* Section divider */
    .section-title {
        color: #cbd5e1;
        font-size: 1.1rem;
        font-weight: 700;
        border-left: 4px solid #3b82f6;
        padding-left: 0.7rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Fraud / Legit verdict boxes */
    .verdict-fraud {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #fca5a5;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
    }
    .verdict-legit {
        background: linear-gradient(135deg, #14532d, #166534);
        border: 1px solid #22c55e;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #86efac;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────────────────────────────────────
model, scaler, test_samples, training_summary = load_ml_assets()

FEATURE_COLS = [c for c in test_samples.columns if c != "Class"]

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-badge">🛡️ AI-Powered Fraud Intelligence</div>
        <h1>Explainable Financial Fraud Detection</h1>
        <p>XGBoost + SHAP · Credit Card Transactions · Real-time Risk Scoring</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL KPI STRIP  (always visible)
# ─────────────────────────────────────────────────────────────────────────────
if training_summary:
    m = training_summary["best_xgb_metrics"]
    kpi_cols = st.columns(6)
    kpis = [
        ("Precision",   f"{m['Precision']:.4f}",   "@ threshold 0.50"),
        ("Recall",      f"{m['Recall']:.4f}",       "Fraud Detection Rate"),
        ("F1 Score",    f"{m['F1 Score']:.4f}",     "Harmonic Mean"),
        ("ROC-AUC",     f"{m['ROC-AUC']:.4f}",      "Ranking Quality"),
        ("PR-AUC",      f"{m['PR-AUC']:.4f}",       "Imbalanced Perf."),
        ("Brier Score", f"{m['Brier Score']:.4f}",  "Calibration Quality"),
    ]
    for col, (lbl, val, sub) in zip(kpi_cols, kpis):
        col.markdown(
            f'<div class="metric-card"><div class="label">{lbl}</div>'
            f'<div class="value">{val}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["🔍 Predict Transaction", "📊 Model Metrics", "🌐 Global Explainability"]
)

# ══════════════════════════════════════════════════════════════════
# TAB 1 – PREDICT TRANSACTION
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Transaction Prediction</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Input mode",
        ["📋 Test Dataset Explorer (Recommended)", "✏️ Advanced – Manual Feature Entry"],
        horizontal=True,
    )

    # ── MODE A: Test Dataset Explorer ─────────────────────────────
    if "Test Dataset" in mode:
        st.markdown(
            "_Select any transaction from the held-out test set. "
            "Fraud transactions are highlighted in the table._"
        )

        # Filter controls
        filter_col, idx_col = st.columns([2, 2])
        with filter_col:
            show_filter = st.selectbox(
                "Filter by class",
                ["All Transactions", "Fraudulent Only", "Legitimate Only"],
            )
        
        if show_filter == "Fraudulent Only":
            display_df = test_samples[test_samples["Class"] == 1]
        elif show_filter == "Legitimate Only":
            display_df = test_samples[test_samples["Class"] == 0]
        else:
            display_df = test_samples

        # Show mini-table (limited columns for readability)
        preview_cols = ["Time", "Amount", "V1", "V2", "V3", "V4", "V14", "V17", "Class"]
        st.dataframe(
            display_df[preview_cols].style.apply(
                lambda row: [
                    "background-color: #7f1d1d44; color: #fca5a5"
                    if row["Class"] == 1
                    else ""
                    for _ in row
                ],
                axis=1,
            ),
            use_container_width=True,
            height=280,
        )

        selected_idx = st.selectbox(
            "Select a transaction index to analyse:",
            options=display_df.index.tolist(),
            format_func=lambda i: f"Index {i}  |  {'🔴 FRAUD' if test_samples.loc[i,'Class']==1 else '🟢 Legitimate'}  |  Amount: ${test_samples.loc[i,'Amount']:.2f}",
        )

        raw_row = test_samples.loc[[selected_idx], FEATURE_COLS]
        actual_label = int(test_samples.loc[selected_idx, "Class"])

    # ── MODE B: Manual Entry ───────────────────────────────────────
    else:
        st.markdown("_Enter raw (unscaled) feature values. Time is in seconds since first tx; Amount in USD._")
        with st.expander("⚙️ Feature Input Panel", expanded=True):
            c1, c2, c3 = st.columns(3)
            time_val   = c1.number_input("Time (seconds)",  value=50000.0, step=1000.0)
            amount_val = c2.number_input("Amount (USD)",     value=50.0,    step=0.01)
            c3.markdown("<br>", unsafe_allow_html=True)

            st.markdown("**PCA Features V1 – V28** _(anonymous, standardised by issuer)_")
            v_vals = {}
            vcols = st.columns(7)
            for i in range(1, 29):
                v_vals[f"V{i}"] = vcols[(i - 1) % 7].number_input(f"V{i}", value=0.0, step=0.01, key=f"v{i}")

        raw_row = pd.DataFrame(
            [{"Time": time_val, **v_vals, "Amount": amount_val}], columns=FEATURE_COLS
        )
        actual_label = None  # unknown in manual mode

    # ── RUN PREDICTION ────────────────────────────────────────────
    st.markdown("---")
    predict_btn = st.button("🚀 Run Fraud Detection", type="primary", use_container_width=True)

    if predict_btn:
        result = predict_transaction(model, scaler, raw_row)

        col_a, col_b = st.columns(2)
        with col_a:
            if result["class"] == 1:
                st.markdown('<div class="verdict-fraud">⚠️ FRAUDULENT TRANSACTION DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-legit">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown(get_risk_badge_html(result["risk_level"], result["probability"]), unsafe_allow_html=True)
            st.progress(result["probability"], text=f"Fraud Probability: {result['probability']*100:.2f}%")

        if actual_label is not None:
            correct = result["class"] == actual_label
            st.info(
                f"**Ground Truth:** {'🔴 Fraud' if actual_label==1 else '🟢 Legitimate'} | "
                f"**Prediction:** {'Correct ✅' if correct else 'Incorrect ❌'}"
            )

        # ── EXPLAIN PREDICTION ────────────────────────────────────
        st.markdown('<div class="section-title">SHAP Local Explanation</div>', unsafe_allow_html=True)
        st.markdown(
            "_The waterfall plot shows which features pushed the model's prediction toward "
            "fraud (red, positive SHAP) or away from fraud (blue, negative SHAP)._"
        )
        with st.spinner("Computing SHAP values…"):
            fig = get_local_shap_waterfall_fig(model, result["scaled_df"])
            st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 – MODEL METRICS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Baseline Model Comparison</div>', unsafe_allow_html=True)
    if training_summary and "baselines" in training_summary:
        baseline_df = pd.DataFrame(training_summary["baselines"])
        st.dataframe(
            baseline_df.style.highlight_max(
                subset=["Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
                color="#1d4ed844",
            ).highlight_min(
                subset=["Brier Score"],
                color="#14532d44",
            ).format(
                {c: "{:.4f}" for c in ["Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC", "Brier Score"]}
            ),
            use_container_width=True,
        )

    st.markdown('<div class="section-title">Threshold Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "_Lowering the threshold increases recall (catches more fraud) at the cost of precision (more false alarms)._"
    )
    if training_summary and "thresholds" in training_summary:
        thresh_df = pd.DataFrame(training_summary["thresholds"])
        st.dataframe(
            thresh_df.style.format(
                {c: "{:.4f}" for c in ["Precision", "Recall", "F1 Score"]}
            ),
            use_container_width=True,
        )

    st.markdown('<div class="section-title">Evaluation Curves</div>', unsafe_allow_html=True)
    curve_paths = {
        "Confusion Matrix":       os.path.join(config.FIGURES_DIR, "confusion_matrix.png"),
        "ROC Curve":              os.path.join(config.FIGURES_DIR, "roc_curve.png"),
        "Precision-Recall Curve": os.path.join(config.FIGURES_DIR, "precision_recall_curve.png"),
        "Calibration Curve":      os.path.join(config.FIGURES_DIR, "calibration_curve.png"),
    }

    img_cols = st.columns(2)
    for idx, (title, path) in enumerate(curve_paths.items()):
        if os.path.exists(path):
            with img_cols[idx % 2]:
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 – GLOBAL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">SHAP Global Summary Plot</div>', unsafe_allow_html=True)
    st.markdown(
        "_Each dot is one transaction. Horizontal position = SHAP value (impact on prediction). "
        "Color = actual feature value (red = high, blue = low). Features ranked by average absolute SHAP._"
    )
    shap_summary_path = os.path.join(config.FIGURES_DIR, "shap_summary.png")
    if os.path.exists(shap_summary_path):
        st.image(shap_summary_path, use_container_width=True)

    col_fi, col_force = st.columns(2)
    with col_fi:
        st.markdown('<div class="section-title">XGBoost Feature Importance (Gain)</div>', unsafe_allow_html=True)
        fi_path = os.path.join(config.FIGURES_DIR, "xgboost_feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)

    with col_force:
        st.markdown('<div class="section-title">SHAP Waterfall – Sample Fraud</div>', unsafe_allow_html=True)
        wf_path = os.path.join(config.FIGURES_DIR, "shap_waterfall.png")
        if os.path.exists(wf_path):
            st.image(wf_path, use_container_width=True)

    st.markdown('<div class="section-title">SHAP Force Plot – Sample Fraud</div>', unsafe_allow_html=True)
    force_path = os.path.join(config.FIGURES_DIR, "shap_force.png")
    if os.path.exists(force_path):
        st.image(force_path, use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="color:#64748b;font-size:0.82rem;text-align:center;">
        Built with XGBoost · SHAP · scikit-learn · Streamlit &nbsp;|&nbsp;
        Dataset: Kaggle Credit Card Fraud Detection (284,807 transactions, 0.17% fraud)
        </div>
        """,
        unsafe_allow_html=True,
    )
