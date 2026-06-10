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
    initial_sidebar_state="collapsed",
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

    /* Style the main app container background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(30, 27, 75, 0.35) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(15, 23, 42, 0.8) 0%, #030712 100%) !important;
        background-attachment: fixed !important;
        color: #f1f5f9 !important;
    }

    /* Background animated aurora glow blobs */
    .stApp::before {
        content: "";
        position: fixed;
        top: -15%;
        left: -15%;
        width: 50%;
        height: 50%;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.05) 0%, transparent 70%);
        animation: aurora 25s infinite alternate ease-in-out;
        z-index: -2;
        pointer-events: none;
    }
    
    .stApp::after {
        content: "";
        position: fixed;
        bottom: -20%;
        right: -10%;
        width: 60%;
        height: 60%;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.04) 0%, transparent 70%);
        animation: aurora 30s infinite alternate-reverse ease-in-out;
        z-index: -2;
        pointer-events: none;
    }
    
    @keyframes aurora {
        0% { transform: translate(0px, 0px) scale(1) rotate(0deg); }
        50% { transform: translate(30px, -40px) scale(1.1) rotate(90deg); }
        100% { transform: translate(0px, 0px) scale(1) rotate(180deg); }
    }

    /* Fade-in and slide animations */
    .animate-fade-in {
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        opacity: 0;
    }
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.2s; }
    .delay-3 { animation-delay: 0.3s; }
    .delay-4 { animation-delay: 0.4s; }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(17, 24, 39, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(6, 182, 212, 0.25);
        box-shadow: 0 12px 40px 0 rgba(6, 182, 212, 0.15);
    }

    /* Gradient hero header */
    .hero-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.4) 100%) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    }
    .hero-header h1 {
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.8px;
        background: linear-gradient(120deg, #f8fafc 30%, #38bdf8 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 999px;
        margin-bottom: 0.75rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Metric KPI cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    .metric-card.cyan-glow:hover { border-color: rgba(6, 182, 212, 0.4); box-shadow: 0 10px 30px rgba(6, 182, 212, 0.15); }
    .metric-card.green-glow:hover { border-color: rgba(16, 185, 129, 0.4); box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15); }
    .metric-card.orange-glow:hover { border-color: rgba(245, 158, 11, 0.4); box-shadow: 0 10px 30px rgba(245, 158, 11, 0.15); }
    .metric-card.purple-glow:hover { border-color: rgba(168, 85, 247, 0.4); box-shadow: 0 10px 30px rgba(168, 85, 247, 0.15); }
    .metric-card.blue-glow:hover { border-color: rgba(59, 130, 246, 0.4); box-shadow: 0 10px 30px rgba(59, 130, 246, 0.15); }
    .metric-card.emerald-glow:hover { border-color: rgba(52, 211, 153, 0.4); box-shadow: 0 10px 30px rgba(52, 211, 153, 0.15); }

    .metric-card .label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.5px;
    }
    .metric-card .sub {
        color: #475569;
        font-size: 0.72rem;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    /* Result box */
    .result-box {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 1.8rem !important;
        margin-top: 1.2rem !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3) !important;
    }
    .result-box h3 {
        color: #f1f5f9;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0 0 0.85rem 0;
    }

    /* Section divider */
    .section-title {
        color: #cbd5e1;
        font-size: 1.15rem;
        font-weight: 700;
        border-left: 4px solid #3b82f6;
        padding-left: 0.8rem;
        margin: 2rem 0 1.2rem 0;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
    }

    /* Fraud / Legit verdict boxes */
    .verdict-fraud {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.6), rgba(185, 28, 28, 0.7)) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid #ef4444 !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.6rem !important;
        color: #fecaca !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.25) !important;
        letter-spacing: 0.5px !important;
    }
    
    .verdict-legit {
        background: linear-gradient(135deg, rgba(20, 83, 45, 0.6), rgba(21, 128, 61, 0.7)) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid #22c55e !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.6rem !important;
        color: #d1fae5 !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        box-shadow: 0 0 25px rgba(34, 197, 94, 0.25) !important;
        letter-spacing: 0.5px !important;
    }

    /* Transaction parameter grid details */
    .transaction-detail-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
        width: 100%;
    }
    .detail-tile {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        transition: all 0.3s ease;
    }
    .detail-tile:hover {
        background: rgba(30, 41, 59, 0.6);
        border-color: rgba(59, 130, 246, 0.2);
        transform: translateY(-2px);
    }
    .detail-tile.highlight {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
    }
    .detail-tile .tile-lbl {
        color: #64748b;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .detail-tile .tile-val {
        color: #f1f5f9;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .detail-tile .tile-sub {
        color: #475569;
        font-size: 0.72rem;
        margin-top: 0.25rem;
    }
    .detail-tile .tile-badge {
        margin-top: 0.5rem;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 999px;
        text-transform: uppercase;
    }
    .badge-red {
        background: rgba(239, 68, 68, 0.15) !important;
        color: #ef4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15) !important;
        color: #10b981 !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
    }

    /* Sidebar container override */
    section[data-testid="stSidebar"] {
        background: #090d16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    
    /* Input element customizations for transparent dark/glassy theme */
    div[data-baseweb="select"], div[data-baseweb="input"], input[type="number"] {
        background-color: rgba(17, 24, 39, 0.6) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #f1f5f9 !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-baseweb="select"]:hover, div[data-baseweb="input"]:hover, input[type="number"]:hover {
        border-color: rgba(6, 182, 212, 0.3) !important;
    }
    
    div[role="combobox"] {
        color: #f1f5f9 !important;
    }
    
    /* Custom button styling to render glossy premium widgets */
    .stButton>button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #60a5fa 100%) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5), 0 0 15px rgba(6, 182, 212, 0.4) !important;
        transform: translateY(-2px) !important;
        border-color: rgba(255,255,255,0.2) !important;
    }
    
    .stButton>button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.2) !important;
    }

    /* Tabs styling override */
    button[data-baseweb="tab"] {
        background: transparent !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #38bdf8 !important;
    }
    
    button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* Images — border + shadow only, NO background that could cover the img */
    div[data-testid="stImage"], div[class*="stImage"] {
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45) !important;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease !important;
        margin-bottom: 1.5rem !important;
    }
    div[data-testid="stImage"]:hover, div[class*="stImage"]:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(6, 182, 212, 0.3) !important;
        box-shadow: 0 10px 36px rgba(6, 182, 212, 0.12) !important;
    }

    /* Pyplot/Matplotlib — border + shadow only, NO background */
    div[data-testid="stPlot"], div[class*="stPlot"],
    div[data-testid="stPyplot"], div[class*="stPyplot"], .stPlot {
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45) !important;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease !important;
        margin-bottom: 1.5rem !important;
    }
    div[data-testid="stPlot"]:hover, div[class*="stPlot"]:hover,
    div[data-testid="stPyplot"]:hover, div[class*="stPyplot"]:hover, .stPlot:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(6, 182, 212, 0.3) !important;
        box-shadow: 0 10px 36px rgba(6, 182, 212, 0.12) !important;
    }

    /* DataFrames — border + shadow only, NO background so iframe content is fully visible */
    div[data-testid="stDataFrame"], div[class*="stDataFrame"],
    div[data-testid="stTable"], div[class*="stTable"] {
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 16px !important;
        padding: 0.75rem !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 1.5rem !important;
        overflow: hidden !important;
    }

    /* Style alerts / notification boxes */
    div[data-testid="stNotification"], div[class*="stNotification"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        color: #f1f5f9 !important;
    }

    /* Custom progress bar styles */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%) !important;
        border-radius: 999px !important;
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.4) !important;
    }
    div[role="progressbar"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 999px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Style expanders */
    div[data-testid="stExpander"], div[class*="stExpander"] {
        background: rgba(17, 24, 39, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 25px rgba(0,0,0,0.2) !important;
    }
    div[data-testid="stExpander"] details summary, div[class*="stExpander"] details summary {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    /* Style radio button groups */
    div[data-testid="stRadio"] label, div[class*="stRadio"] label {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"], div[class*="stRadio"] div[role="radiogroup"] {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
    }

    /* Hide default Streamlit branding */
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
    glow_classes = ["orange-glow", "green-glow", "cyan-glow", "purple-glow", "blue-glow", "emerald-glow"]
    for idx, (col, (lbl, val, sub)) in enumerate(zip(kpi_cols, kpis)):
        glow = glow_classes[idx]
        col.markdown(
            f'<div class="metric-card {glow} animate-fade-in delay-{idx+1}"><div class="label">{lbl}</div>'
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

        # Display selected transaction parameters grid
        st.markdown('<div class="section-title">🔍 Selected Transaction Parameters</div>', unsafe_allow_html=True)
        t_val = test_samples.loc[selected_idx, 'Time']
        a_val = test_samples.loc[selected_idx, 'Amount']
        v14_val = test_samples.loc[selected_idx, 'V14']
        v17_val = test_samples.loc[selected_idx, 'V17']
        v12_val = test_samples.loc[selected_idx, 'V12']
        v4_val = test_samples.loc[selected_idx, 'V4']
        
        v14_badge = "🔴 Risk Factor" if v14_val < -2 else "🟢 Normal"
        v17_badge = "🔴 Risk Factor" if v17_val < -2 else "🟢 Normal"
        v12_badge = "🔴 Risk Factor" if v12_val < -2 else "🟢 Normal"
        v4_badge = "🔴 Risk Factor" if v4_val > 2 else "🟢 Normal"
        
        st.markdown(
            f"""
            <div class="transaction-detail-grid animate-fade-in delay-1">
                <div class="detail-tile">
                    <span class="tile-lbl">TRANSACTION TIME</span>
                    <span class="tile-val">{int(t_val):,}s</span>
                    <span class="tile-sub">Relative timestamp</span>
                </div>
                <div class="detail-tile highlight">
                    <span class="tile-lbl">TRANSACTION AMOUNT</span>
                    <span class="tile-val">${a_val:.2f}</span>
                    <span class="tile-sub">USD currency</span>
                </div>
                <div class="detail-tile">
                    <span class="tile-lbl">V14 (IMPACT LEVEL)</span>
                    <span class="tile-val">{v14_val:.4f}</span>
                    <span class="tile-badge badge-{"red" if v14_val < -2 else "green"}">{v14_badge}</span>
                </div>
                <div class="detail-tile">
                    <span class="tile-lbl">V17 (ANOMALY SCORE)</span>
                    <span class="tile-val">{v17_val:.4f}</span>
                    <span class="tile-badge badge-{"red" if v17_val < -2 else "green"}">{v17_badge}</span>
                </div>
                <div class="detail-tile">
                    <span class="tile-lbl">V12 (VARIATION RATIO)</span>
                    <span class="tile-val">{v12_val:.4f}</span>
                    <span class="tile-badge badge-{"red" if v12_val < -2 else "green"}">{v12_badge}</span>
                </div>
                <div class="detail-tile">
                    <span class="tile-lbl">V4 (SENSITIVITY)</span>
                    <span class="tile-val">{v4_val:.4f}</span>
                    <span class="tile-badge badge-{"red" if v4_val > 2 else "green"}">{v4_badge}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

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

        # Display configured transaction parameters grid
        st.markdown('<div class="section-title">✏️ Configured Transaction Parameters</div>', unsafe_allow_html=True)
        v14_val = v_vals["V14"]
        v17_val = v_vals["V17"]
        v12_val = v_vals["V12"]
        v4_val = v_vals["V4"]
        
        v14_badge = "🔴 Risk Factor" if v14_val < -2 else "🟢 Normal"
        v17_badge = "🔴 Risk Factor" if v17_val < -2 else "🟢 Normal"
        v12_badge = "🔴 Risk Factor" if v12_val < -2 else "🟢 Normal"
        v4_badge = "🔴 Risk Factor" if v4_val > 2 else "🟢 Normal"
        
        st.markdown(
            f"""
            <div class="transaction-detail-grid animate-fade-in delay-1">
                <div class="detail-tile">
                    <span class="tile-lbl">TRANSACTION TIME</span>
                    <span class="tile-val">{int(time_val):,}s</span>
                    <span class="tile-sub">Manual setting</span>
                </div>
                <div class="detail-tile highlight">
                    <span class="tile-lbl">TRANSACTION AMOUNT</span>
                    <span class="tile-val">${amount_val:.2f}</span>
                    <span class="tile-sub">USD currency</span>
                </div>
                <div class="detail-tile">
                    <span class="tile-lbl">V14 (IMPACT LEVEL)</span>
                    <span class="tile-val">{v14_val:.4f}</span>
                    <span class="tile-badge badge-{"red" if v14_val < -2 else "green"}">{v14_badge}</span>
                </div>
                <div class="detail-tile">
                    <span class="tile-lbl">V17 (ANOMALY SCORE)</span>
                    <span class="tile-val">{v17_val:.4f}</span>
                    <span class="tile-badge badge-{"red" if v17_val < -2 else "green"}">{v17_badge}</span>
                </div>
                <div class="detail-tile">
                    <span class="tile-lbl">V12 (VARIATION RATIO)</span>
                    <span class="tile-val">{v12_val:.4f}</span>
                    <span class="tile-badge badge-{"red" if v12_val < -2 else "green"}">{v12_badge}</span>
                </div>
                <div class="detail-tile">
                    <span class="tile-lbl">V4 (SENSITIVITY)</span>
                    <span class="tile-val">{v4_val:.4f}</span>
                    <span class="tile-badge badge-{"red" if v4_val > 2 else "green"}">{v4_badge}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

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
                st.markdown(f'<h4 style="color:#f1f5f9;margin-top:0;margin-bottom:0.75rem;font-weight:700;">{title}</h4>', unsafe_allow_html=True)
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
