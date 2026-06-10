"""
app/utils.py
Helper utilities for the Streamlit dashboard.
Handles model loading, feature scaling, prediction, and SHAP explanation generation.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import streamlit as st

# Allow importing from the src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


@st.cache_resource
def load_ml_assets():
    """
    Loads and caches the trained XGBoost model, RobustScaler, test samples CSV,
    and optional training summary metadata.
    Uses st.cache_resource so heavy objects are loaded only once per session.
    """
    model = joblib.load(config.XGB_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    test_samples = pd.read_csv(config.TEST_DATA_PATH, index_col=0)

    summary_path = os.path.join(config.MODELS_DIR, "training_summary.joblib")
    training_summary = joblib.load(summary_path) if os.path.exists(summary_path) else None

    return model, scaler, test_samples, training_summary


def scale_features(raw_features_df: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Scales the raw Time and Amount columns using the fitted RobustScaler.
    V1-V28 are already PCA-transformed and do not require further scaling.
    """
    scaled = raw_features_df.copy()
    scaled[["Time", "Amount"]] = scaler.transform(raw_features_df[["Time", "Amount"]])
    return scaled


def predict_transaction(model, scaler, raw_features_df: pd.DataFrame) -> dict:
    """
    Accepts a single-row raw feature DataFrame, scales it, and returns:
      - class (0 or 1)
      - probability (float)
      - risk_level ('Low' | 'Medium' | 'High')
      - risk_color  ('green' | 'orange' | 'red')
      - scaled_df   (the scaled DataFrame for downstream SHAP use)
    """
    scaled_df = scale_features(raw_features_df, scaler)
    prob = float(model.predict_proba(scaled_df)[0, 1])

    if prob < 0.30:
        risk_level, risk_color = "Low", "green"
    elif prob < 0.70:
        risk_level, risk_color = "Medium", "orange"
    else:
        risk_level, risk_color = "High", "red"

    return {
        "class": int(prob >= 0.50),
        "probability": prob,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "scaled_df": scaled_df,
    }


def get_local_shap_waterfall_fig(model, scaled_features_df: pd.DataFrame):
    """
    Dynamically creates a SHAP TreeExplainer and returns a matplotlib figure
    showing the waterfall plot for the first row of scaled_features_df.

    WHY NO SERIALIZED EXPLAINER:
    We intentionally avoid pickling the TreeExplainer. Serialized SHAP explainers
    can cause version-mismatch errors and inflate storage. Because TreeExplainer
    is cheap to instantiate (it just reads tree structure from the model), we
    recreate it on-demand.
    """
    # Use dark style context for the plot
    plt.style.use("dark_background")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(scaled_features_df)

    # For binary XGBoost classifiers shap_values may be 3-D: (n, features, 2)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    # Recreate figure with dark slate facecolor matching glass card backgrounds
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#0b0f19")
    ax.set_facecolor("#111827")
    
    # Generate waterfall plot inside our axis
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    
    # Target all text elements and set high-contrast colors
    ax = fig.gca()
    ax.set_facecolor("#111827")
    current_title = ax.get_title()
    ax.set_title(current_title, color="#f1f5f9", fontweight="bold", fontsize=13, pad=15)
    
    # Format labels, axis text and tick marks
    ax.tick_params(colors="#94a3b8", labelsize=10)
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    
    # Lighten text elements inside the waterfall plot
    for txt in ax.texts:
        txt.set_color("#f8fafc")
        txt.set_fontsize(9)
        
    plt.tight_layout()
    return fig


def get_risk_badge_html(risk_level: str, probability: float) -> str:
    """Returns styled HTML for a colour-coded risk badge."""
    color_map = {
        "Low": {
            "bg": "rgba(16, 185, 129, 0.12)",
            "border": "rgba(16, 185, 129, 0.4)",
            "text": "#10b981",
            "glow": "rgba(16, 185, 129, 0.25)"
        },
        "Medium": {
            "bg": "rgba(245, 158, 11, 0.12)",
            "border": "rgba(245, 158, 11, 0.4)",
            "text": "#f59e0b",
            "glow": "rgba(245, 158, 11, 0.25)"
        },
        "High": {
            "bg": "rgba(239, 68, 68, 0.12)",
            "border": "rgba(239, 68, 68, 0.4)",
            "text": "#ef4444",
            "glow": "rgba(239, 68, 68, 0.35)"
        }
    }
    style = color_map.get(risk_level, {
        "bg": "rgba(107, 114, 128, 0.12)",
        "border": "rgba(107, 114, 128, 0.4)",
        "text": "#9ca3af",
        "glow": "rgba(107, 114, 128, 0.15)"
    })
    return (
        f'<div style="display:inline-block;padding:8px 22px;border-radius:12px;'
        f'background:{style["bg"]};border:1px solid {style["border"]};color:{style["text"]};'
        f'font-size:18px;font-weight:700;box-shadow: 0 0 20px {style["glow"]};'
        f'backdrop-filter: blur(12px);-webkit-backdrop-filter: blur(12px);'
        f'letter-spacing: 0.5px;text-transform: uppercase;">'
        f"{risk_level} Risk &nbsp;•&nbsp; {probability*100:.1f}%</div>"
    )
