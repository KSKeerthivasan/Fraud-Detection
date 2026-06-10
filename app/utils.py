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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(scaled_features_df)

    # For binary XGBoost classifiers shap_values may be 3-D: (n, features, 2)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    fig, _ = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    plt.title("SHAP Local Feature Attribution", fontweight="bold", fontsize=13, pad=15)
    plt.tight_layout()
    return fig


def get_risk_badge_html(risk_level: str, probability: float) -> str:
    """Returns styled HTML for a colour-coded risk badge."""
    color_map = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
    color = color_map.get(risk_level, "#6b7280")
    return (
        f'<div style="display:inline-block;padding:8px 20px;border-radius:8px;'
        f'background:{color};color:white;font-size:18px;font-weight:700;">'
        f"{risk_level} Risk &nbsp;|&nbsp; {probability*100:.1f}%</div>"
    )
