import os
import sys
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set non-interactive backend
matplotlib.use('Agg')

# Append project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from app.utils import load_ml_assets, predict_transaction, get_local_shap_waterfall_fig

app = FastAPI(
    title="Fraud Detection API",
    description="REST API for predicting credit card fraud with SHAP explainability.",
    version="1.0.0"
)

# Load assets on startup
model, scaler, test_samples, training_summary = load_ml_assets()

# Dynamically create Pydantic model for V1-V28, Time, Amount
fields = {
    'Time': (float, ...),
    'Amount': (float, ...)
}
for i in range(1, 29):
    fields[f'V{i}'] = (float, ...)

TransactionInput = create_model('TransactionInput', **fields)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(transaction: TransactionInput):
    try:
        # Convert input to DataFrame (Pydantic v2 syntax)
        data_dict = transaction.model_dump() 
        
        # Order columns correctly (assuming Time, V1..V28, Amount)
        cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = pd.DataFrame([data_dict], columns=cols)
        
        # Run prediction pipeline
        result = predict_transaction(model, scaler, df)
        
        # Generate SHAP waterfall plot and encode as base64
        fig = get_local_shap_waterfall_fig(model, result["scaled_df"])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        
        return {
            "prediction_class": int(result["class"]),
            "probability": float(result["probability"]),
            "risk_level": result["risk_level"],
            "risk_color": result["risk_color"],
            "shap_waterfall_base64": img_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
