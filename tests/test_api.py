from fastapi.testclient import TestClient
import sys
import os

# Ensure we can import the api module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_predict_endpoint_success():
    # Create a synthetic valid payload
    payload = {"Time": 50000.0, "Amount": 50.0}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0

    # Intentionally skew a variable to ensure the model runs
    payload["V14"] = -5.0

    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify the structure of the returned JSON
    assert "prediction_class" in data
    assert "probability" in data
    assert "risk_level" in data
    assert "risk_color" in data
    assert "shap_waterfall_base64" in data
    
    # Verify data types
    assert isinstance(data["prediction_class"], int)
    assert isinstance(data["probability"], float)
    assert data["probability"] >= 0.0 and data["probability"] <= 1.0

def test_predict_endpoint_validation_error():
    # Missing required 'Amount' field
    payload = {"Time": 50000.0}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0

    response = client.post("/predict", json=payload)
    
    # FastAPI should return 422 Unprocessable Entity for Pydantic validation failure
    assert response.status_code == 422
