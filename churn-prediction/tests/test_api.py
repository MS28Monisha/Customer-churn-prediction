"""
Integration tests for the FastAPI app.

Requires a trained model to exist (run `python -m src.train_model` first),
since these tests exercise the real prediction path rather than mocking it.
"""

import io
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import config

client = TestClient(app)

SAMPLE_RECORD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 477.5,
    "customerID": "TEST-001",
}


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Model not trained yet")
def test_predict_single_returns_valid_response():
    response = client.post("/predict", json=SAMPLE_RECORD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_level"] in ("Low", "Medium", "High")
    assert body["customerID"] == "TEST-001"


def test_predict_single_invalid_payload_returns_422():
    bad_record = dict(SAMPLE_RECORD)
    del bad_record["tenure"]
    response = client.post("/predict", json=bad_record)
    assert response.status_code == 422


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Model not trained yet")
def test_predict_batch_with_valid_csv():
    header = ",".join(config.ALL_FEATURES + [config.ID_COLUMN])
    row = ",".join(
        [
            "Female", "Yes", "No", "Yes", "No", "No", "Fiber optic", "No", "No", "No",
            "No", "Yes", "No", "Month-to-month", "Yes", "Electronic check", "0", "5", "95.5",
            "477.5", "TEST-002",
        ]
    )
    # Build CSV matching config.ALL_FEATURES order exactly to avoid mismatches
    csv_content = _build_matching_csv()
    files = {"file": ("customers.csv", csv_content, "text/csv")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert 0.0 <= body["predictions"][0]["churn_probability"] <= 1.0


def test_predict_batch_rejects_non_csv():
    files = {"file": ("customers.txt", b"not,a,csv", "text/plain")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400


def _build_matching_csv() -> bytes:
    import pandas as pd

    row = {f: SAMPLE_RECORD[f] for f in config.ALL_FEATURES}
    row[config.ID_COLUMN] = "TEST-002"
    df = pd.DataFrame([row])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@pytest.mark.skipif(not config.METRICS_PATH.exists(), reason="Model not trained yet")
def test_model_metrics_endpoint():
    response = client.get("/model/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "best_model" in body
    assert "all_models" in body


@pytest.mark.skipif(not config.FEATURE_IMPORTANCE_PATH.exists(), reason="Model not trained yet")
def test_feature_importance_endpoint():
    response = client.get("/model/feature-importance")
    assert response.status_code == 200
    body = response.json()
    assert "features" in body
    assert len(body["features"]) > 0
