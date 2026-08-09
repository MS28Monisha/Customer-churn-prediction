"""
FastAPI application entrypoint for the Customer Churn Prediction API.

Run locally with:
    uvicorn api.main:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import predictions, model_info
from api.schemas import HealthResponse
from src.predict import predictor
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability using a trained ML model.",
    version="1.0.0",
)

# Allow the Streamlit / React frontend (running on a different port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)
app.include_router(model_info.router)


@app.get("/", response_model=HealthResponse, tags=["health"])
def health_check():
    """Basic health check + whether the model artifact is available."""
    model_loaded = config.MODEL_PATH.exists() and config.PREPROCESSOR_PATH.exists()
    return {"status": "ok", "model_loaded": model_loaded}
