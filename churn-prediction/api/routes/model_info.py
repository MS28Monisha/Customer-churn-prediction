"""Model metadata endpoints: performance metrics and feature importance."""

import logging
from fastapi import APIRouter, HTTPException

from api.schemas import ModelMetricsResponse, FeatureImportanceResponse
from src.predict import predictor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metrics", response_model=ModelMetricsResponse)
def get_metrics():
    """Return evaluation metrics (accuracy, precision, recall, F1, ROC-AUC) for all trained models."""
    try:
        return predictor.get_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
def get_feature_importance():
    """Return the top churn drivers ranked by importance for the best model."""
    try:
        metrics = predictor.get_metrics()
        features = predictor.get_feature_importance()
        return {"best_model": metrics["best_model"], "features": features}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
