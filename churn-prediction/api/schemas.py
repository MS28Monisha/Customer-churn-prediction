"""Pydantic request/response schemas for the Churn Prediction API."""

from typing import Optional, List
from pydantic import BaseModel, Field


class CustomerRecord(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    customerID: Optional[str] = None


class PredictionResponse(BaseModel):
    customerID: Optional[str] = None
    churn_probability: float
    churn_prediction: int
    risk_level: str


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]


class ModelMetricsResponse(BaseModel):
    best_model: str
    all_models: dict


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    best_model: str
    features: List[FeatureImportanceItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
