"""
Central configuration for the Customer Churn Prediction project.
Keeping paths and constants here avoids hardcoding them across modules.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_PATH = DATA_RAW_DIR / "telco_churn.csv"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"

# --- Schema --------------------------------------------------------------
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

RANDOM_SEED = 42
TEST_SIZE = 0.2

RISK_THRESHOLDS = {
    "low": 0.33,
    "medium": 0.66,
}
