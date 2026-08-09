"""
Data loading, cleaning, and preprocessing pipeline for the churn model.
"""

import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw churn CSV from disk."""
    path = path or config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python src/generate_sample_data.py` "
            "or place the Telco Customer Churn CSV there."
        )
    logger.info("Loading raw data from %s", path)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean known data quality issues in the Telco churn dataset:
    - TotalCharges is sometimes blank/whitespace for new customers (tenure=0)
    - Standardize target column to binary 0/1
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if config.TARGET_COLUMN in df.columns:
        df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    # Drop exact duplicate rows and rows missing an identifier
    df = df.drop_duplicates()
    if config.ID_COLUMN in df.columns:
        df = df.dropna(subset=[config.ID_COLUMN])

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer that handles imputing, scaling,
    and one-hot encoding. Kept separate from the model so it can be
    persisted and reused identically at inference time.
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
        ]
    )
    return preprocessor


def get_feature_and_target(df: pd.DataFrame):
    """Split a cleaned dataframe into X (features) and y (target)."""
    missing = [c for c in config.ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input data: {missing}")

    X = df[config.ALL_FEATURES]
    y = df[config.TARGET_COLUMN] if config.TARGET_COLUMN in df.columns else None
    return X, y


def validate_input_row(record: dict) -> None:
    """
    Validate a single incoming record (e.g. from the API) has the
    minimum required fields before running inference.
    """
    missing = [f for f in config.ALL_FEATURES if f not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
