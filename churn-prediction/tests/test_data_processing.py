"""Unit tests for src/data_processing.py"""

import pandas as pd
import pytest

from src.data_processing import clean_data, build_preprocessor, get_feature_and_target, validate_input_row
from src import config


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "customerID": ["A1", "A2", "A3"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [1, 24, 60],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service"],
            "StreamingTV": ["Yes", "No", "No internet service"],
            "StreamingMovies": ["Yes", "No", "No internet service"],
            "Contract": ["Month-to-month", "Two year", "One year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
            "MonthlyCharges": [80.5, 45.0, 20.0],
            "TotalCharges": [" ", "1080.0", "1200.0"],  # blank simulates real Telco data quirk
            "Churn": ["Yes", "No", "No"],
        }
    )


def test_clean_data_handles_blank_total_charges(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned["TotalCharges"].isnull().sum() == 0
    assert cleaned.loc[cleaned["customerID"] == "A1", "TotalCharges"].iloc[0] == 0


def test_clean_data_encodes_target_as_binary(sample_df):
    cleaned = clean_data(sample_df)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    assert cleaned.loc[cleaned["customerID"] == "A1", "Churn"].iloc[0] == 1
    assert cleaned.loc[cleaned["customerID"] == "A2", "Churn"].iloc[0] == 0


def test_clean_data_drops_duplicates(sample_df):
    dupe_df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(dupe_df)
    assert len(cleaned) == len(sample_df)


def test_get_feature_and_target_splits_correctly(sample_df):
    cleaned = clean_data(sample_df)
    X, y = get_feature_and_target(cleaned)
    assert list(X.columns) == config.ALL_FEATURES
    assert y.tolist() == [1, 0, 0]


def test_get_feature_and_target_missing_column_raises():
    bad_df = pd.DataFrame({"gender": ["Female"]})
    with pytest.raises(ValueError):
        get_feature_and_target(bad_df)


def test_build_preprocessor_fits_and_transforms(sample_df):
    cleaned = clean_data(sample_df)
    X, y = get_feature_and_target(cleaned)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(X)


def test_validate_input_row_passes_with_all_fields():
    record = {f: "x" for f in config.ALL_FEATURES}
    validate_input_row(record)  # should not raise


def test_validate_input_row_raises_on_missing_field():
    record = {f: "x" for f in config.ALL_FEATURES if f != "tenure"}
    with pytest.raises(ValueError):
        validate_input_row(record)
