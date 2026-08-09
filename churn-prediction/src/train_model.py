"""
Trains and compares multiple churn-prediction models (Logistic Regression,
Random Forest, XGBoost), selects the best by ROC-AUC, and persists the
winning model + preprocessor + evaluation metrics + feature importances.

Usage:
    python -m src.train_model
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src import config
from src.data_processing import load_raw_data, clean_data, build_preprocessor, get_feature_and_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_candidate_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=config.RANDOM_SEED, n_jobs=-1
        ),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=config.RANDOM_SEED,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        logger.warning("xgboost not installed; skipping XGBoost candidate model.")
    return models


def evaluate_model(model, X_test_transformed, y_test) -> dict:
    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist()[:50], "tpr": tpr.tolist()[:50]},
    }


def get_feature_importance(model, preprocessor, model_name: str) -> list:
    """Extract feature importance / coefficients, mapped back to readable names."""
    try:
        cat_names = list(
            preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(config.CATEGORICAL_FEATURES)
        )
    except Exception:
        cat_names = config.CATEGORICAL_FEATURES
    feature_names = cat_names + config.NUMERIC_FEATURES

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top_n = pairs[:15]
    total = sum(v for _, v in pairs) or 1.0
    return [{"feature": name, "importance": round(float(val) / float(total), 4)} for name, val in top_n]


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and cleaning data...")
    raw_df = load_raw_data()
    df = clean_data(raw_df)
    X, y = get_feature_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    logger.info("Fitting preprocessor...")
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    candidates = get_candidate_models()
    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        logger.info("Training model: %s", name)
        model.fit(X_train_t, y_train)
        metrics = evaluate_model(model, X_test_t, y_test)
        results[name] = metrics
        fitted_models[name] = model
        logger.info("%s -> ROC-AUC: %.4f | F1: %.4f", name, metrics["roc_auc"], metrics["f1_score"])

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = fitted_models[best_name]
    logger.info("Best model selected: %s (ROC-AUC=%.4f)", best_name, results[best_name]["roc_auc"])

    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    metrics_output = {
        "best_model": best_name,
        "all_models": results,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=2)

    feature_importance = get_feature_importance(best_model, preprocessor, best_name)
    with open(config.FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(feature_importance, f, indent=2)

    logger.info("Saved model to %s", config.MODEL_PATH)
    logger.info("Saved preprocessor to %s", config.PREPROCESSOR_PATH)
    logger.info("Saved metrics to %s", config.METRICS_PATH)
    logger.info("Saved feature importance to %s", config.FEATURE_IMPORTANCE_PATH)


if __name__ == "__main__":
    main()
