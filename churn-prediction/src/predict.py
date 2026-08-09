"""
Loads the persisted model + preprocessor and runs predictions.
Shared by the FastAPI backend and any batch/CLI usage.
"""

import json
import joblib
import pandas as pd

from src import config


class ChurnPredictor:
    """Wraps the trained model + preprocessor for easy reuse."""

    def __init__(self):
        self._model = None
        self._preprocessor = None
        self._metrics = None
        self._feature_importance = None

    def _ensure_loaded(self):
        if self._model is None or self._preprocessor is None:
            if not config.MODEL_PATH.exists() or not config.PREPROCESSOR_PATH.exists():
                raise FileNotFoundError(
                    "Model artifacts not found. Run `python -m src.train_model` first."
                )
            self._model = joblib.load(config.MODEL_PATH)
            self._preprocessor = joblib.load(config.PREPROCESSOR_PATH)

    @staticmethod
    def risk_level(probability: float) -> str:
        if probability < config.RISK_THRESHOLDS["low"]:
            return "Low"
        elif probability < config.RISK_THRESHOLDS["medium"]:
            return "Medium"
        return "High"

    def predict_one(self, record: dict) -> dict:
        self._ensure_loaded()
        df = pd.DataFrame([record])[config.ALL_FEATURES]
        X_t = self._preprocessor.transform(df)
        proba = float(self._model.predict_proba(X_t)[0, 1])
        return {
            "churn_probability": round(proba, 4),
            "churn_prediction": int(proba >= 0.5),
            "risk_level": self.risk_level(proba),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_loaded()
        X = df[config.ALL_FEATURES]
        X_t = self._preprocessor.transform(X)
        probas = self._model.predict_proba(X_t)[:, 1]
        result = df.copy()
        result["churn_probability"] = probas.round(4)
        result["churn_prediction"] = (probas >= 0.5).astype(int)
        result["risk_level"] = [self.risk_level(p) for p in probas]
        return result

    def get_metrics(self) -> dict:
        if not config.METRICS_PATH.exists():
            raise FileNotFoundError("Metrics not found. Run `python -m src.train_model` first.")
        with open(config.METRICS_PATH) as f:
            return json.load(f)

    def get_feature_importance(self) -> list:
        if not config.FEATURE_IMPORTANCE_PATH.exists():
            raise FileNotFoundError(
                "Feature importance not found. Run `python -m src.train_model` first."
            )
        with open(config.FEATURE_IMPORTANCE_PATH) as f:
            return json.load(f)


# Module-level singleton so FastAPI routes share one loaded model in memory.
predictor = ChurnPredictor()
