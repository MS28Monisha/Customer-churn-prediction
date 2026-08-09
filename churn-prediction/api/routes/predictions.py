"""Prediction endpoints: single record and batch CSV upload."""

import io
import logging
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

from api.schemas import CustomerRecord, PredictionResponse, BatchPredictionResponse
from src.predict import predictor
from src import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
def predict_single(customer: CustomerRecord):
    """Predict churn probability for a single customer."""
    try:
        record = customer.model_dump()
        result = predictor.predict_one(record)
        return {**result, "customerID": customer.customerID}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Predict churn for a batch of customers uploaded as a CSV file."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        missing = [c for c in config.ALL_FEATURES if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"CSV missing required columns: {missing}")

        result_df = predictor.predict_batch(df)

        predictions = []
        for _, row in result_df.iterrows():
            predictions.append(
                {
                    "customerID": row.get(config.ID_COLUMN, None),
                    "churn_probability": float(row["churn_probability"]),
                    "churn_prediction": int(row["churn_prediction"]),
                    "risk_level": row["risk_level"],
                }
            )
        return {"count": len(predictions), "predictions": predictions}
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")
