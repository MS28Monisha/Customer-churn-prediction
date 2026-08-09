# Dockerfile for the Customer Churn Prediction API (FastAPI backend)
FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost / scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Train the model at build time if no artifact is already present, then serve the API.
CMD sh -c "if [ ! -f models/churn_model.pkl ]; then python -m src.train_model; fi && \
    uvicorn api.main:app --host 0.0.0.0 --port 8000"
