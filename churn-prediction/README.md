# Customer Churn Prediction System — Full Stack ML Web Application

Predicts whether a customer is likely to churn, using a trained ML model served
via a FastAPI backend and visualized through an interactive Streamlit dashboard.

## Architecture

```
┌─────────────────┐      HTTP       ┌──────────────────┐      loads      ┌───────────────────┐
│ Streamlit        │ ─────────────▶ │ FastAPI          │ ──────────────▶ │ Trained model      │
│ Frontend         │ ◀───────────── │ Backend (API)    │ ◀────────────── │ (.pkl) + metrics   │
│ (dashboard)      │     JSON        │ /predict         │                 │ produced by        │
└─────────────────┘                 │ /predict/batch   │                 │ src/train_model.py │
                                     │ /model/metrics   │                 └───────────────────┘
                                     │ /model/feature-  │
                                     │  importance      │
                                     └──────────────────┘
```

- **`src/`** — data cleaning, preprocessing pipeline, model training, and prediction logic (framework-agnostic, reused by both the API and any scripts/notebooks).
- **`api/`** — FastAPI backend exposing REST endpoints with auto-generated Swagger docs.
- **`frontend/`** — Streamlit dashboard for uploads, single predictions, analytics, and model performance.
- **`models/`** — persisted trained artifacts (`churn_model.pkl`, `preprocessor.pkl`, metrics, feature importance).
- **`tests/`** — pytest unit + integration tests.
- **`notebooks/`** — EDA and modeling exploration notebook.

## Dataset

This project ships with a **synthetic, Telco-style dataset generator**
(`src/generate_sample_data.py`) so it runs out of the box with no external
downloads. The schema matches the real Kaggle **"Telco Customer Churn"**
dataset column-for-column, so you can drop the real CSV into
`data/raw/telco_churn.csv` and everything downstream works unchanged.

| Column | Description |
|---|---|
| customerID | Unique customer identifier |
| gender, SeniorCitizen, Partner, Dependents | Demographics |
| tenure | Months as a customer |
| PhoneService, MultipleLines | Phone plan details |
| InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Internet add-ons |
| Contract, PaperlessBilling, PaymentMethod | Billing details |
| MonthlyCharges, TotalCharges | Charges |
| Churn | Target: Yes/No |

## Setup (VS Code / local machine)

### 1. Clone / open the project folder in VS Code

```bash
cd churn-prediction
code .
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate it:
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If `xgboost` fails to install on your machine (e.g. missing compiler), the
> training script automatically skips it and compares only Logistic
> Regression and Random Forest — no code changes needed.

### 4. Generate sample data (skip this if you're using the real Kaggle CSV)

```bash
python src/generate_sample_data.py
```

This creates `data/raw/telco_churn.csv`.

### 5. Train the model

```bash
python -m src.train_model
```

This trains Logistic Regression, Random Forest, and XGBoost (if available),
picks the best by ROC-AUC, and saves to `models/`:
- `churn_model.pkl` — the winning model
- `preprocessor.pkl` — the fitted preprocessing pipeline
- `metrics.json` — evaluation metrics for all candidates
- `feature_importance.json` — top churn drivers

### 6. Run the API backend

```bash
uvicorn api.main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 7. Run the frontend dashboard (in a second terminal)

```bash
streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501 in your browser.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check + whether a model is loaded |
| POST | `/predict` | Predict churn for a single customer (JSON body) |
| POST | `/predict/batch` | Predict churn for a CSV upload of customers |
| GET | `/model/metrics` | Evaluation metrics for all trained candidate models |
| GET | `/model/feature-importance` | Top churn drivers for the best model |

Example single-prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "CUST-9001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 286.5
  }'
```

## Running Tests

```bash
pytest tests/ -v
```

Note: some tests are automatically skipped until you've run
`python -m src.train_model` at least once (they need real model artifacts).

## Running with Docker

Builds and runs both the API and the Streamlit dashboard together:

```bash
docker-compose up --build
```

- API: http://localhost:8000
- Dashboard: http://localhost:8501

> The API container automatically trains a model on first startup if
> `models/churn_model.pkl` doesn't already exist.

## Retraining with new data

1. Replace `data/raw/telco_churn.csv` with your own data (same column schema).
2. Re-run `python -m src.train_model`.
3. Restart the API (`uvicorn` picks up the new `.pkl` files on restart, or
   automatically with `--reload` during development).

## Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                     # Input CSVs (telco_churn.csv)
│   └── processed/                # Reserved for any exported/cleaned datasets
├── notebooks/
│   └── eda_and_modeling.ipynb    # EDA + inline model comparison
├── src/
│   ├── config.py                 # Paths, schema, constants
│   ├── generate_sample_data.py   # Synthetic dataset generator
│   ├── data_processing.py        # Cleaning + preprocessing pipeline
│   ├── train_model.py            # Trains/evaluates/persists best model
│   └── predict.py                # Loads model, runs inference
├── models/
│   ├── churn_model.pkl
│   ├── preprocessor.pkl
│   ├── metrics.json
│   └── feature_importance.json
├── api/
│   ├── main.py                   # FastAPI app entrypoint
│   ├── schemas.py                # Pydantic request/response models
│   └── routes/
│       ├── predictions.py        # /predict, /predict/batch
│       └── model_info.py         # /model/metrics, /model/feature-importance
├── frontend/
│   ├── streamlit_app.py          # Dashboard (predict, batch, analytics, performance)
│   └── Dockerfile
├── tests/
│   ├── test_data_processing.py
│   └── test_api.py
├── requirements.txt
├── Dockerfile                     # API container
├── docker-compose.yml             # Runs API + frontend together
├── .gitignore
└── README.md
```

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **ML:** scikit-learn, XGBoost (optional), pandas, numpy, joblib
- **Frontend:** Streamlit, Plotly
- **Testing:** pytest, httpx
- **Deployment:** Docker, docker-compose
