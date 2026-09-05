Project Description

The Customer Churn Prediction System helps businesses identify customers
who are likely to cancel their subscription or service. It trains and
compares multiple ML models (Logistic Regression, Random Forest,
XGBoost), automatically selects the best-performing one, and serves
predictions through a REST API and an interactive dashboard.

Features

-**Single prediction** — enter one customer's details, get churn
  probability + risk level (Low/Medium/High)
 **Batch prediction** — upload a CSV, score hundreds of customers
  at once, download results
 **Analytics dashboard** — churn rate by contract type, tenure,
  internet service, monthly charges
 **Model performance** — accuracy, precision, recall, F1, ROC-AUC,
  confusion matrix, ROC curve
 **Feature importance** — see the top drivers behind churn
 **Dockerized** — run the whole stack with one command
 **Tested** — pytest suite for data pipeline + API endpoints



Architecture


┌──────────────────┐      HTTP       ┌──────────────────┐      loads      ┌───────────────────┐
│ Streamlit         │ ─────────────▶ │ FastAPI          │ ──────────────▶ │ Trained model      │
│ Frontend          │ ◀───────────── │ Backend (API)    │ ◀────────────── │ (.pkl) + metrics   │
│ (dashboard)       │     JSON        │ /predict         │                 │ produced by        │
└──────────────────┘                 │ /predict/batch   │                 │ src/train_model.py │
                                      │ /model/metrics   │                 └───────────────────┘
                                      │ /model/feature-  │
                                      │  importance      │
                                      └──────────────────┘




Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| ML | scikit-learn, XGBoost, pandas, NumPy, joblib |
| Frontend | Streamlit, Plotly |
| Testing | pytest, httpx |
| Deployment | Docker, Docker Compose |


Folder Structure


churn-prediction/
├── data/
│   ├── raw/                     # Input CSVs (telco_churn.csv)
│   └── processed/                # Reserved for cleaned/exported datasets
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
└── README.md



Setup & Installation

1. Clone / open the project
```bash
cd churn-prediction
code .
```

2. Create & activate a virtual environment
```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Generate sample data (skip if using real Kaggle Telco dataset)
```bash
python src/generate_sample_data.py
```

5. Train the model
```bash
python -m src.train_model
```

6. Run the API backend
```bash
uvicorn api.main:app --reload --port 8000
```
Swagger docs: `http://localhost:8000/docs`

7. Run the frontend dashboard (second terminal)
```bash
streamlit run frontend/streamlit_app.py
```
Dashboard: `http://localhost:8501`



Possible Future Extensions

-  Add authentication (JWT) to protect prediction endpoints
-  Add PostgreSQL to store prediction history & customer records
-  Add SHAP-based explainability per individual prediction
-  Email/Slack alerts for high-risk customers
-  Automated retraining pipeline (scheduled or trigger-based)
-  Deploy to cloud (Render, Railway, AWS/GCP) with CI/CD
-  Replace Streamlit with a React frontend for a production-grade UI
-  Multi-tenant support for multiple businesses/datasets
-  Mobile-friendly dashboard view



