"""
Generates a synthetic Telco-style customer churn dataset for local development
and testing when the real Kaggle 'Telco Customer Churn' CSV is not available.

Usage:
    python src/generate_sample_data.py

Output:
    data/raw/telco_churn.csv

NOTE: For a real project, download the actual dataset from Kaggle
("Telco Customer Churn" by blastchar) and place it at data/raw/telco_churn.csv
instead of using this synthetic generator. The column schema matches the real
dataset so either source works with the rest of the pipeline unchanged.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42


def generate_synthetic_churn_data(n_rows: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    genders = rng.choice(["Male", "Female"], size=n_rows)
    senior_citizen = rng.choice([0, 1], size=n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_rows)
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7])
    tenure = rng.integers(0, 73, size=n_rows)
    phone_service = rng.choice(["Yes", "No"], size=n_rows, p=[0.9, 0.1])
    multiple_lines = rng.choice(["Yes", "No", "No phone service"], size=n_rows, p=[0.42, 0.48, 0.1])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.35, 0.44, 0.21])
    online_security = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.29, 0.5, 0.21])
    online_backup = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.34, 0.45, 0.21])
    device_protection = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.34, 0.45, 0.21])
    tech_support = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.29, 0.5, 0.21])
    streaming_tv = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.38, 0.41, 0.21])
    streaming_movies = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.39, 0.4, 0.21])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.21, 0.24])
    paperless_billing = rng.choice(["Yes", "No"], size=n_rows, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_rows,
    )
    monthly_charges = np.round(rng.uniform(18.0, 120.0, size=n_rows), 2)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 20, size=n_rows).clip(min=0), 2)

    # Build a churn probability that depends on realistic risk factors,
    # then sample a binary outcome from it (keeps the dataset learnable).
    risk = np.zeros(n_rows)
    risk += (contract == "Month-to-month") * 0.35
    risk += (contract == "One year") * 0.10
    risk += (internet_service == "Fiber optic") * 0.15
    risk += (payment_method == "Electronic check") * 0.15
    risk += (online_security == "No") * 0.10
    risk += (tech_support == "No") * 0.10
    risk += (senior_citizen == 1) * 0.08
    risk += (tenure < 12) * 0.20
    risk += (paperless_billing == "Yes") * 0.05
    risk -= (partner == "Yes") * 0.05
    risk -= (dependents == "Yes") * 0.05
    risk = np.clip(risk + rng.normal(0, 0.08, size=n_rows), 0, 1)

    churn = rng.binomial(1, risk)
    churn_labels = np.where(churn == 1, "Yes", "No")

    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_rows + 1)]

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": genders,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn_labels,
        }
    )
    return df


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "telco_churn.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_churn_data()
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset written to: {out_path}")
    print(f"Rows: {len(df)}, Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
