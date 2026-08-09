"""
Streamlit dashboard for the Customer Churn Prediction System.

Talks to the FastAPI backend over HTTP, so start the API first:
    uvicorn api.main:app --reload --port 8000

Then run this app:
    streamlit run frontend/streamlit_app.py
"""

import os
import io
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

API_URL = os.environ.get("CHURN_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def predict_single(payload: dict):
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def predict_batch(file_bytes: bytes, filename: str):
    files = {"file": (filename, file_bytes, "text/csv")}
    r = requests.post(f"{API_URL}/predict/batch", files=files, timeout=60)
    r.raise_for_status()
    return r.json()


def get_metrics():
    r = requests.get(f"{API_URL}/model/metrics", timeout=10)
    r.raise_for_status()
    return r.json()


def get_feature_importance():
    r = requests.get(f"{API_URL}/model/feature-importance", timeout=10)
    r.raise_for_status()
    return r.json()


st.title("📉 Customer Churn Prediction Dashboard")

health = check_api_health()
if health is None:
    st.error(
        f"Cannot reach the API at {API_URL}. Start it with "
        "`uvicorn api.main:app --reload --port 8000` and refresh this page."
    )
    st.stop()
elif not health.get("model_loaded"):
    st.warning("API is up, but no trained model was found. Run `python -m src.train_model` first.")

tab_predict, tab_batch, tab_analytics, tab_performance = st.tabs(
    ["🔮 Single Prediction", "📁 Batch Upload", "📊 Analytics", "🧠 Model Performance"]
)

# ---------------------------------------------------------------- Tab 1
with tab_predict:
    st.subheader("Predict churn risk for one customer")
    with st.form("single_predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            customer_id = st.text_input("Customer ID (optional)", value="CUST-00001")
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"])
            dependents = st.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with col2:
            phone_service = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["PhoneService"])
            multiple_lines = st.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["MultipleLines"])
            internet_service = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["InternetService"])
            online_security = st.selectbox("Online Security", CATEGORICAL_OPTIONS["OnlineSecurity"])
            online_backup = st.selectbox("Online Backup", CATEGORICAL_OPTIONS["OnlineBackup"])
            device_protection = st.selectbox("Device Protection", CATEGORICAL_OPTIONS["DeviceProtection"])
        with col3:
            tech_support = st.selectbox("Tech Support", CATEGORICAL_OPTIONS["TechSupport"])
            streaming_tv = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])
            streaming_movies = st.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"])
            contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
            paperless = st.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["PaperlessBilling"])
            payment_method = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["PaymentMethod"])

        col4, col5 = st.columns(2)
        with col4:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
        with col5:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)

        submitted = st.form_submit_button("Predict Churn Risk", type="primary")

    if submitted:
        payload = {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
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
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        try:
            result = predict_single(payload)
            prob = result["churn_probability"]
            risk = result["risk_level"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Churn Probability", f"{prob:.1%}")
            c2.metric("Prediction", "Will Churn" if result["churn_prediction"] else "Will Stay")
            c3.markdown(
                f"<h3 style='color:{RISK_COLORS[risk]}'>Risk: {risk}</h3>", unsafe_allow_html=True
            )
            st.progress(min(prob, 1.0))
        except requests.HTTPError as e:
            st.error(f"Prediction failed: {e.response.text}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------- Tab 2
with tab_batch:
    st.subheader("Upload a CSV of customers for bulk prediction")
    st.caption(
        "CSV must include all required columns (see the sample dataset in "
        "`data/raw/telco_churn.csv` for the expected schema)."
    )
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is not None:
        if st.button("Run Batch Prediction", type="primary"):
            try:
                result = predict_batch(uploaded.getvalue(), uploaded.name)
                df_result = pd.DataFrame(result["predictions"])
                st.success(f"Scored {result['count']} customers.")

                risk_filter = st.multiselect(
                    "Filter by risk level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"]
                )
                filtered = df_result[df_result["risk_level"].isin(risk_filter)]
                filtered = filtered.sort_values("churn_probability", ascending=False)
                st.dataframe(filtered, use_container_width=True)

                csv_bytes = filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_bytes,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )

                fig = px.histogram(
                    df_result, x="risk_level", color="risk_level",
                    color_discrete_map=RISK_COLORS, title="Customer Risk Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            except requests.HTTPError as e:
                st.error(f"Batch prediction failed: {e.response.text}")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# ---------------------------------------------------------------- Tab 3
with tab_analytics:
    st.subheader("Churn analytics (from local sample dataset)")
    st.caption(
        "This tab visualizes the training dataset directly (not API-served) "
        "so it works even before you upload new data."
    )
    data_path = "data/raw/telco_churn.csv"
    try:
        df = pd.read_csv(data_path)
        colA, colB = st.columns(2)
        with colA:
            fig1 = px.histogram(
                df, x="Contract", color="Churn", barmode="group", title="Churn by Contract Type"
            )
            st.plotly_chart(fig1, use_container_width=True)
        with colB:
            fig2 = px.box(df, x="Churn", y="MonthlyCharges", color="Churn", title="Monthly Charges vs Churn")
            st.plotly_chart(fig2, use_container_width=True)

        colC, colD = st.columns(2)
        with colC:
            fig3 = px.histogram(df, x="tenure", color="Churn", nbins=30, title="Tenure Distribution by Churn")
            st.plotly_chart(fig3, use_container_width=True)
        with colD:
            churn_by_internet = (
                df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean()).reset_index()
            )
            churn_by_internet.columns = ["InternetService", "ChurnRate"]
            fig4 = px.bar(
                churn_by_internet, x="InternetService", y="ChurnRate",
                title="Churn Rate by Internet Service"
            )
            st.plotly_chart(fig4, use_container_width=True)
    except FileNotFoundError:
        st.info("No local dataset found. Run `python src/generate_sample_data.py` to create one.")

# ---------------------------------------------------------------- Tab 4
with tab_performance:
    st.subheader("Model performance")
    try:
        metrics = get_metrics()
        best = metrics["best_model"]
        st.markdown(f"**Best model selected:** `{best}`")

        rows = []
        for name, m in metrics["all_models"].items():
            rows.append(
                {
                    "model": name,
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1_score": m["f1_score"],
                    "roc_auc": m["roc_auc"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        best_metrics = metrics["all_models"][best]
        col1, col2 = st.columns(2)
        with col1:
            cm = best_metrics["confusion_matrix"]
            fig_cm = px.imshow(
                cm, text_auto=True, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                title=f"Confusion Matrix — {best}", color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        with col2:
            roc = best_metrics["roc_curve"]
            fig_roc = px.line(x=roc["fpr"], y=roc["tpr"], title=f"ROC Curve — {best}",
                               labels={"x": "False Positive Rate", "y": "True Positive Rate"})
            fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)

        st.subheader("Top Churn Drivers")
        fi = get_feature_importance()
        fi_df = pd.DataFrame(fi["features"])
        fig_fi = px.bar(
            fi_df.sort_values("importance"), x="importance", y="feature", orientation="h",
            title=f"Feature Importance — {fi['best_model']}"
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    except requests.HTTPError:
        st.warning("Model metrics not available yet. Run `python -m src.train_model` first.")
    except Exception as e:
        st.error(f"Could not load model performance: {e}")
