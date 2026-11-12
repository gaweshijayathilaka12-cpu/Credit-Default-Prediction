

import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳"
)

st.title("💳 Credit Card Default Prediction")
st.write("Predict whether a customer may default on their next credit card payment.")

# Load trained model and feature names
model = joblib.load("credit_default_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.subheader("Customer Information")

limit_bal = st.number_input(
    "Credit Limit", min_value=0.0, value=50000.0
)

sex = st.selectbox("Sex", ["Male", "Female"])

education = st.selectbox(
    "Education",
    ["Graduate School", "University", "High School", "Other"]
)

marriage = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Other"]
)

age = st.number_input(
    "Age", min_value=18, max_value=100, value=30
)

# Convert categories into dataset codes
sex_code = {"Male": 1, "Female": 2}[sex]

education_code = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Other": 4
}[education]

marriage_code = {
    "Married": 1,
    "Single": 2,
    "Other": 3
}[marriage]

# Payment history
st.subheader("Payment History")

st.caption(
    "Repayment status: -2 = no consumption, "
    "-1 = paid duly, 0 = revolving credit, "
    "1–8 = months delayed."
)

payment_status = {}

for col in [
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]:
    payment_status[col] = st.selectbox(
        f"{col} Repayment Status",
        list(range(-2, 9)),
        index=2
    )

# Bill amounts
st.subheader("Previous Six Months' Bills")

bill_amounts = {}

for i in range(1, 7):
    bill_amounts[f"BILL_AMT{i}"] = st.number_input(
        f"Bill Amount - Month {i}",
        value=0.0
    )

# Payment amounts
st.subheader("Previous Six Months' Payments")

payment_amounts = {}

for i in range(1, 7):
    payment_amounts[f"PAY_AMT{i}"] = st.number_input(
        f"Payment Amount - Month {i}",
        min_value=0.0,
        value=0.0
    )

# Prediction button
if st.button("Predict Credit Default"):

    customer = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex_code,
        "EDUCATION": education_code,
        "MARRIAGE": marriage_code,
        "AGE": age
    }

    customer.update(payment_status)
    customer.update(bill_amounts)
    customer.update(payment_amounts)

    # Feature Engineering
    customer["TOTAL_BILL"] = sum(bill_amounts.values())
    customer["TOTAL_PAYMENT"] = sum(payment_amounts.values())

    # Prepare input
    input_df = pd.DataFrame([customer])
    input_df = input_df[feature_columns]

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("Predicted: Payment Default")
    else:
        st.success("Predicted: No Payment Default")

    st.metric(
        "Estimated Default Probability",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "Academic demonstration only. "
        "Not intended for real credit decisions."
    )
