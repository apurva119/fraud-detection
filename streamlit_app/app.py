import streamlit as st
import requests

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details and click **Predict**.")

API_URL = "https://fraud-detection-41y3.onrender.com/predict"

# ----------------------------
# Inputs
# ----------------------------

col1, col2 = st.columns(2)

with col1:
    Time = st.number_input("Time", value=0.0)
    Amount = st.number_input("Amount", value=0.0)

inputs = {}

for i in range(1, 29):
    with col2 if i > 14 else col1:
        inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

# ----------------------------
# Predict Button
# ----------------------------

if st.button("Predict"):

    payload = {
        "Time": Time,
        **inputs,
        "Amount": Amount
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            probability = result["fraud_probability"]

            st.success("Prediction completed!")

            if prediction == 1:
                st.error("🚨 Fraudulent Transaction")
            else:
                st.success("✅ Legitimate Transaction")

            st.metric(
                label="Fraud Probability",
                value=f"{probability*100:.2f}%"
            )

            st.json(result)

        else:
            st.error(response.text)

    except Exception as e:
        st.error(str(e))