import streamlit as st
import requests

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

API_URL = "https://fraud-detection-41y3.onrender.com/predict"

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("💳 Fraud Detection")

    st.markdown("""
### About
This application predicts whether a **credit card transaction** is:

- ✅ Legitimate
- 🚨 Fraudulent

using a Machine Learning model.

### Tech Stack
- Python
- FastAPI
- Streamlit
- Scikit-learn
""")

    st.markdown("---")

    st.link_button(
        "📄 API Documentation",
        "https://fraud-detection-41y3.onrender.com/docs"
    )

# ----------------------------
# Main Page
# ----------------------------
st.title("💳 Credit Card Fraud Detection")

st.write(
    "Fill in the transaction details below and click **Predict**."
)

# ----------------------------
# Inputs
# ----------------------------
left, right = st.columns(2)

with left:
    Time = st.number_input("Time", value=0.0)
    Amount = st.number_input("Amount", value=0.0)

inputs = {}

for i in range(1, 15):
    with left:
        inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

for i in range(15, 29):
    with right:
        inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🔍 Predict", use_container_width=True):

    payload = {
        "Time": Time,
        **inputs,
        "Amount": Amount
    }

    try:

        with st.spinner("Analyzing transaction..."):

            response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            probability = result["fraud_probability"]

            st.success("Prediction completed!")

            if prediction == 1:
                st.error("🚨 Fraudulent Transaction Detected")
                st.warning(
                    "This transaction has a high probability of fraud."
                )
            else:
                st.success("✅ Legitimate Transaction")
                st.info(
                    "No fraud indicators were detected."
                )

            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

            st.progress(float(min(probability, 1.0)))

            with st.expander("Raw API Response"):
                st.json(result)

        else:
            st.error(f"API Error ({response.status_code})")
            st.code(response.text)

    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error:\n{e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "Built with ❤️ using Streamlit, FastAPI and Scikit-learn"
)