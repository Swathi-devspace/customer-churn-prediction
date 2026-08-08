import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    
    layout="wide"
)

# --------------------------------------------------
# Load Saved Model, Scaler and Feature Names
# --------------------------------------------------

with open("models/churn_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("models/feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("Customer Churn Prediction")

st.write(
    "Enter the customer's details below to predict whether "
    "the customer is likely to churn."
)

st.divider()

# --------------------------------------------------
# Customer Information
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# --------------------------------------------------
# Charges
# --------------------------------------------------

col3, col4 = st.columns(2)

with col3:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

with col4:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Churn", use_container_width=True):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # One-hot encode categorical variables
    input_data = pd.get_dummies(input_data)

    # Make sure input has exactly the same columns as training data
    input_data = input_data.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Scale input using the SAME scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Prediction probability
    probability = model.predict_proba(input_scaled)[0]

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.divider()

    if prediction == 1:

        st.error("⚠️ Customer is likely to CHURN.")

        st.write(
            f"Churn Probability: **{probability[1] * 100:.2f}%**"
        )

    else:

        st.success("✅ Customer is likely to STAY.")

        st.write(
            f"Churn Probability: **{probability[1] * 100:.2f}%**"
        )