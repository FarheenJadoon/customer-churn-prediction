import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("Customer Churn Predictor")
st.write("Fill in customer details to predict churn risk.")

# --- Account Info ---
st.header("Account Info")
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", [
    "Bank transfer (automatic)", "Credit card (automatic)",
    "Electronic check", "Mailed check"
])

# --- Personal Info ---
st.header("Personal Info")
gender = st.selectbox("Gender", ["Female", "Male"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])

# --- Services ---
st.header("Services")
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

# --- Build payload matching main.py's expected field names ---
if st.button("Predict Churn"):
    payload = {
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        "MonthlyCharges": monthly_charges,
        "gender_Male": gender == "Male",
        "Partner_Yes": partner == "Yes",
        "Dependents_Yes": dependents == "Yes",
        "PhoneService_Yes": phone_service == "Yes",
        "MultipleLines_No phone service": multiple_lines == "No phone service",
        "MultipleLines_Yes": multiple_lines == "Yes",
        "InternetService_Fiber optic": internet_service == "Fiber optic",
        "InternetService_No": internet_service == "No",
        "OnlineSecurity_No internet service": online_security == "No internet service",
        "OnlineSecurity_Yes": online_security == "Yes",
        "OnlineBackup_No internet service": online_backup == "No internet service",
        "OnlineBackup_Yes": online_backup == "Yes",
        "DeviceProtection_No internet service": device_protection == "No internet service",
        "DeviceProtection_Yes": device_protection == "Yes",
        "TechSupport_No internet service": tech_support == "No internet service",
        "TechSupport_Yes": tech_support == "Yes",
        "StreamingTV_No internet service": streaming_tv == "No internet service",
        "StreamingTV_Yes": streaming_tv == "Yes",
        "StreamingMovies_No internet service": streaming_movies == "No internet service",
        "StreamingMovies_Yes": streaming_movies == "Yes",
        "PaperlessBilling_Yes": paperless == "Yes",
        "PaymentMethod_Credit card (automatic)": payment_method == "Credit card (automatic)",
        "PaymentMethod_Electronic check": payment_method == "Electronic check",
        "PaymentMethod_Mailed check": payment_method == "Mailed check",
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        prob = result["churn_probability"] * 100
        st.metric("Churn Probability", f"{prob:.1f}%")
        if result["churn_prediction"]:
            st.error(" High risk of churn")
        else:
            st.success(" Low risk of churn")
    else:
        st.error(f"API error: {response.text}")