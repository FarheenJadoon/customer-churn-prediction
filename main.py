from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json

app = FastAPI(title="Churn Prediction API")

model = joblib.load("churn_model.pkl")
with open("model_columns.json") as f:
    model_columns = json.load(f)

class CustomerInput(BaseModel):
    SeniorCitizen: int
    tenure: int
    Contract: int
    MonthlyCharges: float
    gender_Male: bool
    Partner_Yes: bool
    Dependents_Yes: bool
    PhoneService_Yes: bool
    MultipleLines_No_phone_service: bool = Field(alias="MultipleLines_No phone service")
    MultipleLines_Yes: bool
    InternetService_Fiber_optic: bool = Field(alias="InternetService_Fiber optic")
    InternetService_No: bool
    OnlineSecurity_No_internet_service: bool = Field(alias="OnlineSecurity_No internet service")
    OnlineSecurity_Yes: bool
    OnlineBackup_No_internet_service: bool = Field(alias="OnlineBackup_No internet service")
    OnlineBackup_Yes: bool
    DeviceProtection_No_internet_service: bool = Field(alias="DeviceProtection_No internet service")
    DeviceProtection_Yes: bool
    TechSupport_No_internet_service: bool = Field(alias="TechSupport_No internet service")
    TechSupport_Yes: bool
    StreamingTV_No_internet_service: bool = Field(alias="StreamingTV_No internet service")
    StreamingTV_Yes: bool
    StreamingMovies_No_internet_service: bool = Field(alias="StreamingMovies_No internet service")
    StreamingMovies_Yes: bool
    PaperlessBilling_Yes: bool
    PaymentMethod_Credit_card: bool = Field(alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: bool = Field(alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: bool = Field(alias="PaymentMethod_Mailed check")

    class Config:
        populate_by_name = True

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: CustomerInput):
    input_dict = customer.dict(by_alias=True)
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]   # enforce correct column order

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": bool(prediction),
        "churn_probability": round(float(probability), 4)
    }