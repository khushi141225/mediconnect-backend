from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODELS & ENCODERS
# =========================

hospital_model = joblib.load("models/hospital_recommendation_model.pkl")

age_group_encoder = joblib.load("models/Age_Group_encoder.pkl")
bmi_category_encoder = joblib.load("models/BMI_Category_encoder.pkl")
city_encoder = joblib.load("models/City_encoder.pkl")
distance_category_encoder = joblib.load("models/Distance_Category_encoder.pkl")
hospitalization_burden_encoder = joblib.load("models/Hospitalization_Burden_encoder.pkl")

patient_model = joblib.load("models/patient_risk_best_model.pkl")
gender_encoder = joblib.load("models/Gender_encoder.pkl")
symptom_category_encoder = joblib.load("models/Symptom_Category_encoder.pkl")
patient_scaler = joblib.load("models/Patient_risk_scaler.pkl")
risk_level_encoder = joblib.load("models/Patient_Risk_Level_encoder.pkl")

ambulance_model = joblib.load("models/ambulance_dispatch_model.pkl")
dispatch_priority_encoder = joblib.load("models/Dispatch_Priority_encoder.pkl")
emergency_level_encoder = joblib.load("models/Emergency_Level_encoder.pkl")
equipment_level_encoder = joblib.load("models/Equipment_Level_encoder.pkl")
road_type_encoder = joblib.load("models/Road_Type_encoder.pkl")
traffic_level_encoder = joblib.load("models/Traffic_Level_encoder.pkl")
weather_condition_encoder = joblib.load("models/Weather_Condition_encoder.pkl")
zone_encoder = joblib.load("models/Zone_encoder.pkl")

# =========================
# SCHEMAS
# =========================

class HospitalInput(BaseModel):
    age_group: str
    bmi_category: str
    city: str
    distance_category: str
    hospitalization_burden: str


class PatientRiskInput(BaseModel):
    age: int
    gender: Literal["Male","Female"]
    bmi: float
    blood_pressure: float
    heart_rate: float
    oxygen_level: float
    body_temperature: float
    respiratory_rate: float
    diabetes: int
    hypertension: int
    heart_disease: int
    smoking: int
    alcohol_use: int
    previous_hospitalizations: int
    symptoms_severity: Literal["Mild","Moderate","Severe"]


class AmbulanceDispatchInput(BaseModel):
    emergency_level: str
    equipment_level: str
    gender: str
    road_type: str
    symptom_category: str
    traffic_level: str
    weather_condition: str
    zone: str


# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "Mediconnect Backend Running"}


# =========================
# HOSPITAL PREDICTION
# =========================

@app.post("/predict-hospital")
def predict_hospital(data: HospitalInput):

    input_data = np.array([[
        age_group_encoder.transform([data.age_group])[0],
        bmi_category_encoder.transform([data.bmi_category])[0],
        city_encoder.transform([data.city])[0],
        distance_category_encoder.transform([data.distance_category])[0],
        hospitalization_burden_encoder.transform([data.hospitalization_burden])[0]
    ]])

    prediction = hospital_model.predict(input_data)

    return {"recommended_hospital": int(prediction[0])}


# =========================
# PATIENT RISK
# =========================

@app.post("/predict-patient-risk")
def predict_patient_risk(data: PatientRiskInput):

    input_df = pd.DataFrame([{
        "age": data.age,
        "gender": gender_encoder.transform([data.gender])[0],
        "bmi": data.bmi,
        "blood_pressure": data.blood_pressure,
        "heart_rate": data.heart_rate,
        "oxygen_level": data.oxygen_level,
        "body_temperature": data.body_temperature,
        "respiratory_rate": data.respiratory_rate,
        "diabetes": data.diabetes,
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "smoking": data.smoking,
        "alcohol_use": data.alcohol_use,
        "previous_hospitalizations": data.previous_hospitalizations,
        "symptoms_severity": symptom_category_encoder.transform([data.symptoms_severity])[0],
    }])

    input_scaled = patient_scaler.transform(input_df)

    prediction = patient_model.predict(input_scaled)
    risk_label = risk_level_encoder.inverse_transform(prediction)[0]

    return {"risk_level": risk_label}


# =========================
# AMBULANCE DISPATCH
# =========================

@app.post("/predict-ambulance-dispatch")
def predict_ambulance_dispatch(data: AmbulanceDispatchInput):

    input_df = pd.DataFrame([{
        "Emergency_Level": emergency_level_encoder.transform([data.emergency_level])[0],
        "Equipment_Level": equipment_level_encoder.transform([data.equipment_level])[0],
        "Gender": gender_encoder.transform([data.gender])[0],
        "Road_Type": road_type_encoder.transform([data.road_type])[0],
        "Symptom_Category": symptom_category_encoder.transform([data.symptom_category])[0],
        "Traffic_Level": traffic_level_encoder.transform([data.traffic_level])[0],
        "Weather_Condition": weather_condition_encoder.transform([data.weather_condition])[0],
        "Zone": zone_encoder.transform([data.zone])[0]
    }])

    prediction = ambulance_model.predict(input_df)
    dispatch_label = dispatch_priority_encoder.inverse_transform(prediction)[0]

    return {"dispatch_priority": dispatch_label}