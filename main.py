from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# model load
model = joblib.load("hospital_recommendation_model.pkl")

class HospitalInput(BaseModel):
    icu_beds: int
    ventilators: int
    doctors_total: int
    distance_km: float
    ambulance_count: int
    total_beds: int
    hospital_type: int
    specialization_available: int
    emergency_services: int
    icu_ratio: float
    ventilator_ratio: float

@app.get("/")
def home():
    return {"message": "MediConnect AI API Running"}

@app.post("/predict")
def predict(data: HospitalInput):

    input_dict = data.dict()

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return {
        "recommended": int(prediction),
        "confidence": float(probability)
    }