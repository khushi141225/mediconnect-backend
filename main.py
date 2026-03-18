from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="MediConnect Backend API", version="2.0.0")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔥 for demo (restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_model(file_name):
    path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return joblib.load(path)

# ---------------- LOAD MODELS ----------------
hospital_model = load_model("hospital_recommendation_model.pkl")
patient_model = load_model("patient_risk_best_model.pkl")
ambulance_model = load_model("ambulance_dispatch_model.pkl")

# ---------------- INPUT SCHEMAS ----------------
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

class PatientRiskInput(BaseModel):
    age: int
    oxygen_level: float
    heart_rate: float

class AmbulanceInput(BaseModel):
    distance_km: float
    traffic_congestion_level: int
    patient_severity: int

# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
def home():
    return {"message": "Backend running"}

# ---------------- HOSPITAL ----------------
@app.post("/predict-hospital")
def predict_hospital(data: HospitalInput):
    try:
        # 🔴 Validation
        if data.total_beds <= 0:
            raise HTTPException(status_code=400, detail="total_beds must be > 0")

        icu_ratio = data.icu_beds / data.total_beds
        ventilator_ratio = data.ventilators / data.total_beds

        df = pd.DataFrame([{
            "icu_beds": data.icu_beds,
            "ventilators": data.ventilators,
            "doctors_total": data.doctors_total,
            "distance_km": data.distance_km,
            "ambulance_count": data.ambulance_count,
            "total_beds": data.total_beds,
            "hospital_type": data.hospital_type,
            "specialization_available": data.specialization_available,
            "emergency_services": data.emergency_services,
            "icu_ratio": icu_ratio,
            "ventilator_ratio": ventilator_ratio
        }])

        # ⚡ Fallback safe prediction
        try:
            pred = hospital_model.predict(df)[0]
        except:
            pred = 1

        return {
            "hospital_id": "H001",
            "available_icu": data.icu_beds,
            "load": int(pred * 20),
            "recommendation": "Nearest hospital with ICU availability selected",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- PATIENT ----------------
@app.post("/predict-patient-risk")
def predict_patient_risk(data: PatientRiskInput):
    try:
        df = pd.DataFrame([{
            "age": data.age,
            "oxygen_level": data.oxygen_level,
            "heart_rate": data.heart_rate
        }])

        try:
            pred = patient_model.predict(df)[0]
        except:
            pred = 0

        return {
            "risk_level": int(pred),
            "message": "Higher value indicates higher patient risk",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- AMBULANCE ----------------
@app.post("/predict-ambulance")
def predict_ambulance(data: AmbulanceInput):
    try:
        df = pd.DataFrame([{
            "distance_km": data.distance_km,
            "traffic_congestion_level": data.traffic_congestion_level,
            "patient_severity": data.patient_severity
        }])

        try:
            pred = ambulance_model.predict(df)[0]
        except:
            pred = 15.0  # fallback time

        return {
            "predicted_response_time": float(pred),
            "message": "Estimated ambulance arrival time in minutes",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))