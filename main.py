from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os
import glob

# xgboost model ko load karne ke liye package installed hona chahiye
# direct import optional hai, but safe hai
try:
    import xgboost  # noqa: F401
except Exception:
    pass


app = FastAPI(
    title="Mediconnect Backend API",
    version="1.0.0",
    description="Hospital Recommendation + Patient Risk + Ambulance Dispatch API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"


# -----------------------------
# Utility functions
# -----------------------------
def find_file_by_prefix(prefix: str) -> str:
    """
    models folder me prefix se matching file dhundta hai.
    """
    pattern = os.path.join(MODELS_DIR, f"{prefix}*")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No file found for prefix: {prefix}")
    return matches[0]


def load_artifact(prefix: str):
    path = find_file_by_prefix(prefix)
    return joblib.load(path)


def get_classes(encoder):
    try:
        return list(map(str, encoder.classes_))
    except Exception:
        return []


def encode_value(encoder, value: str, field_name: str):
    allowed = get_classes(encoder)
    if allowed and value not in allowed:
        raise ValueError(
            f"Invalid value for '{field_name}': '{value}'. Allowed values: {allowed}"
        )
    return encoder.transform([value])[0]


def safe_predict_proba(model, processed_input):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(processed_input)[0]
        return round(float(np.max(probs) * 100), 2)
    return None


# -----------------------------
# Load models / encoders
# -----------------------------
# Hospital
hospital_model = load_artifact("hospital_recommendation")
age_group_encoder = load_artifact("Age_Group_encoder")
bmi_category_encoder = load_artifact("BMI_Category_encoder")
city_encoder = load_artifact("City_encoder")
distance_category_encoder = load_artifact("Distance_Category_encoder")
hospitalization_burden_encoder = load_artifact("Hospitalization_Burden")

# Patient
patient_model = load_artifact("patient_risk_best_model")
gender_encoder = load_artifact("Gender_encoder")
symptom_category_encoder = load_artifact("Symptom_Category_encoder")
patient_risk_scaler = load_artifact("Patient_risk_scaler")

# risk level encoder ke 2 versions tumhare folder me dikh rahe the,
# isliye exact file na mile to fallback use kiya hai
try:
    risk_level_encoder = load_artifact("risk_level_encoder")
except Exception:
    risk_level_encoder = load_artifact("Patient_Risk_Level_encoder")

# Ambulance
ambulance_model = load_artifact("ambulance_dispatch_model")
dispatch_priority_encoder = load_artifact("Dispatch_Priority_encoder")
emergency_level_encoder = load_artifact("Emergency_Level_encoder")
equipment_level_encoder = load_artifact("Equipment_Level_encoder")
road_type_encoder = load_artifact("Road_Type_encoder")
traffic_level_encoder = load_artifact("Traffic_Level_encoder")
weather_condition_encoder = load_artifact("Weather_Condition_encoder")
zone_encoder = load_artifact("Zone_encoder")


# -----------------------------
# Request Schemas
# -----------------------------
class HospitalInput(BaseModel):
    age_group: str
    bmi_category: str
    city: str
    distance_category: str
    hospitalization_burden: str


from typing import Literal

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


# -----------------------------
# Preprocessing functions
# -----------------------------
def preprocess_hospital_input(data: HospitalInput):
    age_group = encode_value(age_group_encoder, data.age_group, "age_group")
    bmi_category = encode_value(bmi_category_encoder, data.bmi_category, "bmi_category")
    city = encode_value(city_encoder, data.city, "city")
    distance_category = encode_value(
        distance_category_encoder, data.distance_category, "distance_category"
    )
    hospitalization_burden = encode_value(
        hospitalization_burden_encoder,
        data.hospitalization_burden,
        "hospitalization_burden"
    )

    # Feature order training ke hisaab se maintain karna hota hai
    processed = np.array([[
        age_group,
        bmi_category,
        city,
        distance_category,
        hospitalization_burden
    ]])

    return processed


def preprocess_patient_input(data: PatientRiskInput):
    gender = encode_value(gender_encoder, data.gender, "gender")
    symptom_category = encode_value(
        symptom_category_encoder,
        data.symptoms_severity,
        "symptoms_severity"
    )

    # Ye order tumhare earlier patient request ke hisaab se set kiya gaya hai
    df = pd.DataFrame([{
        "age": data.age,
        "gender": gender,
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
        "symptoms_severity": symptom_category,
    }])

    processed = patient_risk_scaler.transform(df)
    return processed


def preprocess_ambulance_input(data: AmbulanceDispatchInput):
    emergency_level = encode_value(
        emergency_level_encoder,
        data.emergency_level,
        "emergency_level"
    )
    equipment_level = encode_value(
        equipment_level_encoder,
        data.equipment_level,
        "equipment_level"
    )
    gender = encode_value(gender_encoder, data.gender, "gender")
    road_type = encode_value(road_type_encoder, data.road_type, "road_type")
    symptom_category = encode_value(
        symptom_category_encoder,
        data.symptom_category,
        "symptom_category"
    )
    traffic_level = encode_value(
        traffic_level_encoder,
        data.traffic_level,
        "traffic_level"
    )
    weather_condition = encode_value(
        weather_condition_encoder,
        data.weather_condition,
        "weather_condition"
    )
    zone = encode_value(zone_encoder, data.zone, "zone")

    processed = np.array([[
        emergency_level,
        equipment_level,
        gender,
        road_type,
        symptom_category,
        traffic_level,
        weather_condition,
        zone
    ]])

    return processed


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to Mediconnect Backend API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "hospital_model_loaded": hospital_model is not None,
        "patient_model_loaded": patient_model is not None,
        "ambulance_model_loaded": ambulance_model is not None
    }


@app.get("/metadata")
def metadata():
    return {
        "hospital": {
            "age_group": get_classes(age_group_encoder),
            "bmi_category": get_classes(bmi_category_encoder),
            "city": get_classes(city_encoder),
            "distance_category": get_classes(distance_category_encoder),
            "hospitalization_burden": get_classes(hospitalization_burden_encoder),
        },
        "patient": {
            "gender": get_classes(gender_encoder),
            "symptoms_severity": get_classes(symptom_category_encoder),
            "risk_levels": get_classes(risk_level_encoder),
        },
        "ambulance": {
            "emergency_level": get_classes(emergency_level_encoder),
            "equipment_level": get_classes(equipment_level_encoder),
            "gender": get_classes(gender_encoder),
            "road_type": get_classes(road_type_encoder),
            "symptom_category": get_classes(symptom_category_encoder),
            "traffic_level": get_classes(traffic_level_encoder),
            "weather_condition": get_classes(weather_condition_encoder),
            "zone": get_classes(zone_encoder),
            "dispatch_priority": get_classes(dispatch_priority_encoder),
        }
    }


@app.post("/predict")
def predict_hospital(data: HospitalInput):
    try:
        processed_input = preprocess_hospital_input(data)
        prediction = hospital_model.predict(processed_input)[0]

        response = {
            "recommended": int(prediction) if str(prediction).isdigit() else str(prediction)
        }

        confidence = safe_predict_proba(hospital_model, processed_input)
        if confidence is not None:
            response["confidence"] = confidence

        return response

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict-patient-risk")
def predict_patient_risk(data: PatientRiskInput):
    try:
        processed_input = preprocess_patient_input(data)
        prediction = patient_model.predict(processed_input)

        try:
            risk_label = risk_level_encoder.inverse_transform(prediction)[0]
        except Exception:
            risk_label = prediction[0]

        response = {
            "risk_level": str(risk_label)
        }

        confidence = safe_predict_proba(patient_model, processed_input)
        if confidence is not None:
            response["confidence"] = confidence

        risk_text = str(risk_label).lower()

        if risk_text == "low":
            response["recommendation"] = "Maintain healthy lifestyle and routine monitoring."
        elif risk_text in ["medium", "moderate"]:
            response["recommendation"] = "Moderate risk. Doctor consultation recommended."
        elif risk_text == "high":
            response["recommendation"] = "High risk. Immediate medical attention advised."
        else:
            response["recommendation"] = "Consult a healthcare professional for further evaluation."

        return response

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict-ambulance-dispatch")
def predict_ambulance_dispatch(data: AmbulanceDispatchInput):
    try:
        processed_input = preprocess_ambulance_input(data)
        prediction = ambulance_model.predict(processed_input)

        try:
            dispatch_label = dispatch_priority_encoder.inverse_transform(prediction)[0]
        except Exception:
            dispatch_label = prediction[0]

        response = {
            "dispatch_priority": str(dispatch_label)
        }

        confidence = safe_predict_proba(ambulance_model, processed_input)
        if confidence is not None:
            response["confidence"] = confidence

        dispatch_text = str(dispatch_label).lower()

        if dispatch_text in ["high", "critical", "priority_1", "p1"]:
            response["recommendation"] = "Dispatch nearest ambulance immediately."
        elif dispatch_text in ["medium", "priority_2", "p2"]:
            response["recommendation"] = "Dispatch ambulance with moderate urgency."
        elif dispatch_text in ["low", "priority_3", "p3"]:
            response["recommendation"] = "Schedule standard ambulance response."
        else:
            response["recommendation"] = "Dispatch decision generated successfully."

        return response

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}