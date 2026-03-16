from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
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
# LOAD MODELS FROM models/
# =========================
hospital_model = joblib.load("models/hospital_recommendation_model.pkl")

patient_model = joblib.load("models/patient_risk_best_model.pkl")
patient_scaler = joblib.load("models/patient_risk_scaler.pkl")

age_group_encoder = joblib.load("models/Age_Group_encoder.pkl")
bmi_category_encoder = joblib.load("models/BMI_Category_encoder.pkl")
gender_encoder = joblib.load("models/Gender_encoder.pkl")
symptom_category_encoder = joblib.load("models/Symptom_Category_encoder.pkl")
hospitalization_burden_encoder = joblib.load("models/Hospitalization_Burden_encoder.pkl")
risk_level_encoder = joblib.load("models/risk_level_encoder.pkl")


# =========================
# INPUT SCHEMAS
# =========================
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


class PatientRiskInput(BaseModel):
    age: int
    gender: str
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
    symptoms_severity: str

# =========================
# HELPER FUNCTIONS
# =========================
def get_age_group(age: int) -> str:
    if age <= 12:
        return "Child"
    elif age <= 25:
        return "Young"
    elif age <= 45:
        return "Adult"
    elif age <= 60:
        return "Middle_Aged"
    else:
        return "Senior"


def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def preprocess_patient_input(data: PatientRiskInput):
    age_group = get_age_group(data.age)
    bmi_category = get_bmi_category(data.bmi)

    fever = 1 if data.body_temperature > 99 else 0
    low_oxygen = 1 if data.oxygen_level < 94 else 0
    high_bp = 1 if data.blood_pressure > 140 else 0
    high_hr = 1 if data.heart_rate > 100 else 0

    comorbidity_count = data.diabetes + data.hypertension + data.heart_disease
    lifestyle_risk = data.smoking + data.alcohol_use
    vital_score = fever + low_oxygen + high_bp + high_hr

    input_dict = {
        "Age": data.age,
        "Gender": gender_encoder.transform([data.gender])[0],
        "BMI": data.bmi,
        "Blood_Pressure": data.blood_pressure,
        "Heart_Rate": data.heart_rate,
        "Oxygen_Level": data.oxygen_level,
        "Body_Temperature": data.body_temperature,
        "Respiratory_Rate": data.respiratory_rate,
        "Diabetes": data.diabetes,
        "Hypertension": data.hypertension,
        "Heart_Disease": data.heart_disease,
        "Smoking": data.smoking,
        "Alcohol_Use": data.alcohol_use,
        "Previous_Hospitalizations": data.previous_hospitalizations,
        "Symptoms_Severity": symptom_category_encoder.transform([data.symptoms_severity])[0],
        "Age_Group": age_group_encoder.transform([age_group])[0],
        "BMI_Category": bmi_category_encoder.transform([bmi_category])[0],
        "Fever": fever,
        "Low_Oxygen": low_oxygen,
        "High_BP": high_bp,
        "High_Heart_Rate": high_hr,
        "Comorbidity_Count": comorbidity_count,
        "Lifestyle_Risk": lifestyle_risk,
        "Vital_Instability_Score": vital_score,
        "Hospitalization_Burden": hospitalization_burden_encoder.transform(["Moderate"])[0],
        "Symptom_Category": symptom_category_encoder.transform(["Moderate"])[0],
    }

    input_df = pd.DataFrame([input_dict])
    scaled_input = patient_scaler.transform(input_df)
    return scaled_input
# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "MediConnect AI API Running"}


@app.post("/predict-hospital")
def predict_hospital(data: HospitalInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = hospital_model.predict(input_df)[0]
    probability = hospital_model.predict_proba(input_df)[0][1]

    return {
        "recommended": int(prediction),
        "confidence": float(probability)
    }


@app.post("/predict-patient-risk")
def predict_patient_risk(data: PatientRiskInput):
    try:
        processed_input = preprocess_patient_input(data)

        prediction = patient_model.predict(processed_input)[0]
        risk_label = risk_level_encoder.inverse_transform([prediction])[0]

        response = {
            "risk_level": str(risk_label)
        }

        if hasattr(patient_model, "predict_proba"):
            probabilities = patient_model.predict_proba(processed_input)[0]
            confidence = max(probabilities) * 100
            response["confidence"] = round(float(confidence), 2)

        risk_text = str(risk_label).lower()

        if risk_text == "low":
            response["recommendation"] = "Maintain healthy lifestyle and regular checkups."
        elif risk_text in ["medium", "moderate"]:
            response["recommendation"] = "Consult a doctor and monitor vitals regularly."
        else:
            response["recommendation"] = "Immediate medical consultation recommended."

        return response

    except Exception as e:
        return {"error": str(e)}