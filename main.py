from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import xgboost  # required for loading xgboost model

app = FastAPI(title="MediConnect Backend API", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_model(file_name):
    path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return joblib.load(path)


# ---------------- LOAD MODELS ----------------
try:
    hospital_model = load_model("hospital_recommendation_model.pkl")
    print("✅ Hospital model loaded")
except Exception as e:
    print("❌ Hospital model error:", e)
    raise

try:
    patient_model = load_model("patient_risk_best_model.pkl")
    gender_encoder = load_model("Gender_encoder.pkl")
    symptom_category_encoder = load_model("Symptom_Category_encoder.pkl")
    age_group_encoder = load_model("Age_Group_encoder.pkl")
    bmi_category_encoder = load_model("BMI_Category_encoder.pkl")
    hospitalization_burden_encoder = load_model("Hospitalization_Burden_encoder.pkl")
    patient_scaler = load_model("patient_risk_scaler.pkl")
    risk_level_encoder = load_model("risk_level_encoder.pkl")
    print("✅ Patient model loaded")
except Exception as e:
    print("❌ Patient model error:", e)
    raise


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
    gender: str
    bmi: float
    oxygen_level: float
    respiratory_rate: int
    diabetes: int
    hypertension: int
    heart_disease: int
    smoking: int
    alcohol_use: int
    blood_pressure: float
    heart_rate: float
    body_temperature: float
    glucose_level: float
    previous_hospitalizations: int
    symptoms_severity: str


# ---------------- HELPERS ----------------
def safe_transform(encoder, value, field_name):
    try:
        return encoder.transform([value])[0]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: {value}. Allowed: {list(encoder.classes_)}"
        )


def safe_decode(encoder, value):
    try:
        return encoder.inverse_transform([int(value)])[0]
    except Exception:
        return str(value)


def get_age_group(age: int) -> str:
    if age < 18:
        return "Child"
    elif age < 35:
        return "Young Adult"
    elif age < 50:
        return "Adult"
    elif age < 65:
        return "Middle Aged"
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


def get_fever(temp: float) -> int:
    return 1 if temp >= 100.4 else 0


def get_low_oxygen(o2: float) -> int:
    return 1 if o2 < 95 else 0


def get_high_bp(bp: float) -> int:
    return 1 if bp >= 140 else 0


def get_high_hr(hr: float) -> int:
    return 1 if hr >= 100 else 0


def get_comorbidity(d: int, h: int, hd: int) -> int:
    return int(d) + int(h) + int(hd)


def get_lifestyle(sm: int, al: int) -> int:
    return int(sm) + int(al)


def get_vital_score(lo: int, bp: int, hr: int, fever: int) -> int:
    return int(lo) + int(bp) + int(hr) + int(fever)


def get_hospitalization_burden(ph: int) -> str:
    if ph == 0:
        return "Low"
    elif ph <= 2:
        return "Moderate"
    else:
        return "High"


def get_symptom_category(s: str) -> str:
    s = str(s).strip().lower()
    if s in ["mild", "low"]:
        return "Mild"
    elif s in ["moderate", "medium"]:
        return "Moderate"
    else:
        return "Severe"


# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"message": "Backend running"}


@app.get("/debug/patient-features")
def debug_patient_features():
    return {
        "scaler_features": list(patient_scaler.feature_names_in_) if hasattr(patient_scaler, "feature_names_in_") else [],
        "model_features": list(patient_model.feature_names_in_) if hasattr(patient_model, "feature_names_in_") else [],
        "gender_classes": list(gender_encoder.classes_),
        "symptom_classes": list(symptom_category_encoder.classes_),
        "age_group_classes": list(age_group_encoder.classes_),
        "bmi_category_classes": list(bmi_category_encoder.classes_),
        "hospitalization_burden_classes": list(hospitalization_burden_encoder.classes_)
    }


@app.post("/predict-hospital")
def predict_hospital(data: HospitalInput):
    try:
        icu_ratio = data.icu_beds / data.total_beds if data.total_beds else 0
        ventilator_ratio = data.ventilators / data.total_beds if data.total_beds else 0

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

        pred = hospital_model.predict(df)[0]

        result = {
            "prediction": int(pred) if isinstance(pred, (int, np.integer)) else str(pred)
        }

        if hasattr(hospital_model, "predict_proba"):
            try:
                result["probabilities"] = [float(x) for x in hospital_model.predict_proba(df)[0]]
            except Exception:
                pass

        return result

    except Exception as e:
        print("HOSPITAL ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-patient-risk")
def predict_patient_risk(data: PatientRiskInput):
    try:
        # -------- derived labels --------
        age_group_label = get_age_group(data.age)
        bmi_cat_label = get_bmi_category(data.bmi)
        hosp_label = get_hospitalization_burden(data.previous_hospitalizations)
        symptom_label = get_symptom_category(data.symptoms_severity)

        # -------- derived numeric features --------
        fever = get_fever(data.body_temperature)
        low_o2 = get_low_oxygen(data.oxygen_level)
        high_bp = get_high_bp(data.blood_pressure)
        high_hr = get_high_hr(data.heart_rate)
        comorb = get_comorbidity(data.diabetes, data.hypertension, data.heart_disease)
        life = get_lifestyle(data.smoking, data.alcohol_use)
        vital = get_vital_score(low_o2, high_bp, high_hr, fever)

        # -------- encodings --------
        gender = safe_transform(gender_encoder, data.gender, "gender")
        age_group = safe_transform(age_group_encoder, age_group_label, "age_group")
        bmi_cat = safe_transform(bmi_category_encoder, bmi_cat_label, "bmi_category")
        hosp = safe_transform(hospitalization_burden_encoder, hosp_label, "hospitalization_burden")
        symptom = safe_transform(symptom_category_encoder, symptom_label, "symptom_category")

        # -------- master row --------
        # Include BOTH Symptom_Category and Symptoms_Severity to avoid mismatch
        row = {
            "Age": data.age,
            "Gender": gender,
            "BMI": data.bmi,
            "Oxygen_Level": data.oxygen_level,
            "Respiratory_Rate": data.respiratory_rate,
            "Diabetes": data.diabetes,
            "Hypertension": data.hypertension,
            "Heart_Disease": data.heart_disease,
            "Smoking": data.smoking,
            "Alcohol_Use": data.alcohol_use,
            "Blood_Pressure": data.blood_pressure,
            "Heart_Rate": data.heart_rate,
            "Body_Temperature": data.body_temperature,
            "Glucose_Level": data.glucose_level,
            "Previous_Hospitalizations": data.previous_hospitalizations,
            "Age_Group": age_group,
            "BMI_Category": bmi_cat,
            "Fever": fever,
            "Low_Oxygen": low_o2,
            "High_BP": high_bp,
            "High_Heart_Rate": high_hr,
            "Comorbidity_Count": comorb,
            "Lifestyle_Risk": life,
            "Vital_Instability_Score": vital,
            "Hospitalization_Burden": hosp,
            "Symptom_Category": symptom,
            "Symptoms_Severity": symptom
        }

        df = pd.DataFrame([row])

        # -------- align to scaler features --------
        if hasattr(patient_scaler, "feature_names_in_"):
            expected_cols = list(patient_scaler.feature_names_in_)

            # Add any still-missing expected columns with safe default 0
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0

            df = df[expected_cols]

        print("\nPATIENT DF COLUMNS:", list(df.columns))
        print(df)

        scaled = patient_scaler.transform(df)
        pred = patient_model.predict(scaled)[0]

        result = {
            "prediction": safe_decode(risk_level_encoder, pred),
            "encoded_prediction": int(pred) if isinstance(pred, (int, np.integer)) else str(pred)
        }

        if hasattr(patient_model, "predict_proba"):
            try:
                result["probabilities"] = [float(x) for x in patient_model.predict_proba(scaled)[0]]
            except Exception:
                pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        print("PATIENT ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))