from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import traceback

app = FastAPI(
    title="MediConnect Backend",
    version="1.0.0"
)

# =========================================================
# LOAD MODELS
# =========================================================
try:
    # Hospital model files
    hospital_model = joblib.load("models/hospital_recommendation_model.pkl")
    age_group_encoder = joblib.load("models/Age_Group_encoder.pkl")
    bmi_category_encoder = joblib.load("models/BMI_Category_encoder.pkl")
    city_encoder = joblib.load("models/City_encoder.pkl")
    distance_category_encoder = joblib.load("models/Distance_Category_encoder.pkl")
    hospitalization_burden_encoder = joblib.load("models/Hospitalization_Burden_encoder.pkl")

    print("✅ Hospital model files loaded successfully")

except Exception as e:
    print("❌ Hospital model loading error:", str(e))
    hospital_model = None
    age_group_encoder = None
    bmi_category_encoder = None
    city_encoder = None
    distance_category_encoder = None
    hospitalization_burden_encoder = None

try:
    # Patient model files
    patient_model = joblib.load("models/patient_risk_best_model.pkl")
    gender_encoder = joblib.load("models/Gender_encoder.pkl")
    symptom_category_encoder = joblib.load("models/Symptom_Category_encoder.pkl")
    patient_scaler = joblib.load("models/patient_risk_scaler.pkl")
    risk_level_encoder = joblib.load("models/risk_level_encoder.pkl")

    print("✅ Patient model files loaded successfully")

except Exception as e:
    print("❌ Patient model loading error:", str(e))
    patient_model = None
    gender_encoder = None
    symptom_category_encoder = None
    patient_scaler = None
    risk_level_encoder = None


# =========================================================
# INPUT SCHEMAS
# =========================================================
class HospitalInput(BaseModel):
    age_group: str
    bmi_category: str
    city: str
    distance_category: str
    hospitalization_burden: str


class PatientRiskInput(BaseModel):
    age: int
    gender: str
    bmi: float
    oxygen_level: float
    respiratory_rate: float
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


# =========================================================
# BASIC ROUTES
# =========================================================
@app.get("/")
def home():
    return {"message": "Backend running"}

@app.get("/test")
def test():
    return {"status": "ok"}

@app.get("/hospital-labels")
def get_hospital_labels():
    try:
        return {
            "age_group": age_group_encoder.classes_.tolist(),
            "bmi_category": bmi_category_encoder.classes_.tolist(),
            "city": city_encoder.classes_.tolist(),
            "distance_category": distance_category_encoder.classes_.tolist(),
            "hospitalization_burden": hospitalization_burden_encoder.classes_.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# HOSPITAL RECOMMENDATION ENDPOINT
# =========================================================
@app.post("/predict")
def predict_hospital(data: HospitalInput):
    try:
        if hospital_model is None:
            raise Exception("Hospital model files not loaded properly")

        print("===== NEW HOSPITAL REQUEST =====")
        print("Raw input:", data.model_dump())

        age_group_encoded = age_group_encoder.transform([data.age_group])[0]
        bmi_category_encoded = bmi_category_encoder.transform([data.bmi_category])[0]
        city_encoded = city_encoder.transform([data.city])[0]
        distance_category_encoded = distance_category_encoder.transform([data.distance_category])[0]
        hospitalization_burden_encoded = hospitalization_burden_encoder.transform([data.hospitalization_burden])[0]

        print("Age Group encoded:", age_group_encoded)
        print("BMI Category encoded:", bmi_category_encoded)
        print("City encoded:", city_encoded)
        print("Distance Category encoded:", distance_category_encoded)
        print("Hospitalization Burden encoded:", hospitalization_burden_encoded)

        features = [[
            age_group_encoded,
            bmi_category_encoded,
            city_encoded,
            distance_category_encoded,
            hospitalization_burden_encoded
        ]]

        print("Hospital features:", features)

        prediction = hospital_model.predict(features)
        print("Hospital raw prediction:", prediction)

        return {
            "recommended_hospital": str(prediction[0]),
            "status": "success"
        }

    except Exception:
        error_text = traceback.format_exc()
        print("❌ FULL HOSPITAL PREDICTION ERROR:\n", error_text)
        raise HTTPException(status_code=500, detail=error_text)


# =========================================================
# PATIENT RISK PREDICTION ENDPOINT
# =========================================================
@app.post("/predict-patient-risk")
def predict_patient_risk(data: PatientRiskInput):
    try:
        if patient_model is None:
            raise Exception("Patient model files not loaded properly")

        print("===== NEW PATIENT REQUEST =====")
        print("Raw input:", data.model_dump())

        gender_encoded = gender_encoder.transform([data.gender])[0]
        print("Gender encoded:", gender_encoded)

        symptom_encoded = symptom_category_encoder.transform([data.symptoms_severity])[0]
        print("Symptom severity encoded:", symptom_encoded)

        # Feature order must match training order exactly
        features = [[
            data.age,
            gender_encoded,
            data.bmi,
            data.oxygen_level,
            data.respiratory_rate,
            data.diabetes,
            data.hypertension,
            data.heart_disease,
            data.smoking,
            data.alcohol_use,
            data.blood_pressure,
            data.heart_rate,
            data.body_temperature,
            data.glucose_level,
            data.previous_hospitalizations,
            symptom_encoded
        ]]

        print("Patient features before scaling:", features)

        scaled_features = patient_scaler.transform(features)
        print("Scaled features:", scaled_features)

        prediction = patient_model.predict(scaled_features)
        print("Patient raw prediction:", prediction)

        result = risk_level_encoder.inverse_transform(prediction)
        print("Decoded result:", result)

        return {
            "prediction": str(result[0]),
            "status": "success"
        }

    except Exception:
        error_text = traceback.format_exc()
        print("❌ FULL PATIENT PREDICTION ERROR:\n", error_text)
        raise HTTPException(status_code=500, detail=error_text)