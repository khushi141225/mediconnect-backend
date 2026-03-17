from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import traceback

app = FastAPI()

# Load files
try:
    patient_model = joblib.load("models/patient_risk_best_model.pkl")
    gender_encoder = joblib.load("models/Gender_encoder.pkl")
    symptom_category_encoder = joblib.load("models/Symptom_Category_encoder.pkl")
    patient_scaler = joblib.load("models/patient_risk_scaler.pkl")
    risk_level_encoder = joblib.load("models/risk_level_encoder.pkl")
    print("All patient files loaded successfully")
except Exception as e:
    print("Model loading error:", str(e))
    patient_model = None

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

@app.get("/")
def home():
    return {"message": "Backend running"}

@app.post("/predict-patient-risk")
def predict_patient_risk(data: PatientRiskInput):
    try:
        if patient_model is None:
            raise Exception("Patient model files not loaded")

        print("Raw input:", data.model_dump())

        gender_encoded = gender_encoder.transform([data.gender])[0]
        print("Gender encoded:", gender_encoded)

        symptom_encoded = symptom_category_encoder.transform([data.symptoms_severity])[0]
        print("Symptom encoded:", symptom_encoded)

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
        print("Features:", features)

        scaled_features = patient_scaler.transform(features)
        print("Scaled features:", scaled_features)

        prediction = patient_model.predict(scaled_features)
        print("Prediction:", prediction)

        result = risk_level_encoder.inverse_transform(prediction)
        print("Decoded result:", result)

        return {"prediction": str(result[0])}

    except Exception as e:
        error_text = traceback.format_exc()
        print("FULL ERROR:\n", error_text)
        raise HTTPException(status_code=500, detail=error_text)