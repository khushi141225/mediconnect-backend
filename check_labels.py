import joblib

age_group_encoder = joblib.load("models/Age_Group_encoder.pkl")
bmi_category_encoder = joblib.load("models/BMI_Category_encoder.pkl")
city_encoder = joblib.load("models/City_encoder.pkl")
distance_category_encoder = joblib.load("models/Distance_Category_encoder.pkl")
hospitalization_burden_encoder = joblib.load("models/Hospitalization_Burden_encoder.pkl")

print("Age Group:", age_group_encoder.classes_)
print("BMI Category:", bmi_category_encoder.classes_)
print("City:", city_encoder.classes_)
print("Distance Category:", distance_category_encoder.classes_)
print("Hospitalization Burden:", hospitalization_burden_encoder.classes_)