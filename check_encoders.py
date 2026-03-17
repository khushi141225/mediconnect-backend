import joblib

print("START")
print("Emergency:", joblib.load("models/Emergency_Level_encoder.pkl").classes_)
print("Equipment:", joblib.load("models/Equipment_Level_encoder.pkl").classes_)
print("Road:", joblib.load("models/Road_Type_encoder.pkl").classes_)
print("Traffic:", joblib.load("models/Traffic_Level_encoder.pkl").classes_)
print("Weather:", joblib.load("models/Weather_Condition_encoder.pkl").classes_)
print("Zone:", joblib.load("models/Zone_encoder.pkl").classes_)
print("Dispatch:", joblib.load("models/Dispatch_Priority_encoder.pkl").classes_)
print("END")