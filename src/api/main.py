from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Model load hoga sirf ek baar, jab server start hoga
model = joblib.load('src/models/trained_credit_risk_model.pkl')

@app.get("/")
def home():
    return {"message": "CreditGuard API is running"}

@app.post("/predict")
def predict(applicant: dict):
    df = pd.DataFrame([applicant])
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    return {
        "prediction": int(prediction[0]),
        "default_probability": float(probability[0])
    }