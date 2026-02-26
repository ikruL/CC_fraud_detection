from fastapi import FastAPI
import joblib
import pandas as pd


app = FastAPI()

model = joblib.load("models/best_model.pkl")
threshold = joblib.load("models/best_threshold.pkl")


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(prob >= threshold)

    return {
        "probability": prob,
        "prediction": pred
    }


@app.get("/")
def root():
    return {"message": "Welcome to the Credit Card Fraud Detection API!"}
