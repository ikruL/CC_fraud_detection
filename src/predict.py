import joblib
import pandas as pd
import os


MODEL_DIR = '../models'
THRESHOLD = 0.5


def load_model():
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Model or scaler not found. Please train the model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def preprocess_input(df, scaler):

    # Scale the 'Time' and 'Amount' features
    scale_features = ['Time', 'Amount']
    df[scale_features] = scaler.transform(df[scale_features])
    return df


def predict(df):
    model, scaler = load_model()

    df = preprocess_input(df, scaler)

    prob = model.predict_proba(df)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    return pred, prob
