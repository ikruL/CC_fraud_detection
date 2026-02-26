import joblib
import pandas as pd
import os


MODEL_DIR = '../models'
THRESHOLD = 0.5


def load_model():

    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Please train the model first.")

    model = joblib.load(model_path)

    return model


def predict(df):
    model = load_model()

    prob = model.predict_proba(df)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)

    return pred, prob
