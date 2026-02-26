import joblib
import pandas as pd
import os


MODEL_DIR = '../models'


def load_model():

    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    threshold_path = os.path.join(MODEL_DIR, 'best_threshold.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Please train the model first.")

    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)

    return model, threshold


def predict(df):
    model, threshold = load_model()

    prob = model.predict_proba(df)[:, 1]
    pred = (prob >= threshold).astype(int)

    return pred, prob
