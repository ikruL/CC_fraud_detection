

import os
import joblib
import models
from preprocessing import load_data, clean_data, preprocess_data, SEED
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score


SEED = 42  # Set a random seed
DATA_PATH = '../data/creditcard.csv'
MODEL_DIR = 'models'

os.makedirs(MODEL_DIR, exist_ok=True)


def train():

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Initialize models
    models_dict = models.get_models()

    best_model = None
    best_score = 0
    best_model_name = ""

    for name, model in models_dict.items():

        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        pr_auc = average_precision_score(y_test, y_proba)

        print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"Average Precision Score for {name}: {pr_auc:.4f}")
        print(
            f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}")
        # Update best model
        if pr_auc > best_score:
            best_score = pr_auc
            best_model = model
            best_model_name = name

    # Save the best model
    print(
        f"Best model: {best_model_name} with Average Precision Score: {best_score:.4f}")
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))


if __name__ == "__main__":
    train()
