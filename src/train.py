import os
import joblib
import models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

from sklearn.pipeline import Pipeline

from preprocessing import load_data, clean_data, preprocess_data, SEED


DATA_PATH = "../data/creditcard.csv"
MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)


def train():

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    pos = sum(y_train)
    neg = len(y_train) - pos
    models_dict = models.get_models(scale_pos_weight=neg/pos)

    best_model = None
    best_score = -1
    best_name = ""

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    for name, model in models_dict.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        # Cross validation score
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring="average_precision",
            n_jobs=-1
        )

        print(f"CV PR-AUC: {cv_scores.mean():.4f}")

        # Train full pipeline
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

        print(classification_report(y_test, y_pred))

        if cv_scores.mean() > best_score:

            best_score = cv_scores.mean()
            best_model = pipeline
            best_name = name

    print(f"\nBest model: {best_name}")
    print(f"Best CV PR-AUC: {best_score:.4f}")

    print("Saving best model ...")
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")


if __name__ == "__main__":
    train()
