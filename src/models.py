
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import SEED


def get_models():
    models = {
        'Logistic Regression': LogisticRegression(random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=5, random_state=SEED),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=SEED)
    }
    return models
