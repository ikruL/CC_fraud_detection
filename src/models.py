
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import SEED


def get_models(scale_pos_weight=None):
    models = {
        'Logistic Regression': LogisticRegression(random_state=SEED, class_weight='balanced', max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=5, random_state=SEED, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=5, random_state=SEED, scale_pos_weight=scale_pos_weight)
    }
    return models
