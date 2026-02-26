import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(y_test, y_proba):
    thresholds = np.arange(0.01, 0.99, 0.01)

    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        f1 = f1_score(y_test, y_pred)

        results.append({
            "threshold": t,
            "f1": f1
        })

    # sort by best f1
    best = sorted(results, key=lambda x: x["f1"], reverse=True)[0]

    return best["threshold"]
