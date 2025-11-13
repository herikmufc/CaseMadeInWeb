
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

def cv_rmse(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
    return -scores.mean()

def evaluate_models(models: dict, X, y, cv=5):
    results = {}
    for name, m in models.items():
        results[name] = cv_rmse(m, X, y, cv=cv)
    return results
