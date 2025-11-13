
import optuna
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

def tune_with_optuna(model_builder, param_space, X, y, cv=5, n_trials=40, direction="minimize"):
    def objective(trial):
        params = {k: v(trial) for k, v in param_space.items()}
        model = model_builder(**params)
        scores = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=KFold(cv, shuffle=True, random_state=42))
        return -scores.mean()
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value, study
