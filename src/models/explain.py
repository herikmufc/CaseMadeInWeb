
import pandas as pd
from sklearn.inspection import permutation_importance

def permutation_importances(model, X, y, n_repeats=5, random_state=42):
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, scoring="neg_root_mean_squared_error")
    imp = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean}).sort_values("importance", ascending=False)
    return imp
