
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

def select_k_best(X: pd.DataFrame, y: pd.Series, k: int = 20):
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    support = selector.get_support()
    selected_cols = X.columns[support].tolist()
    X_sel = pd.DataFrame(X_new, columns=selected_cols, index=X.index)
    return X_sel, selected_cols
