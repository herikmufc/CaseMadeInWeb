
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

NUMERIC_IMPUTER = SimpleImputer(strategy="median")

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop obvious IDs not useful for training
    for col in ["id", "date"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    # Remove duplicates
    df = df.drop_duplicates()
    return df

def zscore_outlier_removal(df: pd.DataFrame, threshold: float = 4.0, target_col: str = "price") -> pd.DataFrame:
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric:
        numeric = [c for c in numeric if c != target_col]
    # Compute zscores only on numeric predictors
    if len(numeric):
        z = (df[numeric] - df[numeric].mean())/df[numeric].std(ddof=0)
        mask = (z.abs() < threshold).all(axis=1)
        df = df[mask]
    return df

def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns
    if len(numeric):
        df[numeric] = NUMERIC_IMPUTER.fit_transform(df[numeric])
    # Fill non-numeric with mode
    non_num = df.select_dtypes(exclude=[np.number]).columns
    for col in non_num:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")
    return df

def merge_zip_demographics(house_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize zip column names
    z1 = None
    for cand in ["zipcode", "zip", "ZIPCode"]:
        if cand in house_df.columns:
            z1 = cand
            break
    z2 = None
    for cand in ["zipcode", "zip", "ZIPCode"]:
        if cand in demo_df.columns:
            z2 = cand
            break
    if z1 is None or z2 is None:
        return house_df
    demo = demo_df.copy()
    demo[z2] = demo[z2].astype(int)
    hh = house_df.copy()
    hh[z1] = hh[z1].astype(int)
    merged = hh.merge(demo, left_on=z1, right_on=z2, how="left", suffixes=("", "_zip"))
    return merged
