
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os, json
import numpy as np
import pandas as pd

app = FastAPI(title="HousePriceAPI")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_order.json")

class PredictRequest(BaseModel):
    instances: list[list[float]]

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_order = None
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            feature_order = json.load(f)
    return model, feature_order

model, feature_order = load_artifacts()

@app.get("/health")
def health():
    return {"model_loaded": model is not None, "n_features": len(feature_order) if feature_order else None}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    X = np.array(req.instances)
    if feature_order and X.shape[1] != len(feature_order):
        raise HTTPException(status_code=400, detail=f"Número de features incorreto. Esperado {len(feature_order)}")
    # Build DataFrame to preserve feature names when available
    if feature_order:
        X_df = pd.DataFrame(X, columns=feature_order)
    else:
        X_df = pd.DataFrame(X)
    preds = model.predict(X_df)
    return {"predictions": preds.tolist()}
