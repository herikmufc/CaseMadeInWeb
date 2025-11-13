
import os, json, joblib, mlflow, mlflow.sklearn

def save_pipeline_and_log(pipeline, X, y, run_name="house_price_best"):
    os.makedirs("models", exist_ok=True)
    pipeline.fit(X, y)
    local_path = "models/best_model.joblib"
    joblib.dump(pipeline, local_path)
    # Save feature order
    try:
        features = list(X.columns)
    except Exception:
        features = None
    if features:
        with open("models/feature_order.json", "w") as f:
            json.dump(features, f)
    with mlflow.start_run(run_name=run_name):
        mlflow.sklearn.log_model(pipeline, "model")
    return local_path
