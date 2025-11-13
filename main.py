# ============================================================
# 🏠 Previsão de Preços de Casas - Projeto MadeinWeb
# ============================================================

import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import optuna
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
import uvicorn


# ============================================================
# CONFIGURAÇÕES DE LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. LIMPEZA E PRÉ-PROCESSAMENTO DE DADOS
# ============================================================
def clean_data(file_path, is_future=False):
    """
    Limpa, trata outliers e normaliza seletivamente as variáveis numéricas.
    Mantém colunas de identificação e preço em escala original.
    """

    df = pd.read_csv(file_path)
    df = df.dropna(how="all")

    # --- Conversão de data em variáveis derivadas ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df = df.drop(columns=["date"])

    # --- Definição de exclusões ---
    exclude_cols = ["id", "zipcode"]
    if not is_future and "price" in df.columns:
        exclude_cols.append("price")

    num_cols = [col for col in df.select_dtypes(include=["float64", "int64"]).columns if col not in exclude_cols]

    # --- Tratamento de outliers via Z-score ---
    for col in num_cols:
        zscore = (df[col] - df[col].mean()) / df[col].std()
        df = df[(zscore.abs() <= 3) | (zscore.isna())]

    # --- Normalização seletiva ---
    scaler = StandardScaler()
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # --- Reorganização de colunas (flexível) ---
    ordered_cols = ["id"] + [c for c in df.columns if c not in ["id", "zipcode"]] + ["zipcode"]
    df = df[[c for c in ordered_cols if c in df.columns]]

    # --- Versão interpretável (para Power BI) ---
    if not is_future:
        df_original = pd.read_csv(file_path)
        if "date" in df_original.columns:
            df_original["date"] = pd.to_datetime(df_original["date"], errors="coerce")
            df_original["year"] = df_original["date"].dt.year
            df_original["month"] = df_original["date"].dt.month
            df_original["day"] = df_original["date"].dt.day
            df_original = df_original.drop(columns=["date"])
        df_original.to_csv("outputs/dashboard_dataset.csv", index=False)
        logger.info("📊 Versão original salva em 'dashboard_dataset.csv' para uso no Power BI.")

    logger.info(
        f"✅ Dados limpos e normalizados seletivamente ({len(df)} linhas, "
        f"{'sem preço' if is_future else 'com preço'})."
    )
    return df, scaler


# ============================================================
# 2. SELEÇÃO DE FEATURES
# ============================================================
def select_features(df):
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    if "price" not in numeric_df.columns:
        raise ValueError("A coluna 'price' não foi encontrada para calcular correlação.")
    corr = numeric_df.corr()
    top_features = corr["price"].abs().sort_values(ascending=False).head(10).index
    top_features = [f for f in top_features if f != "price_dollar"]
    if "zipcode" in df.columns and "zipcode" not in top_features:
        top_features = list(top_features) + ["zipcode"]
    logger.info(f"🎯 Features selecionadas: {top_features}")
    return df[top_features]


# ============================================================
# 3. TREINAMENTO E AVALIAÇÃO DE MODELOS
# ============================================================
def train_models(X, y):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "RandomForest": RandomForestRegressor(random_state=42),
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)
        results[name] = abs(scores.mean())
        logger.info(f"{name}: MSE médio = {results[name]:.2f}")

    best_model_name = min(results, key=results.get)
    best_model = models[best_model_name]
    best_model.fit(X, y)
    logger.info(f"✅ Melhor modelo: {best_model_name}")
    return best_model_name, best_model


# ============================================================
# 4. OTIMIZAÇÃO DE HIPERPARÂMETROS (Optuna)
# ============================================================
def optimize_model(X, y, model_type="RandomForest"):
    def objective(trial):
        if model_type == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
        else:
            alpha = trial.suggest_float("alpha", 0.0001, 10.0)
            model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)
        return abs(scores.mean())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    logger.info(f"🔧 Melhores parâmetros: {study.best_params}")
    return study.best_params


# ============================================================
# 5. PIPELINE PRINCIPAL
# ============================================================
def main():
    os.makedirs("outputs", exist_ok=True)
    data_path = "data/raw/kc_house_data.csv"
    unseen_path = "data/raw/future_unseen_examples.csv"

    logger.info("🧹 Iniciando limpeza e padronização...")
    clean_df, scaler = clean_data(data_path)

    logger.info("🎯 Selecionando features relevantes...")
    selected_df = select_features(clean_df)
    X = selected_df.drop(["price", "zipcode"], axis=1)
    y = selected_df["price"]

    logger.info("🤖 Treinando modelos base...")
    best_name, best_model = train_models(X, y)

    logger.info("⚙️ Otimizando hiperparâmetros com Optuna...")
    best_params = optimize_model(X, y, best_name)

    mlflow.set_experiment("house_price_prediction")
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(best_model, "best_model")

    logger.info("🔮 Aplicando modelo aos dados futuros...")
    future_df, _ = clean_data(unseen_path, is_future=True)
    train_features = [col for col in X.columns if col in future_df.columns]
    preds = best_model.predict(future_df[train_features])
    future_df["pred_price"] = preds

    # ========================================================
    # 🚀 DASHBOARD PARA POWER BI
    # ========================================================
    logger.info("📊 Gerando tabela consolidada para o Power BI...")
    if "zipcode" in clean_df.columns and "zipcode" in future_df.columns:
        summary_real = clean_df.groupby("zipcode", as_index=False).agg(
            avg_price=("price", "mean"), houses_count=("zipcode", "count")
        )
        summary_pred = future_df.groupby("zipcode", as_index=False).agg(
            avg_pred_price=("pred_price", "mean"),
        )
        summary = pd.merge(summary_real, summary_pred, on="zipcode", how="outer")
        summary["diff_%"] = (
            (summary["avg_pred_price"] - summary["avg_price"]) / summary["avg_price"]
        ) * 100
        summary.to_csv("outputs/predictions_dashboard.csv", index=False)
        logger.info("✅ 'predictions_dashboard.csv' criado com sucesso!")

    # ========================================================
    # 💾 SALVAR RESULTADOS
    # ========================================================
    clean_df.to_csv("outputs/clean_dataset.csv", index=False)
    selected_df.to_csv("outputs/features_selected.csv", index=False)
    future_df.to_csv("outputs/future_predictions.csv", index=False)
    logger.info("🏁 Processo finalizado com sucesso! Arquivos disponíveis em /outputs.")


# ============================================================
# 6. API FASTAPI PARA CONSULTAS
# ============================================================
app = FastAPI(title="House Price Prediction API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "API de previsão de preços de imóveis (MadeinWeb)"}

@app.get("/predict")
def predict_endpoint():
    df = pd.read_csv("outputs/future_predictions.csv")
    return df.head(10).to_dict(orient="records")


if __name__ == "__main__":
    main()
    uvicorn.run(app, host="0.0.0.0", port=8000)
