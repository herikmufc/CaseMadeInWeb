# 🏠 Predição de Preços de Imóveis — Case MadeinWeb

Este projeto implementa um pipeline completo de **Machine Learning + Power BI**, com foco na previsão de preços de imóveis e na comunicação executiva dos resultados.  
A solução combina **modelagem preditiva em Python** e **visualização analítica em Power BI**, refletindo boas práticas de engenharia, versionamento e storytelling de dados.

---

## 📂 Estrutura do Projeto

```
Case - Made in web/
├─ data/
│  └─ raw/
│     ├─ kc_house_data.csv
│     ├─ zipcode_demographics.csv
│     └─ future_unseen_examples.csv
│
├─ outputs/
│  ├─ clean_dataset.csv
│  ├─ dashboard_dataset.csv
│  ├─ features_selected.csv
│  ├─ future_predictions.csv
│  └─ predictions_dashboard.csv
│
├─ models/
│  ├─ best_model.joblib
│  └─ feature_order.json
│
├─ mlruns/
│  └─ ... (experimentos versionados pelo MLflow)
│
├─ src/
│  ├─ data_preprocessing/
│  │  └─ cleaner.py
│  ├─ feature_selection/
│  │  └─ selector.py
│  ├─ models/
│  │  ├─ evaluate.py
│  │  ├─ tune.py
│  │  └─ registry.py
│  └─ api/
│     └─ app.py
│
├─ Madeinweb.pbix
├─ main.py
├─ requirements.txt
└─ README.md
```

**Características da arquitetura**
- Modular: cada função isolada em `src/`.
- Reprodutível: dependências e MLflow para versionamento.
- Integrável: outputs otimizados para Power BI.

---

## 🎯 Objetivo

Prever **preços de imóveis** com base em atributos estruturais e geográficos, entregando:
- Modelagem de regressão robusta e interpretável;
- Métricas de erro e acurácia claras;
- **Dashboard executivo** em Power BI, com análise espacial e comparativo real vs previsto.

---

## ⚙️ Pipeline de Execução

### 1️⃣ Limpeza e Pré-Processamento — `src/data_preprocessing/cleaner.py`
- Conversão de data → ano, mês, dia.  
- Remoção de *outliers* (Z-Score ±3).  
- Padronização seletiva via `StandardScaler` (mantendo `id`, `price`, `zipcode`, datas).  
- Saídas:
  - `outputs/clean_dataset.csv`
  - `outputs/dashboard_dataset.csv`

### 2️⃣ Seleção de Variáveis — `src/feature_selection/selector.py`
- Correlação absoluta com `price`;
- Retém variáveis mais preditivas + `zipcode`;
- Exporta:
  - `outputs/features_selected.csv`
  - `models/feature_order.json`

### 3️⃣ Treinamento e Avaliação — `src/models/evaluate.py`
- Modelos testados:
  - `LinearRegression`
  - `Ridge`
  - `Lasso`
  - `RandomForestRegressor`
- Validação cruzada (RMSE, MAE).
- Seleciona o melhor baseline.

### 4️⃣ Otimização de Hiperparâmetros — `src/models/tune.py`
- Otimização automática via **Optuna**.  
- Parâmetros ajustados: `n_estimators`, `max_depth`.  
- Registro de *trials* no **MLflow** (`mlruns/`).

### 5️⃣ Registro e Versionamento — `src/models/registry.py`
- Exporta o modelo final:
  - `models/best_model.joblib`
- Log completo de métricas e parâmetros no MLflow.

### 6️⃣ Scoring e Integração com BI — `main.py`
- Gera previsões sobre `future_unseen_examples.csv`;
- Exporta:
  - `future_predictions.csv`
  - `predictions_dashboard.csv`
- Este último é usado no Power BI, contendo:
  ```csv
  zipcode, avg_price, avg_pred_price, diff_%, houses_count
  ```

### 7️⃣ API de Predição — `src/api/app.py`
- Implementação em **FastAPI**;
- Endpoint `/predict` para scoring online com `best_model.joblib`.

---

## 🤪 Execução Local

```bash
# Criar ambiente
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Rodar pipeline completo
python main.py

# Abrir interface do MLflow
mlflow ui --port 5000
# Acesse: http://localhost:5000

# API de predição
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
# Acesse: http://localhost:8000/docs
```

---

## 📊 Power BI (Madeinweb.pbix)

As visualizações foram criadas em **duas páginas principais**, utilizando os arquivos do diretório `outputs/`.

---

### 📘 Página 1 — Dashboard de Previsão de Preços (Case)

**Objetivo:** Apresentar a performance geral do modelo e a distribuição espacial das previsões.

**Visuais incluídos:**
- Cards de indicadores:
  - Preço Médio Real (BRL)
  - Preço Médio Previsto (BRL)
  - Diferença Média (%)
  - Total de Regiões
  - Erro Médio Absoluto (MAE)
- Mapa interativo por latitude/longitude
- Gráfico de barras comparando preço real vs previsto por `zipcode`

📷 **Visualização:**
![Dashboard Página 1]([https://imgur.com/gFMbi2M.png[)

---

### 📗 Página 2 — Análise Detalhada

**Objetivo:** Explorar as variáveis e entender os fatores que influenciam o preço.

**Visuais incluídos:**
- **Dispersão:** `sqft_living` × `price` (Color = `grade`)  
  → Mostra a relação direta entre área construída e preço.  
- **Boxplot:** `price` × `bedrooms`  
  → Demonstra o impacto do número de quartos no preço.  
- **Linha:** Média de `price` por `year`  
  → Exibe tendências de valorização ao longo dos anos.

📷 **Visualização:**
![Dashboard Página 2]([https://imgur.com/VraSz4Y.png])

> O layout segue a identidade visual MadeinWeb, com cabeçalho azul e paleta consistente.  
> Todos os indicadores estão formatados em milhar (K) e percentual com duas casas decimais.

---

## 📈 Resultados Principais

| Métrica | Valor |
|----------|--------|
| Modelo vencedor | RandomForestRegressor |
| RMSE médio | ≈ 104.000 |
| MAE | ≈ 278.150 |
| Diferença média (%) | 3.25% |
| Regiões avaliadas | 70 |

**Top variáveis preditivas:**  
`sqft_living`, `grade`, `zipcode`, `bathrooms`, `year`.

---

## 🧭 Decisões Técnicas

- **Padronização seletiva:** preserva variáveis identificadoras e monetárias.  
- **Outputs compatíveis com BI:** colunas renomeadas e agregadas para visualização direta.  
- **Controle de versão:** MLflow garante reprodutibilidade completa dos experimentos.  
- **Pipeline escalável:** estrutura modular em `src/` permite substituição e evolução dos modelos.

---

## 🚀 Próximos Passos

- Monitoramento de *data drift* no MLflow.  
- Teste de modelos avançados (CatBoost, XGBoost).  
- Integração direta com Power BI Service para atualização automática.  
- Deploy da API no ambiente de produção com autenticação JWT.

---

## 👤 Autor

**Herik Ramos**  
📍 Niterói — RJ  
MBA em Ciência de Dados | Especialista em BI e Modelagem Preditiva  
📧 [herikramos.dev@gmail.com](mailto:herikramos@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/herikramos)

---

## 🧾 Créditos

Projeto desenvolvido como parte do **Case Técnico da MadeinWeb & Mobile**, unindo modelagem preditiva e visualização executiva de dados em um mesmo fluxo.
