# Walmart Demand Forecasting — Time-Series Models

Company-wide weekly sales forecasting for Walmart using classical and deep-learning
time-series models: **ARIMA, SARIMA, Prophet, LSTM, and GRU**. Built with data-leakage,
missing-value, and negative-sales issues fixed, and full stationarity/diagnostic
verification performed before every classical model fit.

## Problem Statement

Retailers like Walmart need accurate weekly sales forecasts to plan inventory, staffing,
and promotions. Sales are affected by seasonality, holidays, markdown promotions, and
macro-economic factors. Store/Dept-level data is aggregated into a single total weekly
demand series because ARIMA, SARIMA, and Prophet are univariate (per-series) models —
running 45 stores × 81 departments as 3,645 separate ARIMA models is out of scope. The
models forecast **company-wide weekly demand**, with LSTM/GRU also taking macro/holiday
features as inputs.

## Repository Contents

| File | Description |
|---|---|
| `walmart_demand_forecasting_timeseries.ipynb` | End-to-end Jupyter notebook: data loading, cleaning, EDA, feature engineering, 5 forecasting models, evaluation, and SHAP explainability |
| `Walmart_Demand_Forecasting_Process_Flow_Document.pdf` | Technical blueprint of the full pipeline (ingestion → cleaning → feature engineering → training/validation → prediction/post-processing), with a process-flow diagram |
| `requirements.txt` | Pinned Python dependencies needed to run the notebook |
| `train.csv`, `test.csv`, `features.csv`, `stores.csv` | Source Kaggle Walmart Recruiting datasets (not included — see **Data** below) |

## Data

This project uses the [Walmart Recruiting – Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
Kaggle dataset. Download the following files and place them in the working directory
referenced by the notebook (`/content/` for Colab, or update the paths for local use):

- `train.csv` — historical weekly sales by Store/Dept
- `test.csv` — forecast horizon (Store/Dept/Date, no target)
- `features.csv` — Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment, IsHoliday
- `stores.csv` — Store Type and Size

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then open `walmart_demand_forecasting_timeseries.ipynb` in Jupyter, JupyterLab, or
Google Colab and run all cells top to bottom. The notebook is self-contained and
executes without errors given the four CSVs above.

## Pipeline Overview

1. **Data Ingestion** — load and merge `train`/`test` with `features` (on Store, Date,
   IsHoliday) and `stores` (on Store).
2. **Data Cleaning** — negative `Weekly_Sales` (0.305% of rows) investigated and clipped
   at 0; `MarkDown1–5` NaNs filled with 0 *before* aggregation; duplicates checked.
3. **EDA** — sales trends by store/department/time, holiday effect, correlation matrix,
   distribution of the target.
4. **Weekly Aggregation & Feature Engineering** — Store/Dept rows collapsed into one
   company-wide weekly series (`y`); lag features (1, 2, 4, 52 weeks), rolling
   mean/std (4, 12 weeks), holiday-proximity, store-level aggregates, and interaction
   terms (`temp_x_fuel`, `markdown_x_holiday`).
5. **Chronological Train/Test Split** — last 13 weeks held out; no random shuffling.
6. **Statistical Verification** — seasonal decomposition, ADF/KPSS stationarity tests,
   ACF/PACF, all computed on the training set only.
7. **Model Training** — ARIMA and SARIMA via `pmdarima.auto_arima` (AIC-optimized order
   search); Prophet with a custom holiday calendar and exogenous regressors; LSTM and
   GRU on MinMax-scaled multivariate sliding windows (`LOOKBACK = 8` weeks), scaler fit
   on train only.
8. **Evaluation** — MAE, RMSE, MAPE, and R² computed on the 13-week holdout for every
   model, plus Ljung-Box residual diagnostics for ARIMA/SARIMA.
9. **Model Explainability** — SHAP summary, bar, waterfall, dependence, and decision
   plots for the exogenous drivers of the best model (ARIMA).

## Results (13-week holdout)

| Model   | MAE          | RMSE         | MAPE (%) | R²    |
|---------|-------------:|-------------:|---------:|------:|
| ARIMA   | 1,388,939.53 | 1,767,506.77 | 3.07     | -0.44 |
| SARIMA  | 1,464,710.11 | 1,928,347.35 | 3.24     | -0.72 |
| LSTM    | 1,787,817.33 | 2,163,275.10 | 3.94     | -1.16 |
| Prophet | 1,913,532.62 | 2,318,128.28 | 4.14     | -1.48 |
| GRU     | 2,020,551.30 | 2,321,596.32 | 4.45     | -1.49 |

**ARIMA(2,0,2)** is the best-performing model on this holdout across every metric.
See the process flow document and notebook for trade-off analysis, residual diagnostics,
and business interpretation.

## Notes on Leakage & Validity

- No target-derived aggregate (e.g., a per-store average sales column) is computed
  before the train/test split.
- Every scaler, holiday list, and decomposition is fit on the training portion only,
  then applied to the test portion.
- The train/test split is strictly chronological (last 13 weeks held out) — never
  randomized — since a random split would leak future trend/seasonality into training.

## License / Data Attribution

Data © Walmart, distributed via the Kaggle "Walmart Recruiting – Store Sales
Forecasting" competition. This repository contains only code and documentation; the
raw CSVs must be downloaded separately from Kaggle under its terms of use.
