# Retail Demand Forecasting — Walmart Store Sales (Time-Series Models)

Forecasting company-wide weekly demand for Walmart using classical and deep-learning
time-series models — **ARIMA, SARIMA, Prophet, LSTM, GRU** — with data-leakage,
missing-value, and negative-sales issues fixed, and full stationarity/diagnostic
verification performed before each model is fit.

## Contents

| File | Description |
|---|---|
| `vertopal_com_walmart_demand_forecasting_timeseries__1_.pdf` | Source Colab notebook export (code, output, plots) — the analysis this project is based on. |
| `Walmart_Demand_Forecasting_Technical_Report.pdf` | Technical blueprint & evaluation report: process flow diagram, model selection reasoning, evaluation/validation methodology, and explainability writeup. |
| `README.md` | This file. |

## Problem Statement

Retailers need accurate weekly sales forecasts to plan inventory, staffing, and
promotions. Sales are affected by seasonality, holidays, markdown promotions, and
macroeconomic factors. Store/Dept-level data (45 stores × ~81 departments) is
aggregated into a single **company-wide weekly demand series**, since ARIMA, SARIMA,
and Prophet are univariate models — running 3,645 separate per-series models is out
of scope. LSTM/GRU forecast the same aggregate series but additionally take
macro/holiday features as extra input channels.

## Data

Source: [Walmart Recruiting — Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) (Kaggle)

- `train.csv` — historical Store × Dept × Week sales (target: `Weekly_Sales`)
- `stores.csv` — store `Type` (A/B/C) and `Size`
- `features.csv` — `Temperature`, `Fuel_Price`, `MarkDown1–5`, `CPI`, `Unemployment`, `IsHoliday`
- `test.csv` — competition's future grid (not used for training here; an internal
  chronological holdout is used instead)

## Pipeline Summary

1. **Ingestion** — merge `train` + `features` + `stores` on `Store`/`Date`/`IsHoliday`; validate dtypes, nulls, duplicates.
2. **Cleaning** — negative `Weekly_Sales` (0.305% of rows) investigated and clipped at 0 (not dropped, to preserve the weekly calendar); `MarkDown1–5` NaNs filled with 0 **before** aggregation; explicit zero-NaN assertion.
3. **Feature engineering** — `Total_MarkDown` = sum of markdown columns; row-level data aggregated to a 143-week company-wide series (`y`, plus mean/sum-aggregated exogenous features). No target-derived aggregate is included anywhere — the leakage source (`Store_Avg_Sales`-style features) from an earlier regression notebook is fully removed.
4. **Chronological split** — last 13 weeks held out as test (130 train / 13 test weeks); all scalers/holiday lists fit on train only.
5. **Statistical verification** — seasonal decomposition, ADF + KPSS stationarity tests, ACF/PACF plots, all computed on train only.
6. **Modeling** — ARIMA and SARIMA via `pmdarima.auto_arima`; Prophet with a holiday calendar and 5 regressors; 2-layer LSTM and GRU on 8-week lookback windows of scaled multivariate sequences.
7. **Evaluation** — MAE, RMSE, MAPE, R² on the held-out 13 weeks; Ljung-Box residual test for ARIMA/SARIMA.

## Results (13-week holdout, ranked by RMSE)

| Rank | Model | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|---|
| 1 | **ARIMA(1,0,0)** | 1,388,939.53 | 1,767,506.77 | 3.07 | -0.44 |
| 2 | LSTM | 1,489,494.16 | 1,865,483.60 | 3.29 | -0.61 |
| 3 | SARIMA(2,0,2)(2,0,1)[13] | 1,464,710.11 | 1,928,347.35 | 3.24 | -0.72 |
| 4 | GRU | 1,560,700.72 | 2,135,763.41 | 3.47 | -1.10 |
| 5 | Prophet | 1,913,532.62 | 2,318,128.28 | 4.14 | -1.48 |

**Best model: ARIMA(1,0,0)** on point-accuracy metrics — but it fails the Ljung-Box
white-noise test on residuals (p=0.0016), meaning it under-fits the holiday spikes;
SARIMA passes this test (p=0.187). See the technical report for the full trade-off
discussion and the recommendation to treat ARIMA's holiday-week intervals with caution.

## Key Takeaways

- Aggregating to a single company-wide series removes the store/dept-level leakage problem entirely.
- All models post negative R² on this holdout — expected, since the calm Aug–Oct test window has low variance relative to the holiday-dominated training period; MAE/RMSE/MAPE are the more trustworthy metrics here.
- ARIMA/SARIMA are fast, interpretable baselines; Prophet's decomposition is the most business-legible; LSTM/GRU need more than ~130 training weeks to reliably beat the classical models.

## Limitations & Next Steps

- Add a **WMAE** metric (5× weight on holiday weeks, the Kaggle competition's own metric) to re-rank models on the weeks that matter most operationally.
- Run an **expanding-window backtest** (not just a single 13-week holdout) to confirm ARIMA's win is stable.
- Segment evaluation by **holiday vs. non-holiday**, **store Type**, and **markdown intensity** to check for uneven performance (see Model Fairness section of the report).
- Extend to a **global LSTM/GRU with Store/Dept embeddings** to forecast at the granularity retailers actually plan against, and to enable SHAP-based feature attribution.

## Reproducing This Work

Open the source notebook in Google Colab or Jupyter with the four Kaggle CSVs placed
at `/content/*.csv` (or update the paths), then run cells top-to-bottom. Key
dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `pmdarima`,
`prophet`, `tensorflow`, `scikit-learn`.

```bash
pip install pandas numpy matplotlib seaborn statsmodels pmdarima prophet tensorflow scikit-learn
```

For the full technical writeup — process-flow diagram, best-model reasoning,
evaluation methodology, and explainability plan — see
`Walmart_Demand_Forecasting_Technical_Report.pdf`.
