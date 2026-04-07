# S&P 500 Market Intelligence System

> Forecast S&P 500 price direction and generate data-driven sector rotation
> recommendations using machine learning and macroeconomic regime detection.

[![Python](https://img.shields.io/badge/Python-3.12-green?style=for-the-badge&logo=python)](https://python.org)
[![Darts](https://img.shields.io/badge/Time%20Series-Darts-blue?style=for-the-badge)](https://unit8co.github.io/darts/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Deployed-Docker-2496ED?style=for-the-badge&logo=docker)](https://docker.com)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?style=for-the-badge)](https://xgboost.readthedocs.io)

---

## What is the S&P 500 Market Intelligence System

This is a two-layer quantitative market intelligence system that forecasts
S&P 500 price direction 30 days ahead and recommends which of the 11 S&P 500
sector ETFs to overweight or underweight based on current market conditions.
Unlike rule-based systems, every recommendation is driven entirely by model
predictions, historical regime analysis, and risk-adjusted scoring — not
manual rules or gut feeling.

The system combines time series forecasting, SHAP-based feature selection,
Bayesian hyperparameter optimization, K-Means market regime detection, and
a 5-factor sector scoring model into a unified pipeline that generates 339
weekly recommendations across a 7-year backtesting period with no
look-ahead bias.

---

## Key Results

| Metric | Value |
|--------|-------|
| Best Validation MAPE | 0.65% (Stacking Ensemble) |
| Improvement Over Baseline | 65.8% vs NaiveSeasonal |
| In-Regime Direction Accuracy | 76.6% (2023 walk forward windows) |
| Market Regimes Detected | 5 (automatic via silhouette score) |
| Weekly Recommendations | 339 pre-computed 2018-2024 |
| Total Models Trained | 18 across 6 families |
| SHAP Selected Features | 7 from 46 candidates |

---

## System Architecture
Raw Data (S&P 500 + 11 Sector ETFs + 16 Macro Indicators)
↓
Data Collection and Preprocessing
yfinance + FRED API + XLC reconstruction
Business day alignment + release-aware forward fill
↓
Feature Engineering (46 features)
Technical indicators + Macro features
Calendar features + Event flags (FOMC CPI holidays)
↓
SHAP Feature Selection (46 → 7 optimal features)
Correlation filter + SHAP importance threshold
↓
┌─────────────────────────────┐    ┌─────────────────────────────────┐
│  Layer 1 — Price Forecast   │    │  Layer 2 — Sector Rotation      │
│                             │    │                                  │
│  18 Models across 6 families│    │  K-Means Regime Detection        │
│  Optuna Hyperparameter Tune │───▶│  5 Market Regimes (silhouette)  │
│  Stacking Ensemble          │    │  5-Factor Composite Scoring      │
│  MAPE 0.65%                 │    │  Ridge Regression Weights        │
└─────────────────────────────┘    └─────────────────────────────────┘
↓
339 Weekly Recommendations (2018-2024)
No look-ahead bias — simulates real deployment

---

## Features

### Layer 1 — S&P 500 Price Forecasting
- **18 models** across 6 families: 4 baseline, 7 statistical,
  1 probabilistic, 5 ML, and 1 Prophet
- **SHAP feature selection**: reduced 46 features to 7 optimal
  macro and technical indicators improving MAPE from 1.10% to 0.95%
  and R2 from -0.14 to +0.21
- **Optuna Bayesian optimization**: TPESampler with 20 trials per
  model improved XGBoost by 63% from MAPE 1.88% to 0.70%
- **Stacking ensemble**: Ridge Regression meta-model achieves
  MAPE 0.65% — 7.09% better than the best individual model

### Layer 2 — Intelligent Sector Rotation
- **K-Means regime detection**: silhouette score automatically
  selected k=5 market regimes across 2018-2024
- **5 detected regimes**: Bear Market High Volatility Moderate
  Growth Recovery and Stimulus Bull Market
- **5-factor composite scoring**: historical regime return,
  momentum, Sharpe ratio, VIX alignment, and forecast alignment
- **Ridge Regression weights**: learned optimal factor combination
  from last 120 days of historical data
- **Regime attribution analysis**: reveals that the same regime
  label can produce different sector outcomes depending on macro
  context within the regime
- **339 pre-computed weekly recommendations**: no look-ahead bias,
  simulates real-time deployment from 2018 to 2024

---

## Dashboard

The interactive Streamlit dashboard brings the entire system together:

- **Market Timeline** — S&P 500 price with regime background
  shading across 2018-2024 with VIX overlay and key event annotations
- **Timeline Explorer** — Select any date range and see week-by-week
  regime changes sector recommendations and what actually happened
- **Regime Deep Dive** — Full 2018-2024 analysis of all 5 regimes
  showing which sectors outperformed the S&P 500 in each environment
  and the macro conditions that defined each regime
- **Model Performance** — Honest walk forward validation across 12
  windows showing both in-regime success and out-of-distribution
  limitations

---

## Walk Forward Validation

| Period | Avg MAPE | Avg Direction Accuracy |
|--------|----------|----------------------|
| 2023 (in-regime) | 2.78% | 76.6% ✅ |
| 2024 (AI boom out-of-distribution) | 17.7% | 18.4% ❌ |
| Overall (12 windows) | 12.68% | 38.8% |

The 2024 performance degradation reflects the AI-driven bull market
regime not represented in training data — validating the need for
regime confidence scoring. When the system detects an unfamiliar
macro environment it signals lower confidence allowing users to
reduce position sizes accordingly.

---

## Data Sources

| Source | Data | Period |
|--------|------|--------|
| Yahoo Finance | S&P 500 price (^GSPC) | 2018-2024 |
| Yahoo Finance | 11 sector ETFs (XLK XLV XLF XLE XLY XLP XLI XLU XLB XLRE XLC) | 2018-2024 |
| Yahoo Finance | VIX TNX IRX OIL DXY | 2018-2024 |
| FRED API | CPI FED_RATE UNEMPLOYMENT PPI RETAIL_SALES and others | 2018-2024 |

**Note:** XLC was reconstructed from 16 verified SPDR constituents
for the pre-launch period before June 2018 achieving 0.9566 return
correlation with the actual ETF.

---

## Feature Engineering

46 features engineered across 4 categories:

| Category | Features |
|----------|---------|
| Technical | MA5 MA20 MA50 EMA12 EMA26 MACD RSI BB_UPPER BB_LOWER |
| Macro | YIELD_CURVE YIELD_CURVE_INVERTED VIX_MA20 OIL_MOM30 CPI_MOM NFP_MOM |
| Calendar | DAY_OF_WEEK MONTH QUARTER |
| Events | IS_FOMC_DATE IS_CPI_RELEASE IS_HOLIDAY_ADJACENT |

SHAP analysis identified 7 optimal features:
**RSI MACD VIX VIX_MA20 BREAKEVEN OIL OIL_MOM30**

---

## External Factor Impact Analysis

| Model | FOMC Days | CPI Days | Holiday Adjacent |
|-------|-----------|----------|-----------------|
| RandomForest | 73% WORSE ❌ | 7% better | 78% better |
| LightGBM | 44% BETTER ✅ | 21% better | 76% better |
| XGBoost | 55% better | 2% better | 7% better |

LightGBM correctly uses FOMC and CPI event flags as predictive
signals — validating our event flag feature engineering.

---

## Quick Start
```bash
# Clone repository.
git clone https://github.com/RushilJoshi07/sp500-market-intelligence-system.git
cd sp500-market-intelligence-system

# Create virtual environment.
python -m venv venv
source venv/bin/activate

# Install dependencies.
pip install -r requirements.txt

# Set FRED API key.
export FRED_API_KEY=your_api_key_here

# Run dashboard.
streamlit run dashboard/app.py
```

### Docker
```bash
# Build image.
docker build -t sp500-intelligence .

# Run container.
docker run -p 8501:8501 \
    -e FRED_API_KEY=your_api_key_here \
    sp500-intelligence
```

---

## Project Structure
sp500-market-intelligence-system/
├── notebook/
│   ├── darts.example.ipynb     # Full analysis notebook (13 phases)
│   └── darts.example.py        # Python version
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   ├── data_loader.py          # Data loading functions
│   └── charts.py               # Plotly chart functions
├── data/
│   └── dashboard/              # Pre-computed results (CSVs)
│       ├── weekly_recommendations.csv
│       ├── walk_forward_results.csv
│       ├── attribution_df.csv
│       ├── sector_scores.csv
│       ├── regime_stats.csv
│       ├── regime_labels.csv
│       ├── sp500.csv
│       ├── sectors.csv
│       └── macro_daily.csv
├── utils.py                    # 45 helper functions
├── requirements.txt            # Pinned dependencies
└── Dockerfile                  # Docker configuration

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Time Series Library | Darts 0.30.0 |
| ML Models | XGBoost LightGBM CatBoost RandomForest |
| Statistical Models | ARIMA AutoARIMA ExponentialSmoothing Theta TBATS |
| Hyperparameter Tuning | Optuna 3.6.1 (Bayesian TPESampler) |
| Feature Explainability | SHAP |
| Regime Detection | Scikit-learn K-Means + Silhouette Score |
| Sector Scoring | Ridge Regression |
| Dashboard | Streamlit + Plotly |
| Data Sources | yfinance + FRED API |
| Containerization | Docker |

---

## What I Learned Building This

The most important finding was that feature quality matters more than
feature quantity. Using all 40 macro and technical features produced
MAPE 1.10% with R2 -0.14. Using only 7 SHAP-selected features
produced MAPE 0.95% with R2 +0.21. Adding irrelevant features
actively hurts ML model performance by diluting the signal.

The most important limitation discovered was that macro-driven ML
models struggle during unprecedented regime transitions. The 2024
AI boom created a market environment not represented in training data
where traditional macro signals were overwhelmed by AI earnings
surprises. This validated the core design decision to build regime
detection with confidence scoring — the system correctly signals
uncertainty when it detects an unfamiliar macro environment.

---

## Author

**Rushil Joshi**

[GitHub](https://github.com/RushilJoshi07)
