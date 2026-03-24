# Sales Forecasting and Demand Prediction with LightGBM + SHAP

A portfolio-ready machine learning project for **daily sales forecasting** with a strong focus on **time-aware validation**, **leakage-safe feature engineering**, and **model explainability**.

This repository shows an end-to-end workflow from raw sales data to forecasting artifacts that can be reused in downstream dashboards or business reporting.

## Why this project matters

Forecasting projects often look good on paper but fail in practice because of two common issues:

- **data leakage** from features that accidentally use future information
- **unreliable validation** that tunes on the final test split instead of a proper time-based backtest

This version addresses both problems and reframes the original PoC into a cleaner portfolio project suitable for:

- **GitHub showcase**
- **CV / internship portfolio**
- **technical interviews**
- **future production hardening**

## Project highlights

- Built a **daily sales forecasting pipeline** using **LightGBM**
- Added **Optuna** for time-series-aware hyperparameter tuning
- Reworked feature engineering to avoid **target leakage**
- Added lag, rolling, EWM, and hierarchical aggregate features
- Used **SHAP** to explain global and local model behavior
- Included a modular training path under `src/optimized/`
- Exported reusable artifacts for prediction and evaluation

## Tech stack

- Python
- Pandas / NumPy
- LightGBM
- Optuna
- SHAP
- Scikit-learn
- Streamlit
- Matplotlib

## Repository structure

```text
sales_forecasting_xai-master/
├── app.py                        # Streamlit UI from the original PoC
├── data/                         # Input data files
│   ├── 2016_sales.csv
│   ├── 2017_sales.csv
│   └── weather_data.csv
├── docs/
│   ├── github_portfolio_guide.md # CV/GitHub presentation notes
│   ├── optimization_audit.md     # What was fixed and why
│   ├── project_description_poc_phase.md
│   └── shap_analysis_summary_report.md
├── figures/                      # EDA and SHAP output images
├── notebooks/                    # Original notebook workflow
│   ├── 01_preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelling.ipynb
│   └── 05_explain_model.ipynb
├── scripts/
│   ├── predict_with_bundle.py    # Predict from a saved model bundle
│   └── train_optimized.py        # Main optimized training entrypoint
├── src/
│   ├── data_generator/
│   ├── data_loader/
│   ├── optimized/
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── features.py
│   │   ├── forecast.py
│   │   ├── metrics.py
│   │   └── model.py
│   ├── ui_builder/
│   ├── ui_predictor/
│   └── utils/
├── .gitignore
├── environment.yml
├── environment_macm1.yml
├── requirements.txt
└── README.md
```

## Problem statement

Given daily sales records for 2016-2017 and external weather data, the objective is to forecast future item-level/store-level demand with a model that is both:

- **accurate enough for operational planning**
- **interpretable enough for stakeholder trust**

## Modeling approach

### 1) Data preparation

- merged raw sales and weather inputs
- standardized date handling
- prepared categorical dimensions such as store, item, category, and province

### 2) Leakage-safe feature engineering

The optimized pipeline avoids using current-day target information when constructing predictors.

Feature groups include:

- calendar features
- lag features: `1 / 7 / 14 / 28`
- rolling statistics
- exponentially weighted moving averages
- hierarchical historical rollups by store, item, category, and province
- categorical features handled natively by LightGBM

### 3) Time-aware tuning and training

Instead of tuning on the final test set, the project uses **time-based validation folds** during Optuna search.

This makes the evaluation more trustworthy and closer to real forecasting behavior.

### 4) Explainability

SHAP is used to inspect:

- which features influence the model most overall
- why a specific prediction goes up or down
- whether the model is driven more by item history, store dynamics, or external factors

## Key improvement over the original PoC

The original notebook-based flow was useful as a proof of concept, but the optimized path improves it in several important ways:

- removes target leakage from aggregate features
- separates tuning from final holdout evaluation
- adds stronger time-series features
- provides a more reusable code layout for repeatable training

More detail is documented in [`docs/optimization_audit.md`](docs/optimization_audit.md).

## Quick start

### 1) Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run a quick smoke test

```bash
python scripts/train_optimized.py --n-trials 3 --ensemble-size 1 --n-splits 2 --validation-days 7 --gap-days 1 --min-train-days 60
```

### 4) Run a stronger training job

```bash
python scripts/train_optimized.py --n-trials 60 --ensemble-size 3
```

## Generated outputs

After training, the following files are written to `models/`:

- `optimized_sales_forecast_bundle.pkl`
- `optimized_metrics.json`
- `optimized_holdout_predictions.csv`
- `optimized_optuna_trials.csv`

## Suggested GitHub presentation

For a strong GitHub profile, pin this repository and describe it as:

> Leakage-safe sales forecasting with LightGBM, Optuna, and SHAP for interpretable retail demand prediction.

Add topics such as:

- `time-series`
- `sales-forecasting`
- `lightgbm`
- `optuna`
- `shap`
- `xai`
- `machine-learning`
- `streamlit`

See [`docs/github_portfolio_guide.md`](docs/github_portfolio_guide.md) for CV bullets and portfolio wording.

## Suggested CV bullets

- Built a leakage-safe daily sales forecasting pipeline using **LightGBM**, **Optuna**, and **SHAP** for interpretable retail demand prediction.
- Redesigned time-series feature engineering with lag, rolling, EWM, and hierarchical history features to improve modeling realism and evaluation trustworthiness.
- Implemented time-based validation and exported reusable forecasting artifacts for downstream analysis and deployment.

## Streamlit note

`app.py` belongs to the original PoC interface. The optimized training path is the most reliable workflow for model evaluation and portfolio presentation.

## Screenshots / visuals

Use the images in `figures/` for your README or portfolio page:

- global feature importance
- dependency plots
- local explanation examples
- category-level sales visuals

## Next steps

Good extensions for future versions:

- probabilistic forecasting
- walk-forward backtesting report
- model registry + experiment tracking
- Docker packaging
- CI checks for training scripts
- API serving for prediction

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

