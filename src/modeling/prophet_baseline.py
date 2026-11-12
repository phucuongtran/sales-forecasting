"""Prophet baseline: one Prophet model per store-item series.

Prophet is an optional dependency (installed by ``environment.yml`` but not by
``requirements.txt``), so it is imported lazily.
"""

from __future__ import annotations

import pandas as pd

from src.modeling.metrics import score, wape

REGRESSORS = ["is_weekend", "is_holiday", "temperature"]


def to_prophet_frame(features: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, store_item) in Prophet's ``ds``/``y`` format."""
    return (
        features.groupby(["date", "store_item"])
        .agg(
            y=("sales", "sum"),
            is_weekend=("is_weekend", "first"),
            is_holiday=("is_holiday", "first"),
            temperature=("temperature", "first"),
            humidity=("humidity", "first"),
        )
        .reset_index()
        .rename(columns={"date": "ds"})
    )


def fit_predict_series(series: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame | None:
    """Fit Prophet on one series before ``test_start`` and forecast the rest."""
    from prophet import Prophet

    train = series[series["ds"] < test_start]
    test = series[series["ds"] >= test_start]
    if train.empty or test.empty:
        return None

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="multiplicative",
    )
    for regressor in REGRESSORS:
        model.add_regressor(regressor)
    model.fit(train)

    forecast = model.predict(test[["ds", *REGRESSORS]])
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(test[["ds", "y"]], on="ds")


def run_prophet_baseline(features: pd.DataFrame):
    """Forecast every series; return (all forecasts, per-series metrics, overall metrics).

    Overall MAE/RMSE are the mean of the per-series values (as in the original
    notebook); overall WAPE is computed over all forecasts pooled together.
    """
    data = to_prophet_frame(features)
    test_start = features.loc[features["is_test"], "date"].min()

    forecasts, rows = [], []
    for store_item, series in data.groupby("store_item"):
        try:
            forecast = fit_predict_series(series, test_start)
        except Exception as exc:  # Prophet occasionally fails on a series
            print(f"Prophet failed for {store_item}: {exc}")
            continue
        if forecast is None:
            continue

        forecast["store_item"] = store_item
        forecasts.append(forecast)
        rows.append({"store_item": store_item, **vars(score(forecast["y"], forecast["yhat"]))})

    all_forecasts = pd.concat(forecasts, ignore_index=True)
    per_series = pd.DataFrame(rows)
    overall = {
        "mae": per_series["mae"].mean(),
        "rmse": per_series["rmse"].mean(),
        "wape": wape(all_forecasts["y"], all_forecasts["yhat"]),
    }
    return all_forecasts, per_series, overall
