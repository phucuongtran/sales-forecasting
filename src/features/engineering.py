"""Step 2 of the pipeline: turn cleaned sales + weather into model features.

    python -m src.pipeline --steps features

Each feature family is a small function that takes and returns a DataFrame, so
the notebook (03_feature_engineering) can call and inspect them one by one,
while ``build_features`` chains them for the pipeline.

The column order produced here is the column order the model is trained on.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data import io

GROUP_KEY = "store_item"  # one time series per (store, item)


# --------------------------------------------------------------------------- #
# Base table
# --------------------------------------------------------------------------- #
def mark_test_period(df: pd.DataFrame, test_start: str = config.TEST_START_DATE) -> pd.DataFrame:
    """Add ``is_test``: True for rows in the hold-out period (Oct-Dec 2017)."""
    df = df.copy()
    df["is_test"] = df["date"] >= pd.Timestamp(test_start)
    return df


def merge_weather(sales: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach temperature, humidity and season of the store's province."""
    columns = ["date", "province", "temperature", "humidity", "season"]
    return sales.merge(weather[columns], on=["date", "province"], how="left")


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = df["date"].dt
    df["year"] = dates.year
    df["month"] = dates.month
    df["day"] = dates.day
    df["day_of_week"] = dates.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = dates.quarter
    df["is_holiday"] = df["date"].isin(pd.to_datetime(config.HOLIDAYS)).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def temperature_category(temperature: pd.Series) -> pd.Series:
    return pd.cut(temperature, bins=config.TEMPERATURE_BINS, labels=config.TEMPERATURE_LABELS)


def humidity_level(humidity: pd.Series) -> pd.Series:
    return pd.cut(humidity, bins=config.HUMIDITY_BINS, labels=config.HUMIDITY_LABELS)


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket temperature / humidity and one-hot encode them together with season."""
    df = df.copy()
    df["temp_category"] = temperature_category(df["temperature"])
    df["humidity_level"] = humidity_level(df["humidity"])
    return pd.get_dummies(
        df, columns=["temp_category", "humidity_level", "season"], drop_first=True
    )


# --------------------------------------------------------------------------- #
# Sales history (computed per store-item series, always shifted by >= 1 day)
# --------------------------------------------------------------------------- #
def add_series_key(df: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically and add the ``store_item`` series identifier."""
    df = df.sort_values(["date", "store_name", "item_name"]).copy()
    df[GROUP_KEY] = df["store_name"] + "_" + df["item_name"]
    return df


def add_lag_features(df: pd.DataFrame, lags=config.LAG_DAYS) -> pd.DataFrame:
    """``sales_lag_n``: sales of the same store-item n days earlier."""
    df = df.copy()
    grouped = df.groupby(GROUP_KEY)["sales"]
    for n in lags:
        df[f"sales_lag_{n}"] = grouped.shift(n)
    return df


def add_rolling_features(df: pd.DataFrame, windows=config.ROLLING_WINDOWS) -> pd.DataFrame:
    """Mean / min / max / std of the previous ``window`` days (yesterday excluded today)."""
    df = df.copy()
    grouped = df.groupby(GROUP_KEY)["sales"]
    for window in windows:
        for stat in ("mean", "min", "max", "std"):
            df[f"sales_{stat}_{window}d"] = grouped.transform(
                lambda s, w=window, f=stat: getattr(
                    s.shift(1).rolling(window=w, min_periods=1), f
                )()
            )
    return df


def add_ewma_features(
    df: pd.DataFrame, alphas=config.EWMA_ALPHAS, windows=config.EWMA_WINDOWS
) -> pd.DataFrame:
    """Exponentially weighted moving average of past sales.

    NOTE (kept from the original PoC for reproducibility): ``ewm(alpha=...)``
    does not use a window, so the 7d/14d/28d columns for the same alpha are
    identical. See REFACTORING.md -> "Known issues".
    """
    df = df.copy()
    grouped = df.groupby(GROUP_KEY)["sales"]
    for alpha in alphas:
        ewma = grouped.transform(lambda s, a=alpha: s.shift(1).ewm(alpha=a).mean())
        alpha_str = str(alpha).replace(".", "")
        for window in windows:
            df[f"sales_ewma_{window}d_a{alpha_str}"] = ewma
    return df


def add_store_item_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Store-level and item-level aggregates.

    WARNING (kept from the original PoC): despite the ``_7d`` suffix these are
    aggregates over the *same day* and include the target itself, i.e. they leak
    the answer into the features. See REFACTORING.md -> "Known issues".
    """
    df = df.copy()
    by_store_day = df.groupby(["store_name", "date"])["sales"]
    by_item_day = df.groupby(["item_name", "date"])["sales"]
    df["store_mean_7d"] = by_store_day.transform("mean")
    df["store_sum_7d"] = by_store_day.transform("sum")
    df["item_mean_7d"] = by_item_day.transform("mean")
    df["item_sum_7d"] = by_item_day.transform("sum")
    return df


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def build_features(sales: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature step and drop the warm-up rows that have NaN lags."""
    df = mark_test_period(sales)
    df = merge_weather(df, weather)
    df = add_calendar_features(df)
    df = add_weather_features(df)
    df = add_series_key(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_ewma_features(df)
    df = add_store_item_aggregates(df)
    return df.dropna()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns used as model inputs (everything except ids, flags and target)."""
    excluded = set(config.NON_FEATURE_COLUMNS) | {config.TARGET}
    return [col for col in df.columns if col not in excluded]


def run() -> pd.DataFrame:
    """Load preprocessed data, build features and save them as Feather."""
    features = build_features(io.load_preprocessed_sales(), io.load_preprocessed_weather())
    io.save_dataframe(features, config.FEATURES_FILE)
    print(f"{features.shape[1]} columns, {len(feature_columns(features))} model features")
    return features
