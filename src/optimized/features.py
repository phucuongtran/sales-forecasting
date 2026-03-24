from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
except Exception:  # pragma: no cover - fallback if dependency missing
    holidays_lib = None

from .config import OptimizedTrainingConfig


def _make_holiday_set(years: Iterable[int]) -> set:
    if holidays_lib is None:
        return set()
    try:
        vn_holidays = holidays_lib.country_holidays("VN", years=list(years))
        return {pd.Timestamp(d).normalize() for d in vn_holidays.keys()}
    except Exception:
        return set()


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    holiday_dates = _make_holiday_set(df["year"].unique())
    if holiday_dates:
        df["is_holiday"] = df["date"].dt.normalize().isin(holiday_dates).astype(int)
    else:
        df["is_holiday"] = 0

    # Cyclical encodings.
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "temperature" in df.columns:
        df["temp_x_weekend"] = df["temperature"] * df["is_weekend"]
        df["temp_bin"] = pd.cut(
            df["temperature"], bins=[-100, 15, 25, 100], labels=["cool", "mild", "hot"]
        ).astype(str)
    if "humidity" in df.columns:
        df["humidity_bin"] = pd.cut(
            df["humidity"], bins=[-1, 40, 70, 101], labels=["low", "mid", "high"]
        ).astype(str)
        if "temperature" in df.columns:
            df["temp_humidity_ratio"] = _safe_ratio(df["temperature"], df["humidity"])
    return df


def _add_series_features(df: pd.DataFrame, config: OptimizedTrainingConfig) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    group = df.groupby(["store_id", "item_id"], sort=False)["sales"]

    for lag in config.lag_days:
        df[f"sales_lag_{lag}"] = group.shift(lag)

    for window in config.rolling_windows:
        df[f"sales_mean_{window}d"] = group.transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"sales_min_{window}d"] = group.transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).min()
        )
        df[f"sales_max_{window}d"] = group.transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).max()
        )
        df[f"sales_std_{window}d"] = group.transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).std()
        ).fillna(0.0)

    for alpha in config.ewm_alphas:
        alpha_key = str(alpha).replace(".", "")
        df[f"sales_ewm_a{alpha_key}"] = group.transform(
            lambda s, alpha=alpha: s.shift(1).ewm(alpha=alpha, adjust=False).mean()
        )

    if "sales_lag_1" in df.columns and "sales_mean_7d" in df.columns:
        df["lag1_vs_mean7_ratio"] = _safe_ratio(df["sales_lag_1"], df["sales_mean_7d"])
    if "sales_mean_7d" in df.columns and "sales_mean_28d" in df.columns:
        df["trend_7_over_28"] = _safe_ratio(df["sales_mean_7d"], df["sales_mean_28d"])
    return df


def _merge_daily_rollups(
    df: pd.DataFrame,
    level_cols: List[str],
    prefix: str,
    windows: List[int],
) -> pd.DataFrame:
    daily = (
        df.groupby(level_cols + ["date"], as_index=False)["sales"].sum().sort_values(level_cols + ["date"])
    )
    grp = daily.groupby(level_cols, sort=False)["sales"]
    daily[f"{prefix}_lag_1"] = grp.shift(1)
    for window in windows:
        daily[f"{prefix}_mean_{window}d"] = grp.transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        daily[f"{prefix}_sum_{window}d"] = grp.transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).sum()
        )
    return df.merge(daily.drop(columns=["sales"]), on=level_cols + ["date"], how="left")


def build_feature_frame(base_df: pd.DataFrame, config: OptimizedTrainingConfig) -> pd.DataFrame:
    """Build a leakage-safe feature frame suitable for LightGBM forecasting."""
    df = base_df.copy().sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)
    df = _add_calendar_features(df)
    df = _add_weather_features(df)
    df = _add_series_features(df, config)

    # Hierarchical leakage-safe rollups (all shifted by 1 day before rolling).
    df = _merge_daily_rollups(df, ["store_id"], "store", config.global_windows)
    df = _merge_daily_rollups(df, ["item_id"], "item", config.global_windows)
    df = _merge_daily_rollups(df, ["category"], "category", config.global_windows)
    df = _merge_daily_rollups(df, ["province"], "province", config.global_windows)

    if {"store_mean_7d", "item_mean_7d"}.issubset(df.columns):
        df["store_item_balance"] = _safe_ratio(df["item_mean_7d"], df["store_mean_7d"])
    if {"item_sum_7d", "category_sum_7d"}.issubset(df.columns):
        df["item_share_in_category_7d"] = _safe_ratio(
            df["item_sum_7d"], df["category_sum_7d"]
        )

    cutoff_ts = pd.Timestamp(config.cutoff_date)
    df["is_test"] = df["date"] >= cutoff_ts

    # Drop rows where lag features are unavailable.
    essential = ["sales_lag_1", "sales_mean_7d", "sales_mean_28d"]
    df = df.dropna(subset=[c for c in essential if c in df.columns]).reset_index(drop=True)

    # Fill remaining numerical NaNs caused by ratios or sparse rollups.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fill_cols = [c for c in num_cols if c not in ["sales"]]
    df[fill_cols] = df[fill_cols].replace([np.inf, -np.inf], np.nan)
    df[fill_cols] = df[fill_cols].fillna(0.0)

    # Cast categorical columns for native LightGBM handling.
    for col in ["province", "category", "season", "temp_bin", "humidity_bin"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
