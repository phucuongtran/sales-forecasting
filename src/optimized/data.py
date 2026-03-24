from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_raw_inputs(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw or preprocessed sales/weather inputs from the repository."""
    sales_preprocessed = os.path.join(data_dir, "sales_data_preprocessed.csv")
    weather_preprocessed = os.path.join(data_dir, "weather_preprocessed.csv")
    sales_2016 = os.path.join(data_dir, "2016_sales.csv")
    sales_2017 = os.path.join(data_dir, "2017_sales.csv")
    weather_raw = os.path.join(data_dir, "weather_data.csv")

    if os.path.exists(sales_preprocessed):
        sales = pd.read_csv(sales_preprocessed, parse_dates=["date"])
    elif os.path.exists(sales_2016) and os.path.exists(sales_2017):
        sales = pd.concat(
            [
                pd.read_csv(sales_2016, parse_dates=["date"]),
                pd.read_csv(sales_2017, parse_dates=["date"]),
            ],
            ignore_index=True,
        )
    else:
        raise FileNotFoundError(
            "Missing sales inputs. Expected sales_data_preprocessed.csv or 2016_sales.csv + 2017_sales.csv"
        )

    if os.path.exists(weather_preprocessed):
        weather = pd.read_csv(weather_preprocessed, parse_dates=["date"])
    elif os.path.exists(weather_raw):
        weather = pd.read_csv(weather_raw, parse_dates=["date"])
    else:
        raise FileNotFoundError(
            "Missing weather inputs. Expected weather_preprocessed.csv or weather_data.csv"
        )

    return sales, weather


def _clip_group_outliers(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return series.fillna(series.median())
    lower = max(0.0, q1 - 3.0 * iqr)
    upper = q3 + 3.0 * iqr
    return series.clip(lower, upper)


def prepare_base_frame(sales: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Clean the base sales table and merge weather."""
    sales = sales.copy()
    weather = weather.copy()

    sales["date"] = pd.to_datetime(sales["date"])
    weather["date"] = pd.to_datetime(weather["date"])

    if "city" in weather.columns and "province" not in weather.columns:
        weather = weather.rename(columns={"city": "province"})

    required_sales_cols = [
        "date",
        "province",
        "store_id",
        "store_name",
        "category",
        "item_id",
        "item_name",
        "sales",
    ]
    missing_cols = [c for c in required_sales_cols if c not in sales.columns]
    if missing_cols:
        raise ValueError(f"Sales data missing required columns: {missing_cols}")

    sales = sales[required_sales_cols].copy()
    sales["sales"] = pd.to_numeric(sales["sales"], errors="coerce")

    sales = sales.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    sales["store_item_id"] = (
        sales["store_id"].astype(str) + "__" + sales["item_id"].astype(str)
    )

    # Missing values: per series interpolation, then series median, then global median.
    sales["sales"] = sales.groupby("store_item_id")["sales"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    sales["sales"] = sales.groupby("store_item_id")["sales"].transform(
        lambda s: s.fillna(s.median())
    )
    sales["sales"] = sales["sales"].fillna(sales["sales"].median()).clip(lower=0)

    # Outlier clipping within each store-item series.
    sales["sales"] = sales.groupby("store_item_id")["sales"].transform(_clip_group_outliers)

    weather_cols = [c for c in ["date", "province", "temperature", "humidity", "season"] if c in weather.columns]
    weather = weather[weather_cols].drop_duplicates(["date", "province"])

    if "temperature" in weather.columns:
        weather["temperature"] = pd.to_numeric(weather["temperature"], errors="coerce")
    if "humidity" in weather.columns:
        weather["humidity"] = pd.to_numeric(weather["humidity"], errors="coerce")

    base = sales.merge(weather, on=["date", "province"], how="left")
    for col in ["temperature", "humidity"]:
        if col in base.columns:
            base[col] = base.groupby("province")[col].transform(
                lambda s: s.interpolate(limit_direction="both")
            )
            base[col] = base[col].fillna(base[col].median())

    if "season" in base.columns:
        base["season"] = base["season"].fillna("unknown")

    return base.drop(columns=["store_item_id"])
