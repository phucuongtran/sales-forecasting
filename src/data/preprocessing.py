"""Step 1 of the pipeline: clean the raw sales and weather data.

    python -m src.pipeline --steps preprocess
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data import io


def fill_missing_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing ``sales`` values with the global mean of sales."""
    df = df.copy()
    df["sales"] = df["sales"].fillna(df["sales"].mean())
    return df


def correct_outliers(
    df: pd.DataFrame, z_threshold: float = config.OUTLIER_Z_THRESHOLD
) -> pd.DataFrame:
    """Replace sales whose |z-score| exceeds ``z_threshold`` by the global mean."""
    df = df.copy()
    mean, std = df["sales"].mean(), df["sales"].std()
    is_outlier = np.abs((df["sales"] - mean) / std) > z_threshold
    df.loc[is_outlier, "sales"] = mean
    return df


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Full sales cleaning: fill missing values, then correct outliers."""
    return correct_outliers(fill_missing_sales(df))


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ``city`` to ``province`` so weather can be joined with sales."""
    return df.rename(columns={"city": "province"})


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw files, clean them and save the preprocessed CSVs."""
    sales = clean_sales(io.load_raw_sales())
    weather = clean_weather(io.load_raw_weather())

    io.save_dataframe(sales, config.SALES_PREPROCESSED_FILE)
    io.save_dataframe(weather, config.WEATHER_PREPROCESSED_FILE)
    return sales, weather
