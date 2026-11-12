"""Small, framework-free helpers to read and write project data and models."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------------- #
# DataFrames
# --------------------------------------------------------------------------- #
def load_raw_sales() -> pd.DataFrame:
    """Concatenate the yearly raw sales files and parse dates."""
    frames = [pd.read_csv(path) for path in config.RAW_SALES_FILES.values()]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_raw_weather() -> pd.DataFrame:
    return pd.read_csv(config.RAW_WEATHER_FILE, parse_dates=["date"])


def load_preprocessed_sales() -> pd.DataFrame:
    return pd.read_csv(config.SALES_PREPROCESSED_FILE, parse_dates=["date"])


def load_preprocessed_weather() -> pd.DataFrame:
    return pd.read_csv(config.WEATHER_PREPROCESSED_FILE, parse_dates=["date"])


def load_features() -> pd.DataFrame:
    return pd.read_feather(config.FEATURES_FILE)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save as CSV or Feather depending on the file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".feather":
        df.reset_index(drop=True).to_feather(path)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file type '{path.suffix}' (use .csv or .feather)")

    print(f"Saved {len(df):,} rows -> {path}")
    return path


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Number and percentage of missing values per column."""
    counts = df.isna().sum()
    return pd.DataFrame(
        {"counts": counts, "ratio (%)": np.round(counts / len(df) * 100, 2)}
    )


# --------------------------------------------------------------------------- #
# Model artifacts
# --------------------------------------------------------------------------- #
def save_model(model, path: str | Path = config.MODEL_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {path}")
    return path


def load_model(path: str | Path = config.MODEL_FILE):
    """Load a pickled model (LightGBM scikit-learn API)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved -> {path}")
    return path
