"""SHAP helpers used by notebook 05_explain_model."""

from __future__ import annotations

import numpy as np
import pandas as pd

SALES_HISTORY_PREFIXES = (
    "sales_lag",
    "sales_mean",
    "sales_min",
    "sales_max",
    "sales_std",
    "sales_ewma",
)
DATE_KEYWORDS = ("day_of_week", "month", "year", "is_weekend", "is_holiday")
WEATHER_KEYWORDS = ("temp", "humidity", "season")


def feature_group(feature: str) -> str:
    """Classify a feature name into a business-friendly group."""
    if any(key in feature for key in DATE_KEYWORDS):
        return "date_time"
    if any(key in feature for key in WEATHER_KEYWORDS):
        return "weather"
    if feature.startswith(SALES_HISTORY_PREFIXES):
        return "sales_history"
    if "store_" in feature:
        return "store"
    if "item_" in feature:
        return "item"
    return "other"


def global_importance(shap_values: np.ndarray, feature_names) -> pd.DataFrame:
    """Mean |SHAP| per feature, sorted descending."""
    return (
        pd.DataFrame(
            {"feature": list(feature_names), "importance": np.abs(shap_values).mean(axis=0)}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def importance_by_group(shap_values: np.ndarray, feature_names) -> pd.DataFrame:
    """Sum of mean |SHAP| per feature group, with its share in percent."""
    importance = global_importance(shap_values, feature_names)
    importance["type"] = importance["feature"].map(feature_group)
    summary = (
        importance.groupby("type")
        .agg(importance=("importance", "sum"), feature=("feature", "count"))
        .reset_index()
        .sort_values("importance", ascending=False)
    )
    summary["importance_pct"] = 100 * summary["importance"] / summary["importance"].sum()
    return summary
