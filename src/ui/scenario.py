"""Pure prediction logic behind the "Sales Prediction" page (no Streamlit here).

A prediction is made in two clearly separated stages:

1. **Model prediction** - the LightGBM model is fed the most recent feature row
   of the selected store-item series, with the calendar and weather features
   replaced by the user's inputs. Weather categories and season are encoded
   with the *same* rules as in training (``src.features.engineering``).
2. **Scenario adjustment** - optional what-if multipliers for things the model
   has never seen (promotions, competition, supply, weather condition). These
   are business assumptions, not learned from data, and are shown separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

import numpy as np
import pandas as pd

from src.data.catalog import season_for
from src.features.engineering import humidity_level, temperature_category

# --------------------------------------------------------------------------- #
# What-if multipliers (business assumptions)
# --------------------------------------------------------------------------- #
# event -> (slider min %, slider max %, default %)
SPECIAL_EVENTS = {
    "None": None,
    "Sale/Promotion": (-50, 100, 20),
    "Local Event": (-20, 50, 10),
    "Inventory Clearance": (-70, 30, -10),
    "New Product Launch": (0, 200, 50),
}
WEATHER_CONDITION_MULTIPLIER = {
    "Clear": 1.0, "Cloudy": 0.95, "Rainy": 0.9, "Snowy": 0.8, "Stormy": 0.7,
}
COMPETITION_MULTIPLIER = {"Low": 1.1, "Medium": 1.0, "High": 0.9}
SUPPLY_MULTIPLIER = {"Constrained": 0.9, "Normal": 1.0, "Abundant": 1.05}


@dataclass
class Scenario:
    """Everything the user enters on the prediction page."""

    date: Date
    is_holiday: bool = False
    temperature: float = 25.0
    humidity: float = 70.0
    special_event: str = "None"
    event_impact_pct: float = 0.0
    weather_condition: str = "Clear"
    competition_level: str = "Medium"
    supply_chain: str = "Normal"

    @property
    def temperature_category(self) -> str:
        return str(temperature_category(pd.Series([self.temperature])).iloc[0])

    @property
    def humidity_level(self) -> str:
        return str(humidity_level(pd.Series([self.humidity])).iloc[0])

    def adjustment_multipliers(self) -> dict[str, float]:
        event = 1.0 if self.special_event == "None" else 1.0 + self.event_impact_pct / 100
        return {
            "Special event": event,
            "Weather condition": WEATHER_CONDITION_MULTIPLIER[self.weather_condition],
            "Competition": COMPETITION_MULTIPLIER[self.competition_level],
            "Supply chain": SUPPLY_MULTIPLIER[self.supply_chain],
        }

    def adjustment_factor(self) -> float:
        return float(np.prod(list(self.adjustment_multipliers().values())))


# --------------------------------------------------------------------------- #
# Model input
# --------------------------------------------------------------------------- #
def _set_one_hot(row: pd.DataFrame, prefix: str, active_value: str) -> None:
    """Set every ``{prefix}_*`` dummy column to False except ``{prefix}_{active_value}``.

    If the active value is the reference level dropped by ``get_dummies``
    (e.g. temperature "Cold"), all dummies end up False, as in training.
    """
    for column in row.columns:
        if column.startswith(f"{prefix}_"):
            row[column] = column == f"{prefix}_{active_value}"


def build_model_input(series_history: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """One-row feature frame for the prediction date.

    ``series_history`` holds the feature rows of one store-item series. Sales
    history features (lags, rolling stats, ...) are taken from its latest row;
    calendar and weather features come from the scenario.
    """
    row = series_history.sort_values("date").iloc[[-1]].copy()
    when = pd.Timestamp(scenario.date)
    province = row["province"].iloc[0]

    row["date"] = when
    row["year"] = when.year
    row["month"] = when.month
    row["day"] = when.day
    row["day_of_week"] = when.weekday()
    row["is_weekend"] = int(when.weekday() >= 5)
    row["quarter"] = when.quarter
    row["is_holiday"] = int(scenario.is_holiday)
    row["temperature"] = scenario.temperature
    row["humidity"] = scenario.humidity

    _set_one_hot(row, "temp_category", scenario.temperature_category)
    _set_one_hot(row, "humidity_level", scenario.humidity_level)
    _set_one_hot(row, "season", season_for(province, when.month))
    return row


@dataclass
class PredictionResult:
    model_prediction: float
    adjusted_prediction: float
    multipliers: dict[str, float]
    contributions: pd.Series  # per-feature SHAP values, sorted by |value|


def predict(model, series_history: pd.DataFrame, scenario: Scenario) -> PredictionResult:
    """Model prediction, scenario-adjusted prediction and its explanation."""
    X = build_model_input(series_history, scenario)[model.feature_name_]

    model_prediction = float(model.predict(X)[0])

    # LightGBM computes exact TreeSHAP values natively: one value per feature
    # plus the expected value (last column). They sum to the prediction.
    shap_row = model.predict(X, pred_contrib=True)[0][:-1]
    contributions = pd.Series(shap_row, index=model.feature_name_)
    contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index)

    return PredictionResult(
        model_prediction=model_prediction,
        adjusted_prediction=model_prediction * scenario.adjustment_factor(),
        multipliers=scenario.adjustment_multipliers(),
        contributions=contributions,
    )
