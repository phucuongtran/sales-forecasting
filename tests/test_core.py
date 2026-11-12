"""Fast unit tests for the pure (non-UI) logic.  Run:  pytest"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.catalog import season_for
from src.data.preprocessing import correct_outliers, fill_missing_sales
from src.features import engineering as fe
from src.modeling.metrics import score, wape
from src.ui.scenario import Scenario, build_model_input


def _toy_sales(days: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", periods=days)
    return pd.DataFrame(
        {
            "date": dates,
            "store_name": "S",
            "item_name": "I",
            "sales": np.arange(1, days + 1, dtype=float),
        }
    )


def test_wape_and_score():
    assert wape([10, 10], [5, 15]) == pytest.approx(50.0)
    s = score([1, 2, 3], [1, 2, 3])
    assert (s.mae, s.rmse, s.wape) == (0, 0, 0)


def test_cleaning_fills_nan_and_caps_outliers():
    df = pd.DataFrame({"sales": [1.0] * 50 + [np.nan, 1000.0]})
    clean = correct_outliers(fill_missing_sales(df))
    assert clean["sales"].isna().sum() == 0
    assert clean["sales"].max() < 1000


def test_lag_and_rolling_features_only_use_the_past():
    df = fe.add_rolling_features(fe.add_lag_features(fe.add_series_key(_toy_sales())))
    row = df.iloc[7]  # sales = 8, previous days = 1..7
    assert row["sales_lag_1"] == 7
    assert row["sales_lag_7"] == 1
    assert row["sales_mean_7d"] == pytest.approx(np.mean(range(1, 8)))
    assert row["sales_max_7d"] == 7  # today's value (8) is not included


def test_weather_buckets_match_training_bins():
    assert list(fe.temperature_category(pd.Series([20, 20.1, 30.5])).astype(str)) == [
        "Cold", "Cool", "Hot",
    ]
    assert list(fe.humidity_level(pd.Series([60, 70, 90])).astype(str)) == [
        "Low", "Medium", "High",
    ]


def test_season_depends_on_province():
    assert season_for("Hanoi", 7) == "summer"
    assert season_for("Ho Chi Minh City", 7) == "wet"


def test_build_model_input_encodes_scenario_like_training():
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-12-30", "2017-12-31"]),
            "province": "Ho Chi Minh City",
            "sales": [10.0, 12.0],
            "temp_category_Cool": [True, True],
            "temp_category_Hot": [False, False],
            "temp_category_Warm": [False, False],
            "humidity_level_Medium": [True, True],
            "humidity_level_High": [False, False],
            "season_wet": [False, False],
            "season_winter": [False, False],
        }
    )
    scenario = Scenario(date=date(2018, 6, 2), temperature=31, humidity=90, is_holiday=True)
    row = build_model_input(history, scenario).iloc[0]

    assert row["day_of_week"] == 5 and row["is_weekend"] == 1 and row["is_holiday"] == 1
    assert row["temp_category_Hot"] and not row["temp_category_Cool"]
    assert row["humidity_level_High"] and not row["humidity_level_Medium"]
    assert row["season_wet"]  # June in Ho Chi Minh City
    assert row["sales"] == 12.0  # history features come from the latest row


def test_scenario_adjustment_factor():
    neutral = Scenario(date=date(2018, 1, 1))
    assert neutral.adjustment_factor() == 1.0
    promo = Scenario(date=date(2018, 1, 1), special_event="Sale/Promotion",
                     event_impact_pct=20, competition_level="High")
    assert promo.adjustment_factor() == pytest.approx(1.2 * 0.9)
