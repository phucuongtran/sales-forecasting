"""'Sales Prediction' page: input widgets and result display.

All computation lives in ``src.ui.scenario``; this module only draws the UI.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from src.ui import charts
from src.ui.scenario import (
    COMPETITION_MULTIPLIER,
    SPECIAL_EVENTS,
    SUPPLY_MULTIPLIER,
    WEATHER_CONDITION_MULTIPLIER,
    PredictionResult,
    Scenario,
    predict,
)

HISTORY_DAYS = 60
TOP_CONTRIBUTIONS = 8


def render(features: pd.DataFrame, model) -> None:
    st.title("Sales Prediction Tool")

    series = select_series(features)
    last_date = series["date"].max().date()

    st.subheader("Prediction Parameters")
    scenario = scenario_inputs(default_date=last_date + timedelta(days=1))

    if st.button("Predict Sales"):
        result = predict(model, series, scenario)
        show_result(result, series, scenario)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def select_series(features: pd.DataFrame) -> pd.DataFrame:
    """Sidebar store + product pickers; returns the feature rows of that series."""
    stores = features[["store_id", "store_name"]].drop_duplicates().sort_values("store_id")

    with st.sidebar:
        st.header("Product Selection")
        store_id = st.selectbox(
            "Select Store",
            stores["store_id"],
            format_func=dict(zip(stores["store_id"], stores["store_name"])).get,
        )
        store_rows = features[features["store_id"] == store_id]
        items = store_rows[["item_id", "item_name"]].drop_duplicates().sort_values("item_id")
        item_id = st.selectbox(
            "Select Product",
            items["item_id"],
            format_func=dict(zip(items["item_id"], items["item_name"])).get,
        )

    return store_rows[store_rows["item_id"] == item_id]


def scenario_inputs(default_date) -> Scenario:
    col1, col2, col3 = st.columns(3)

    with col1:
        date = st.date_input("Prediction Date", default_date)
        is_holiday = st.checkbox("Holiday", value=False)
        special_event = st.selectbox("Special Event", list(SPECIAL_EVENTS))
        event_impact = 0.0
        if SPECIAL_EVENTS[special_event]:
            low, high, default = SPECIAL_EVENTS[special_event]
            event_impact = st.slider(f"{special_event} Impact (%)", low, high, default)

    with col2:
        temperature = st.slider("Temperature (°C)", -10.0, 40.0, 25.0)
        weather_condition = st.selectbox("Weather Condition", list(WEATHER_CONDITION_MULTIPLIER))

    with col3:
        humidity = st.slider("Humidity (%)", 0, 100, 70)
        competition = st.select_slider(
            "Competition Level", options=list(COMPETITION_MULTIPLIER), value="Medium"
        )
        supply = st.select_slider(
            "Supply Chain Status", options=list(SUPPLY_MULTIPLIER), value="Normal"
        )

    scenario = Scenario(
        date=date,
        is_holiday=is_holiday,
        temperature=temperature,
        humidity=humidity,
        special_event=special_event,
        event_impact_pct=event_impact,
        weather_condition=weather_condition,
        competition_level=competition,
        supply_chain=supply,
    )
    col2.caption(f"Temperature category: **{scenario.temperature_category}**")
    col3.caption(f"Humidity level: **{scenario.humidity_level}**")
    return scenario


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def show_result(result: PredictionResult, series: pd.DataFrame, scenario: Scenario) -> None:
    st.header("Prediction Results")
    first = series.iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Prediction", f"{result.model_prediction:,.1f} units")
        st.metric(
            "Scenario-Adjusted Prediction",
            f"{result.adjusted_prediction:,.1f} units",
            delta=(
                f"x{scenario.adjustment_factor():.2f} what-if adjustment"
                if scenario.adjustment_factor() != 1.0
                else None
            ),
            delta_color="off",
        )
        st.write(f"**Store:** {first['store_name']}  \n**Product:** {first['item_name']}")
        st.write(f"**Date:** {scenario.date:%B %d, %Y}")

        with st.expander("What-if adjustment details"):
            st.caption(
                "These multipliers are business assumptions applied on top of the "
                "model output; they are not learned from data."
            )
            st.table(
                pd.DataFrame(
                    {"Multiplier": result.multipliers.values()}, index=result.multipliers.keys()
                )
            )

    with col2:
        latest = series.sort_values("date").iloc[-1]
        peak = series.loc[series["sales"].idxmax()]
        st.metric("Historical Average", f"{series['sales'].mean():,.1f} units")
        st.caption(f"{series['date'].min():%b %d, %Y} to {series['date'].max():%b %d, %Y}")
        st.metric("Last Recorded Sales", f"{latest['sales']:,.1f} units")
        st.caption(f"{latest['date']:%b %d, %Y}")
        st.metric("Historical Maximum", f"{peak['sales']:,.1f} units")
        st.caption(f"{peak['date']:%b %d, %Y}")

    recent = series[series["date"] >= series["date"].max() - pd.Timedelta(days=HISTORY_DAYS)]
    st.subheader(f"Last {HISTORY_DAYS} Days")
    st.pyplot(charts.recent_history_with_prediction(recent, scenario.date, result.model_prediction))

    st.subheader("Weekly Sales Pattern")
    st.pyplot(charts.sales_by_weekday(recent, highlight_day=scenario.date.weekday()))

    st.subheader("Key Factors Behind This Prediction")
    st.caption("SHAP values: how much each feature pushed this prediction up (green) or down (red).")
    st.pyplot(charts.prediction_contributions(result.contributions.head(TOP_CONTRIBUTIONS)))
