"""Matplotlib figures used by the Streamlit pages.

Every function takes data and returns a ``Figure``; none of them touches
Streamlit, so they are easy to reuse and test. Sales are quantities (units).
"""

from __future__ import annotations

from datetime import date as Date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
UNITS = "Units sold"


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
def daily_sales_trend(data: pd.DataFrame, title: str) -> plt.Figure:
    """Total sales per day with a 7-day moving average."""
    daily = data.groupby("date")["sales"].sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(daily.index, daily.values, "b-", label="Daily sales")
    if len(daily) > 7:
        ax.plot(daily.index, daily.rolling(7).mean(), "r--", label="7-Day Moving Avg")
    ax.set(ylabel=UNITS, title=title)
    ax.legend()
    fig.autofmt_xdate()
    return fig


def sales_by_weekday(data: pd.DataFrame, highlight_day: int | None = None) -> plt.Figure:
    """Average sales per weekday.

    Without ``highlight_day`` the best day is green and the worst orange;
    with it, that weekday (0 = Monday) is highlighted in red.
    """
    by_day = (
        data.assign(day_name=data["date"].dt.day_name())
        .groupby("day_name")["sales"].mean()
        .reindex(DAY_NAMES)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(6, 4 if highlight_day is None else 2.5))
    bars = ax.bar(by_day.index, by_day.values, color="skyblue")

    for bar, day in zip(bars, by_day.index):
        if highlight_day is not None:
            if day == DAY_NAMES[highlight_day]:
                bar.set_color("red")
        elif day == by_day.idxmax():
            bar.set_color("green")
        elif day == by_day.idxmin():
            bar.set_color("orange")

    if highlight_day is None:
        ax.axhline(by_day.mean(), color="red", linestyle="--", label="Daily Average")
        ax.legend()
    ax.set(ylabel=f"Avg {UNITS.lower()}", title="Sales by Day of Week")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def category_share(data: pd.DataFrame, top_n: int = 5) -> plt.Figure:
    """Pie chart of sales by category (small categories grouped as 'Others')."""
    by_category = data.groupby("category")["sales"].sum().sort_values(ascending=False)
    shares = by_category.head(top_n)
    others = by_category.iloc[top_n:].sum()
    if others > 0:
        shares = pd.concat([shares, pd.Series({"Others": others})])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(shares, labels=shares.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    ax.set_title("Sales by Category")
    return fig


def top_stores(data: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Horizontal bar chart of the stores with the highest total sales."""
    by_store = data.groupby("store_name")["sales"].sum().nlargest(top_n)

    fig, ax = plt.subplots(figsize=(6, 6))
    y = np.arange(len(by_store))
    ax.barh(y, by_store.values)
    ax.set_yticks(y, by_store.index)
    ax.invert_yaxis()
    ax.set(xlabel=UNITS, title=f"Top {top_n} Stores by Sales")
    return fig


def sales_distribution(data: pd.DataFrame) -> plt.Figure:
    """Histogram + KDE with mean and median markers."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data["sales"], bins=30, kde=True, ax=ax)
    median, mean = data["sales"].median(), data["sales"].mean()
    ax.axvline(median, color="r", linestyle="--", label=f"Median: {median:.2f}")
    ax.axvline(mean, color="g", linestyle="--", label=f"Mean: {mean:.2f}")
    ax.set(xlabel=UNITS, ylabel="Frequency", title="Sales Distribution")
    ax.legend()
    return fig


# --------------------------------------------------------------------------- #
# Prediction page
# --------------------------------------------------------------------------- #
def recent_history_with_prediction(
    history: pd.DataFrame, prediction_date: Date, prediction: float
) -> plt.Figure:
    """Recent sales of one series, its 7-day average and the predicted point."""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(history["date"], history["sales"], "b-", label="Sales")
    if len(history) > 7:
        ax.plot(history["date"], history["sales"].rolling(7).mean(), "g--", label="7-Day Avg")
    ax.scatter(pd.Timestamp(prediction_date), prediction, color="red", s=60, label="Prediction")
    ax.set(ylabel=UNITS, title="Recent Sales History")
    ax.legend(loc="upper left", fontsize="x-small")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig


def prediction_contributions(contributions: pd.Series) -> plt.Figure:
    """Signed per-feature contributions (SHAP values) of a single prediction."""
    contributions = contributions.iloc[::-1]  # largest at the top
    colors = ["tab:green" if v > 0 else "tab:red" for v in contributions.values]
    labels = [name.replace("_", " ") for name in contributions.index]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, contributions.values, color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set(xlabel="Contribution to prediction (units)", title="Why the model predicts this value")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig
