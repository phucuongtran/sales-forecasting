"""Batch-export forecast evaluation plots to PNG files.

Replaces the former ``utils/visualization_code.py`` (520 lines, 6 copies of the
same "actual vs predicted" plot) with one shared plotting helper.

``test_results`` is the DataFrame returned by
``src.modeling.train.build_test_results`` (date, store_name, item_name,
category, sales, prediction).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _safe_name(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_actual_vs_predicted(df: pd.DataFrame, title: str, path: Path) -> None:
    """Line plot of ``sales`` vs ``prediction`` over ``date``, saved to ``path``."""
    df = df.sort_values("date")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["date"], df["sales"], "o-", label="Actual", alpha=0.7, linewidth=2)
    ax.plot(df["date"], df["prediction"], "s--", label="Predicted", alpha=0.7, linewidth=2)
    ax.set(title=title, xlabel="Date", ylabel="Sales")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    _save(fig, path)


def export_store_item_plots(test_results: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    """One plot per store-item series, for the ``top_n`` series by total sales."""
    top = (
        test_results.groupby(["store_name", "item_name"])["sales"].sum()
        .nlargest(top_n).index
    )
    for store, item in top:
        group = test_results[(test_results["store_name"] == store) & (test_results["item_name"] == item)]
        mae = np.mean(np.abs(group["sales"] - group["prediction"]))
        plot_actual_vs_predicted(
            group,
            f"Store: {store}, Item: {item}\nMAE: {mae:.2f}",
            output_dir / f"{_safe_name(store)}_{_safe_name(item)}.png",
        )


def export_aggregated_plots(test_results: pd.DataFrame, output_dir: Path) -> None:
    """Daily totals overall, per store and per item."""
    def daily_totals(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("date")[["sales", "prediction"]].sum().reset_index()

    plot_actual_vs_predicted(
        daily_totals(test_results), "Total Daily Sales: Actual vs Predicted",
        output_dir / "total_daily_sales.png",
    )
    for column, prefix in (("store_name", "store"), ("item_name", "item")):
        for name, group in test_results.groupby(column):
            plot_actual_vs_predicted(
                daily_totals(group), f"{prefix.title()}: {name} - Total Daily Sales",
                output_dir / f"{prefix}_{_safe_name(name)}_total.png",
            )


def export_error_plots(test_results: pd.DataFrame, output_dir: Path) -> None:
    """Distribution of errors and errors by sales level, date, weekday and category."""
    df = test_results.copy()
    df["error"] = df["sales"] - df["prediction"]
    df["pct_error"] = df["error"] / df["sales"] * 100
    df["day_name"] = df["date"].dt.day_name()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df["error"], kde=True, bins=50, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set(title="Distribution of Prediction Errors", xlabel="Error (Actual - Predicted)")
    _save(fig, output_dir / "error_distribution.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df["sales"], df["error"], alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set(title="Prediction Error vs Actual Sales", xlabel="Actual Sales", ylabel="Error")
    _save(fig, output_dir / "error_vs_sales.png")

    daily = df.groupby("date")["error"].mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily.index, daily.values, "o-")
    ax.axhline(0, color="red", linestyle="--")
    ax.set(title="Mean Prediction Error Over Time", xlabel="Date", ylabel="Mean Error")
    _save(fig, output_dir / "error_over_time.png")

    breakdowns = [("day_name", DAY_ORDER, "error_by_day_of_week.png")]
    if "category" in df.columns:
        breakdowns.append(("category", None, "error_by_category.png"))
    for column, order, filename in breakdowns:
        values = df.groupby(column)["pct_error"].mean()
        values = values.reindex(order) if order else values.sort_values()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=values.index, y=values.values, ax=ax)
        ax.axhline(0, color="red", linestyle="--")
        ax.set(title=f"Mean Percentage Error by {column}", ylabel="Mean Percentage Error (%)")
        ax.tick_params(axis="x", rotation=45)
        _save(fig, output_dir / filename)


def export_forecast_report(test_results: pd.DataFrame, output_dir: str | Path = "visualizations") -> None:
    """Export all evaluation plots to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_store_item_plots(test_results, output_dir)
    export_aggregated_plots(test_results, output_dir)
    export_error_plots(test_results, output_dir)
    print(f"Forecast report saved to {output_dir}/")
