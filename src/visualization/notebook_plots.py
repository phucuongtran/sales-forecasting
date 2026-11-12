"""Quick plots used inside the notebooks (they call ``plt.show()``)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_store_item_sales(df: pd.DataFrame, store_id: int = 1, item_id: int = 1) -> None:
    """Sales of one store-item series; missing days are marked with red dots."""
    series = df.query("store_id == @store_id and item_id == @item_id").sort_values("date")
    store_name, item_name = series["store_name"].iloc[-1], series["item_name"].iloc[-1]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(series["date"], series["sales"], label="sales")

    missing = series["sales"].isna()
    if missing.any():
        filled = series["sales"].ffill()
        ax.scatter(series.loc[missing, "date"], filled[missing], color="red", s=15, label="missing")

    ax.set(xlabel="Date", ylabel="Sales", title=f"Store: {store_name} - Item: {item_name}")
    ax.legend()
    plt.show()


def plot_prophet_forecast(forecasts: pd.DataFrame, store_item: str) -> None:
    """Actual vs Prophet forecast (with confidence band) for one series."""
    df = forecasts[forecasts["store_item"] == store_item]
    if df.empty:
        print(f"No data found for: {store_item}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["ds"], df["y"], color="black", label="Actual")
    ax.plot(df["ds"], df["yhat"], color="blue", label="Forecast")
    ax.fill_between(
        df["ds"], df["yhat_lower"], df["yhat_upper"], color="blue", alpha=0.2,
        label="Confidence Interval",
    )
    ax.set(title=f"Forecast vs Actual for {store_item}", xlabel="Date", ylabel="Sales")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_store_predictions(
    predictions: pd.DataFrame, store_id: int = 1, nrows: int = 6, ncols: int = 5,
    figsize=(20, 20),
) -> None:
    """Grid of actual vs predicted sales for every item of one store.

    ``predictions`` needs columns: store_id, store_name, item_id, item_name,
    date, sales, prediction.
    """
    store = predictions[predictions["store_id"] == store_id]
    store_name = store["store_name"].iloc[-1]
    item_ids = sorted(store["item_id"].unique())

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, item_id in zip(axes, item_ids):
        item = store[store["item_id"] == item_id]
        ax.plot(item["date"], item["sales"], color="blue", label="Actual")
        ax.plot(item["date"], item["prediction"], color="red", linestyle="--", marker=".",
                label="Forecast")
        ax.set(title=f"Item: {item['item_name'].iloc[-1]}", ylabel="Sales")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True)

    for ax in axes[len(item_ids):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(f"Sales Forecast vs Actual - Store {store_name}", fontsize=16, fontweight="bold")
    plt.show()
