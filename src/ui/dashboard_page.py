"""'Historical Sales Analysis' page."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

import pandas as pd
import streamlit as st

from src.ui import charts

ALL_STORES = "All Stores"


@dataclass
class Filters:
    start: Date
    end: Date
    store: str  # a store name or ALL_STORES
    categories: list[str]

    @property
    def all_stores(self) -> bool:
        return self.store == ALL_STORES


def render(sales: pd.DataFrame) -> None:
    st.title("Store Sales Dashboard")

    filters = sidebar_filters(sales)
    data = apply_filters(sales, filters)
    if data.empty:
        st.warning("No data available for the selected filters.")
        return

    kpi_section(data, filters)
    trend_section(data, filters)
    breakdown_section(data, filters)

    st.header("Sales Distribution")
    st.pyplot(charts.sales_distribution(data))

    with st.expander("View Detailed Sales Data"):
        st.dataframe(data.sort_values("date", ascending=False), use_container_width=True)


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def sidebar_filters(sales: pd.DataFrame) -> Filters:
    min_date, max_date = sales["date"].min().date(), sales["date"].max().date()
    categories = sorted(sales["category"].unique())

    with st.sidebar:
        st.header("Dashboard Filters")
        st.subheader("Date Range")
        start = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
        end = st.date_input("To", max_date, min_value=min_date, max_value=max_date)

        st.subheader("Store Selection")
        store = st.selectbox(
            "Select Store", [ALL_STORES, *sorted(sales["store_name"].unique())]
        )

        st.subheader("Product Categories")
        selected = st.multiselect("Select Categories", categories, default=categories)

    return Filters(start=start, end=end, store=store, categories=selected)


def apply_filters(sales: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    dates = sales["date"].dt.date
    mask = (dates >= filters.start) & (dates <= filters.end)
    if not filters.all_stores:
        mask &= sales["store_name"] == filters.store
    if filters.categories:
        mask &= sales["category"].isin(filters.categories)
    return sales[mask]


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def growth_between_halves(data: pd.DataFrame, filters: Filters) -> float | None:
    """% change of total sales between the 2nd and the 1st half of the date range."""
    midpoint = filters.start + (filters.end - filters.start) / 2
    dates = data["date"].dt.date
    first_half = data.loc[dates <= midpoint, "sales"].sum()
    second_half = data.loc[dates > midpoint, "sales"].sum()
    if first_half <= 0 or second_half <= 0:
        return None
    return (second_half - first_half) / first_half * 100


def kpi_section(data: pd.DataFrame, filters: Filters) -> None:
    st.header("Key Performance Indicators")

    total = data["sales"].sum()
    growth = growth_between_halves(data, filters)
    records = len(data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Units Sold", f"{total:,.0f}",
        delta=f"{growth:.1f}%" if growth is not None else None,
        help="Delta: change between the 2nd and the 1st half of the selected period.",
    )
    col2.metric("Avg Units per Day", f"{data.groupby('date')['sales'].sum().mean():,.1f}")
    col3.metric("Store-Item-Days", f"{records:,}")
    col4.metric("Avg Units per Store-Item-Day", f"{total / records:,.2f}")


def trend_section(data: pd.DataFrame, filters: Filters) -> None:
    st.header("Sales Trends")
    col1, col2 = st.columns(2)
    col1.pyplot(charts.daily_sales_trend(data, f"Daily Sales - {filters.store}"))
    if data["date"].nunique() >= 7:
        col2.pyplot(charts.sales_by_weekday(data))


def breakdown_section(data: pd.DataFrame, filters: Filters) -> None:
    st.header("Performance Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        if data["category"].nunique() > 1:
            st.subheader("Category Performance")
            st.dataframe(share_table(data, "category"), use_container_width=True)
            st.pyplot(charts.category_share(data))

    with col2:
        if filters.all_stores:
            st.subheader("Store Comparison")
            st.dataframe(share_table(data, "store_name").head(10), use_container_width=True)
            st.pyplot(charts.top_stores(data))


def share_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Total units and share of total per value of ``column``."""
    totals = data.groupby(column)["sales"].sum().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "Units Sold": totals.map("{:,.0f}".format),
            "Share": (totals / totals.sum() * 100).round(1).astype(str) + "%",
        }
    ).reset_index()
