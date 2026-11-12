"""Generate the synthetic sales and weather datasets used by the project.

Run from the project root:

    python -m src.data.generator

Daily sales of an item in a store are modelled as a product of independent
multiplicative factors (store size, weekday, month, quarter, holidays, weather,
year-over-year growth and random noise). Each factor has its own small function
below so the simulation logic is easy to read and tweak.
"""

from __future__ import annotations

from datetime import date as Date

import numpy as np
import pandas as pd

from src import config
from src.data.catalog import (
    CATEGORIES,
    ITEMS,
    PROVINCE_CLIMATE,
    PROVINCES,
    STORES,
    season_for,
)
from src.data.io import check_missing_values

# Probability that a store carries a given item.
ASSORTMENT_PROBABILITY = 0.8

# Year-over-year growth applied to 2017 sales, per category.
YOY_GROWTH_2017 = {
    "Staples": 1.03,
    "Dairy & Frozen": 1.05,
    "Beverages & Snacks": 1.08,
    "Household & Personal Care": 1.05,
    "Baby & Health": 1.07,
}


# --------------------------------------------------------------------------- #
# Sales factors
# --------------------------------------------------------------------------- #
def store_factor(store: dict) -> float:
    """Bigger stores sell more: 0.9 for store 1 ... 1.7 for store 9, 0.8 for store 10."""
    return 0.8 + (store["id"] % 10) / 10


def weekday_factor(day: Date) -> float:
    """Weekend boost."""
    return 1.3 if day.weekday() >= 5 else 1.0


def month_factor(day: Date) -> float:
    """Higher sales in December, lower in February."""
    return 1.0 + 0.3 * (day.month == 12) - 0.1 * (day.month == 2)


def quarter_factor(day: Date) -> float:
    """Slight upward business cycle over the year (Q1 lowest, Q4 highest)."""
    quarter = (day.month - 1) // 3 + 1
    return 1.0 + 0.05 * (quarter - 2.5)


def holiday_factor(day: Date) -> float:
    """Tet (late Jan / early Feb), National Day (Sep 2) and year-end shopping."""
    if (day.month == 1 and day.day >= 27) or (day.month == 2 and day.day <= 5):
        return 1.5
    if day.month == 9 and day.day == 2:
        return 1.3
    if day.month == 12 and day.day >= 20:
        return 1.4
    return 1.0


def weather_factor(item: dict, weather: dict | None) -> float:
    """Hot weather boosts drinks/frozen goods; humid (rainy) days boost indoor goods."""
    if not weather:
        return 1.0

    factor = 1.0
    temperature, humidity = weather["temperature"], weather["humidity"]

    if item["category"] == "Beverages & Snacks":
        if temperature > 28:
            factor *= 1.3
        elif temperature < 18:
            factor *= 0.9
    elif item["category"] == "Dairy & Frozen":
        if temperature > 28:
            factor *= 1.4
        elif temperature < 18:
            factor *= 0.8

    if humidity > 80 and item["category"] in (
        "Beverages & Snacks",
        "Household & Personal Care",
    ):
        factor *= 1.2

    return factor


def growth_factor(day: Date, item: dict) -> float:
    """Year-over-year growth: 2017 sales grow by a category-specific rate."""
    if day.year == 2017:
        return YOY_GROWTH_2017.get(item["category"], 1.05)
    return 1.0


def daily_sales(
    day: Date,
    store: dict,
    item: dict,
    weather: dict | None,
    rng: np.random.Generator,
) -> int:
    """Simulated number of units sold (integer, at least 1)."""
    noise = rng.normal(1.0, item["volatility"])
    sales = (
        item["base_sales"]
        * store_factor(store)
        * weekday_factor(day)
        * month_factor(day)
        * quarter_factor(day)
        * holiday_factor(day)
        * weather_factor(item, weather)
        * growth_factor(day, item)
        * noise
    )
    return max(1, int(round(sales)))


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def store_assortment(store: dict) -> list[dict]:
    """Items carried by a store (~80% of the catalogue).

    A dedicated RNG seeded with the store id makes the assortment deterministic
    without touching the main random generator.
    """
    store_rng = np.random.RandomState(store["id"] * 10)
    return [item for item in ITEMS if store_rng.random_sample() < ASSORTMENT_PROBABILITY]


def generate_weather_data(
    start: str, end: str, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict]:
    """Daily weather per province.

    Returns the weather as a DataFrame and as a ``{(date_str, province): row}``
    lookup used while simulating sales.
    """
    rows = []
    for day in pd.date_range(start, end):
        for province in PROVINCES:
            climate = PROVINCE_CLIMATE[province]
            month_idx = day.month - 1

            temp_var = climate["temp_variation"]
            hum_var = climate["humidity_variation"]
            temperature = climate["base_temp"][month_idx] + rng.uniform(-temp_var, temp_var)
            humidity = climate["base_humidity"][month_idx] + rng.uniform(-hum_var, hum_var)

            rows.append(
                {
                    "city": province,
                    "date": day.strftime("%Y-%m-%d"),
                    "temperature": round(temperature, 1),
                    "humidity": min(95, max(40, round(humidity, 1))),
                    "season": season_for(province, day.month),
                }
            )

    weather_df = pd.DataFrame(rows)
    lookup = {(row["date"], row["city"]): row for row in rows}
    return weather_df, lookup


def generate_sales_data(
    start: str, end: str, weather_lookup: dict, rng: np.random.Generator
) -> pd.DataFrame:
    """Daily sales for every (store, carried item) pair between start and end."""
    assortments = {store["id"]: store_assortment(store) for store in STORES}

    rows = []
    for day in pd.date_range(start, end):
        date_str = day.strftime("%Y-%m-%d")
        for store in STORES:
            weather = weather_lookup.get((date_str, store["province"]))
            for item in assortments[store["id"]]:
                rows.append(
                    {
                        "date": date_str,
                        "province": store["province"],
                        "store_id": store["id"],
                        "store_name": store["name"],
                        "category": item["category"],
                        "item_id": item["id"],
                        "item_name": item["name"],
                        "sales": daily_sales(day, store, item, weather, rng),
                    }
                )
    return pd.DataFrame(rows)


def add_outliers_and_nans(
    df: pd.DataFrame,
    rng: np.random.Generator,
    outlier_pct: float = 0.5,
    nan_pct: float = 1.0,
) -> pd.DataFrame:
    """Inject data-quality problems so the cleaning step has something to do.

    ``outlier_pct`` / ``nan_pct`` are percentages of rows (0.5 means 0.5%).
    Outlier rows get 3x their sales; NaN rows lose their sales value.
    """
    df = df.copy()
    n_rows = len(df)

    outlier_idx = rng.choice(n_rows, int(n_rows * outlier_pct / 100), replace=False)
    df.loc[outlier_idx, "sales"] *= 3

    nan_idx = rng.choice(n_rows, int(n_rows * nan_pct / 100), replace=False)
    df.loc[nan_idx, "sales"] = np.nan
    return df


def main(seed: int = config.RANDOM_SEED) -> None:
    """Generate all raw CSV files into ``data/``."""
    rng = np.random.default_rng(seed)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{len(STORES)} stores, {len(ITEMS)} items, {len(CATEGORIES)} categories")

    print("Generating weather data...")
    weather_df, weather_lookup = generate_weather_data("2016-01-01", "2017-12-31", rng)
    weather_df.to_csv(config.RAW_WEATHER_FILE, index=False)
    print(f"  -> {config.RAW_WEATHER_FILE} ({len(weather_df):,} rows)")

    for year, path in config.RAW_SALES_FILES.items():
        print(f"Generating {year} sales data...")
        sales = generate_sales_data(f"{year}-01-01", f"{year}-12-31", weather_lookup, rng)
        sales = add_outliers_and_nans(sales, rng)
        sales.to_csv(path, index=False)
        print(
            f"  -> {path} ({len(sales):,} rows, "
            f"mean={sales['sales'].mean():.2f}, max={sales['sales'].max():.0f})"
        )
        print(check_missing_values(sales).query("counts > 0").to_string())


if __name__ == "__main__":
    main()
