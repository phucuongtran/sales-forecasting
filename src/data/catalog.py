"""Master data of the synthetic retail chain: stores, items and climate.

These tables used to live inside ``data_generator.py`` as ~300 lines of nested
dicts. They are now compact tables so they can be read at a glance and reused
(e.g. the Streamlit app uses ``PROVINCE_SEASONS`` to derive the season of a
prediction date exactly the way the training data does).
"""

PROVINCES = ["Hanoi", "Ho Chi Minh City"]

CATEGORIES = [
    "Staples",
    "Dairy & Frozen",
    "Beverages & Snacks",
    "Household & Personal Care",
    "Baby & Health",
]

# (store_id, store_name, province)
_STORE_TABLE = [
    (1, "Hoan Kiem Market", "Hanoi"),
    (2, "Ba Dinh Supermarket", "Hanoi"),
    (3, "Dong Da Mall", "Hanoi"),
    (4, "Tay Ho Store", "Hanoi"),
    (5, "Long Bien Shop", "Hanoi"),
    (6, "District 1 Market", "Ho Chi Minh City"),
    (7, "Ben Thanh Store", "Ho Chi Minh City"),
    (8, "Saigon Supermarket", "Ho Chi Minh City"),
    (9, "Phu Nhuan Shop", "Ho Chi Minh City"),
    (10, "Binh Thanh Market", "Ho Chi Minh City"),
]

# (item_id, item_name, category, base_price, base_sales, volatility)
_ITEM_TABLE = [
    (1, "Rice", "Staples", 20.0, 15, 0.30),
    (2, "Noodles", "Staples", 15.0, 12, 0.25),
    (3, "Bread", "Staples", 10.0, 20, 0.40),
    (4, "Flour", "Staples", 12.0, 8, 0.20),
    (5, "Cooking Oil", "Staples", 25.0, 10, 0.15),
    (6, "Sugar", "Staples", 8.0, 7, 0.10),
    (7, "Milk", "Dairy & Frozen", 18.0, 30, 0.35),
    (8, "Cheese", "Dairy & Frozen", 35.0, 12, 0.30),
    (9, "Yogurt", "Dairy & Frozen", 12.0, 25, 0.40),
    (10, "Ice Cream", "Dairy & Frozen", 30.0, 15, 0.50),
    (11, "Frozen Vegetables", "Dairy & Frozen", 22.0, 10, 0.25),
    (12, "Soda", "Beverages & Snacks", 15.0, 40, 0.45),
    (13, "Juice", "Beverages & Snacks", 20.0, 30, 0.40),
    (14, "Water", "Beverages & Snacks", 10.0, 50, 0.30),
    (15, "Coffee", "Beverages & Snacks", 45.0, 20, 0.25),
    (16, "Tea", "Beverages & Snacks", 35.0, 15, 0.20),
    (17, "Chips", "Beverages & Snacks", 12.0, 35, 0.45),
    (18, "Cookies", "Beverages & Snacks", 18.0, 30, 0.40),
    (19, "Chocolate", "Beverages & Snacks", 22.0, 25, 0.35),
    (20, "Soap", "Household & Personal Care", 8.0, 20, 0.20),
    (21, "Shampoo", "Household & Personal Care", 25.0, 15, 0.25),
    (22, "Toothpaste", "Household & Personal Care", 15.0, 18, 0.15),
    (23, "Laundry Detergent", "Household & Personal Care", 40.0, 12, 0.20),
    (24, "Paper Towels", "Household & Personal Care", 20.0, 14, 0.30),
    (25, "Toilet Paper", "Household & Personal Care", 25.0, 16, 0.25),
    (26, "Trash Bags", "Household & Personal Care", 18.0, 10, 0.15),
    (27, "Dishwashing Liquid", "Household & Personal Care", 15.0, 11, 0.20),
    (28, "All-Purpose Cleaner", "Household & Personal Care", 22.0, 9, 0.15),
    (29, "Diapers", "Baby & Health", 45.0, 25, 0.30),
    (30, "Baby Food", "Baby & Health", 20.0, 15, 0.25),
    (31, "Baby Wipes", "Baby & Health", 15.0, 20, 0.20),
    (32, "Pain Relievers", "Baby & Health", 30.0, 10, 0.15),
    (33, "Vitamins", "Baby & Health", 40.0, 8, 0.20),
    (34, "Cold & Flu Medicine", "Baby & Health", 35.0, 7, 0.40),
    (35, "First Aid Kit", "Baby & Health", 50.0, 5, 0.10),
]

STORES = [
    {"id": sid, "name": name, "province": province}
    for sid, name, province in _STORE_TABLE
]

ITEMS = [
    {
        "id": iid,
        "name": name,
        "category": category,
        "base_price": price,
        "base_sales": base_sales,
        "volatility": volatility,
    }
    for iid, name, category, price, base_sales, volatility in _ITEM_TABLE
]

# --------------------------------------------------------------------------- #
# Climate (one value per month, index 0 = January)
# --------------------------------------------------------------------------- #
PROVINCE_CLIMATE = {
    "Hanoi": {
        "base_temp": [16, 17, 20, 24, 28, 30, 30, 29, 28, 25, 21, 18],
        "temp_variation": 3.5,
        "base_humidity": [80, 83, 85, 85, 80, 80, 83, 85, 83, 78, 75, 77],
        "humidity_variation": 10,
    },
    "Ho Chi Minh City": {
        "base_temp": [26, 27, 28, 29, 29, 28, 28, 28, 28, 27, 27, 26],
        "temp_variation": 2.0,
        "base_humidity": [70, 70, 70, 75, 80, 83, 85, 85, 88, 85, 80, 75],
        "humidity_variation": 8,
    },
}

# Season name per province and month (index 0 = January).
# Hanoi has four seasons; Ho Chi Minh City has a dry and a wet season.
PROVINCE_SEASONS = {
    "Hanoi": [
        "winter", "winter", "spring", "spring", "summer", "summer",
        "summer", "summer", "fall", "fall", "fall", "winter",
    ],
    "Ho Chi Minh City": [
        "dry", "dry", "dry", "dry", "wet", "wet",
        "wet", "wet", "wet", "wet", "wet", "dry",
    ],
}


def season_for(province: str, month: int) -> str:
    """Return the season label used in the data for a province and month (1-12)."""
    return PROVINCE_SEASONS[province][month - 1]
