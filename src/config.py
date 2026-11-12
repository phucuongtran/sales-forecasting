"""Central configuration: file paths and modelling constants.

Every module imports paths and "magic numbers" from here, so there is exactly one
place to change them. Paths are absolute (resolved from the project root), which
means the app, the notebooks and the CLI all work regardless of the current
working directory.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Raw (synthetic) data
RAW_SALES_FILES = {
    2016: DATA_DIR / "2016_sales.csv",
    2017: DATA_DIR / "2017_sales.csv",
}
RAW_WEATHER_FILE = DATA_DIR / "weather_data.csv"

# Preprocessed data
SALES_PREPROCESSED_FILE = DATA_DIR / "sales_data_preprocessed.csv"
WEATHER_PREPROCESSED_FILE = DATA_DIR / "weather_preprocessed.csv"

# Feature-engineered data (input of the model)
FEATURES_FILE = DATA_DIR / "feature_engineered_data.feather"

# Model artifacts
MODEL_FILE = MODELS_DIR / "sales_forecast_model.pkl"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 2025

# --------------------------------------------------------------------------- #
# Data cleaning
# --------------------------------------------------------------------------- #
OUTLIER_Z_THRESHOLD = 3.0  # |z-score| above this is treated as an outlier

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
TARGET = "sales"
TEST_START_DATE = "2017-10-01"  # Oct-Dec 2017 is the hold-out test period

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]
EWMA_ALPHAS = [0.5, 0.75]
EWMA_WINDOWS = [7, 14, 28]  # only used to name the EWMA columns (see features)

# Bins used to turn temperature / humidity into categories.
# NOTE: categories are one-hot encoded with drop_first=True, so the first label
# of each list ("Cold", "Low") is the reference level and has no column.
TEMPERATURE_BINS = [float("-inf"), 20, 25, 30, float("inf")]
TEMPERATURE_LABELS = ["Cold", "Cool", "Warm", "Hot"]
HUMIDITY_BINS = [float("-inf"), 60, 75, float("inf")]
HUMIDITY_LABELS = ["Low", "Medium", "High"]

# Simplified list of Vietnamese public holidays used in this PoC.
HOLIDAYS = [
    "2016-01-01", "2016-01-02", "2016-01-03",  # New Year
    "2016-04-16", "2016-04-17", "2016-04-18",  # Hung Kings Festival
    "2016-04-30", "2016-05-01", "2016-05-02",  # Reunification Day & May Day
    "2016-09-02", "2016-09-03", "2016-09-04",  # National Day
    "2017-01-01", "2017-01-02",                # New Year
]

# Columns that identify a row but are not model inputs.
NON_FEATURE_COLUMNS = [
    "date",
    "province",
    "store_name",
    "item_name",
    "store_item",
    "is_test",
    "category",
]

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
LGBM_BASE_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "n_estimators": 100,
    "verbose": -1,
}
CV_SPLITS = 5
OPTUNA_TRIALS = 20
