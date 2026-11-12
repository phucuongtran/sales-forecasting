"""Cached loaders for the Streamlit app.

Each loader returns ``None`` when its file is missing; the pages then show a
message explaining how to generate it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.data import io

PIPELINE_HINT = "Run `python -m src.pipeline` from the project root to generate it."


@st.cache_data
def load_sales() -> pd.DataFrame | None:
    if not config.SALES_PREPROCESSED_FILE.exists():
        return None
    return io.load_preprocessed_sales()


@st.cache_data
def load_features() -> pd.DataFrame | None:
    if not config.FEATURES_FILE.exists():
        return None
    return io.load_features()


@st.cache_resource
def load_model():
    if not config.MODEL_FILE.exists():
        return None
    return io.load_model(config.MODEL_FILE)


def missing_file_error(path) -> None:
    st.error(f"File not found: `{path.relative_to(config.PROJECT_ROOT)}`. {PIPELINE_HINT}")
