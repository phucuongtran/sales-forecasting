"""Streamlit entry point:  streamlit run app.py"""

import streamlit as st

from src import config
from src.ui import dashboard_page, loaders, prediction_page

PAGES = ["Historical Sales Analysis", "Sales Prediction"]


def main() -> None:
    st.set_page_config(page_title="Sales Forecasting App", page_icon="📈", layout="wide")
    st.sidebar.title("Sales Forecasting App")
    page = st.sidebar.selectbox("Choose a page", PAGES)

    if page == "Historical Sales Analysis":
        sales = loaders.load_sales()
        if sales is None:
            loaders.missing_file_error(config.SALES_PREPROCESSED_FILE)
            return
        dashboard_page.render(sales)
    else:
        features, model = loaders.load_features(), loaders.load_model()
        if features is None:
            loaders.missing_file_error(config.FEATURES_FILE)
            return
        if model is None:
            loaders.missing_file_error(config.MODEL_FILE)
            return
        prediction_page.render(features, model)


if __name__ == "__main__":
    main()
