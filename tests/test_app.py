"""Smoke tests that render both Streamlit pages headlessly.

Skipped until the data and model exist (run ``python -m src.pipeline`` first).
"""

import pytest
from streamlit.testing.v1 import AppTest

from src import config

pytestmark = pytest.mark.skipif(
    not (config.SALES_PREPROCESSED_FILE.exists() and config.MODEL_FILE.exists()),
    reason="run `python -m src.pipeline` first",
)

APP = str(config.PROJECT_ROOT / "app.py")


def test_dashboard_page_renders():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Store Sales Dashboard"
    assert len(at.metric) == 4


def test_prediction_page_predicts():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.selectbox[0].select("Sales Prediction").run()
    assert not at.exception

    at.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Model Prediction" in labels and "Scenario-Adjusted Prediction" in labels
