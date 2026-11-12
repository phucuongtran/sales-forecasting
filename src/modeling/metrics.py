"""Forecast accuracy metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error, in percent: sum|error| / sum|actual|."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


@dataclass(frozen=True)
class Scores:
    mae: float
    rmse: float
    wape: float

    def __str__(self) -> str:
        return f"MAE: {self.mae:.2f} | RMSE: {self.rmse:.2f} | WAPE: {self.wape:.2f}%"


def score(y_true, y_pred) -> Scores:
    return Scores(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        wape=float(wape(y_true, y_pred)),
    )
