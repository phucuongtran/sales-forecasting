from __future__ import annotations

from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


def load_bundle(path: str) -> Dict:
    return joblib.load(path)


def predict_with_bundle(bundle: Dict, frame: pd.DataFrame) -> np.ndarray:
    X = frame[bundle["feature_names"]].copy()
    for col in bundle.get("categorical_features", []):
        if col in X.columns:
            X[col] = X[col].astype("category")
    preds = []
    for model in bundle["models"]:
        pred = model.predict(X, num_iteration=getattr(model, "best_iteration_", None))
        if bundle.get("config", {}).get("use_log_target", False):
            pred = np.expm1(pred)
        preds.append(np.clip(pred, 0.0, None))
    return np.mean(np.vstack(preds), axis=0)
