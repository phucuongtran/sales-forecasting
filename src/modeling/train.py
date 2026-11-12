"""Step 3 of the pipeline: train, evaluate and save the LightGBM model.

    python -m src.pipeline --steps train            # baseline LightGBM
    python -m src.pipeline --steps train --tune     # + Optuna tuning
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src import config
from src.data import io
from src.features.engineering import feature_columns
from src.modeling.metrics import Scores, score


@dataclass
class TrainTestSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def split_train_test(features: pd.DataFrame) -> TrainTestSplit:
    """Split on the ``is_test`` flag (time-based split, no shuffling)."""
    columns = feature_columns(features)
    train = features[~features["is_test"]]
    test = features[features["is_test"]]
    return TrainTestSplit(
        X_train=train[columns],
        y_train=train[config.TARGET],
        X_test=test[columns],
        y_test=test[config.TARGET],
    )


# --------------------------------------------------------------------------- #
# Baseline LightGBM
# --------------------------------------------------------------------------- #
def cross_validate(params: dict, X: pd.DataFrame, y: pd.Series, n_splits=config.CV_SPLITS) -> Scores:
    """Average scores over an expanding-window time-series cross-validation."""
    fold_scores = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = lgbm.LGBMRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_scores.append(score(y.iloc[val_idx], model.predict(X.iloc[val_idx])))

    return Scores(
        mae=float(np.mean([s.mae for s in fold_scores])),
        rmse=float(np.mean([s.rmse for s in fold_scores])),
        wape=float(np.mean([s.wape for s in fold_scores])),
    )


def train_lightgbm(split: TrainTestSplit, params: dict | None = None):
    """Cross-validate, then fit on the full training set and score on the test set."""
    params = params or config.LGBM_BASE_PARAMS

    cv_scores = cross_validate(params, split.X_train, split.y_train)
    print(f"LightGBM CV   - {cv_scores}")

    model = lgbm.LGBMRegressor(**params)
    model.fit(split.X_train, split.y_train)
    test_scores = score(split.y_test, model.predict(split.X_test))
    print(f"LightGBM Test - {test_scores}")
    return model, test_scores


# --------------------------------------------------------------------------- #
# Optuna tuning
# --------------------------------------------------------------------------- #
def tune_lightgbm(split: TrainTestSplit, n_trials: int = config.OPTUNA_TRIALS):
    """Search LightGBM hyper-parameters with Optuna, then refit with the best ones.

    NOTE (kept from the original PoC): trials are early-stopped and scored on the
    *test* set, so the reported test score is optimistic. See REFACTORING.md.
    """
    import optuna  # imported lazily: only needed when tuning

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "n_estimators": 1000,
            "verbose": -1,
        }
        model = lgbm.LGBMRegressor(**params)
        model.fit(
            split.X_train,
            split.y_train,
            eval_set=[(split.X_test, split.y_test)],
            callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)],
        )
        return score(split.y_test, model.predict(split.X_test)).rmse

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = {
        **study.best_params,
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbose": -1,
    }
    print("Best parameters:", best_params)

    model = lgbm.LGBMRegressor(**best_params)
    model.fit(split.X_train, split.y_train)
    test_scores = score(split.y_test, model.predict(split.X_test))
    print(f"Tuned LightGBM Test - {test_scores}")
    return model, test_scores, best_params


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def build_test_results(model, features: pd.DataFrame, split: TrainTestSplit) -> pd.DataFrame:
    """Test rows with predictions and errors, for error analysis and plots."""
    results = features[features["is_test"]].copy()
    results["prediction"] = model.predict(split.X_test)
    results["error"] = results["sales"] - results["prediction"]
    results["abs_error"] = results["error"].abs()
    results["month_name"] = results["date"].dt.strftime("%B")
    return results


def errors_by_month(test_results: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        test_results.groupby("month_name")
        .agg(abs_error=("abs_error", "mean"), error=("error", "mean"), sales=("sales", "mean"))
        .reset_index()
    )
    monthly["error_pct"] = 100 * monthly["abs_error"] / monthly["sales"]
    return monthly


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_artifacts(model, feature_names: list[str], test_scores: Scores, params: dict) -> None:
    """Save the model and a JSON file describing how it was trained."""
    io.save_model(model, config.MODEL_FILE)
    io.save_json(
        {
            "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "features": list(feature_names),
            "params": params,
            "test_scores": vars(test_scores),
            "test_start_date": config.TEST_START_DATE,
        },
        config.MODEL_METADATA_FILE,
    )


def run(tune: bool = False):
    """Train (optionally tune) the model on the saved features and persist it."""
    features = io.load_features()
    split = split_train_test(features)

    if tune:
        model, test_scores, params = tune_lightgbm(split)
    else:
        params = config.LGBM_BASE_PARAMS
        model, test_scores = train_lightgbm(split, params)

    print(errors_by_month(build_test_results(model, features, split)).to_string(index=False))
    save_artifacts(model, split.X_train.columns, test_scores, params)
    return model
