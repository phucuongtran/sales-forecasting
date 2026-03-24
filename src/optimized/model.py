from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from .config import OptimizedTrainingConfig
from .metrics import mae, rmse, smape, wape


def _build_time_folds(
    train_dates: List[pd.Timestamp],
    n_splits: int,
    validation_days: int,
    gap_days: int,
    min_train_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    train_dates = sorted(pd.to_datetime(pd.Index(train_dates)).unique())
    if len(train_dates) < (min_train_days + validation_days + gap_days):
        raise ValueError("Not enough history to build time folds.")

    folds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    end_idx = len(train_dates) - 1
    for split_idx in range(n_splits, 0, -1):
        val_end_idx = end_idx - (n_splits - split_idx) * validation_days
        val_start_idx = val_end_idx - validation_days + 1
        train_end_idx = val_start_idx - gap_days - 1
        if train_end_idx < min_train_days - 1 or val_start_idx < 0:
            continue
        folds.append(
            (
                train_dates[0],
                train_dates[train_end_idx],
                train_dates[val_start_idx],
            )
        )
    # Convert each tuple into train_start, train_end, val_start; val_end implied by next segment length.
    output = []
    for _, train_end, val_start in folds:
        val_end = val_start + pd.Timedelta(days=validation_days - 1)
        output.append((train_dates[0], train_end, val_start, val_end))
    return output


def _prepare_train_test(df_features: pd.DataFrame):
    meta_cols = [
        "date",
        "store_name",
        "item_name",
        "is_test",
    ]
    feature_df = df_features.copy()
    feature_cols = [c for c in feature_df.columns if c not in meta_cols + ["sales"]]
    categorical_cols = [
        c
        for c in feature_cols
        if str(feature_df[c].dtype) == "category" or feature_df[c].dtype == object
    ]

    train_mask = ~feature_df["is_test"]
    test_mask = feature_df["is_test"]
    X_train = feature_df.loc[train_mask, feature_cols].copy()
    y_train = feature_df.loc[train_mask, "sales"].copy()
    X_test = feature_df.loc[test_mask, feature_cols].copy()
    y_test = feature_df.loc[test_mask, "sales"].copy()

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    train_dates = feature_df.loc[train_mask, "date"]
    test_dates = feature_df.loc[test_mask, "date"]
    return X_train, y_train, train_dates, X_test, y_test, test_dates, feature_cols, categorical_cols


def _lgb_params_from_trial(trial: optuna.trial.Trial, config: OptimizedTrainingConfig) -> Dict:
    objective = trial.suggest_categorical("objective", config.objective_options)
    params = {
        "objective": objective,
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1600),
        "random_state": config.random_seed,
        "num_threads": 4,
        "verbosity": -1,
    }
    if objective == "tweedie":
        params["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power", 1.1, 1.7
        )
    return params


def _fit_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Dict,
    categorical_cols: List[str],
    use_log_target: bool,
):
    apply_log_target = use_log_target and params.get("objective", "regression") == "regression"
    y_train_fit = np.log1p(y_train) if apply_log_target else y_train
    y_valid_fit = np.log1p(y_valid) if apply_log_target else y_valid

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train_fit,
        eval_set=[(X_valid, y_valid_fit)],
        eval_metric="rmse",
        categorical_feature=categorical_cols or "auto",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    pred = model.predict(X_valid, num_iteration=model.best_iteration_)
    if apply_log_target:
        pred = np.expm1(pred)
    pred = np.clip(pred, 0.0, None)
    return model, pred


def _optimize(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_dates: pd.Series,
    categorical_cols: List[str],
    config: OptimizedTrainingConfig,
):
    unique_dates = sorted(pd.to_datetime(train_dates.unique()))
    folds = _build_time_folds(
        unique_dates,
        n_splits=config.n_splits,
        validation_days=config.validation_days,
        gap_days=config.gap_days,
        min_train_days=config.min_train_days,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = _lgb_params_from_trial(trial, config)
        fold_scores = []
        for _, train_end, val_start, val_end in folds:
            train_mask = train_dates <= train_end
            valid_mask = (train_dates >= val_start) & (train_dates <= val_end)
            X_tr = X_train.loc[train_mask]
            y_tr = y_train.loc[train_mask]
            X_va = X_train.loc[valid_mask]
            y_va = y_train.loc[valid_mask]
            if X_tr.empty or X_va.empty:
                continue
            _, pred = _fit_single_model(
                X_tr, y_tr, X_va, y_va, params, categorical_cols, config.use_log_target
            )
            fold_scores.append(0.7 * wape(y_va, pred) + 0.3 * rmse(y_va, pred))
        if not fold_scores:
            return float("inf")
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.n_optuna_trials, show_progress_bar=False)
    return study, folds


def _train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    best_params: Dict,
    categorical_cols: List[str],
    config: OptimizedTrainingConfig,
):
    models = []
    preds = []
    for seed_offset in range(config.ensemble_size):
        params = dict(best_params)
        params["random_state"] = config.random_seed + seed_offset
        model, pred = _fit_single_model(
            X_train,
            y_train,
            X_valid,
            y_valid,
            params,
            categorical_cols,
            config.use_log_target,
        )
        models.append(model)
        preds.append(pred)
    ensemble_pred = np.mean(np.vstack(preds), axis=0)
    return models, ensemble_pred


def train_optimized_pipeline(df_features: pd.DataFrame, config: OptimizedTrainingConfig):
    X_train, y_train, train_dates, X_test, y_test, test_dates, feature_cols, categorical_cols = _prepare_train_test(df_features)
    study, folds = _optimize(X_train, y_train, train_dates, categorical_cols, config)
    best_params = _lgb_params_from_trial(study.best_trial, config)

    models, test_pred = _train_ensemble(
        X_train, y_train, X_test, y_test, best_params, categorical_cols, config
    )

    metrics = {
        "mae": mae(y_test, test_pred),
        "rmse": rmse(y_test, test_pred),
        "wape": wape(y_test, test_pred),
        "smape": smape(y_test, test_pred),
    }

    bundle = {
        "models": models,
        "feature_names": feature_cols,
        "categorical_features": categorical_cols,
        "best_params": best_params,
        "metrics": metrics,
        "config": asdict(config),
    }
    return bundle, study, folds, X_test, y_test, test_pred


def save_training_outputs(
    bundle: Dict,
    study: optuna.study.Study,
    folds,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_pred: np.ndarray,
    df_features: pd.DataFrame,
    config: OptimizedTrainingConfig,
) -> Dict[str, str]:
    os.makedirs(config.models_dir, exist_ok=True)
    bundle_path = os.path.join(config.models_dir, "optimized_sales_forecast_bundle.pkl")
    metrics_path = os.path.join(config.models_dir, "optimized_metrics.json")
    leaderboard_path = os.path.join(config.models_dir, "optimized_holdout_predictions.csv")
    study_path = os.path.join(config.models_dir, "optimized_optuna_trials.csv")

    joblib.dump(bundle, bundle_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": bundle["metrics"],
                "best_params": bundle["best_params"],
                "folds": [
                    {
                        "train_end": str(train_end.date()),
                        "val_start": str(val_start.date()),
                        "val_end": str(val_end.date()),
                    }
                    for _, train_end, val_start, val_end in folds
                ],
            },
            f,
            indent=2,
        )

    holdout = df_features[df_features["is_test"]][
        [
            c
            for c in [
                "date",
                "province",
                "store_id",
                "store_name",
                "category",
                "item_id",
                "item_name",
                "sales",
            ]
            if c in df_features.columns
        ]
    ].copy()
    holdout["prediction"] = test_pred
    holdout.to_csv(leaderboard_path, index=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(study_path, index=False)

    return {
        "bundle": bundle_path,
        "metrics": metrics_path,
        "holdout": leaderboard_path,
        "trials": study_path,
    }
