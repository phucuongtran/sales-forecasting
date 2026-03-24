# Optimization audit and concrete fixes

This repository already has a good PoC structure, but there are **four issues that cap real forecasting performance**.

## 1) Target leakage in engineered features

The original notebooks compute these features using **same-day sales**:

- `store_mean_7d`
- `store_sum_7d`
- `item_mean_7d`
- `item_sum_7d`

Because those features are built from the target day itself, the model can "peek" at the answer. That inflates offline metrics and makes future forecasting less reliable.

### Fix

The optimized pipeline replaces them with **shifted, leakage-safe daily rollups**:

- `store_lag_1`, `store_mean_7d`, `store_sum_7d`
- `item_lag_1`, `item_mean_7d`, `item_sum_7d`
- plus category- and province-level lagged rollups

All of them are computed from **history up to day t-1 only**.

## 2) Hyperparameter tuning used the test set

The Optuna block in the original notebook evaluates candidate parameters directly on `X_test, y_test`. That makes the final test score optimistic because the test set is no longer untouched.

### Fix

The optimized training script uses **expanding time folds** on the training period only:

- last 4 validation windows
- configurable validation length
- configurable temporal gap to reduce leakage from adjacent dates

The holdout test period remains untouched until the end.

## 3) Validation was row-based rather than date-based

`TimeSeriesSplit` on the fully stacked panel can mix many stores/items inside each fold in a way that is not equivalent to a strict future-date validation.

### Fix

The optimized pipeline builds folds from **unique dates**, not row order, then trains on all rows up to `train_end` and validates on the next time block.

## 4) The Streamlit prediction logic reuses the latest row and injects random noise

That makes demos look dynamic, but it is not a true forecasting workflow.

### Fix

The optimized path focuses first on **correct training and holdout scoring**. The new bundle is deterministic and can be used by a future recursive forecaster without random variation.

## Added optimized workflow

New files:

- `src/optimized/data.py`
- `src/optimized/features.py`
- `src/optimized/model.py`
- `src/optimized/forecast.py`
- `scripts/train_optimized.py`
- `scripts/predict_with_bundle.py`

## Recommended command

```bash
python scripts/train_optimized.py --n-trials 60 --ensemble-size 3
```

## Why this should perform better in practice

Not because it "cheats" more, but because it is a stronger and cleaner forecasting setup:

- leakage-safe features
- stronger hierarchical rollups
- cyclical calendar features
- native categorical handling in LightGBM
- log-target option for heavy-tailed sales
- Optuna tuned on proper temporal folds
- small ensemble for better stability
