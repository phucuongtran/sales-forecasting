from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.optimized import OptimizedTrainingConfig
from src.optimized.data import load_raw_inputs, prepare_base_frame
from src.optimized.features import build_feature_frame
from src.optimized.model import save_training_outputs, train_optimized_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an optimized leakage-safe LightGBM sales forecaster.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--cutoff-date", default="2017-10-01")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--validation-days", type=int, default=28)
    parser.add_argument("--gap-days", type=int, default=7)
    parser.add_argument("--min-train-days", type=int, default=180)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--no-log-target", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OptimizedTrainingConfig(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        cutoff_date=args.cutoff_date,
        n_optuna_trials=args.n_trials,
        n_splits=args.n_splits,
        validation_days=args.validation_days,
        gap_days=args.gap_days,
        min_train_days=args.min_train_days,
        ensemble_size=args.ensemble_size,
        use_log_target=not args.no_log_target,
    )

    sales, weather = load_raw_inputs(config.data_dir)
    base = prepare_base_frame(sales, weather)
    features = build_feature_frame(base, config)
    bundle, study, folds, X_test, y_test, test_pred = train_optimized_pipeline(features, config)
    paths = save_training_outputs(bundle, study, folds, X_test, y_test, test_pred, features, config)

    summary = {
        "rows": int(len(features)),
        "n_features": int(len(bundle["feature_names"])),
        "metrics": bundle["metrics"],
        "saved": paths,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
