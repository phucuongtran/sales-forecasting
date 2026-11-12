"""Run the whole project end to end from the command line.

    python -m src.pipeline                      # generate -> preprocess -> features -> train
    python -m src.pipeline --tune               # same, with Optuna tuning
    python -m src.pipeline --steps features train

Steps (in order):
    generate    create synthetic raw CSVs in data/
    preprocess  clean sales (missing values, outliers) and weather
    features    build the feature table (data/feature_engineered_data.feather)
    train       train LightGBM and save models/sales_forecast_model.pkl
"""

from __future__ import annotations

import argparse

from src.data import generator, preprocessing
from src.features import engineering
from src.modeling import train

STEPS = ["generate", "preprocess", "features", "train"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS, help="steps to run")
    parser.add_argument("--tune", action="store_true", help="tune LightGBM with Optuna")
    args = parser.parse_args(argv)

    for step in STEPS:  # always run in pipeline order
        if step not in args.steps:
            continue
        print(f"\n=== {step} ===")
        if step == "generate":
            generator.main()
        elif step == "preprocess":
            preprocessing.run()
        elif step == "features":
            engineering.run()
        elif step == "train":
            train.run(tune=args.tune)


if __name__ == "__main__":
    main()
