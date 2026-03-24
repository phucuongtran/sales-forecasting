from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.optimized.forecast import load_bundle, predict_with_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a prepared feature frame with the optimized ensemble.")
    parser.add_argument("feature_file", help="CSV or feather file containing the exact feature columns expected by the bundle")
    parser.add_argument("--bundle", default="models/optimized_sales_forecast_bundle.pkl")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    if args.feature_file.endswith(".feather"):
        frame = pd.read_feather(args.feature_file)
    else:
        frame = pd.read_csv(args.feature_file)

    pred = predict_with_bundle(bundle, frame)
    out = frame.copy()
    out["prediction"] = pred
    output_path = os.path.splitext(args.feature_file)[0] + "_scored.csv"
    out.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
