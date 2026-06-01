"""Train packaged pretrained curve-imputation models from geolink LAS data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from petrophys.imputation.training import describe_model_pack, train_pretrained_model_pack


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train KNN, RF, and XGBoost curve-imputation model pack."
    )
    parser.add_argument(
        "--data-dir",
        default="geolink_data",
        help="Directory containing LAS files used for training.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output joblib path. Defaults to petrophys/imputation/pretrained.",
    )
    parser.add_argument(
        "--models",
        default="knn,rf,xgboost",
        help="Comma-separated model names: knn, rf, xgboost.",
    )
    parser.add_argument(
        "--max-rows-per-well",
        type=int,
        default=1500,
        help="Maximum sampled rows per LAS file.",
    )
    parser.add_argument(
        "--max-samples-per-target",
        type=int,
        default=12000,
        help="Maximum training rows per target curve.",
    )
    parser.add_argument(
        "--min-samples-per-target",
        type=int,
        default=500,
        help="Skip targets with fewer usable rows than this.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling and model training.",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    pack = train_pretrained_model_pack(
        data_dir=args.data_dir,
        output_path=output,
        model_names=model_names,
        max_rows_per_well=args.max_rows_per_well,
        max_samples_per_target=args.max_samples_per_target,
        min_samples_per_target=args.min_samples_per_target,
        random_state=args.random_state,
    )

    summary = describe_model_pack(pack)
    if summary.empty:
        print("No target models were trained.")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
