"""Measure how well ``pp.impute`` rebuilds a curve in wells it did not learn from.

The wells (LAS files in the folders given) are split at random into training
wells and test wells. In each test well every alias of the curve is removed,
the curve is rebuilt with ``pp.impute(well, curve, train_wells=...)``, and the
rebuilt samples are scored against the measured log: R2 and mean absolute
error over all test samples together. Held or zero-filled readings (runs of 10
or more identical samples) are not scored. The figures on the analysis-tools
page come from the 223 GeoLink North Sea wells with seeds 0, 1 and 2.

Usage::

    python scripts/evaluate_imputation.py FOLDER [FOLDER ...] [--seed 0]
        [--train-fraction 0.8] [--curves RHOB,DTC,NPHI] [--report scores.json]
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np

import petrophys as pp
from petrophys.preprocessing.imputation.impute import _find_curve, _learnable, _standard_values


def score_curve(curve, test, train):
    """R2 and MAE of the rebuilt curve over the test wells, with the wells scored and skipped."""
    truth_all, rebuilt_all, scored, skipped = [], [], [], []
    for well in test:
        mnemonic = _find_curve(well, curve)
        if mnemonic is None:
            skipped.append((str(well.name), "no curve"))
            continue
        truth = _learnable(_standard_values(well, mnemonic, curve)[0])
        if np.isfinite(truth).sum() < 200:
            skipped.append((str(well.name), "fewer than 200 scorable samples"))
            continue
        target = copy.deepcopy(well)
        while _find_curve(target, curve) is not None:
            target.remove_curve(_find_curve(target, curve))
        try:
            result = pp.impute(target, curve, train_wells=train)
        except Exception as error:  # noqa: BLE001 - report and go on
            skipped.append((str(well.name), f"{type(error).__name__}: {error}"))
            continue
        name = next(str(c) for c in result.curves if str(c).endswith("_IMPUTED"))
        rebuilt = np.asarray(result[name].data, dtype=float)
        ok = np.isfinite(truth) & np.isfinite(rebuilt)
        truth_all.append(truth[ok])
        rebuilt_all.append(rebuilt[ok])
        scored.append((str(well.name), int(ok.sum()), float(np.mean(np.abs(rebuilt[ok] - truth[ok])))))
    y = np.concatenate(truth_all)
    p = np.concatenate(rebuilt_all)
    return {
        "wells_scored": len(scored),
        "samples": int(y.size),
        "r2": float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)),
        "mae": float(np.mean(np.abs(p - y))),
        "per_well_mae": scored,
        "skipped": skipped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--curves", default="RHOB,DTC,NPHI")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    warnings.simplefilter("ignore")
    files = sorted(path for folder in args.folders for path in folder.glob("*.las"))
    wells = [pp.read(path) for path in files]
    order = np.random.default_rng(args.seed).permutation(len(wells))
    n_train = int(round(args.train_fraction * len(wells)))
    train = [wells[i] for i in order[:n_train]]
    test = [wells[i] for i in order[n_train:]]
    print(f"{len(wells)} wells: {len(train)} training, {len(test)} test (seed {args.seed})")

    report = {"seed": args.seed, "train_wells": len(train), "test_wells": len(test), "curves": {}}
    for curve in args.curves.split(","):
        scores = score_curve(curve, test, train)
        report["curves"][curve] = scores
        print(f"{curve}: R2 {scores['r2']:.3f}, MAE {scores['mae']:.4g} over {scores['samples']} samples "
              f"of {scores['wells_scored']} wells ({len(scores['skipped'])} skipped)")
    if args.report:
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
