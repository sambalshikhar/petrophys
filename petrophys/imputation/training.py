"""
Training utilities for bundled curve-imputation model packs.

These utilities are intentionally optional. They require scikit-learn, joblib,
and xgboost only when training or loading pretrained ML models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from petrophys.imputation.curve_imputer import (
    DEFAULT_FEATURE_CURVES,
    DEFAULT_TARGET_CURVES,
    _canonical_curve_name,
    default_pretrained_model_path,
)
from petrophys.qaqc.standardization.alias_map import get_standard_mnemonic
from petrophys.qaqc.standardization.alias_map import get_standard_unit
from petrophys.qaqc.standardization.curve_standardizer import (
    CURVE_PLAUSIBLE_RANGES,
    POROSITY_FRACTION_CURVES,
)
from petrophys.qaqc.standardization.unit_converter import get_conversion_factor


def _require_training_dependencies():
    try:
        import joblib
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "Training curve-imputation models requires scikit-learn and joblib. "
            "Install the ML extras with: pip install petrophys[ml]"
        ) from exc

    return {
        "joblib": joblib,
        "BaseEstimator": BaseEstimator,
        "TransformerMixin": TransformerMixin,
        "TransformedTargetRegressor": TransformedTargetRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "SimpleImputer": SimpleImputer,
        "mean_absolute_error": mean_absolute_error,
        "r2_score": r2_score,
        "train_test_split": train_test_split,
        "KNeighborsRegressor": KNeighborsRegressor,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def _require_xgboost():
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "The xgboost model option requires xgboost. "
            "Install the ML extras with: pip install petrophys[ml]"
        ) from exc
    return XGBRegressor
def _normalize_model_names(model_names: Sequence[str]) -> List[str]:
    aliases = {
        "random_forest": "rf",
        "forest": "rf",
        "xgb": "xgboost",
        "xgbregressor": "xgboost",
    }
    normalized = []
    for name in model_names:
        key = str(name).strip().lower().replace("-", "_")
        normalized.append(aliases.get(key, key))
    return normalized


TRAINING_PLAUSIBLE_RANGES = {
    **CURVE_PLAUSIBLE_RANGES,
    "SP": (-500.0, 500.0),
}
RESISTIVITY_CURVES = {"RDEP", "RMED", "RSHA", "RXO"}


def _log_resistivity_target(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.log10(np.clip(values, 1.0e-3, None))


def _inverse_log_resistivity_target(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.power(10.0, values)


class CurveFeatureTransformer:
    """Apply curve-specific transforms before model fitting."""

    def __init__(self, features: Sequence[str]):
        self.features = list(features)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"features": self.features}

    def set_params(self, **params: Any):
        if "features" in params:
            self.features = list(params["features"])
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64).copy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for idx, feature in enumerate(self.features):
            if feature in RESISTIVITY_CURVES and idx < arr.shape[1]:
                column = arr[:, idx]
                valid = np.isfinite(column)
                column[valid] = np.log10(np.clip(column[valid], 1.0e-3, None))
                arr[:, idx] = column
        return arr


def _safe_numeric(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def _standardize_training_values(
    values: Any,
    standard_name: str,
    unit: str = "",
) -> np.ndarray:
    data = _safe_numeric(values)
    standard_name = _canonical_curve_name(standard_name)
    data[np.isfinite(data) & (data <= -900.0)] = np.nan

    standard_unit = get_standard_unit(standard_name)
    if unit and standard_unit:
        factor = get_conversion_factor(unit, standard_unit)
        if factor is not None:
            data = data * factor

    if standard_name in POROSITY_FRACTION_CURVES:
        unit_text = str(unit or "").lower().strip()
        finite = data[np.isfinite(data)]
        if finite.size:
            median = float(np.nanmedian(finite))
            if unit_text in {"%", "pu", "percent"} or median > 1.0:
                data = data / 100.0

    limits = TRAINING_PLAUSIBLE_RANGES.get(standard_name)
    if limits is not None:
        low, high = limits
        invalid = np.isfinite(data) & ((data < low) | (data > high))
        data[invalid] = np.nan
    return data


def _extract_standard_curves(
    las: Any,
    curves: Sequence[str],
) -> Dict[str, np.ndarray]:
    requested = {_canonical_curve_name(curve) for curve in curves}
    candidates: Dict[str, Tuple[int, np.ndarray]] = {}

    for curve in las.curves:
        mnemonic = str(getattr(curve, "mnemonic", "") or "").strip().upper()
        standard = _canonical_curve_name(mnemonic)
        if standard not in requested:
            continue
        try:
            unit = str(getattr(curve, "unit", "") or "")
            values = _standardize_training_values(las[mnemonic], standard, unit)
        except Exception:
            continue
        valid_count = int(np.sum(~np.isnan(values)))
        previous = candidates.get(standard)
        if previous is None or valid_count > previous[0]:
            candidates[standard] = (valid_count, values)

    return {name: values for name, (_, values) in candidates.items()}


def build_training_table(
    data_dir: Union[str, Path],
    curves: Optional[Sequence[str]] = None,
    max_rows_per_well: Optional[int] = 1500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a sampled training table from LAS files.

    Curves are mapped through the project alias map, so equivalent mnemonics
    such as ``DT`` and ``DTC`` share the same training column.
    """
    try:
        import lasio
    except ImportError as exc:
        raise ImportError(
            "lasio is required to build curve-imputation training tables."
        ) from exc

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_path}")

    curves = list(curves or DEFAULT_FEATURE_CURVES)
    curves = sorted({_canonical_curve_name(curve) for curve in curves})
    frames: List[pd.DataFrame] = []
    rng = np.random.default_rng(random_state)

    for las_path in sorted(data_path.glob("*.las")):
        try:
            las = lasio.read(str(las_path), ignore_header_errors=True)
            extracted = _extract_standard_curves(las, curves)
        except Exception:
            continue

        if len(extracted) < 2:
            continue

        n_rows = max(len(values) for values in extracted.values())
        row = {
            curve: extracted.get(curve, np.full(n_rows, np.nan, dtype=np.float64))
            for curve in curves
        }
        frame = pd.DataFrame(row)
        frame = frame.dropna(how="all")
        if frame.empty:
            continue

        if max_rows_per_well is not None and len(frame) > max_rows_per_well:
            selected = rng.choice(len(frame), size=max_rows_per_well, replace=False)
            frame = frame.iloc[np.sort(selected)]

        frames.append(frame.reset_index(drop=True))

    if not frames:
        raise ValueError(f"No usable LAS training rows found under {data_path}")

    return pd.concat(frames, axis=0, ignore_index=True)


def _make_estimator(
    model_name: str,
    deps: Dict[str, Any],
    random_state: int,
    features: Sequence[str],
    target: str,
):
    Pipeline = deps["Pipeline"]
    SimpleImputer = deps["SimpleImputer"]
    transform_step = ("curve_transform", CurveFeatureTransformer(features))

    if model_name == "knn":
        estimator = Pipeline(
            steps=[
                transform_step,
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", deps["StandardScaler"]()),
                (
                    "regressor",
                    deps["KNeighborsRegressor"](n_neighbors=15, weights="distance"),
                ),
            ]
        )
        return _wrap_target_transform(estimator, deps, target)

    if model_name == "rf":
        estimator = Pipeline(
            steps=[
                transform_step,
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    deps["RandomForestRegressor"](
                        n_estimators=80,
                        min_samples_leaf=3,
                        max_features=0.8,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return _wrap_target_transform(estimator, deps, target)

    if model_name == "xgboost":
        XGBRegressor = _require_xgboost()
        estimator = Pipeline(
            steps=[
                transform_step,
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=160,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="reg:squarederror",
                        tree_method="hist",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return _wrap_target_transform(estimator, deps, target)

    raise ValueError(f"Unsupported model name: {model_name}")


def _wrap_target_transform(estimator: Any, deps: Dict[str, Any], target: str):
    if target not in RESISTIVITY_CURVES:
        return estimator
    return deps["TransformedTargetRegressor"](
        regressor=estimator,
        func=_log_resistivity_target,
        inverse_func=_inverse_log_resistivity_target,
        check_inverse=False,
    )


def train_pretrained_model_pack(
    data_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    model_names: Sequence[str] = ("knn", "rf", "xgboost"),
    feature_curves: Sequence[str] = DEFAULT_FEATURE_CURVES,
    target_curves: Sequence[str] = DEFAULT_TARGET_CURVES,
    max_rows_per_well: Optional[int] = 1500,
    max_samples_per_target: int = 12000,
    min_samples_per_target: int = 500,
    min_available_features: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
    compress: int = 3,
) -> Dict[str, Any]:
    """
    Train KNN, random forest, and XGBoost regressors for target curves.

    The resulting joblib pack can be consumed by ``PretrainedCurveImputer``.
    """
    deps = _require_training_dependencies()
    model_names = _normalize_model_names(model_names)
    feature_curves = sorted({_canonical_curve_name(curve) for curve in feature_curves})
    target_curves = [_canonical_curve_name(curve) for curve in target_curves]
    output = Path(output_path) if output_path is not None else default_pretrained_model_path()
    output.parent.mkdir(parents=True, exist_ok=True)

    table = build_training_table(
        data_dir=data_dir,
        curves=feature_curves,
        max_rows_per_well=max_rows_per_well,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    pack: Dict[str, Any] = {
        "schema_version": 1,
        "name": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_data": str(Path(data_dir).resolve()),
        "feature_curves": feature_curves,
        "target_curves": [],
        "models": {name: {} for name in model_names},
        "metadata": {
            "max_rows_per_well": max_rows_per_well,
            "max_samples_per_target": max_samples_per_target,
            "min_samples_per_target": min_samples_per_target,
            "min_available_features": min_available_features,
            "targets": {},
        },
    }

    for target in target_curves:
        if target not in table.columns:
            continue
        features = [curve for curve in feature_curves if curve != target]
        target_frame = table.loc[
            table[target].notna() & (table[features].notna().sum(axis=1) >= min_available_features),
            [*features, target],
        ]
        target_frame = target_frame.replace([np.inf, -np.inf], np.nan)
        target_frame = target_frame.dropna(subset=[target])

        if len(target_frame) < min_samples_per_target:
            continue
        if len(target_frame) > max_samples_per_target:
            selected = rng.choice(
                len(target_frame),
                size=max_samples_per_target,
                replace=False,
            )
            target_frame = target_frame.iloc[np.sort(selected)]

        X = target_frame[features]
        y = target_frame[target].to_numpy(dtype=np.float64)
        if len(target_frame) > 10 and 0 < test_size < 1:
            X_train, X_test, y_train, y_test = deps["train_test_split"](
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
            held_out = True
        else:
            # Too few samples to hold out a test set: metrics are in-sample
            # (train == test), so flag them rather than report them as test scores.
            X_train, X_test, y_train, y_test = X, X, y, y
            held_out = False

        target_metadata = {
            "samples": int(len(target_frame)),
            "features": features,
            "models": {},
        }

        for model_name in model_names:
            estimator = _make_estimator(model_name, deps, random_state, features, target)
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
            metrics = {
                "r2": float(deps["r2_score"](y_test, predictions))
                if len(np.unique(y_test)) > 1
                else float("nan"),
                "mae": float(deps["mean_absolute_error"](y_test, predictions)),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "evaluation": "held_out_test" if held_out else "in_sample_train",
            }
            pack["models"][model_name][target] = {
                "estimator": estimator,
                "features": features,
            }
            target_metadata["models"][model_name] = metrics

        pack["target_curves"].append(target)
        pack["metadata"]["targets"][target] = target_metadata

    deps["joblib"].dump(pack, output, compress=compress)
    return pack


def describe_model_pack(pack: Mapping[str, Any]) -> pd.DataFrame:
    """Return a compact tabular summary of a trained model pack."""
    rows: List[Dict[str, Any]] = []
    target_meta = pack.get("metadata", {}).get("targets", {})
    for target, metadata in target_meta.items():
        for model_name, metrics in metadata.get("models", {}).items():
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "samples": metadata.get("samples"),
                    "features": len(metadata.get("features", [])),
                    "r2": metrics.get("r2"),
                    "mae": metrics.get("mae"),
                }
            )
    return pd.DataFrame(rows)
