"""
Pretrained curve imputation for LAS well logs.

The imputer fills null samples in an existing target curve from companion logs.
It does not create a new target curve when the target is absent from the well.
"""

from __future__ import annotations

import copy
from importlib import metadata as importlib_metadata
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from petrophys.preprocessing.standardization.alias_map import (
    get_aliases,
    get_standard_mnemonic,
    get_standard_unit,
)
from petrophys.preprocessing.standardization.curve_standardizer import (
    CURVE_PLAUSIBLE_RANGES,
    POROSITY_FRACTION_CURVES,
)
from petrophys.preprocessing.standardization.unit_converter import get_conversion_factor

if TYPE_CHECKING:
    from petrophys.model.well import Well


ML_IMPUTATION_NOTES_PARAMETER = "ML_IMPUTATION_NOTES"
DEFAULT_MODEL_PACK_NAME = "geolink_curve_imputation.joblib"
DEFAULT_MIN_VALIDATION_R2 = 0.5
DEFAULT_FEATURE_CURVES = [
    "BS",
    "CALI",
    "DRHO",
    "DTC",
    "DTS",
    "GR",
    "NPHI",
    "PEF",
    "RDEP",
    "RHOB",
    "RMED",
    "RSHA",
    "RXO",
    "SP",
]
DEFAULT_TARGET_CURVES = [
    "DTC",
    "NPHI",
    "RHOB",
]


class ModelCompatibilityError(RuntimeError):
    """Raised when a serialized model pack and runtime dependencies differ."""


@dataclass
class CurveImputationResult:
    """Summary of a curve-imputation operation."""

    target_curve: str
    actual_curve: str
    model_name: str
    model_pack: str
    missing_before: int
    imputed_count: int
    missing_after: int
    skipped_count: int
    intervals: List[Dict[str, Any]]
    features_used: List[str]
    note: str
    output_las: Optional[str] = None
    well: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable result summary."""
        return {
            "target_curve": self.target_curve,
            "actual_curve": self.actual_curve,
            "model_name": self.model_name,
            "model_pack": self.model_pack,
            "missing_before": self.missing_before,
            "imputed_count": self.imputed_count,
            "missing_after": self.missing_after,
            "skipped_count": self.skipped_count,
            "intervals": list(self.intervals),
            "features_used": list(self.features_used),
            "note": self.note,
            "output_las": self.output_las,
        }


def default_pretrained_model_path() -> Path:
    """Return the bundled pretrained geolink model-pack path."""
    return Path(__file__).resolve().parent / "pretrained" / DEFAULT_MODEL_PACK_NAME


def _load_joblib():
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load pretrained curve-imputation models. "
            "Install the ML extras (Python 3.11+) with: pip install 'petrophys[ml]'"
        ) from exc
    return joblib


def _version_series(version: str) -> Optional[Tuple[int, int]]:
    match = re.match(r"\s*(\d+)\.(\d+)", str(version))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _validate_model_pack_dependencies(
    model_pack: Mapping[str, Any],
    *,
    include_joblib: bool,
) -> None:
    recorded = model_pack.get("metadata", {}).get("dependency_versions", {})
    if not isinstance(recorded, Mapping) or not recorded:
        return

    packages = ["scikit-learn"]
    if include_joblib:
        packages.append("joblib")
    if "xgboost" in model_pack.get("models", {}):
        packages.append("xgboost")

    issues = []
    for package in packages:
        trained_version = recorded.get(package)
        if not trained_version:
            continue
        try:
            runtime_version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            issues.append(f"{package} is not installed")
            continue

        trained_series = _version_series(str(trained_version))
        runtime_series = _version_series(runtime_version)
        if trained_series is None or runtime_series != trained_series:
            required = (
                f"{trained_series[0]}.{trained_series[1]}.x"
                if trained_series is not None
                else str(trained_version)
            )
            issues.append(
                f"{package} {runtime_version} is installed; {required} is required"
            )

    if issues:
        pack_name = str(model_pack.get("name", "model pack"))
        details = "; ".join(issues)
        raise ModelCompatibilityError(
            f"Incompatible dependencies for {pack_name}: {details}. "
            "Install petrophys[ml] in Python 3.11+ or train a model pack "
            "with the active environment."
        )


def _load_serialized_model_pack(path: Path) -> Mapping[str, Any]:
    joblib = _load_joblib()
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:
        InconsistentVersionWarning = None

    try:
        with warnings.catch_warnings():
            if InconsistentVersionWarning is not None:
                warnings.simplefilter("ignore", InconsistentVersionWarning)
            model_pack = joblib.load(path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Loading pretrained curve-imputation models requires the ML extras "
            "(Python 3.11+). Install with: pip install 'petrophys[ml]'"
        ) from exc
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        from importlib.metadata import PackageNotFoundError, version

        installed = []
        for package in ("scikit-learn", "joblib", "xgboost"):
            try:
                installed.append(f"{package} {version(package)}")
            except PackageNotFoundError:
                installed.append(f"{package} (not installed)")
        raise ModelCompatibilityError(
            f"The pretrained models in {path.name} cannot be loaded with "
            f"{', '.join(installed)}. They require scikit-learn 1.8.x, joblib 1.5.x "
            "and xgboost 3.2.x (Python 3.11+): pip install 'petrophys[ml]'. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(model_pack, Mapping):
        raise ValueError(f"Model pack {path} does not contain a mapping.")
    _validate_model_pack_dependencies(model_pack, include_joblib=True)
    return model_pack


def _canonical_curve_name(mnemonic: str) -> str:
    return get_standard_mnemonic(str(mnemonic).strip().upper()).upper()


def _normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower().replace("-", "_")
    aliases = {
        "random_forest": "rf",
        "forest": "rf",
        "xgb": "xgboost",
        "xgbregressor": "xgboost",
        "knneighbors": "knn",
        "k_neighbors": "knn",
    }
    return aliases.get(normalized, normalized)


def _safe_curve_values(values: Any, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) != length:
        return np.full(length, np.nan, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def _values_in_model_units(
    values: Any,
    standard_name: str,
    unit: str = "",
    *,
    length: Optional[int] = None,
    strict_units: bool = True,
) -> Tuple[np.ndarray, float, str]:
    """Convert a curve to the physical-unit contract used by model packs.

    Returns standardized values, the multiplicative source-to-model factor,
    and the model unit. The factor can be inverted when predictions must be
    written back without changing the input curve's declared unit.
    """
    data = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if length is not None and len(data) != length:
        return np.full(length, np.nan, dtype=np.float64), 1.0, ""
    data[~np.isfinite(data)] = np.nan
    data[np.isfinite(data) & (data <= -900.0)] = np.nan

    standard_name = _canonical_curve_name(standard_name)
    standard_unit = str(get_standard_unit(standard_name) or "")
    source_unit = str(unit or "").strip()
    factor = 1.0
    if source_unit and standard_unit:
        conversion = get_conversion_factor(source_unit, standard_unit)
        if conversion is None:
            if strict_units:
                raise ValueError(
                    f"Cannot safely use {standard_name} in ML model: unit "
                    f"{source_unit!r} cannot be converted to {standard_unit!r}"
                )
        else:
            factor = float(conversion)
            data *= factor

    # Unitless vendor porosity curves are common. Apply the magnitude heuristic
    # only after declared-unit conversion so percent units are never divided twice.
    if standard_name in POROSITY_FRACTION_CURVES:
        finite = data[np.isfinite(data)]
        if finite.size and float(np.nanmedian(finite)) > 1.0:
            data /= 100.0
            factor *= 0.01

    limits = CURVE_PLAUSIBLE_RANGES.get(standard_name)
    if limits is None and standard_name == "SP":
        limits = (-500.0, 500.0)
    if limits is not None:
        low, high = limits
        invalid = np.isfinite(data) & ((data < low) | (data > high))
        data[invalid] = np.nan
    return data, factor, standard_unit


def _format_interval(interval: Mapping[str, Any], depth_unit: str) -> str:
    samples = int(interval.get("samples", 0))
    if interval.get("top") is None or interval.get("bottom") is None:
        start = interval.get("start_index", "")
        end = interval.get("end_index", "")
        return f"index {start}-{end} ({samples} samples)"

    top = float(interval["top"])
    bottom = float(interval["bottom"])
    if np.isclose(top, bottom):
        return f"{top:.3f} {depth_unit} ({samples} samples)"
    return f"{top:.3f}-{bottom:.3f} {depth_unit} ({samples} samples)"


def _build_intervals(mask: np.ndarray, depth: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    intervals: List[Dict[str, Any]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        if idx == previous + 1:
            previous = idx
            continue
        intervals.append(_make_interval(start, previous, depth))
        start = previous = idx
    intervals.append(_make_interval(start, previous, depth))
    return intervals


def _make_interval(
    start_index: int,
    end_index: int,
    depth: Optional[np.ndarray],
) -> Dict[str, Any]:
    interval = {
        "start_index": int(start_index),
        "end_index": int(end_index),
        "samples": int(end_index - start_index + 1),
        "top": None,
        "bottom": None,
    }
    if depth is not None and len(depth) > end_index:
        interval["top"] = float(depth[start_index])
        interval["bottom"] = float(depth[end_index])
    return interval


class PretrainedCurveImputer:
    """
    Fill missing samples in existing curves using packaged pretrained models.

    Parameters
    ----------
    model_pack : dict, optional
        In-memory model pack. Mainly useful for tests and custom loading.
    model_path : str or Path, optional
        Path to a joblib model pack. If omitted, the bundled geolink pack is
        loaded.

    Security
    --------
    Model packs are deserialized with ``joblib.load`` (pickle), which executes
    arbitrary code embedded in the file. Only load packs from a trusted source.
    Loading a ``model_path`` that did not come from you is unsafe; a warning is
    emitted whenever a non-default path is loaded.
    """

    def __init__(
        self,
        model_pack: Optional[Mapping[str, Any]] = None,
        model_path: Optional[Union[str, Path]] = None,
    ):
        loaded_from_file = model_pack is None
        if model_pack is None:
            default_path = default_pretrained_model_path()
            path = Path(model_path) if model_path is not None else default_path
            if not path.exists():
                raise FileNotFoundError(
                    f"Pretrained curve-imputation model pack not found: {path}. "
                    "Generate it with scripts/train_curve_imputation_models.py."
                )
            if path != default_path:
                warnings.warn(
                    f"Loading a curve-imputation model pack from {path} with "
                    "joblib.load (pickle). This executes arbitrary code in the "
                    "file — only load packs you trust.",
                    stacklevel=2,
                )
            model_pack = _load_serialized_model_pack(path)
            self.model_path = str(path)
        else:
            self.model_path = str(model_path or "<in-memory>")

        self.model_pack: Dict[str, Any] = dict(model_pack)
        self.model_pack.setdefault("name", Path(self.model_path).name)
        if not loaded_from_file:
            _validate_model_pack_dependencies(
                self.model_pack,
                include_joblib=False,
            )

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, Path]] = None,
    ) -> "PretrainedCurveImputer":
        """Load the bundled or user-specified pretrained model pack."""
        return cls(model_path=model_path)

    @property
    def model_names(self) -> List[str]:
        """Return algorithm names available in the loaded model pack."""
        return sorted(str(name) for name in self.model_pack.get("models", {}).keys())

    @property
    def target_curves(self) -> List[str]:
        """Return target curves available in at least one algorithm."""
        targets = set()
        for model_targets in self.model_pack.get("models", {}).values():
            targets.update(str(target).upper() for target in model_targets.keys())
        return sorted(targets)

    def available_targets(self, model_name: Optional[str] = None) -> List[str]:
        """Return target curves available for a model or for the whole pack."""
        if model_name is None:
            return self.target_curves
        name = _normalize_model_name(model_name)
        return sorted(
            str(target).upper()
            for target in self.model_pack.get("models", {}).get(name, {}).keys()
        )

    def impute_missing(
        self,
        well: "Well",
        target_curve: str,
        model_name: str = "auto",
        min_available_features: int = 2,
        min_validation_r2: float = DEFAULT_MIN_VALIDATION_R2,
        inplace: bool = True,
    ) -> CurveImputationResult:
        """
        Impute null samples in an existing target curve.

        Existing non-null target values are left unchanged. If the target curve
        is absent, a ``KeyError`` is raised instead of creating a new curve.
        """
        working_well = well if inplace else copy.deepcopy(well)
        target = _canonical_curve_name(target_curve)
        actual_target = self._resolve_curve_name(working_well, target)
        if actual_target is None:
            raise KeyError(
                f"Target curve {target_curve!r} is not present in the well; "
                "the pretrained imputer only fills null samples in existing curves."
            )

        model_name = self._select_model(model_name, target, min_validation_r2)
        estimator, features = self._get_estimator(model_name, target)
        target_curve_object = working_well[actual_target]
        raw_target_values = _safe_curve_values(
            target_curve_object.data, working_well.n_samples
        )
        target_values, target_scale, _ = _values_in_model_units(
            raw_target_values,
            target,
            getattr(target_curve_object, "unit", ""),
            length=working_well.n_samples,
            strict_units=True,
        )
        missing_mask = np.isnan(target_values)
        missing_before = int(np.sum(missing_mask))

        if missing_before == 0:
            note = f"ML_IMPUTATION: {actual_target} had no missing samples to impute."
            return CurveImputationResult(
                target_curve=target,
                actual_curve=actual_target,
                model_name=model_name,
                model_pack=str(self.model_pack.get("name", self.model_path)),
                missing_before=0,
                imputed_count=0,
                missing_after=0,
                skipped_count=0,
                intervals=[],
                features_used=[],
                note=note,
                well=working_well,
            )

        feature_frame, features_used = self._build_feature_frame(
            working_well,
            target,
            features,
        )
        available_feature_counts = feature_frame.notna().sum(axis=1).to_numpy()
        impute_mask = missing_mask & (available_feature_counts >= min_available_features)
        imputed_count = int(np.sum(impute_mask))
        skipped_count = missing_before - imputed_count

        if imputed_count:
            predictions = estimator.predict(feature_frame.loc[impute_mask, features])
            model_predictions = np.asarray(predictions, dtype=np.float64)
            valid_predictions = np.isfinite(model_predictions)
            if self.model_pack.get("unit_contract"):
                limits = CURVE_PLAUSIBLE_RANGES.get(target)
                if limits is not None:
                    low, high = limits
                    valid_predictions &= (
                        (model_predictions >= low) & (model_predictions <= high)
                    )
            prediction_rows = np.where(impute_mask)[0]
            accepted_rows = prediction_rows[valid_predictions]
            rejected_count = int(len(prediction_rows) - len(accepted_rows))
            impute_mask = np.zeros_like(impute_mask)
            impute_mask[accepted_rows] = True
            imputed_count = int(len(accepted_rows))
            skipped_count = missing_before - imputed_count

            updated = raw_target_values.copy()
            if imputed_count:
                updated[accepted_rows] = model_predictions[valid_predictions] / target_scale
            working_well[actual_target].data = updated
            if rejected_count:
                warnings.warn(
                    f"Rejected {rejected_count} non-finite or physically implausible "
                    f"{target} predictions from model {model_name}.",
                    stacklevel=2,
                )
            self._record_curve_metadata(
                working_well,
                actual_target,
                model_name,
                features_used,
                imputed_count,
            )
        else:
            updated = raw_target_values

        missing_after = int(np.sum(np.isnan(updated)))
        intervals = _build_intervals(impute_mask, working_well.depth)
        note = self._make_note(
            actual_target,
            model_name,
            imputed_count,
            skipped_count,
            intervals,
            features_used,
            getattr(working_well, "depth_unit", ""),
        )
        if imputed_count:
            self._append_imputation_note(working_well, note)

        return CurveImputationResult(
            target_curve=target,
            actual_curve=actual_target,
            model_name=model_name,
            model_pack=str(self.model_pack.get("name", self.model_path)),
            missing_before=missing_before,
            imputed_count=imputed_count,
            missing_after=missing_after,
            skipped_count=skipped_count,
            intervals=intervals,
            features_used=features_used,
            note=note,
            well=working_well,
        )

    def _select_model(
        self,
        requested: str,
        target: str,
        min_validation_r2: float,
    ) -> str:
        """Resolve an explicit model or choose the best well-held-out model."""
        normalized = _normalize_model_name(requested)
        if normalized != "auto":
            return normalized

        metadata = (
            self.model_pack.get("metadata", {})
            .get("targets", {})
            .get(target, {})
            .get("models", {})
        )
        candidates = []
        for name in self.model_names:
            if target not in self.model_pack.get("models", {}).get(name, {}):
                continue
            metrics = metadata.get(name, {})
            evaluation = str(metrics.get("evaluation", ""))
            r2 = metrics.get("r2")
            if not evaluation.startswith("held_out_wells"):
                continue
            try:
                score = float(r2)
            except (TypeError, ValueError):
                continue
            if np.isfinite(score) and score >= min_validation_r2:
                candidates.append((score, name))

        if not candidates:
            raise ValueError(
                f"No grouped held-out model for {target} meets R2 >= "
                f"{min_validation_r2:.3f}. Select a model explicitly only after "
                "reviewing its validation metrics."
            )
        return max(candidates)[1]

    def _get_estimator(
        self,
        model_name: str,
        target: str,
    ) -> Tuple[Any, List[str]]:
        models = self.model_pack.get("models", {})
        if model_name not in models:
            available = ", ".join(self.model_names) or "none"
            raise KeyError(f"Model {model_name!r} is not available. Available: {available}")

        target_models = models[model_name]
        if target not in target_models:
            available = ", ".join(sorted(target_models.keys())) or "none"
            raise KeyError(
                f"Target {target!r} is not available for model {model_name!r}. "
                f"Available targets: {available}"
            )

        entry = target_models[target]
        if isinstance(entry, Mapping):
            estimator = entry.get("estimator") or entry.get("model")
            features = list(entry.get("features") or self.model_pack.get("feature_curves", []))
        else:
            estimator = entry
            features = list(self.model_pack.get("feature_curves", []))

        features = [_canonical_curve_name(feature) for feature in features]
        features = [feature for feature in features if feature != target]
        if estimator is None:
            raise ValueError(f"Model entry for {model_name}/{target} has no estimator.")
        if not features:
            raise ValueError(f"Model entry for {model_name}/{target} has no features.")
        return estimator, features

    def _build_feature_frame(
        self,
        well: "Well",
        target: str,
        features: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        n_samples = well.n_samples
        data: Dict[str, np.ndarray] = {}
        features_used: List[str] = []
        for feature in features:
            if feature == target:
                continue
            actual = self._resolve_curve_name(well, feature)
            if actual is None:
                data[feature] = np.full(n_samples, np.nan, dtype=np.float64)
                continue
            values = _safe_curve_values(well[actual].data, n_samples)
            try:
                values, _, _ = _values_in_model_units(
                    values,
                    feature,
                    getattr(well[actual], "unit", ""),
                    length=n_samples,
                    strict_units=True,
                )
            except ValueError as exc:
                warnings.warn(str(exc), stacklevel=2)
                values = np.full(n_samples, np.nan, dtype=np.float64)
            data[feature] = values
            if np.any(~np.isnan(values)):
                features_used.append(feature)

        frame = pd.DataFrame(data, columns=list(data.keys()))
        return frame, features_used

    def _resolve_curve_name(self, well: "Well", standard_name: str) -> Optional[str]:
        standard_name = _canonical_curve_name(standard_name)
        if hasattr(well, "has_curve") and well.has_curve(standard_name):
            return standard_name

        aliases = [standard_name, *get_aliases(standard_name)]
        for alias in aliases:
            alias = alias.upper()
            if hasattr(well, "has_curve") and well.has_curve(alias):
                return alias

        for curve_name in getattr(well, "curves", []):
            if _canonical_curve_name(curve_name) == standard_name:
                return str(curve_name).upper()
        return None

    def _record_curve_metadata(
        self,
        well: "Well",
        actual_target: str,
        model_name: str,
        features_used: Sequence[str],
        imputed_count: int,
    ) -> None:
        curve = well[actual_target]
        custom = getattr(curve.metadata, "custom", None)
        if custom is None:
            return
        custom["ml_imputation"] = {
            "model": model_name,
            "model_pack": self.model_pack.get("name", self.model_path),
            "imputed_count": int(imputed_count),
            "features_used": list(features_used),
        }

    def _append_imputation_note(self, well: "Well", note: str) -> None:
        if not hasattr(well, "get_parameter") or not hasattr(well, "set_parameter"):
            return
        notes = well.get_parameter(ML_IMPUTATION_NOTES_PARAMETER, [])
        if isinstance(notes, str):
            notes = [notes]
        elif not isinstance(notes, list):
            notes = list(notes) if isinstance(notes, Iterable) else []
        notes.append(note)
        well.set_parameter(ML_IMPUTATION_NOTES_PARAMETER, notes)

    def _make_note(
        self,
        actual_target: str,
        model_name: str,
        imputed_count: int,
        skipped_count: int,
        intervals: Sequence[Mapping[str, Any]],
        features_used: Sequence[str],
        depth_unit: str,
    ) -> str:
        pack_name = str(self.model_pack.get("name", self.model_path))
        if imputed_count == 0:
            return (
                f"ML_IMPUTATION: {actual_target} had missing samples, but none met "
                f"the companion-log availability rule for pretrained {model_name}."
            )

        listed = list(intervals[:10])
        interval_text = "; ".join(
            _format_interval(interval, depth_unit or "depth units") for interval in listed
        )
        remaining = len(intervals) - len(listed)
        if remaining > 0:
            interval_text = f"{interval_text}; +{remaining} more intervals"

        feature_text = ",".join(features_used) if features_used else "none"
        skipped_text = f"; skipped {skipped_count} missing samples" if skipped_count else ""
        interval_label = "interval" if len(intervals) == 1 else "intervals"
        return (
            f"ML_IMPUTATION: {actual_target} missing samples were imputed with "
            f"pretrained {model_name} model {pack_name} using {feature_text}; "
            f"imputed {imputed_count} samples in {len(intervals)} {interval_label}"
            f"{skipped_text}: {interval_text}"
        )


def load_pretrained_imputer(
    model_path: Optional[Union[str, Path]] = None,
) -> PretrainedCurveImputer:
    """Load the bundled or user-specified pretrained curve imputer."""
    return PretrainedCurveImputer.from_pretrained(model_path=model_path)


def impute_missing_curve(
    well: "Well",
    target_curve: str,
    model_name: str = "auto",
    model_path: Optional[Union[str, Path]] = None,
    min_available_features: int = 2,
    min_validation_r2: float = DEFAULT_MIN_VALIDATION_R2,
    inplace: bool = True,
) -> CurveImputationResult:
    """Convenience wrapper for pretrained curve imputation on a Well object."""
    imputer = load_pretrained_imputer(model_path=model_path)
    return imputer.impute_missing(
        well=well,
        target_curve=target_curve,
        model_name=model_name,
        min_available_features=min_available_features,
        min_validation_r2=min_validation_r2,
        inplace=inplace,
    )


def impute_las_file(
    input_las: Union[str, Path],
    output_las: Union[str, Path],
    target_curve: str,
    model_name: str = "auto",
    model_path: Optional[Union[str, Path]] = None,
    min_available_features: int = 2,
    min_validation_r2: float = DEFAULT_MIN_VALIDATION_R2,
    **export_kwargs: Any,
) -> CurveImputationResult:
    """
    Read a LAS file, impute missing target samples, and write an output LAS.

    The output LAS keeps the original target mnemonic and adds an ML imputation
    note to the LAS header when samples were filled.
    """
    from petrophys.io import read_las
    from petrophys.io.export import export_las

    well = read_las(input_las)
    result = impute_missing_curve(
        well=well,
        target_curve=target_curve,
        model_name=model_name,
        model_path=model_path,
        min_available_features=min_available_features,
        min_validation_r2=min_validation_r2,
        inplace=True,
    )
    export_las(well, output_las, **export_kwargs)
    result.output_las = str(output_las)
    return result
