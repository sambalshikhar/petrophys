"""
Shared helpers for interval-based analysis outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from petrophys.model.data_types import DataType
from petrophys.model.units import normalize_depth_unit
from petrophys._internal.statistics import finite_min_max

if TYPE_CHECKING:
    from petrophys.model.well import Well

_LAS_OCCURRENCE_SUFFIX = re.compile(r":\d+$")


def base_curve_mnemonic(mnemonic: str) -> str:
    """Return a LAS mnemonic without lasio's duplicate-occurrence suffix."""
    return _LAS_OCCURRENCE_SUFFIX.sub("", str(mnemonic).strip().upper())


@dataclass
class DerivedCurve:
    """Derived curve payload to add back onto a Well."""

    data: np.ndarray
    unit: str = ""
    description: str = ""


@dataclass
class AnalysisSection:
    """A single grouped classification section."""

    description: str
    methodology: str
    calculation_thresholds: str
    raw_labels: np.ndarray
    intervals: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "description": self.description,
            "methodology": self.methodology,
            "calculation_thresholds": self.calculation_thresholds,
        }
        payload.update(self.intervals)
        return payload


@dataclass
class CategoryAnalysisResult:
    """Category-level result with grouped sections and derived curves."""

    category_key: str
    sections: dict[str, AnalysisSection]
    derived_curves: dict[str, DerivedCurve] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_curves_to_well(self, well: "Well") -> "Well":
        for mnemonic, curve in self.derived_curves.items():
            try:
                numeric = np.asarray(curve.data, dtype=float)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None and not np.any(np.isfinite(numeric)):
                continue
            well.add_curve(
                mnemonic,
                curve.data,
                curve.unit,
                curve.description,
                data_type=DataType.COMPUTED,
            )
        return well

    def to_dict(self) -> dict[str, Any]:
        payload = {name: section.to_dict() for name, section in self.sections.items()}
        payload.update(self.metadata)
        return {self.category_key: payload}


@dataclass
class ComprehensiveAnalysisResult:
    """Top-level wrapper for all category analyses."""

    categories: dict[str, CategoryAnalysisResult]

    def add_curves_to_well(self, well: "Well") -> "Well":
        for result in self.categories.values():
            result.add_curves_to_well(well)
        return well

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for result in self.categories.values():
            payload.update(result.to_dict())
        return payload


def get_curve_data(
    well: "Well",
    names: Sequence[str],
    require_finite: bool = True,
    combine_aliases: bool = True,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Return usable curve data for a list of aliases.

    By default aliases are combined depth-wise: the first alias supplies each
    sample where finite, and lower-priority aliases fill only missing samples.
    Use ``combine_aliases=False`` to get the first finite curve as a whole, or
    ``require_finite=False`` when inspecting raw curve presence.
    """
    if require_finite and combine_aliases:
        data, mnemonic, _ = get_curve_data_depthwise(well, names)
        return data, mnemonic

    curve = get_curve(well, names, require_finite=require_finite)
    if curve is None:
        return None, None
    return np.asarray(curve.data, dtype=float), curve.mnemonic


def well_depth_array(well: "Well", require_finite: bool = False) -> np.ndarray:
    """Return a well depth array with a clear error when depth is unavailable."""
    depth = getattr(well, "depth", None)
    if depth is None:
        raise ValueError("Well has no depth data")
    depth_arr = np.asarray(depth, dtype=float).reshape(-1)
    if require_finite and not np.any(np.isfinite(depth_arr)):
        raise ValueError("Well has no finite depth values")
    return depth_arr


def resolve_well_depth_range(
    well: "Well",
    top_depth: Optional[float] = None,
    bottom_depth: Optional[float] = None,
) -> tuple[float, float]:
    """Resolve an analysis interval from explicit bounds or finite well depth."""
    top_value = top_depth
    bottom_value = bottom_depth

    if top_value is None:
        top_value = getattr(well, "start_depth", None)
    if bottom_value is None:
        bottom_value = getattr(well, "stop_depth", None)

    if top_value is None or bottom_value is None:
        inferred_top, inferred_bottom = finite_min_max(well_depth_array(well))
        if top_value is None:
            top_value = inferred_top
        if bottom_value is None:
            bottom_value = inferred_bottom

    if top_value is None or bottom_value is None:
        raise ValueError("Well has no finite depth range")
    return float(top_value), float(bottom_value)


def curve_has_finite_data(curve: Any) -> bool:
    """Return True when a curve object has at least one finite numeric sample."""
    if curve is None:
        return False
    try:
        data = np.asarray(curve.data, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.any(np.isfinite(data)))


def curve_source_mnemonic(curve: Any) -> Optional[str]:
    """Return the original input mnemonic behind a canonical working alias."""
    if curve is None:
        return None
    metadata = getattr(curve, "metadata", None)
    custom = getattr(metadata, "custom", {}) or {}
    source = custom.get("working_alias_source")
    if source:
        return str(source)
    mnemonic = getattr(curve, "mnemonic", None)
    return str(mnemonic) if mnemonic else None


def get_curve(
    well: "Well",
    names: Sequence[str],
    require_finite: bool = True,
):
    """Return the first curve matching aliases, preferring finite data."""
    first_existing = None
    for name in names:
        curve = well.get(name)
        if curve is None:
            continue
        if first_existing is None:
            first_existing = curve
        if not require_finite or curve_has_finite_data(curve):
            return curve
    return None if require_finite else first_existing


def get_curve_data_depthwise(
    well: "Well",
    names: Sequence[str],
) -> tuple[Optional[np.ndarray], Optional[str], dict[str, Any]]:
    """Build a depth-wise composite array from aliased curves.

    For aliases such as ``RDEP/RT`` or ``RSHA/RXO/RESS``, this keeps the
    priority order per sample instead of letting sparse primary curves block
    usable fallback aliases at other depths.
    """
    n_samples = int(getattr(well, "n_samples", 0) or 0)
    first_existing = None
    existing_curves: list[str] = []
    used_curves: list[str] = []
    source_per_sample = np.full(n_samples, "", dtype=object)
    result = np.full(n_samples, np.nan, dtype=float)

    for name in names:
        curve = well.get(name)
        if curve is None:
            continue
        if first_existing is None:
            first_existing = curve
        source_name = curve_source_mnemonic(curve) or curve.mnemonic
        existing_curves.append(source_name)
        try:
            data = np.asarray(curve.data, dtype=float)
        except (TypeError, ValueError):
            continue
        if data.ndim == 0:
            data = np.full(n_samples, float(data), dtype=float)
        elif data.size != n_samples:
            if data.size == 1:
                data = np.full(n_samples, float(data.reshape(-1)[0]), dtype=float)
            else:
                continue
        else:
            data = data.reshape(n_samples)

        fill = ~np.isfinite(result) & np.isfinite(data)
        if not np.any(fill):
            continue
        result[fill] = data[fill]
        source_per_sample[fill] = source_name
        used_curves.append(source_name)

    source_counts = {
        name: int(np.count_nonzero(source_per_sample == name))
        for name in used_curves
    }
    metadata: dict[str, Any] = {
        "existing_curves": existing_curves,
        "used_curves": used_curves,
        "source_counts": source_counts,
        "source_per_sample": source_per_sample,
    }
    if not used_curves:
        metadata["first_existing_curve"] = curve_source_mnemonic(first_existing)
        return None, None, metadata
    return result, "+".join(used_curves), metadata


def format_curve_source(source: Optional[str], missing: str = "unavailable") -> str:
    """Return a concise human-readable selected-curve/source label."""
    if not source:
        return missing
    parts = [part for part in str(source).split("+") if part]
    if len(parts) > 1:
        return "depth-wise " + " -> ".join(parts)
    return str(source)


def safe_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    fill_value: float = np.nan,
    min_abs: float = 1e-10,
) -> np.ndarray:
    """Safely divide arrays/scalars and return a float ndarray."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast(num, den).shape, fill_value, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > min_abs)
    result[valid] = num[valid] / den[valid]
    return result


def interval_strings(
    depth: np.ndarray,
    labels: np.ndarray,
    depth_unit: str,
) -> dict[str, list[str]]:
    """Group consecutive identical labels into depth intervals."""
    grouped: dict[str, list[str]] = {}
    if depth is None:
        return grouped

    depth_arr = np.asarray(depth, dtype=float)
    label_arr = np.asarray(labels, dtype=object)
    if len(depth_arr) == 0 or len(label_arr) == 0:
        return grouped

    start = 0
    previous = label_arr[0]

    def _is_missing(value: Any) -> bool:
        return value is None or (isinstance(value, float) and np.isnan(value)) or value == ""

    for idx in range(1, len(label_arr) + 1):
        current = label_arr[idx] if idx < len(label_arr) else None
        if idx < len(label_arr) and current == previous:
            continue

        if not _is_missing(previous):
            top = float(depth_arr[start])
            bottom = float(depth_arr[idx - 1])
            grouped.setdefault(str(previous), []).append(
                f"{top:.1f}-{bottom:.1f} {depth_unit}"
            )

        if idx < len(label_arr):
            start = idx
            previous = current

    return grouped


def build_section(
    well: "Well",
    labels: np.ndarray,
    description: str,
    methodology: str,
    calculation_thresholds: str,
) -> AnalysisSection:
    """Build an interval-grouped section from raw labels."""
    return AnalysisSection(
        description=description,
        methodology=methodology,
        calculation_thresholds=calculation_thresholds,
        raw_labels=np.asarray(labels, dtype=object),
        intervals=interval_strings(well.depth, labels, well.depth_unit),
    )


def depth_step_array(depth: np.ndarray, max_gap_factor: float = 3.0) -> np.ndarray:
    """Return per-sample depth increments using the actual depth spacing.

    A sample that follows a data gap would otherwise be credited the entire gap
    as its thickness (inflating zone/net-pay/HCPV thickness). Steps larger than
    ``max_gap_factor`` times the median sampling are capped to the median, so a
    gap is not counted as a thick cell.
    """
    depth_arr = np.asarray(depth, dtype=float)
    if depth_arr.size == 0:
        return np.array([], dtype=float)
    if depth_arr.size == 1:
        return np.ones(1, dtype=float)
    dz = np.abs(np.diff(depth_arr, prepend=depth_arr[0]))
    positive = dz[np.isfinite(dz) & (dz > 0.0)]
    median_step = float(np.median(positive)) if positive.size else 1.0
    dz[~np.isfinite(dz) | (dz <= 0.0)] = median_step
    # Cap gap-inflated steps so a single sample spanning a gap is not credited
    # the full gap thickness.
    if max_gap_factor and median_step > 0.0:
        dz[dz > max_gap_factor * median_step] = median_step
    return dz


def depth_to_meters(depth: np.ndarray, depth_unit: str) -> np.ndarray:
    """Convert depth values to meters."""
    factor = 0.3048 if normalize_depth_unit(depth_unit, strict=False) == "ft" else 1.0
    return np.asarray(depth, dtype=float) * factor


def rolling_std(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered rolling standard deviation."""
    arr = np.asarray(values, dtype=float)
    result = np.full(arr.shape, np.nan, dtype=float)
    if arr.size == 0:
        return result

    radius = max(window // 2, 0)
    for idx in range(arr.size):
        start = max(0, idx - radius)
        stop = min(arr.size, idx + radius + 1)
        local = arr[start:stop]
        local = local[np.isfinite(local)]
        if local.size > 1:
            result[idx] = float(np.nanstd(local))
    return result


def classification_code(labels: np.ndarray, mapping: dict[str, int]) -> np.ndarray:
    """Convert string labels to numeric codes."""
    codes = np.full(len(labels), np.nan, dtype=float)
    for name, code in mapping.items():
        mask = np.asarray(labels, dtype=object) == name
        codes[mask] = float(code)
    return codes
