"""
Shared helpers for interval-based analysis outputs.
"""

from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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
    #: How this run made the curve, where the run chose it (a datum, the curves a ratio read); the description
    #: includes it. Empty where the curve is made the same way in every run. Kept in the written curve's
    #: ``metadata.custom["basis"]``.
    basis: str = ""


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
            if curve.basis:
                well[mnemonic].metadata.custom["basis"] = curve.basis
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
    num, den = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    result = np.full(np.broadcast(num, den).shape, fill_value, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > min_abs)
    result[valid] = num[valid] / den[valid]
    return result


#: Private well parameter: the depth unit label the analyzers write after a
#: depth interval in place of ``well.depth_unit``, e.g. ``"FT"`` as the source
#: file labelled it. Set for the published context packages, whose intervals
#: carried the source label.
DEPTH_UNIT_LABEL_PARAMETER = "_DEPTH_UNIT_LABEL"


def depth_unit_label(well: "Well") -> str:
    """The depth unit label written after the well's depth intervals."""
    label = well.get_parameter(DEPTH_UNIT_LABEL_PARAMETER) if hasattr(well, "get_parameter") else None
    return str(label) if label else well.depth_unit


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
        intervals=interval_strings(well.depth, labels, depth_unit_label(well)),
    )


def _geolink_train_steps() -> bool:
    """True while the GeoLink training context profile computes its package."""
    from petrophys.workflow.dynamic_workflow import _DATASET_CALCULATIONS_REQUESTED

    return _DATASET_CALCULATIONS_REQUESTED.get() == "geolink_train"


def depth_step_array(depth: np.ndarray, max_gap_factor: float = 3.0) -> np.ndarray:
    """Return per-sample depth increments using the actual depth spacing.

    A sample that follows a data gap would otherwise be credited the entire gap
    as its thickness (inflating zone/net-pay/HCPV thickness). Steps larger than
    ``max_gap_factor`` times the median sampling are capped to the median, so a
    gap is not counted as a thick cell. The per-sample diagnostics (KH, S_V)
    use this capped backward step; thickness sums use :func:`cell_thickness`,
    which also integrates a change of sampling rate. The GeoLink training
    context profile keeps its package's uncapped step.
    """
    depth_arr = np.asarray(depth, dtype=float)
    if depth_arr.size == 0:
        return np.array([], dtype=float)
    if depth_arr.size == 1:
        return np.ones(1, dtype=float)
    dz = np.abs(np.diff(depth_arr, prepend=depth_arr[0]))
    if _geolink_train_steps():
        # The GeoLink training package used every backward step as it is:
        # the first sample takes the median of the others, a zero step the
        # median of the positive steps, and no gap is capped.
        if dz.size > 1 and dz[0] == 0.0:
            dz[0] = float(np.nanmedian(dz[1:])) if np.any(dz[1:] > 0) else 1.0
        dz[dz <= 0.0] = float(np.nanmedian(dz[dz > 0.0])) if np.any(dz > 0.0) else 1.0
        return dz
    positive = dz[np.isfinite(dz) & (dz > 0.0)]
    median_step = float(np.median(positive)) if positive.size else 1.0
    dz[~np.isfinite(dz) | (dz <= 0.0)] = median_step
    # Cap gap-inflated steps so a single sample spanning a gap is not credited
    # the full gap thickness.
    if max_gap_factor and median_step > 0.0:
        dz[dz > max_gap_factor * median_step] = median_step
    return dz


def cell_half_lengths(depth: np.ndarray, max_gap_factor: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Half-cell lengths on the previous-sample and next-sample side of each sample.

    Each sample owns the depth range halfway to each neighbour (midpoint
    cells), so the cells tile the logged interval and a change of sampling
    rate is integrated exactly. A step longer than ``max_gap_factor`` times
    the typical sampling around it is a gap in the data. The typical sampling
    is the larger of the medians of the three steps before and the three
    steps after, so a new sampling rate sets its own scale. A gap is credited
    to neither neighbour; a sample next to a gap, or at either end, mirrors its other
    half-cell, and a sample with gaps on both sides gets the median regular
    step. Depths may increase or decrease. A sample with a non-finite depth
    gets no cell (0); with fewer than two finite depths the cells are
    undefined (NaN).
    """
    depth_arr = np.asarray(depth, dtype=float).reshape(-1)
    before = np.zeros(depth_arr.size, dtype=float)
    after = np.zeros(depth_arr.size, dtype=float)
    finite = np.flatnonzero(np.isfinite(depth_arr))
    if finite.size < 2:
        before[finite] = after[finite] = np.nan
        return before, after

    steps = np.abs(np.diff(depth_arr[finite]))
    gap = np.zeros(steps.size, dtype=bool)
    if max_gap_factor:
        window = 3
        positive = np.where(steps > 0.0, steps, np.nan)
        padded = np.concatenate([np.full(window, np.nan), positive, np.full(window, np.nan)])
        windows = sliding_window_view(padded, window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows at the ends
            reference = np.fmax(
                np.nanmedian(windows[: steps.size], axis=1),
                np.nanmedian(windows[window + 1:], axis=1),
            )
        gap = np.isfinite(reference) & (steps > max_gap_factor * reference)

    half = np.where(gap, np.nan, steps / 2.0)
    half_before = np.concatenate([[np.nan], half])
    half_after = np.concatenate([half, [np.nan]])
    half_before = np.where(np.isnan(half_before), half_after, half_before)
    half_after = np.where(np.isnan(half_after), half_before, half_after)
    regular = steps[~gap & (steps > 0.0)]
    fill = float(np.median(regular)) / 2.0 if regular.size else 0.0
    before[finite] = np.where(np.isnan(half_before), fill, half_before)
    after[finite] = np.where(np.isnan(half_after), fill, half_after)
    return before, after


def cell_thickness(depth: np.ndarray, max_gap_factor: float = 3.0) -> np.ndarray:
    """Per-sample thickness of the midpoint cells from :func:`cell_half_lengths`.

    N regularly spaced samples sum to N steps; sampling-rate changes are
    integrated from the actual depths, and data gaps are not counted.
    """
    before, after = cell_half_lengths(depth, max_gap_factor)
    return before + after


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


def as_well(value):
    """The interpreted well of a run result; anything else as it is.

    A run result is what ``pp.interpret`` returns (it has ``steps_executed``)
    or what a custom ``Workflow`` run returns
    (:class:`~petrophys.workflow.custom.WorkflowRunResult`). Other objects
    that hold a well (viewers, plots, modules) are returned as they are.
    """
    if hasattr(value, "steps_executed") and hasattr(value, "well"):
        return value.well
    # A custom run result exists only once its module is imported.
    custom = sys.modules.get("petrophys.workflow.custom")
    if custom is not None and isinstance(value, custom.WorkflowRunResult):
        return value.well
    return value
