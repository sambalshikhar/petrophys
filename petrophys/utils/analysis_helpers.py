"""
Shared helpers for interval-based analysis outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from petrophys.base.data_types import DataType


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
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Return the first available curve data for a list of aliases."""
    for name in names:
        curve = well.get(name)
        if curve is not None:
            return np.asarray(curve.data, dtype=float), curve.mnemonic
    return None, None


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


def depth_step_array(depth: np.ndarray) -> np.ndarray:
    """Return per-sample depth increments using the actual depth spacing."""
    depth_arr = np.asarray(depth, dtype=float)
    if depth_arr.size == 0:
        return np.array([], dtype=float)
    if depth_arr.size == 1:
        return np.ones(1, dtype=float)
    dz = np.abs(np.diff(depth_arr, prepend=depth_arr[0]))
    if dz.size > 1 and dz[0] == 0.0:
        dz[0] = float(np.nanmedian(dz[1:])) if np.any(dz[1:] > 0) else 1.0
    dz[dz <= 0.0] = float(np.nanmedian(dz[dz > 0.0])) if np.any(dz > 0.0) else 1.0
    return dz


def depth_to_meters(depth: np.ndarray, depth_unit: str) -> np.ndarray:
    """Convert depth values to meters."""
    unit = (depth_unit or "").strip().lower()
    factor = 0.3048 if unit in {"ft", "feet"} else 1.0
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
