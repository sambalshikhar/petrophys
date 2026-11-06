"""JSON-safe conversion and summary statistics shared by result summaries."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert numpy values, paths, dataclasses and non-finite floats to JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        return [json_safe(item) for item in items]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def json_object(value: Dict[str, Any]) -> Dict[str, Any]:
    """:func:`json_safe` for a mapping, typed as a mapping."""
    converted: Dict[str, Any] = json_safe(value)
    return converted


def curve_statistics(values: Any) -> Dict[str, Any]:
    """Sample count, valid count, min, P10, median, mean, P90 and max."""
    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    stats: Dict[str, Any] = {"samples": int(data.size), "valid": int(finite.size)}
    if finite.size:
        p10, p50, p90 = np.percentile(finite, [10, 50, 90])
        stats.update(
            min=float(finite.min()),
            p10=float(p10),
            median=float(p50),
            mean=float(finite.mean()),
            p90=float(p90),
            max=float(finite.max()),
        )
    return stats


def depth_mask(depth: Any, top: Any = None, bottom: Any = None) -> np.ndarray:
    """Boolean mask of finite depths between ``top`` and ``bottom`` (inclusive)."""
    values = np.asarray(depth, dtype=float)
    mask = np.isfinite(values)
    if top is not None:
        mask &= values >= float(top)
    if bottom is not None:
        mask &= values <= float(bottom)
    return np.asarray(mask, dtype=bool)
