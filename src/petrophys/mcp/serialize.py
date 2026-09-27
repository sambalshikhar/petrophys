"""Compact, JSON-safe summaries of wells and interpretation results."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from petrophys.model.well import Well
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowResult

#: Interpretation products summarized after every run.
KEY_CURVES = ("VCL", "PHIT", "PHIE", "SW", "PERM")

#: Header parameters worth showing an analyst, when present.
KEY_PARAMETERS = (
    "BHT",
    "RW",
    "RWT",
    "RMF",
    "RMFT",
    "RM",
    "RMT",
    "MUD",
    "DFD",
    "BS",
    "TD",
)


def json_safe(value: Any) -> Any:
    """Convert numpy scalars/arrays, paths and non-finite floats to JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = sorted(value) if isinstance(value, set) else value
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


def depth_mask(
    well: "Well", top: Optional[float], bottom: Optional[float]
) -> np.ndarray:
    """Boolean mask selecting samples between ``top`` and ``bottom`` (inclusive)."""
    depth = np.asarray(well.depth, dtype=float)
    mask = np.isfinite(depth)
    if top is not None:
        mask &= depth >= float(top)
    if bottom is not None:
        mask &= depth <= float(bottom)
    return mask


def curve_statistics(values: np.ndarray) -> Dict[str, Any]:
    """Summary statistics of the finite samples in ``values``."""
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


def well_summary(well: "Well", well_id: str, source: str) -> Dict[str, Any]:
    """Describe a loaded well: header, depth range, curves and attached geology."""
    curves = []
    for mnemonic in well.curves:
        curve = well.get(mnemonic)
        if curve is None:
            continue
        data = np.asarray(curve.data, dtype=float)
        valid = int(np.isfinite(data).sum())
        curves.append(
            {
                "mnemonic": mnemonic,
                "unit": curve.unit,
                "description": curve.metadata.description,
                "valid_fraction": round(valid / data.size, 3) if data.size else 0.0,
            }
        )
    parameters = {
        name: well.get_parameter(name)
        for name in KEY_PARAMETERS
        if well.get_parameter(name) is not None
    }
    geology = well.get_parameter("_geology_context") or {}
    header = well.header
    return json_object(
        {
            "well_id": well_id,
            "source": source,
            "name": well.name,
            "field": getattr(header, "field", None),
            "depth_unit": well.depth_unit,
            "top": well.top,
            "bottom": well.bottom,
            "step": well.step,
            "n_samples": well.n_samples,
            "curves": curves,
            "header_parameters": parameters,
            "formations": [
                row.get("formation_name") for row in geology.get("formations", [])
            ],
            "lithology_intervals": len(geology.get("lithology", [])),
        }
    )


def _thickness(well: "Well", flag: str, mask: np.ndarray) -> Optional[float]:
    curve = well.get(flag)
    if curve is None or well.step is None:
        return None
    values = np.asarray(curve.data, dtype=float)[mask]
    return float(np.nansum(values == 1) * abs(float(well.step)))


def interpretation_summary(
    result: "DynamicWorkflowResult",
    interpretation_id: str,
    well_id: str,
    *,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    captured_warnings: Sequence[str] = (),
) -> Dict[str, Any]:
    """Summarize an interpretation: status, settings used, steps and key results."""
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig

    well = result.well
    mask = depth_mask(well, top, bottom)
    defaults = DynamicWorkflowConfig.__dataclass_fields__
    settings = {
        name: getattr(result.config, name)
        for name, spec in defaults.items()
        if spec.default is not dataclasses.MISSING
        and getattr(result.config, name) != spec.default
    }
    key_results: Dict[str, Any] = {}
    for name in KEY_CURVES:
        curve = well.get(name)
        if curve is not None:
            key_results[name] = {
                "unit": curve.unit,
                **curve_statistics(np.asarray(curve.data)[mask]),
            }
    net = {
        flag.lower(): _thickness(well, flag, mask)
        for flag in ("NET_ROCK", "NET_RES", "NET_PAY")
    }
    warnings = list(result.warnings)
    warnings.extend(message for message in captured_warnings if message not in warnings)
    return json_object(
        {
            "interpretation_id": interpretation_id,
            "well_id": well_id,
            "status": result.status,
            "interpretation_profile": result.interpretation_profile,
            "resolved": {
                "matrix_type": result.config.matrix_type,
                "lithology": result.config.lithology,
            },
            "settings_used": settings,
            "summary_interval": {
                "top": top,
                "bottom": bottom,
                "depth_unit": well.depth_unit,
            },
            "key_results": key_results,
            "net_thickness": {**net, "depth_unit": well.depth_unit},
            "executed_steps": [
                {"name": step.step, "method": step.method, "success": step.success}
                for step in result.steps_executed
            ],
            "skipped_steps": dict(result.specific_steps_skipped),
            "warnings": warnings,
            "errors": list(result.errors),
        }
    )


def step_detail(result: "DynamicWorkflowResult", step: str) -> Dict[str, Any]:
    """Full trace record of one executed or skipped step."""
    for record in result.steps_executed:
        if record.step == step:
            return json_object({"status": "executed", **record.to_dict()})
    if step in result.specific_steps_skipped:
        return {
            "status": "skipped",
            "step": step,
            "reason": result.specific_steps_skipped[step],
        }
    names = [record.step for record in result.steps_executed] + list(
        result.specific_steps_skipped
    )
    raise KeyError(
        f"Step {step!r} was not part of this run. Steps: {', '.join(names)}."
    )


def decimated_values(
    well: "Well", curves: Sequence[str], mask: np.ndarray, max_points: int
) -> Dict[str, Any]:
    """Curve values at no more than ``max_points`` evenly spaced depths."""
    indices = np.flatnonzero(mask)
    if indices.size > max_points:
        indices = indices[
            np.linspace(0, indices.size - 1, max_points).round().astype(int)
        ]
    values: Dict[str, List[Any]] = {}
    missing = []
    for name in curves:
        curve = well.get(name)
        if curve is None:
            missing.append(name)
            continue
        values[curve.mnemonic] = json_safe(np.asarray(curve.data, dtype=float)[indices])
    return {
        "depth_unit": well.depth_unit,
        "depth": json_safe(np.asarray(well.depth, dtype=float)[indices]),
        "curves": values,
        "missing_curves": missing,
        "points": int(indices.size),
        "decimated": bool(np.count_nonzero(mask) > indices.size),
    }
