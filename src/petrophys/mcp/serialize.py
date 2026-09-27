"""JSON responses of the MCP server, built on the library's own summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

from petrophys._internal.serialization import (
    curve_statistics,
    json_object,
    json_safe,
)
from petrophys._internal.serialization import depth_mask as _depth_mask

if TYPE_CHECKING:
    from petrophys.model.well import Well
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowResult

__all__ = [
    "curve_statistics",
    "decimated_values",
    "depth_mask",
    "interpretation_summary",
    "json_object",
    "json_safe",
    "step_detail",
    "well_summary",
]


def depth_mask(well: "Well", top: Optional[float], bottom: Optional[float]) -> np.ndarray:
    """Boolean mask selecting samples between ``top`` and ``bottom`` (inclusive)."""
    return _depth_mask(well.depth, top, bottom)


def well_summary(well: "Well", well_id: str, source: str) -> Dict[str, Any]:
    """:meth:`Well.describe` plus the session id and source of the well."""
    return {"well_id": well_id, "source": source, **well.describe()}


def interpretation_summary(
    result: "DynamicWorkflowResult",
    interpretation_id: str,
    well_id: str,
    *,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    captured_warnings: Sequence[str] = (),
) -> Dict[str, Any]:
    """:meth:`DynamicWorkflowResult.summary` plus ids and warnings raised while running."""
    summary = result.summary(top=top, bottom=bottom)
    warnings = list(summary["warnings"])
    warnings.extend(message for message in captured_warnings if message not in warnings)
    return {
        "interpretation_id": interpretation_id,
        "well_id": well_id,
        **summary,
        "warnings": warnings,
    }


def step_detail(result: "DynamicWorkflowResult", step: str) -> Dict[str, Any]:
    """Full trace record of one executed or skipped step."""
    for record in result.steps_executed:
        if record.step == step:
            return json_object({"status": "executed", **record.to_dict()})
    if step in result.specific_steps_skipped:
        return {"status": "skipped", "step": step, "reason": result.specific_steps_skipped[step]}
    names = [record.step for record in result.steps_executed] + list(result.specific_steps_skipped)
    raise KeyError(f"Step {step!r} was not part of this run. Steps: {', '.join(names)}.")


def decimated_values(
    well: "Well", curves: Sequence[str], mask: np.ndarray, max_points: int
) -> Dict[str, Any]:
    """Curve values at no more than ``max_points`` evenly spaced depths."""
    indices = np.flatnonzero(mask)
    if indices.size > max_points:
        indices = indices[np.linspace(0, indices.size - 1, max_points).round().astype(int)]
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
