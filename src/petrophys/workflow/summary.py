"""Compact, JSON-serializable summaries of interpretation results."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from petrophys._internal.serialization import curve_statistics, depth_mask, json_object

if TYPE_CHECKING:
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowResult

#: Interpretation products whose statistics are included in a summary.
SUMMARY_CURVES = ("VCL", "PHIT", "PHIE", "SW", "PERM")

#: Net flags whose thickness is included in a summary.
NET_FLAGS = ("NET_ROCK", "NET_RES", "NET_PAY")


def summarize_result(
    result: "DynamicWorkflowResult",
    top: Optional[float] = None,
    bottom: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize an interpretation; see :meth:`DynamicWorkflowResult.summary`."""
    from petrophys.version import __version__
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig

    well = result.well
    mask = depth_mask(well.depth, top, bottom)
    defaults = DynamicWorkflowConfig.__dataclass_fields__
    settings = {
        name: getattr(result.config, name)
        for name, spec in defaults.items()
        if name != "lithology"
        and spec.default is not dataclasses.MISSING
        and getattr(result.config, name) != spec.default
    }
    key_results: Dict[str, Any] = {}
    for name in SUMMARY_CURVES:
        curve = well.get(name)
        if curve is not None:
            key_results[name] = {"unit": curve.unit, **curve_statistics(np.asarray(curve.data)[mask])}
    thickness: Dict[str, Any] = {"depth_unit": well.depth_unit}
    for flag in NET_FLAGS:
        curve = well.get(flag)
        if curve is not None and well.step is not None:
            values = np.asarray(curve.data, dtype=float)[mask]
            thickness[flag.lower()] = round(
                float(np.count_nonzero(values == 1) * abs(float(well.step))), 2
            )
    return json_object({
        "petrophys_version": __version__,
        "well": well.name,
        "status": result.status,
        "lithology_domain": result.lithology_domain,
        "resolved": {
            "matrix_type": result.config.matrix_type,
            "lithology_domain": result.config.lithology_domain,
        },
        "settings_used": settings,
        "summary_interval": {"top": top, "bottom": bottom, "depth_unit": well.depth_unit},
        "key_results": key_results,
        "net_thickness": thickness,
        "executed_steps": [
            {"name": step.step, "method": step.method, "success": step.success}
            for step in result.steps_executed
        ],
        "skipped_steps": dict(result.specific_steps_skipped),
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    })


def summarize_formations(
    result: "DynamicWorkflowResult",
    formations: Any = None,
    *,
    rank: Optional[str] = None,
) -> Any:
    """Per-formation table of an interpretation; see :meth:`DynamicWorkflowResult.formation_summary`."""
    import pandas as pd

    from petrophys.evaluation.summaries.formation_analysis import analyze_formations
    from petrophys.io.geology_reader import read_formations
    from petrophys.model.formation_tops import FormationTops
    from petrophys.model.intervals import Formation

    well = result.well
    if formations is None:
        records = (well.get_parameter("_geology_context") or {}).get("formations")
        if not records:
            raise ValueError(
                "No formations are attached to this well. Pass formations=<file or table>, "
                "or attach them with petrophys.read(path, formations=...)."
            )
        table = pd.DataFrame(records)
    else:
        table = read_formations(formations, well=well)
    if "bottom_depth" in table and table["bottom_depth"].isna().all():
        table = table.drop(columns="bottom_depth")
    tops = FormationTops.from_dataframe(
        table, name_col="formation_name", terminal_depth=well.bottom, rank=rank
    )
    zones = [interval for interval in tops.intervals if isinstance(interval, Formation)]
    return analyze_formations(well, zones).to_dataframe()
