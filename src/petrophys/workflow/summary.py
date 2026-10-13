"""Compact, JSON-serializable summaries of interpretation results.

The helpers work on any interpreted :class:`~petrophys.model.Well`, so the
prebuilt workflow (:class:`~petrophys.workflow.DynamicWorkflowResult`) and
custom workflows (:class:`~petrophys.workflow.WorkflowRunResult`) report the
same statistics in the same layout.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

import numpy as np

from petrophys._internal.analysis_helpers import cell_thickness
from petrophys._internal.serialization import curve_statistics, depth_mask, json_object

if TYPE_CHECKING:
    from petrophys.model.well import Well
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowResult

#: Interpretation products whose statistics are included in a summary.
SUMMARY_CURVES = ("VCL", "PHIT", "PHIE", "SW", "PERM")

#: Net flags whose thickness is included in a summary.
NET_FLAGS = ("NET_ROCK", "NET_RES", "NET_PAY")


def curve_results(well: "Well", names: Iterable[str], mask: np.ndarray) -> Dict[str, Any]:
    """Unit and statistics of each named curve over ``mask``; absent curves are left out."""
    results: Dict[str, Any] = {}
    for name in names:
        curve = well.get(name)
        if curve is not None:
            results[name] = {"unit": curve.unit, **curve_statistics(np.asarray(curve.data)[mask])}
    return results


def net_thickness(well: "Well", mask: np.ndarray, cells: bool = True) -> Dict[str, Any]:
    """Thickness flagged by each net flag over ``mask``, rounded to 0.01 depth units.

    Each flagged sample counts with its midpoint cell from the actual depths,
    or, with ``cells=False`` (the "geolink" calculation profile), with the
    well's depth step, as the net-pay trace counts it.
    """
    thickness: Dict[str, Any] = {"depth_unit": well.depth_unit}
    if well.step is None:
        return thickness
    cell = cell_thickness(well.depth)[mask] if cells else None
    for flag in NET_FLAGS:
        curve = well.get(flag)
        if curve is not None:
            flagged = np.asarray(curve.data, dtype=float)[mask] == 1
            total = np.sum(cell[flagged]) if cells else np.count_nonzero(flagged) * abs(float(well.step))
            thickness[flag.lower()] = round(float(total), 2)
    return thickness


def summarize_result(
    result: "DynamicWorkflowResult",
    top: Optional[float] = None,
    bottom: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize an interpretation; see :meth:`DynamicWorkflowResult.summary`."""
    from petrophys.version import __version__
    from petrophys.workflow.dynamic_workflow import _BENCHMARK_PROFILES, DynamicWorkflowConfig

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
        "key_results": curve_results(well, SUMMARY_CURVES, mask),
        "net_thickness": net_thickness(
            well, mask, cells=result.config.calculation_profile not in _BENCHMARK_PROFILES
        ),
        "executed_steps": [
            {"name": step.step, "method": step.method, "success": step.success}
            for step in result.steps_executed
        ],
        "skipped_steps": dict(result.specific_steps_skipped),
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    })


def formation_table(well: "Well", formations: Any = None, *, rank: Optional[str] = None) -> Any:
    """Per-formation gross, net and pay-averaged properties of an interpreted well.

    Parameters
    ----------
    well : Well
        Interpreted well.
    formations : path, DataFrame, mapping or sequence, optional
        Formation tops or intervals (anything :func:`petrophys.read_formations`
        accepts). Defaults to the formations attached by
        ``petrophys.read(path, formations=...)``.
    rank : str, optional
        Level to summarize when the tops mix levels, e.g. ``"formation"``.

    Returns
    -------
    pandas.DataFrame
        One row per formation, as returned by
        :meth:`DynamicWorkflowResult.formation_summary`.

    Raises
    ------
    ValueError
        If no formations are given or attached to ``well``.
    """
    import pandas as pd

    from petrophys.evaluation.summaries.formation_analysis import analyze_formations
    from petrophys.io.geology_reader import read_formations
    from petrophys.model.formation_tops import FormationTops
    from petrophys.model.intervals import Formation

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


def summarize_formations(
    result: "DynamicWorkflowResult",
    formations: Any = None,
    *,
    rank: Optional[str] = None,
) -> Any:
    """Per-formation table of an interpretation; see :meth:`DynamicWorkflowResult.formation_summary`."""
    return formation_table(result.well, formations, rank=rank)
