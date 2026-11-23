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
from petrophys._internal.advice import Advice

if TYPE_CHECKING:
    from petrophys.model.well import Well
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowResult

#: Interpretation products whose statistics are included in a summary.
SUMMARY_CURVES = ("VCL", "PHIT", "PHIE", "SW", "PERM")

#: Net flags whose thickness is included in a summary.
NET_FLAGS = ("NET_ROCK", "NET_RES", "NET_PAY")


def _summary_curves(result: "DynamicWorkflowResult") -> tuple:
    """The curves a summary describes: SUMMARY_CURVES, with the core-corrected
    PHIE_CC and PERM_CC after their plain curves wherever the pay averages
    use them (not under the benchmark profiles, whose summaries had none)."""
    from petrophys.evaluation.summaries.zone_summary import pay_average_curves
    from petrophys.workflow.dynamic_workflow import _BENCHMARK_PROFILES

    if result.config.calculation_profile in _BENCHMARK_PROFILES:
        return SUMMARY_CURVES
    chosen = pay_average_curves(result.well)
    names = []
    for name in SUMMARY_CURVES:
        names.append(name)
        corrected = chosen.get(name)
        if corrected and corrected != name:
            names.append(corrected)
    return tuple(names)


def depth_window(
    well: "Well", top: Optional[float], bottom: Optional[float],
) -> tuple:
    """The samples of ``well`` between ``top`` and ``bottom`` (inclusive).

    The two depths are read in either order (a top below the bottom is the
    same window). A window that holds no sample of the well gives a
    UserWarning naming the logged interval, since every statistic of it is
    empty.

    Returns
    -------
    tuple
        ``(mask, top, bottom)``: the boolean sample mask and the two depths,
        shallow first.
    """
    import warnings

    if top is not None and bottom is not None and float(top) > float(bottom):
        top, bottom = bottom, top
    mask = depth_mask(well.depth, top, bottom)
    if (top is not None or bottom is not None) and not mask.any() and well.top is not None:
        unit = well.depth_unit or ""
        window = " to ".join(
            f"the {end} of the log" if value is None else f"{float(value):g}"
            for value, end in ((top, "top"), (bottom, "bottom")))
        warnings.warn(
            f"No sample of {well.name} lies in the window {window} {unit}: the well is logged "
            f"from {well.top:g} to {well.bottom:g} {unit}, so the window's statistics are empty.",
            UserWarning, stacklevel=4,
        )
    return mask, top, bottom


def statistics_table(well: "Well", curves: Any = None, top: Optional[float] = None,
                     bottom: Optional[float] = None, default: Iterable[str] = SUMMARY_CURVES) -> Any:
    """A row per curve of ``well`` over a depth window: unit, finite samples,
    mean, p10, median and p90. ``curves`` defaults to those of ``default``
    present on the well."""
    import pandas as pd

    if isinstance(curves, str):
        curves = [curves]
    names = list(curves) if curves is not None else [name for name in default if well.get(name) is not None]
    missing = [name for name in names if well.get(name) is None]
    if missing:
        raise KeyError(f"No curve {', '.join(missing)} in the well")
    window, top, bottom = depth_window(well, top, bottom)
    rows = {}
    for name in names:
        curve = well.get(name)
        assert curve is not None
        values = np.asarray(curve.data, dtype=float)[window]
        values = values[np.isfinite(values)]
        p10, median, p90 = np.percentile(values, [10, 50, 90]) if values.size else (np.nan,) * 3
        rows[name] = {"unit": curve.unit or "", "samples": int(values.size),
                      "mean": float(values.mean()) if values.size else np.nan,
                      "p10": float(p10), "median": float(median), "p90": float(p90)}
    return pd.DataFrame.from_dict(rows, orient="index")


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

    ``net_rock``, ``net_res`` and ``net_pay`` are always present; a flag the
    run did not compute is None. Each flagged sample counts with its midpoint cell from the actual depths,
    or, with ``cells=False`` (the "geolink" calculation profile), with the
    well's depth step, as the net-pay trace counts it.
    """
    thickness: Dict[str, Any] = {"depth_unit": well.depth_unit, **{flag.lower(): None for flag in NET_FLAGS}}
    if well.step is None or well.depth is None:
        return thickness
    cell = cell_thickness(well.depth)[mask] if cells else np.empty(0)
    for flag in NET_FLAGS:
        curve = well.get(flag)
        if curve is not None:
            flagged = np.asarray(curve.data, dtype=float)[mask] == 1
            total = np.sum(cell[flagged]) if cells else np.count_nonzero(flagged) * abs(float(well.step))
            thickness[flag.lower()] = round(float(total), 2)
    return thickness


#: Settings whose unset default (None) a run resolves to this value.
_RESOLVED_DEFAULTS = {"rho_fl": 1.0, "rho_cl": 2.45}


def _differs(value: Any, default: Any) -> bool:
    """Whether a setting differs from its default; a table or array that is set does."""
    try:
        return bool(value != default)
    except (TypeError, ValueError):
        return True


def summarize_result(
    result: "DynamicWorkflowResult",
    top: Optional[float] = None,
    bottom: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize an interpretation; see :meth:`DynamicWorkflowResult.summary`."""
    from petrophys.version import __version__
    from petrophys.workflow.dynamic_workflow import _BENCHMARK_PROFILES, DynamicWorkflowConfig

    well = result.well
    mask, top, bottom = depth_window(well, top, bottom)
    defaults = DynamicWorkflowConfig.__dataclass_fields__
    def shown(value: Any) -> Any:
        # A setting with its unit, as text: "251 degF".
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
            return f"{float(value[0]):g} {value[1]}"
        return value

    settings = {
        name: shown(getattr(result.config, name))
        for name, spec in defaults.items()
        if name != "lithology" and not name.startswith("_")
        and spec.default is not dataclasses.MISSING
        and _differs(getattr(result.config, name), _RESOLVED_DEFAULTS.get(name, spec.default))
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
        "key_results": curve_results(well, _summary_curves(result), mask),
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
        Level to summarize when the tops mix levels, e.g. ``"group"``. Tops
        that mix levels default to ``"formation"``; a rank is not needed when
        the tops carry none and never coincide. Tops at or below the logged
        bottom are left out.

    Returns
    -------
    pandas.DataFrame
        One row per formation, as returned by
        :meth:`DynamicWorkflowResult.formation_summary`: ``N/G (reservoir)``
        is net reservoir / gross and ``Pay/Gross`` net pay / gross; the curve
        of each pay average is named in ``DataFrame.attrs["curves"]``
        (PHIE_CC and PERM_CC where the interpretation produced or used them,
        PHIE and PERM at the samples those do not cover).

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
            raise ValueError(Advice(
                "No formations are attached to this well. Pass formations=<file or table>, "
                "or attach them with petrophys.read(path, formations=...).", ("formations",)
            ))
        table = pd.DataFrame(records)
    else:
        table = read_formations(formations, well=well)
    if "bottom_depth" in table and table["bottom_depth"].isna().all():
        table = table.drop(columns="bottom_depth")
    if "bottom_depth" not in table and well.bottom is not None:
        # Picks at or below the logged bottom (formations the log did not reach) hold no samples.
        inside = table.loc[table["top_depth"] < well.bottom]
        if inside.empty:
            raise ValueError(
                f"No formation top lies above the logged bottom {well.bottom:g} {well.depth_unit}; "
                "check the depth unit of the formation tops."
            )
        inside.attrs = table.attrs
        table = inside
    ranks = (
        {str(value).strip().lower() for value in table["rank"].dropna() if str(value).strip()}
        if "rank" in table else set()
    )
    if rank is None and len(ranks) > 1 and "formation" in ranks:
        # Hierarchical picks (group, formation, member): formations are the usual level.
        rank = "formation"
    elif rank is not None and not ranks and not table["top_depth"].duplicated().any():
        # Flat picks with no rank: every pick is at the one level there is.
        rank = None
    tops = FormationTops.from_dataframe(
        table, name_col="formation_name", terminal_depth=well.bottom, rank=rank
    )
    zones = [interval for interval in tops.intervals if isinstance(interval, Formation)]
    return analyze_formations(well, zones).to_dataframe()


def summarize_zones(result: "DynamicWorkflowResult", zones: Any = None) -> Any:
    """Per-zone table of an interpretation; see :meth:`DynamicWorkflowResult.zone_summary`."""
    from petrophys.evaluation.summaries.formation_analysis import analyze_formations
    from petrophys.model.intervals import Formation
    from petrophys.workflow.dynamic_workflow import _zone_intervals_for_well

    well = result.well
    intervals = result.zone_intervals if zones is None else _zone_intervals_for_well(well, zones)
    if not intervals:
        raise ValueError(Advice(
            "No zones apply to this well. Pass zones=<file or table>, or interpret "
            "with petrophys.interpret(well, zones=...).", ("zones",)
        ))
    formations = [Formation(zone["name"], zone["top"], zone["bottom"]) for zone in intervals]
    return analyze_formations(well, formations).to_dataframe().rename(columns={"Formation": "Zone"})


def summarize_formations(
    result: "DynamicWorkflowResult",
    formations: Any = None,
    *,
    rank: Optional[str] = None,
) -> Any:
    """Per-formation table of an interpretation; see :meth:`DynamicWorkflowResult.formation_summary`."""
    return formation_table(result.well, formations, rank=rank)
