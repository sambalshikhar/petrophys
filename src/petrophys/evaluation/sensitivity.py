"""Sensitivity of reservoir answers to cutoffs, parameters and models.

A net-pay or hydrocarbon-volume number is only as firm as the choices behind
it. This module measures how far those choices move the answer:

- :func:`cutoff_sensitivity` re-applies the net-pay cutoffs over a grid of
  VCL, PHIE and Sw values on fixed curves (fast, no re-interpretation).
- :func:`parameter_sensitivity` runs any evaluation one input at a time at a
  low and a high setting and returns the data for a tornado chart.
- :func:`compare_saturation_models` runs several field-data water-saturation
  models on one interval with a common parameter set and tabulates mean Sw,
  net pay and hydrocarbon pore thickness per model.

All thicknesses are in the depth unit of the input; porosity, saturation and
clay volume are fractions (v/v); resistivities are in ohm.m.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from petrophys._internal.analysis_helpers import as_well as _as_well

from petrophys.evaluation.net_pay.cutoffs import cutoff_fraction
from petrophys.evaluation.net_pay.pay_summary import (
    PHIE_CURVES,
    SW_CURVES,
    VCL_CURVES,
    PaySummary,
    pay_summary_from_arrays,
    resolve_cutoffs,
    resolve_windowed_inputs,
)
from petrophys.interpretation.saturation import (
    archie,
    indonesia,
    nigeria_schlumberger,
    north_sea_shaly_sand,
    simandoux,
    modified_simandoux,
)

RT_CURVES = ("RDEP", "RT")

#: Saturation models :func:`compare_saturation_models` runs by default (it also
#: takes "modified_simandoux"). All need only
#: Rt, PHIE, VCL, Rw and (for the shaly-sand models) a shale resistivity.
SATURATION_MODELS: Tuple[str, ...] = (
    "archie",
    "simandoux",
    "indonesia",
    "north_sea",
    "nigeria_schlumberger",
)

_PAY_COLUMNS = (
    "gross_thickness",
    "net_rock_thickness",
    "net_reservoir_thickness",
    "net_pay_thickness",
    "ntg_reservoir",
    "ntg_pay",
    "avg_phie_pay",
    "avg_sw_pay",
    "hcpt",
    "n_pay_samples",
)


def _cutoff_values(values, default: float, name: str) -> list:
    if values is None:
        return [float(default)]
    array = np.atleast_1d(np.asarray(values, dtype=float)).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite numbers")
    return [cutoff_fraction(value, name) for value in array]


def _base_cutoffs(source: Any, lithology: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    """The cutoffs a sweep starts from, and where each comes from.

    A cutoff set as one number in the settings of an interpretation result
    (``source.config``) is that run's; any other is the standard of the
    ``lithology`` family.
    """
    base = resolve_cutoffs(None, None, None, lithology)
    origin = {name: f"standard, {lithology}" for name in base}
    config = getattr(source, "config", None) if hasattr(source, "steps_executed") else None
    for name in base:
        value = getattr(config, name, None)
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            base[name] = cutoff_fraction(value, name)
            origin[name] = "interpretation setting"
    return base, origin


def cutoff_sensitivity(
    well=None,
    *,
    vcl_cutoffs: Union[float, Sequence[float], None] = None,
    phie_cutoffs: Union[float, Sequence[float], None] = None,
    sw_cutoffs: Union[float, Sequence[float], None] = None,
    vcl=None,
    phie=None,
    sw=None,
    depth=None,
    lithology: str = "clastic",
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    depth_unit: Optional[str] = None,
):
    """Net pay and net-to-gross over a grid of VCL, PHIE and Sw cutoffs.

    The curves stay fixed; only the flags change, so a large grid is fast.
    Each row is one cutoff combination summarized with
    :func:`~petrophys.evaluation.net_pay.summarize_pay`.

    Parameters
    ----------
    well : Well or interpretation result, optional
        Source of the depth and of any curve not passed as an array (curve
        lookup as in :func:`~petrophys.evaluation.calculate_net_pay`).
    vcl_cutoffs, phie_cutoffs, sw_cutoffs : float or sequence of float, optional
        Cutoff values to combine, fractions (v/v); a value above 1 and up to
        100 is read as percent. A single value holds that cutoff fixed. An
        unset cutoff stays at its base value: the cutoff the interpretation
        was run with when an interpretation result is passed and that
        cutoff was set as one number in its settings
        (``pp.interpret(well, sw_cutoff=0.5)``), else the standard value for
        ``lithology``.
    vcl, phie, sw, depth : array-like, optional
        Explicit curves and depths, one value per sample.
    lithology : str, default "clastic"
        Lithology family for the standard cutoffs.
    top, bottom : float, optional
        Depth window (inclusive), in depth units, for example the reservoir.
    depth_unit : str, optional
        Unit label for the thicknesses. Defaults to ``well.depth_unit``.

    Returns
    -------
    pandas.DataFrame
        One row per combination with ``vcl_cutoff``, ``phie_cutoff``,
        ``sw_cutoff``, the gross, net rock, net reservoir and net pay
        thicknesses, ``ntg_reservoir``, ``ntg_pay``, ``avg_phie_pay``,
        ``avg_sw_pay``, ``hcpt`` and ``n_pay_samples``. ``attrs`` records the
        depth unit and window, the base cutoffs (``base_cutoffs``) and
        where each comes from (``base_cutoff_source``: ``"interpretation
        setting"`` or ``"standard, <lithology>"``). Pivot it for a heat map,
        e.g.
        ``grid.pivot(index="vcl_cutoff", columns="sw_cutoff", values="ntg_pay")``.

    Raises
    ------
    ValueError
        If inputs are missing or inconsistent, or a cutoff is not a finite
        number between 0 and 100.

    Notes
    -----
    Every row applies one cutoff set to every sample and leaves no sample
    out. An interpretation run without cutoffs in its settings takes each
    sample's lithology-family standards, and every run leaves evaporite,
    coal, halite recognised on the logs and bad-hole samples out of net
    reservoir and net pay, so the base row can differ from the run's own
    ``NET_PAY`` (``result.summary(top, bottom)``).

    See Also
    --------
    petrophys.evaluation.net_pay.summarize_pay : One cutoff set.
    petrophys.visualization.heatmap : Plot the pivoted grid.

    Examples
    --------
    >>> import numpy as np
    >>> depth = np.arange(1000.0, 1010.0, 1.0)
    >>> vcl = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.1, 0.15, 0.25, 0.6, 0.1])
    >>> phie = np.full(10, 0.2)
    >>> sw = np.array([0.2, 0.3, 0.4, 0.5, 0.9, 0.55, 0.65, 0.7, 1.0, 0.35])
    >>> grid = cutoff_sensitivity(vcl=vcl, phie=phie, sw=sw, depth=depth,
    ...                           vcl_cutoffs=[0.2, 0.4], sw_cutoffs=[0.5, 0.7])
    >>> net = grid.pivot(index="vcl_cutoff", columns="sw_cutoff", values="net_pay_thickness")
    >>> net.to_dict("index")    # rows: VCL cutoff; columns: Sw cutoff
    {0.2: {0.5: 3.0, 0.7: 5.0}, 0.4: {0.5: 4.0, 0.7: 7.0}}

    The cutoffs are strict, so the samples at VCL = 0.2 and at Sw = 0.5 or
    0.7 fail the matching cutoff.
    """
    base, base_source = _base_cutoffs(well, lithology)
    well = _as_well(well)
    import pandas as pd

    _, arrays, dz, unit, _ = resolve_windowed_inputs(
        well, depth,
        {"vcl": (vcl, VCL_CURVES), "phie": (phie, PHIE_CURVES), "sw": (sw, SW_CURVES)},
        top=top, bottom=bottom, depth_unit=depth_unit,
    )
    grid = itertools.product(
        _cutoff_values(vcl_cutoffs, base["vcl_cutoff"], "vcl_cutoffs"),
        _cutoff_values(phie_cutoffs, base["phie_cutoff"], "phie_cutoffs"),
        _cutoff_values(sw_cutoffs, base["sw_cutoff"], "sw_cutoffs"),
    )
    rows = []
    for vcl_cut, phie_cut, sw_cut in grid:
        cutoffs = {"vcl_cutoff": vcl_cut, "phie_cutoff": phie_cut, "sw_cutoff": sw_cut}
        summary = pay_summary_from_arrays(
            arrays["vcl"], arrays["phie"], arrays["sw"], dz, cutoffs, unit
        )
        row = dict(cutoffs)
        row.update({name: getattr(summary, name) for name in _PAY_COLUMNS})
        rows.append(row)
    table = pd.DataFrame(rows, columns=["vcl_cutoff", "phie_cutoff", "sw_cutoff", *_PAY_COLUMNS])
    table.attrs.update({"depth_unit": unit, "top": top, "bottom": bottom,
                        "base_cutoffs": base, "base_cutoff_source": base_source})
    return table


def _outputs(value: Any, selected: Optional[Sequence[str]]) -> Dict[str, float]:
    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        value = value.to_dict()
    if isinstance(value, Mapping):
        values = {
            str(key): float(item) for key, item in value.items()
            if isinstance(item, (int, float, np.integer, np.floating))
            and not isinstance(item, bool)
        }
    else:
        try:
            values = {"value": float(value)}
        except (TypeError, ValueError):
            raise TypeError(
                "evaluate() must return a number, a mapping of numbers, or an object "
                f"with to_dict(); got {type(value).__name__}"
            ) from None
    if selected is not None:
        missing = [name for name in selected if name not in values]
        if missing:
            raise ValueError(
                f"evaluate() did not return output(s) {missing}; available: {sorted(values)}"
            )
        values = {name: values[name] for name in selected}
    if not values:
        raise ValueError("evaluate() returned no numeric outputs")
    return values


def _describe(changes: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in changes.items())


def parameter_sensitivity(
    evaluate: Callable[..., Any],
    cases: Mapping[str, Tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    outputs: Optional[Sequence[str]] = None,
):
    """One-at-a-time sensitivity: the data behind a tornado chart.

    ``evaluate()`` is called once with no arguments for the base case, then
    twice per case with that case's low and high keyword changes; everything
    else stays at base. The widest swing names the assumption the answer
    depends on most, which is usually the measurement worth buying first.

    Parameters
    ----------
    evaluate : callable
        ``evaluate(**changes)`` returns the answer for one set of changes: a
        number, a mapping of named numbers, or an object with ``to_dict()``
        (for example a :class:`~petrophys.evaluation.net_pay.PaySummary`).
        With no arguments it must return the base case.
    cases : mapping
        ``{label: (low_changes, high_changes)}``, each a mapping of keyword
        arguments for ``evaluate``, e.g.
        ``{"Rw 0.05 / 0.09": ({"rw": 0.05}, {"rw": 0.09})}``. "Low" and "high"
        refer to the settings, not to the result, which may move either way.
    outputs : sequence of str, optional
        Outputs to keep when ``evaluate`` returns several. Defaults to every
        numeric output.

    Returns
    -------
    pandas.DataFrame
        One row per case and output: ``case``, ``output``, ``low``, ``base``,
        ``high``, ``swing`` (``|high - low|``), ``low_setting`` and
        ``high_setting``. Rows are grouped by output and sorted by
        decreasing swing. ``attrs["base"]`` holds the base outputs.

    Raises
    ------
    ValueError
        If ``cases`` is empty or malformed, or a requested output is missing.
    TypeError
        If ``evaluate`` returns something that is not numeric.

    See Also
    --------
    petrophys.visualization.tornado_plot : Draw the result.

    Examples
    --------
    >>> def net_pay(rw=0.07, m=2.0):
    ...     return 100.0 - 300.0 * rw - 10.0 * m
    >>> table = parameter_sensitivity(
    ...     net_pay,
    ...     {"Rw 0.05 / 0.09": ({"rw": 0.05}, {"rw": 0.09}),
    ...      "m 1.8 / 2.2": ({"m": 1.8}, {"m": 2.2})})
    >>> table["case"].tolist()                  # widest swing first
    ['Rw 0.05 / 0.09', 'm 1.8 / 2.2']
    >>> table[["low", "base", "high", "swing"]].round(2).values.tolist()
    [[65.0, 59.0, 53.0, 12.0], [61.0, 59.0, 57.0, 4.0]]
    """
    import pandas as pd

    if not callable(evaluate):
        raise TypeError("evaluate must be callable")
    if not isinstance(cases, Mapping) or not cases:
        raise ValueError(
            "cases must be a non-empty mapping of label -> (low_changes, high_changes)"
        )
    pairs = {}
    for label, spec in cases.items():
        if (
            not isinstance(spec, (tuple, list)) or len(spec) != 2
            or not all(isinstance(changes, Mapping) for changes in spec)
        ):
            raise ValueError(
                f"case {label!r} must be a (low_changes, high_changes) pair of mappings"
            )
        pairs[str(label)] = (dict(spec[0]), dict(spec[1]))

    base = _outputs(evaluate(), outputs)
    rows = []
    for label, (low_changes, high_changes) in pairs.items():
        low = _outputs(evaluate(**low_changes), list(base))
        high = _outputs(evaluate(**high_changes), list(base))
        for name, base_value in base.items():
            rows.append({
                "case": label,
                "output": name,
                "low": low[name],
                "base": base_value,
                "high": high[name],
                "swing": abs(high[name] - low[name]),
                "low_setting": _describe(low_changes),
                "high_setting": _describe(high_changes),
            })
    table = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(base)}
    table = (
        table.assign(_order=table["output"].map(order))
        .sort_values(["_order", "swing"], ascending=[True, False], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    table.attrs["base"] = dict(base)
    return table


@dataclass(repr=False)
class SaturationModelComparison:
    """Water saturation from several models over one interval.

    Attributes
    ----------
    sw : dict of str to numpy.ndarray
        Water saturation per model (fraction, clipped to 0-1), aligned with
        ``depth``.
    table : pandas.DataFrame
        One row per model (index ``model``): ``mean_sw`` (thickness-weighted
        over the interval), ``mean_sw_pay``, ``net_pay_thickness``,
        ``ntg_pay`` and ``hcpt`` (hydrocarbon pore thickness, depth units).
    pay : dict of str to PaySummary
        Full net-pay summary per model.
    depth : numpy.ndarray
        Depths of the evaluated samples.
    parameters : dict
        Rw, Rsh, a, m, n and the cutoffs used.
    depth_unit : str
        Unit of the thicknesses.
    """

    sw: Dict[str, np.ndarray]
    table: Any
    pay: Dict[str, PaySummary]
    depth: np.ndarray
    parameters: Dict[str, Any] = field(default_factory=dict)
    depth_unit: str = ""

    def __repr__(self) -> str:
        unit = f" {self.depth_unit}" if self.depth_unit else ""
        net = self.table["net_pay_thickness"]
        hcpt = self.table["hcpt"]
        return (
            f"SaturationModelComparison({len(self.sw)} models over "
            f"{np.nanmin(self.depth):.1f}-{np.nanmax(self.depth):.1f}{unit}; "
            f"net pay {net.min():.2f}-{net.max():.2f}{unit}, "
            f"HCPT {hcpt.min():.3f}-{hcpt.max():.3f}{unit})"
        )


def _run_saturation_model(model, rt, rw, phie, vcl, rsh, a, m, n, vsh_exp, shale_exp):
    if model == "archie":
        return archie(rt, rw, phie, a, m, n)
    if model == "simandoux":
        return simandoux(rt, rw, phie, vcl, rsh, a, m, n)
    if model == "modified_simandoux":
        return modified_simandoux(rt, rw, phie, vcl, rsh, a, m, n)
    if model == "indonesia":
        return indonesia(rt, rw, phie, vcl, rsh, a, m, n)
    if model == "north_sea":
        return north_sea_shaly_sand(rt, rw, phie, vcl, rtsh=rsh, a=a, m=m, n=n, vsh_exp=vsh_exp)
    if model == "nigeria_schlumberger":
        return nigeria_schlumberger(rt, rw, phie, vcl, rsh=rsh, a=a, m=m, n=n, shale_exp=shale_exp)
    raise ValueError(model)


def compare_saturation_models(
    well=None,
    *,
    rw: Union[float, np.ndarray],
    rsh: Optional[float] = None,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    models: Sequence[str] = SATURATION_MODELS,
    rt=None,
    phie=None,
    vcl=None,
    depth=None,
    vsh_exp: float = 1.6,
    shale_exp: float = 1.6,
    vcl_cutoff: Optional[float] = None,
    phie_cutoff: Optional[float] = None,
    sw_cutoff: Optional[float] = None,
    lithology: str = "clastic",
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    depth_unit: Optional[str] = None,
) -> SaturationModelComparison:
    """Compare field-data water-saturation models on one interval.

    Every model runs on the same effective porosity, clay volume, Rt, Rw and
    Archie constants, and the result is summarized with the same cutoffs, so
    the table isolates the effect of the model choice. The models are the
    ones that need no laboratory inputs; Waxman-Smits and Dual-Water need
    CEC or bound-water data and are run through
    :class:`~petrophys.interpretation.saturation.SaturationModule` instead.

    Parameters
    ----------
    well : Well, optional
        Source of the depth and of any curve not passed as an array: Rt from
        RDEP/RT, PHIE from PHIE_CC/PHIE/PHIT, VCL from VCL/VCLAY/VSH.
    rw : float or array-like
        Formation-water resistivity at formation temperature (ohm.m). An
        array must have one value per sample of the full input.
    rsh : float, optional
        Shale (clay) resistivity in ohm.m, used as Rcl by Simandoux and
        Indonesia, Rtsh by North Sea and Rsh by Nigeria-Schlumberger.
        Required when any shaly-sand model is requested.
    a, m, n : float
        Tortuosity factor, cementation and saturation exponents.
    models : sequence of str
        Models to run, from :data:`SATURATION_MODELS`, or
        ``"modified_simandoux"``.
    rt, phie, vcl, depth : array-like, optional
        Explicit deep resistivity (ohm.m), effective porosity and clay volume
        (fractions) and depths.
    vsh_exp, shale_exp : float, default 1.6
        Shale exponents of the North Sea and Nigeria-Schlumberger models.
    vcl_cutoff, phie_cutoff, sw_cutoff : float, optional
        Net-pay cutoffs; unset ones use the standard set for ``lithology``.
    lithology : str, default "clastic"
        Lithology family for unset cutoffs.
    top, bottom : float, optional
        Depth window (inclusive), in depth units.
    depth_unit : str, optional
        Unit label for the thicknesses. Defaults to ``well.depth_unit``.

    Returns
    -------
    SaturationModelComparison
        Per-model Sw curves, a decision table and full pay summaries.

    Raises
    ------
    ValueError
        For an unknown model, a missing or non-positive ``rsh`` with a
        shaly-sand model, a non-positive ``rw``, or inconsistent inputs.

    Examples
    --------
    >>> import numpy as np
    >>> depth = np.arange(2841.0, 2846.0, 1.0)
    >>> result = compare_saturation_models(
    ...     rt=np.array([20.0, 12.0, 5.0, 2.0, 30.0]),
    ...     phie=np.array([0.22, 0.20, 0.18, 0.15, 0.24]),
    ...     vcl=np.array([0.10, 0.20, 0.30, 0.35, 0.05]),
    ...     depth=depth, depth_unit="m", rw=0.07, rsh=3.0, a=0.8,
    ...     models=["archie", "indonesia"])
    >>> result.table.index.tolist()
    ['archie', 'indonesia']
    >>> bool((result.sw["indonesia"] <= result.sw["archie"]).all())
    True
    """
    import pandas as pd

    models = [str(model).strip().lower() for model in models]
    if not models:
        raise ValueError("models must name at least one saturation model")
    unknown = [model for model in models if model not in (*SATURATION_MODELS, "modified_simandoux")]
    if unknown:
        raise ValueError(
            f"Unknown saturation model(s) {unknown}; choose from {', '.join(SATURATION_MODELS)}, modified_simandoux"
        )
    if any(model != "archie" for model in models):
        if rsh is None or not np.isfinite(float(rsh)) or float(rsh) <= 0:
            raise ValueError("rsh (shale resistivity, ohm.m) must be a positive number "
                             "for the shaly-sand models")
    for name, value in (("a", a), ("m", m), ("n", n)):
        if not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be a positive number")

    depth_w, arrays, dz, unit, window = resolve_windowed_inputs(
        well, depth,
        {"rt": (rt, RT_CURVES), "phie": (phie, PHIE_CURVES), "vcl": (vcl, VCL_CURVES)},
        top=top, bottom=bottom, depth_unit=depth_unit,
    )
    rw_array = np.asarray(rw, dtype=float)
    if rw_array.ndim == 0:
        rw_value = float(rw_array)
        if not np.isfinite(rw_value) or rw_value <= 0:
            raise ValueError("rw must be a positive number (ohm.m)")
    else:
        rw_array = rw_array.reshape(-1)
        if rw_array.size != window.size:
            raise ValueError(
                f"rw has {rw_array.size} values but the input has {window.size} samples"
            )
        rw_value = rw_array[window]
    cutoffs = resolve_cutoffs(vcl_cutoff, phie_cutoff, sw_cutoff, lithology)

    rt_w, phie_w, vcl_w = arrays["rt"], arrays["phie"], arrays["vcl"]
    saturations: Dict[str, np.ndarray] = {}
    summaries: Dict[str, PaySummary] = {}
    rows = []
    for model in models:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sw_model = _run_saturation_model(
                model, rt_w, rw_value, phie_w, vcl_w,
                None if rsh is None else float(rsh), float(a), float(m), float(n),
                float(vsh_exp), float(shale_exp),
            )
        sw_model = np.clip(np.asarray(sw_model, dtype=float), 0.0, 1.0)
        # A non-finite Rt is an unknown measurement: Sw is unknown for every model.
        sw_model[~np.isfinite(np.asarray(rt_w, dtype=float))] = np.nan
        saturations[model] = sw_model
        summary = pay_summary_from_arrays(vcl_w, phie_w, sw_model, dz, cutoffs, unit)
        summaries[model] = summary
        finite = np.isfinite(sw_model)
        weight = float(np.sum(dz[finite]))
        rows.append({
            "model": model,
            "mean_sw": float(np.sum(sw_model[finite] * dz[finite]) / weight) if weight else np.nan,
            "mean_sw_pay": summary.avg_sw_pay,
            "net_pay_thickness": summary.net_pay_thickness,
            "ntg_pay": summary.ntg_pay,
            "hcpt": summary.hcpt,
        })
    table = pd.DataFrame(rows).set_index("model")
    table.attrs["depth_unit"] = unit
    return SaturationModelComparison(
        sw=saturations,
        table=table,
        pay=summaries,
        depth=depth_w,
        parameters={
            "rw": rw if np.ndim(rw) == 0 else "per-sample",
            "rsh": rsh, "a": a, "m": m, "n": n,
            "vsh_exp": vsh_exp, "shale_exp": shale_exp, **cutoffs,
        },
        depth_unit=unit,
    )


__all__ = [
    "SATURATION_MODELS",
    "SaturationModelComparison",
    "compare_saturation_models",
    "cutoff_sensitivity",
    "parameter_sensitivity",
]
