"""Built-in steps for custom workflows.

Each function here returns a configured :class:`~petrophys.workflow.Step`
for :meth:`petrophys.workflow.Workflow.add`. The steps wrap the library's tested
calculations (for example :func:`~petrophys.interpretation.porosity.density_porosity`,
:class:`~petrophys.evaluation.NetPayModule` or
:func:`~petrophys.interpretation.geochemistry.interpret_spectral_gamma`); they
choose no methods on their own. Parameters are validated when the step is
created: a typo raises an error that names the closest valid option.

Every step states the curves it ``requires`` and the curves it ``produces``,
so :meth:`petrophys.workflow.Workflow.check` can tell, before running, what a well is
missing. Steps read and write standard curve names (``VCL``, ``PHIE``,
``SW``, ...); where a step reads a curve that a well might hold under another
name, a ``*_curve`` parameter selects it.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Step
     - Calculates
     - Produces
   * - :func:`standardize`
     - Canonical curve names and units
     - canonical aliases
   * - :func:`temperature`
     - Formation temperature profile
     - TEMP
   * - :func:`clay_volume`
     - Clay volume from GR, SP or neutron-density
     - VCL
   * - :func:`porosity`
     - Density, neutron or sonic porosity
     - PHIT, PHIE (+ PHID/PHIN/PHIS)
   * - :func:`formation_water`
     - Rw at formation temperature
     - RW
   * - :func:`mud_filtrate`
     - Rmf at formation temperature
     - RMF
   * - :func:`saturation`
     - Water saturation (Archie or shaly-sand)
     - SW, SXO
   * - :func:`permeability`
     - Empirical or core-calibrated permeability
     - PERM
   * - :func:`rock_quality`
     - RQI, FZI and normalized porosity
     - RQI, FZI, PHI_Z
   * - :func:`bvw`
     - Bulk volume water
     - BVW
   * - :func:`mhi`
     - Moveable hydrocarbon index
     - MHI
   * - :func:`net_pay`
     - Net rock, reservoir and pay flags
     - NET_ROCK, NET_RES, NET_PAY
   * - :func:`spectral_gamma`
     - K-Th-U standardization and ratios
     - as available
   * - :func:`interpret`
     - The whole of :func:`petrophys.interpret`
     - VCL, PHIE, SW, PERM, NET_PAY, ...
   * - :func:`formation_summary`
     - Per-formation table
     - a table in ``run.artifacts``

Examples
--------
>>> from petrophys.workflow import steps
>>> steps.porosity(method="density", matrix_density=2.65)
steps.porosity(method='density', matrix_density=2.65)
>>> steps.porosity(method="densty")
Traceback (most recent call last):
...
ValueError: Invalid steps.porosity() parameters:
  - method='densty' is not valid; use one of: density, neutron_density, neutron, sonic (did you mean 'density'?)
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Set

import numpy as np

from petrophys._internal.advice import Advice
from petrophys.model.well import Well
from petrophys.workflow.custom import (
    Step,
    _curve_requirement,
    _check_parameter_names,
    _json_param,
    _Params,
    _Requirement,
    _StepContext,
    _step_name,
    _suggestion,
)

__all__ = [
    "standardize",
    "temperature",
    "clay_volume",
    "porosity",
    "formation_water",
    "mud_filtrate",
    "saturation",
    "permeability",
    "rock_quality",
    "bvw",
    "mhi",
    "net_pay",
    "spectral_gamma",
    "interpret",
    "formation_summary",
]

#: Built-in step factories by kind; used to rebuild saved workflows.
_REGISTRY: Dict[str, Callable[..., Step]] = {}

#: The built-in step that makes each standard curve, for "how to fix" hints.
_PRODUCERS = {
    "TEMP": "steps.temperature()",
    "VCL": "steps.clay_volume()",
    "PHIT": "steps.porosity()",
    "PHIE": "steps.porosity()",
    "PHID": "steps.porosity(method='density')",
    "PHIN": "steps.porosity(method='neutron')",
    "PHIS": "steps.porosity(method='sonic')",
    "RW": "steps.formation_water(rw=..., rw_temp_f=...)",
    "RMF": "steps.mud_filtrate(rmf=..., rmf_temp_f=...)",
    "SW": "steps.saturation()",
    "SXO": "steps.saturation()",
    "PERM": "steps.permeability()",
    "NET_ROCK": "steps.net_pay()",
    "NET_RES": "steps.net_pay()",
    "NET_PAY": "steps.net_pay()",
}

_GR_FAMILY = ("GR", "CGR", "SGR", "HSGR", "HCGR", "CGR_CALC")
_RT_CURVES = ("RDEP", "RT")
_RXO_CURVES = ("RSHA", "RXO", "RESS")


#: The inputs a producer's call above asks for (its ``name=...`` arguments), as data.
_PRODUCER_INPUTS = {"RW": ("rw", "rw_temp_f"), "RMF": ("rmf", "rmf_temp_f"), "TEMP": ("bht_f",)}


def _producer_hint(curve: str) -> str:
    return _PRODUCERS.get(curve.upper(), "")


def _producer_inputs(curve: str) -> tuple:
    return _PRODUCER_INPUTS.get(curve.upper(), ())


def _builtin(func: Callable[..., Step]) -> Callable[..., Step]:
    """Register a step factory and give it name checking with suggestions."""
    kind = func.__name__
    signature = inspect.signature(func)
    valid = [
        parameter.name for parameter in signature.parameters.values()
        if parameter.name != "name" and parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ]
    open_ended = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    label = f"steps.{kind}()"

    @functools.wraps(func)
    def factory(*args: Any, **kwargs: Any) -> Step:
        _check_parameter_names(label, {key: value for key, value in kwargs.items() if key != "name"},
                               valid, open_ended=open_ended)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"{label}: {exc}") from None
        given: Dict[str, Any] = {}
        for key, value in bound.arguments.items():
            if key == "name":
                continue
            if signature.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
                given.update(value)
            else:
                given[key] = value
        step = func(*args, **kwargs)
        # Record what the caller set: numbers as typed (20, not 20.0), other
        # values in their validated form (normalized choices and curve names).
        params = {}
        for key, original in given.items():
            normalized = step._settings.get(key, original)
            number = isinstance(original, (int, float, np.integer, np.floating)) \
                and not isinstance(original, (bool, np.bool_))
            params[key] = _json_param(original if number else normalized, f"{label[:-2]}({key})")
        step._params = params
        step._factory = factory
        return step

    factory._petrophys_step_factory = True  # type: ignore[attr-defined]
    _REGISTRY[kind] = factory
    return factory


def _make(
    kind: str,
    name: Optional[str],
    settings: Mapping[str, Any],
    *,
    runner: Callable[..., None],
    requirements: Sequence[_Requirement] = (),
    produces: Optional[Mapping[str, str]] = None,
    description: str = "",
    planner: Optional[Callable[[Well, Set[str]], Iterable[str]]] = None,
    produces_text: str = "",
) -> Step:
    """Build a built-in step from validated settings (the runner's keywords).

    :func:`_builtin` then replaces ``params`` with the parameters the caller set.
    """
    return Step(
        kind,
        _step_name(name, kind),
        {},
        settings=dict(settings),
        runner=runner,
        requirements=requirements,
        produces=produces,
        description=description,
        planner=planner,
        produces_text=produces_text,
        label=f"steps.{kind}",
        default_name=kind,
    )


def _unused(check: _Params, method_label: str, given: Mapping[str, Any], allowed: Iterable[str]) -> None:
    """Reject parameters that the chosen method does not use."""
    allowed = set(allowed)
    for key, value in given.items():
        if value is not None and key not in allowed:
            check.add(f"{key} is not used by {method_label}; remove it")


def _tag(well: Well, name: str, step: str) -> None:
    """Record on a curve which workflow step wrote it."""
    curve = well.get(name)
    if curve is not None:
        curve.metadata.custom["workflow_step"] = step


def _required(value: Optional[float], what: str) -> float:
    if value is None:  # prevented by the step's validation and requirements
        raise ValueError(f"{what} is missing")
    return float(value)


def _percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("the curve has no finite samples")
    return float(np.percentile(finite, q))


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


@_builtin
def standardize(*, rename: bool = False, convert_units: bool = True,
                name: Optional[str] = None) -> Step:
    """Standardize curve names and units, as :func:`petrophys.interpret` does first.

    Adds canonical working aliases (``ILD`` is also available as ``RDEP``,
    ``ZDEN`` as ``RHOB``, ...) while keeping the vendor mnemonics, converts
    recognized units (percent porosity to v/v, for example) and removes
    physically implausible values. Put it first when a well uses vendor
    mnemonics, so later steps find the standard names.

    Wraps :func:`petrophys.preprocessing.standardization.standardize_curves`
    and :func:`~petrophys.preprocessing.standardization.add_standard_curve_aliases`.

    Parameters
    ----------
    rename : bool, default False
        Rename curves to canonical names instead of adding aliases.
    convert_units : bool, default True
        Convert recognized units to the library's standard units.
    name : str, optional
        Step name; defaults to ``"standardize"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Requires nothing; produces the canonical aliases of recognized
        curves. Mostly-implausible curves are reported in the warnings.

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.standardize().requires
    ()
    """
    check = _Params("steps.standardize()")
    values = {"rename": check.flag("rename", rename),
              "convert_units": check.flag("convert_units", convert_units)}
    check.raise_if_any()
    return _make(
        "standardize", name, values,
        runner=_run_standardize,
        description="Canonical curve names and units; implausible values removed",
        planner=_plan_standardize,
        produces_text="canonical aliases (e.g. RDEP, RHOB)",
    )


def _plan_standardize(well: Well, available: Set[str]) -> Iterable[str]:
    from petrophys.preprocessing.standardization import get_standard_mnemonic

    return sorted({get_standard_mnemonic(name) for name in available})


def _run_standardize(ctx: _StepContext, *, rename: bool, convert_units: bool) -> None:
    from petrophys.preprocessing.standardization import (
        add_standard_curve_aliases,
        standardize_curves,
    )
    from petrophys.preprocessing.standardization.curve_standardizer import (
        RANGE_SCREENING_WARNING_FRACTION,
    )

    well = ctx.well
    before = {name: str(getattr(well.get(name), "unit", "") or "") for name in well.curves}
    standardize_curves(well, rename_curves=rename, convert_units=convert_units)
    aliases = {} if rename else add_standard_curve_aliases(well)
    renamed = dict(well.get_parameter("_rename_map") or {}) if rename else {}
    ctx.trace.method = "curve_alias_and_unit_standardization"
    ctx.trace.input_curves = sorted(before)
    for mnemonic in well.curves:
        if mnemonic not in before:
            _tag(well, mnemonic, ctx.step.name)
            ctx.output(mnemonic)
    changes = {}
    for original, unit in before.items():
        converted = well.get(renamed.get(original, original)) or well.get(original)
        if converted is not None and str(converted.unit or "") != unit:
            changes[original] = f"{unit or '(none)'} -> {converted.unit}"
    ctx.trace.parameters.update({"aliases": aliases, "renamed": renamed, "unit_changes": changes})
    if aliases:
        ctx.decide("Added aliases: " + ", ".join(f"{alias} from {source}"
                                                 for source, alias in aliases.items()))
    for original, change in changes.items():
        ctx.decide(f"Converted {original} units: {change}")
    for mnemonic in well.curves:
        screened = well.get(mnemonic)
        item = screened.metadata.custom.get("range_screening") if screened is not None else None
        if item and item["fraction"] >= RANGE_SCREENING_WARNING_FRACTION:
            message = (f"{mnemonic}: {item['masked']} of {item['finite']} samples were outside the "
                       f"plausible {item['standard_curve']} range {item['low']:g}-{item['high']:g} "
                       "and were removed")
            if item.get("hint"):
                message += f"; {item['hint']}, check the unit label {item['source_unit']!r}"
            ctx.warn(message)
    if not aliases and not changes:
        ctx.decide("Curve names and units were already standard")


@_builtin
def temperature(*, bht_f: Optional[float] = None, td_ft: Optional[float] = None,
                surface_temp_f: float = 68.0, gradient_f_per_100ft: Optional[float] = None,
                name: Optional[str] = None) -> Step:
    """Formation temperature profile (``TEMP``, degF).

    A linear profile from the surface temperature to the bottom-hole
    temperature at total depth, or from a given gradient. Without either, the
    LAS header BHT is used, and failing that a 30 degC/km gradient (with a
    warning). Wraps
    :func:`petrophys.interpretation.temperature.calculate_temperature_profile`,
    the function :func:`petrophys.interpret` uses.

    :func:`formation_water` and :func:`mud_filtrate` correct resistivities
    along ``TEMP`` when it exists.

    Parameters
    ----------
    bht_f : float, optional
        Bottom-hole temperature, degF. Defaults to the LAS header BHT.
    td_ft : float, optional
        Depth of ``bht_f`` in feet. Defaults to the header TD, then the
        deepest sample.
    surface_temp_f : float, default 68.0
        Surface (mean annual) temperature, degF.
    gradient_f_per_100ft : float, optional
        Geothermal gradient, degF/100 ft, instead of a BHT.
    name : str, optional
        Step name; defaults to ``"temperature"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Requires nothing; produces ``TEMP`` (degF).

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.temperature(bht_f=230).produces
    {'TEMP': 'degF'}
    """
    check = _Params("steps.temperature()")
    values = {
        "bht_f": check.number("bht_f", bht_f, minimum=-60, maximum=800),
        "td_ft": check.number("td_ft", td_ft, positive=True),
        "surface_temp_f": check.number("surface_temp_f", surface_temp_f, optional=False,
                                       minimum=-60, maximum=200),
        "gradient_f_per_100ft": check.number("gradient_f_per_100ft", gradient_f_per_100ft,
                                             positive=True),
    }
    if bht_f is not None and gradient_f_per_100ft is not None:
        check.add("give bht_f or gradient_f_per_100ft, not both")
    check.raise_if_any()
    return _make(
        "temperature", name, values,
        runner=_run_temperature,
        produces={"TEMP": "degF"},
        description="Formation temperature from BHT or a gradient",
    )


def _run_temperature(ctx: _StepContext, *, bht_f: Optional[float], td_ft: Optional[float],
                     surface_temp_f: float, gradient_f_per_100ft: Optional[float]) -> None:
    from petrophys.interpretation.temperature import (
        calculate_temperature_profile,
        gradient_to_celsius_per_km,
    )
    from petrophys.interpretation.temperature.temp_profile import store_temperature_parameters

    result = calculate_temperature_profile(
        ctx.well, bht=bht_f, surface_temp=surface_temp_f, gradient=gradient_f_per_100ft,
        bht_unit="F", surface_temp_unit="F", td=td_ft,
        td_unit="ft" if td_ft is not None else None,
    )
    ctx.add_curve("TEMP", result.temperature, "degF", "Formation Temperature")
    store_temperature_parameters(ctx.well, result)
    ctx.trace.method = result.source
    gradient_c = gradient_to_celsius_per_km(result.gradient)
    ctx.trace.parameters.update({
        "gradient_f_per_100ft": result.gradient,
        "gradient_c_per_km": gradient_c,
        "reference_temperature_f": result.reference_temperature,
        "reference_depth_ft": result.reference_depth,
    })
    if result.source == "fallback_geothermal_gradient":
        ctx.warn("No bottom-hole temperature was found; used the default 30 degC/km gradient. "
                 "Pass bht_f= for a calibrated profile.")
    elif result.reference_temperature is not None:
        ctx.decide(f"Anchored to BHT {result.reference_temperature:.1f} degF at "
                   f"{result.reference_depth:.0f} ft")
    ctx.decide(f"Gradient {result.gradient:.3f} degF/100 ft ({gradient_c:.1f} degC/km); "
               f"TEMP {np.nanmin(result.temperature):.1f}-{np.nanmax(result.temperature):.1f} degF")


# --------------------------------------------------------------------------
# Clay volume and porosity
# --------------------------------------------------------------------------

_CLAY_METHODS = ("gr", "sp", "neutron_density", "minimum")
_ND_MATRICES = ("sandstone", "limestone", "dolomite")


@_builtin
def clay_volume(method: str = "gr", *, gr_clean: Optional[float] = None,
                gr_shale: Optional[float] = None, gr_transform: str = "linear",
                gr_curve: Optional[str] = None, sp_clean: Optional[float] = None,
                sp_shale: Optional[float] = None, nd_matrix: Optional[str] = None,
                name: Optional[str] = None) -> Step:
    """Clay volume (``VCL``) from gamma ray, SP or neutron-density.

    ============================  ==================================================
    ``method``                    Calculation
    ============================  ==================================================
    ``"gr"``                      Gamma-ray index between ``gr_clean`` and
                                  ``gr_shale``, then ``gr_transform``
    ``"sp"``                      SP deflection between ``sp_clean`` and ``sp_shale``
    ``"neutron_density"``         Three-point neutron-density clay volume
    ``"minimum"``                 Lowest of the GR, SP and neutron-density
                                  estimates available at each depth (as in
                                  the :func:`petrophys.interpret` "geolink"
                                  profile); also writes ``VCL_GR``,
                                  ``VCL_SP``, ``VCL_ND``
    ============================  ==================================================

    Endpoints left as None are picked from the log (P5 and P95) and recorded
    in the trace. Explicit endpoints are used exactly as given. With the
    defaults ``"minimum"`` reads low: SP endpoints picked over the whole log
    mostly measure baseline drift, and in sandstone the limestone clean line
    and the fixed clay point (2.62 g/cm3, 0.38) give a low VCL_ND.
    :func:`petrophys.interpret` keeps such an SP out of the minimum and
    picks the clay point in the well's shales; here, give ``sp_clean`` and
    ``sp_shale`` only for a drift-free SP, and the matrix in ``nd_matrix``. Uses
    :func:`~petrophys.interpretation.clay_volume.vcl_linear` and the
    Larionov, Stieber and Clavier transforms,
    :func:`~petrophys.interpretation.clay_volume.vcl_from_sp` and
    :func:`~petrophys.interpretation.clay_volume.vcl_from_nd`.

    Parameters
    ----------
    method : {"gr", "sp", "neutron_density", "minimum"}, default "gr"
        Clay indicator.
    gr_clean, gr_shale : float, optional
        Clean and shale gamma-ray endpoints, API. Default: P5 and P95.
    gr_transform : str, default "linear"
        ``"linear"``, ``"larionov_tertiary"``, ``"larionov_older"``,
        ``"stieber"`` (stored as ``"steiber"``) or ``"clavier"``.
    gr_curve : str, optional
        Gamma-ray curve. Default: the first of GR, CGR, SGR, HSGR, HCGR and
        CGR_CALC with data.
    sp_clean, sp_shale : float, optional
        Clean (static SP) and shale SP baselines, mV. Default: P5 and P95.
    nd_matrix : {"sandstone", "limestone", "dolomite"}, optional
        Clean line of the neutron-density method. Default ``"limestone"``,
        as in the "geolink" profile of :func:`petrophys.interpret`; its
        default profile uses the line of the resolved matrix.
    name : str, optional
        Step name; defaults to ``"clay_volume"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Produces ``VCL`` (v/v).

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.clay_volume(method="gr", gr_clean=20, gr_shale=110).requires
    ('GR',)
    >>> steps.clay_volume(method="neutron_density").requires
    ('NPHI', 'RHOB')
    """
    from petrophys.interpretation.clay_volume.gr_method import GR_VCL_METHODS, gr_method_name

    check = _Params("steps.clay_volume()")
    method = check.choice("method", method, _CLAY_METHODS)
    values = {
        "method": method,
        "gr_clean": check.number("gr_clean", gr_clean),
        "gr_shale": check.number("gr_shale", gr_shale),
        "gr_transform": check.choice("gr_transform", gr_method_name(gr_transform), GR_VCL_METHODS),
        "gr_curve": check.curve("gr_curve", gr_curve, optional=True),
        "sp_clean": check.number("sp_clean", sp_clean),
        "sp_shale": check.number("sp_shale", sp_shale),
        "nd_matrix": check.choice("nd_matrix", nd_matrix, _ND_MATRICES, optional=True),
    }
    if values["gr_clean"] is not None and values["gr_shale"] is not None \
            and values["gr_clean"] >= values["gr_shale"]:
        check.add(f"gr_clean ({gr_clean}) must be less than gr_shale ({gr_shale})")
    if values["sp_clean"] is not None and values["sp_clean"] == values["sp_shale"]:
        check.add("sp_clean and sp_shale must differ")
    uses = {
        "gr": ("gr_clean", "gr_shale", "gr_curve"),
        "sp": ("sp_clean", "sp_shale"),
        "neutron_density": ("nd_matrix",),
        "minimum": ("gr_clean", "gr_shale", "gr_curve", "sp_clean", "sp_shale", "nd_matrix"),
    }.get(method, ())
    _unused(check, f"method={method!r}",
            {key: values[key] for key in ("gr_clean", "gr_shale", "gr_curve", "sp_clean",
                                          "sp_shale", "nd_matrix")}, uses)
    check.raise_if_any()

    requirements = []
    if method in ("gr", "minimum"):
        curve = values["gr_curve"]
        requirements.append(_curve_requirement(curve or "GR", (curve,) if curve else _GR_FAMILY,
                                               "gamma ray"))
    if method == "sp":
        requirements.append(_curve_requirement("SP", what="spontaneous potential"))
    if method == "neutron_density":
        requirements.append(_curve_requirement("NPHI", what="neutron porosity"))
        requirements.append(_curve_requirement("RHOB", what="bulk density"))
    return _make(
        "clay_volume", name, values,
        runner=_run_clay_volume,
        requirements=requirements,
        produces={"VCL": "v/v"},
        description="Clay volume from gamma ray, SP or neutron-density",
    )


def _gr_transform(name: str) -> Callable[[np.ndarray], np.ndarray]:
    from petrophys.interpretation import clay_volume as module

    transforms = {
        "linear": lambda igr: igr,
        "larionov_tertiary": module.vcl_larionov_tertiary,
        "larionov_older": module.vcl_larionov_older,
        "steiber": module.vcl_steiber,
        "clavier": module.vcl_clavier,
    }
    return transforms[name]


def _run_clay_volume(ctx: _StepContext, *, method: str, gr_clean: Optional[float],
                     gr_shale: Optional[float], gr_transform: str, gr_curve: Optional[str],
                     sp_clean: Optional[float], sp_shale: Optional[float],
                     nd_matrix: Optional[str]) -> None:
    from petrophys.interpretation.clay_volume import vcl_from_nd, vcl_from_sp, vcl_linear
    from petrophys.interpretation.clay_volume.nd_method import nd_clean_line

    indicators: Dict[str, np.ndarray] = {}
    parameters = ctx.trace.parameters
    if method in ("gr", "minimum"):
        label = ctx.step._requirements[0].label
        gr = ctx.data(label)
        source = ctx.curve_name(label)
        clean = gr_clean if gr_clean is not None else _percentile(gr, 5)
        shale = gr_shale if gr_shale is not None else _percentile(gr, 95)
        if clean >= shale:
            raise ValueError(f"GR clean endpoint {clean:.1f} must be below the shale endpoint "
                             f"{shale:.1f}; pass gr_clean= and gr_shale=")
        indicators["GR"] = _gr_transform(gr_transform)(np.asarray(vcl_linear(gr, clean, shale)))
        parameters.update({"gr_clean_used": clean, "gr_shale_used": shale})
        ctx.decide(f"GR clean = {clean:.1f} API ({'given' if gr_clean is not None else 'P5 of ' + source}), "
                   f"GR shale = {shale:.1f} API ({'given' if gr_shale is not None else 'P95 of ' + source}); "
                   f"{gr_transform} transform")
    sp_data = None
    if method == "sp":
        sp_data = ("SP", ctx.data("SP"))
    elif method == "minimum":
        sp_data = ctx.optional("SP")
    if sp_data is not None:
        sp = sp_data[1]
        clean = sp_clean if sp_clean is not None else _percentile(sp, 5)
        shale = sp_shale if sp_shale is not None else _percentile(sp, 95)
        indicators["SP"] = np.asarray(vcl_from_sp(sp, clean, shale), dtype=float)
        parameters.update({"sp_clean_used": clean, "sp_shale_used": shale})
        ctx.decide(f"SP clean = {clean:.1f} mV ({'given' if sp_clean is not None else 'P5'}), "
                   f"SP shale = {shale:.1f} mV ({'given' if sp_shale is not None else 'P95'})")
    nd = None
    if method == "neutron_density":
        nd = (ctx.data("NPHI"), ctx.data("RHOB"))
    elif method == "minimum":
        nphi, rhob = ctx.optional("NPHI"), ctx.optional("RHOB")
        if nphi is not None and rhob is not None:
            nd = (nphi[1], rhob[1])
    if nd is not None:
        matrix = nd_matrix or "limestone"
        indicators["ND"] = np.asarray(vcl_from_nd(nd[0], nd[1], *nd_clean_line(matrix)), dtype=float)
        parameters["nd_matrix_used"] = matrix
        ctx.decide(f"Neutron-density clay volume with the {matrix} clean line")
    if method == "minimum":
        stacked = np.vstack(list(indicators.values()))
        vcl = np.full(ctx.well.n_samples, np.nan)
        valid = np.any(np.isfinite(stacked), axis=0)
        vcl[valid] = np.nanmin(stacked[:, valid], axis=0)
        for key, values in indicators.items():
            ctx.add_curve(f"VCL_{key}", values, "v/v", f"Clay volume from {key}")
        ctx.decide("VCL = minimum of " + ", ".join(indicators) + " at each depth")
        ctx.trace.method = "minimum_" + "_".join(key.lower() for key in indicators)
    else:
        vcl = next(iter(indicators.values()))
        ctx.trace.method = {"gr": f"gr_{gr_transform}", "sp": "sp",
                            "neutron_density": "neutron_density"}[method]
    ctx.add_curve("VCL", vcl, "v/v", f"Clay volume ({ctx.trace.method})")


_POROSITY_METHODS = ("density", "neutron_density", "neutron", "sonic")


@_builtin
def porosity(method: str = "neutron_density", *, matrix: str = "sandstone",
             matrix_density: Optional[float] = None, matrix_neutron: Optional[float] = None,
             matrix_slowness: Optional[float] = None, fluid_density: float = 1.0,
             fluid_slowness: float = 189.0, clay_porosity: float = 0.15,
             vcl_curve: Optional[str] = "VCL", hc_type: str = "oil",
             name: Optional[str] = None) -> Step:
    """Total (``PHIT``) and effective (``PHIE``) porosity.

    =====================  ======================================================
    ``method``             ``PHIT``
    =====================  ======================================================
    ``"density"``          ``PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)``
    ``"neutron_density"``  Adaptive combination of ``PHID`` and the
                           matrix-corrected neutron ``PHIN`` (in crossover,
                           their root-mean-square for gas and ``PHID`` for
                           oil), as :func:`petrophys.interpret`
    ``"neutron"``          ``PHIN = NPHI - nphi_ma``
    ``"sonic"``            ``PHIS`` from the Wyllie time average
    =====================  ======================================================

    ``PHIE = PHIT - VCL * clay_porosity``. The component curves (``PHID``,
    ``PHIN``, ``PHIS``) are written too. Uses
    :func:`~petrophys.interpretation.porosity.density_porosity`,
    :func:`~petrophys.interpretation.porosity.neutron_porosity`,
    :func:`~petrophys.interpretation.porosity.total_porosity`,
    :func:`~petrophys.interpretation.porosity.wyllie_time_average` and
    :func:`~petrophys.interpretation.porosity.effective_porosity`.

    Parameters
    ----------
    method : {"density", "neutron_density", "neutron", "sonic"}, default "neutron_density"
        Porosity log(s) to use.
    matrix : str, default "sandstone"
        Mineral whose endpoints are used unless overridden, e.g.
        ``"limestone"`` or ``"dolomite"``; see
        :meth:`petrophys.model.constants.MatrixProperties.names`.
    matrix_density : float, optional
        Matrix density, g/cm3 (density methods). Defaults to the mineral's
        apparent log density, which for evaporites differs from the grain
        density (:meth:`~petrophys.model.constants.MatrixProperties.get_log_density`).
    matrix_neutron : float, optional
        Matrix neutron porosity, v/v (neutron methods).
    matrix_slowness : float, optional
        Matrix slowness, us/ft (sonic).
    fluid_density : float, default 1.0
        Pore-fluid density, g/cm3.
    fluid_slowness : float, default 189.0
        Pore-fluid slowness, us/ft.
    clay_porosity : float, default 0.15
        Clay-bound porosity removed per unit clay volume, v/v.
    vcl_curve : str or None, default "VCL"
        Clay volume curve for the ``PHIE`` correction; None writes
        ``PHIE = PHIT``.
    hc_type : {"oil", "gas"}, default "oil"
        Hydrocarbon type. Where ``PHID > PHIN + 0.05`` (crossover),
        ``"neutron_density"`` takes ``PHIT = sqrt((PHID^2 + PHIN^2) / 2)``
        (Gaymard and Poupon, 1968) for gas and ``PHIT = PHID`` for oil, whose
        crossover comes from the matrix or shale endpoints rather than the fluid.
    name : str, optional
        Step name; defaults to ``"porosity"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Produces ``PHIT`` and ``PHIE`` (v/v) and the component curve(s).

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> step = steps.porosity(method="density", matrix_density=2.65)
    >>> step.requires, list(step.produces)
    (('RHOB', 'VCL'), ['PHID', 'PHIT', 'PHIE'])
    """
    from petrophys.model.constants import MatrixProperties

    check = _Params("steps.porosity()")
    method = check.choice("method", method, _POROSITY_METHODS)
    if not isinstance(matrix, str) or matrix.strip().lower() not in MatrixProperties.names():
        check.add(f"matrix={matrix!r} is not a known mineral; use one of: "
                  f"{', '.join(MatrixProperties.names())}" + _suggestion(matrix, MatrixProperties.names()))
    values = {
        "method": method,
        "matrix": matrix.strip().lower() if isinstance(matrix, str) else matrix,
        "matrix_density": check.number("matrix_density", matrix_density, minimum=1.5, maximum=4.0),
        "matrix_neutron": check.number("matrix_neutron", matrix_neutron, minimum=-0.2, maximum=0.6),
        "matrix_slowness": check.number("matrix_slowness", matrix_slowness, positive=True),
        "fluid_density": check.number("fluid_density", fluid_density, optional=False,
                                      minimum=0.1, maximum=1.5),
        "fluid_slowness": check.number("fluid_slowness", fluid_slowness, optional=False,
                                       positive=True),
        "clay_porosity": check.number("clay_porosity", clay_porosity, optional=False,
                                      fraction=True),
        "vcl_curve": check.curve("vcl_curve", vcl_curve, optional=True),
        "hc_type": check.choice("hc_type", hc_type, ("oil", "gas")),
    }
    uses = {"density": ("matrix_density",), "neutron": ("matrix_neutron",),
            "neutron_density": ("matrix_density", "matrix_neutron"),
            "sonic": ("matrix_slowness",)}.get(method, ())
    _unused(check, f"method={method!r}",
            {key: values[key] for key in ("matrix_density", "matrix_neutron", "matrix_slowness")},
            uses)
    check.raise_if_any()

    requirements = []
    produces: Dict[str, str] = {}
    if method in ("density", "neutron_density"):
        requirements.append(_curve_requirement("RHOB", what="bulk density"))
        produces["PHID"] = "v/v"
    if method in ("neutron", "neutron_density"):
        requirements.insert(0, _curve_requirement("NPHI", what="neutron porosity"))
        produces["PHIN"] = "v/v"
    if method == "sonic":
        requirements.append(_curve_requirement("DTC or DT", ("DTC", "DT"), "compressional slowness"))
        produces["PHIS"] = "v/v"
    clay_curve = values["vcl_curve"]
    if isinstance(clay_curve, str):
        requirements.append(_curve_requirement(clay_curve, what="clay volume"))
    produces.update({"PHIT": "v/v", "PHIE": "v/v"})
    return _make(
        "porosity", name, values,
        runner=_run_porosity,
        requirements=requirements,
        produces=produces,
        description="Total and effective porosity from density, neutron or sonic logs",
    )


def _run_porosity(ctx: _StepContext, *, method: str, matrix: str,
                  matrix_density: Optional[float], matrix_neutron: Optional[float],
                  matrix_slowness: Optional[float], fluid_density: float,
                  fluid_slowness: float, clay_porosity: float, vcl_curve: Optional[str],
                  hc_type: str = "oil") -> None:
    from petrophys.interpretation.porosity.density_porosity import density_porosity
    from petrophys.interpretation.porosity.effective_porosity import effective_porosity
    from petrophys.interpretation.porosity.neutron_porosity import neutron_porosity
    from petrophys.interpretation.porosity.sonic_porosity import wyllie_time_average
    from petrophys.interpretation.porosity.total_porosity import total_porosity
    from petrophys.model.constants import MatrixProperties

    props = MatrixProperties.get(matrix)
    rho_ma = float(matrix_density if matrix_density is not None else MatrixProperties.get_log_density(matrix))
    nphi_ma = float(matrix_neutron if matrix_neutron is not None else props.nphi)
    dt_ma = float(matrix_slowness if matrix_slowness is not None else props.slowness)
    phid: Any = None
    phin: Any = None
    if method in ("density", "neutron_density"):
        rhob = ctx.data("RHOB")
        if np.nanmedian(rhob) > 10:
            ctx.warn("RHOB looks like kg/m3, not g/cm3; add steps.standardize() before porosity")
        phid = density_porosity(rhob, rho_ma, float(fluid_density))
        ctx.add_curve("PHID", phid, "v/v", "Density porosity")
        ctx.trace.parameters["matrix_density_used"] = rho_ma
        ctx.decide(f"PHID = ({rho_ma:g} - RHOB) / ({rho_ma:g} - {float(fluid_density):g})")
    if method in ("neutron", "neutron_density"):
        nphi = ctx.data("NPHI")
        if np.nanmax(nphi) > 1.5:
            ctx.warn("NPHI looks like percent, not v/v; add steps.standardize() before porosity")
        phin = neutron_porosity(nphi, nphi_matrix=nphi_ma)
        ctx.add_curve("PHIN", phin, "v/v", "Neutron porosity")
        ctx.trace.parameters["matrix_neutron_used"] = nphi_ma
        sign = "-" if nphi_ma >= 0 else "+"
        ctx.decide(f"PHIN = NPHI {sign} {abs(nphi_ma):g}, limited to 0-0.6")
    if method == "sonic":
        label = ctx.step._requirements[0].label
        phis: Any = wyllie_time_average(ctx.data(label), dt_ma, float(fluid_slowness))
        ctx.add_curve("PHIS", phis, "v/v", "Sonic porosity")
        ctx.trace.parameters["matrix_slowness_used"] = dt_ma
        ctx.decide(f"PHIS = ({ctx.curve_name(label)} - {dt_ma:g}) / ({float(fluid_slowness):g} - {dt_ma:g})")
        phit = phis
    elif method == "neutron_density":
        gas = hc_type == "gas"
        phit = total_porosity(phin, phid, gas_method="rms" if gas else "density")
        ctx.decide(
            "PHIT = adaptive neutron-density combination ("
            + ("root-mean-square of PHID and PHIN" if gas else "PHID")
            + f" in crossover, hc_type={hc_type!r})"
        )
    else:
        phit = phid if method == "density" else phin
        ctx.decide(f"PHIT = {'PHID' if method == 'density' else 'PHIN'}")
    ctx.trace.method = method
    ctx.add_curve("PHIT", phit, "v/v", f"Total porosity ({method})")
    if vcl_curve:
        phie = effective_porosity(phit, ctx.data(vcl_curve), float(clay_porosity))
        ctx.decide(f"PHIE = PHIT - {ctx.curve_name(vcl_curve)} * {float(clay_porosity):g}")
    else:
        phie = np.asarray(phit, dtype=float)
        ctx.decide("PHIE = PHIT (no clay correction)")
    ctx.add_curve("PHIE", phie, "v/v", f"Effective porosity ({method})")


# --------------------------------------------------------------------------
# Water, mud filtrate and saturation
# --------------------------------------------------------------------------


def _header_value(well: Well, name: str) -> Optional[float]:
    value = well.get_parameter(name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _header_temperature_f(well: Well, name: str) -> Optional[float]:
    from petrophys.model.units import temperature_to_fahrenheit

    value = _header_value(well, name)
    if value is None:
        return None
    return temperature_to_fahrenheit(value, well.get_parameter(f"{name}__UNIT"))


def _header_present(resistivity: str, temperature: str, well: Well,
                    settings: Mapping[str, Any]) -> Optional[str]:
    if _header_value(well, resistivity) is None:
        return None
    explicit = settings.get("rw_temp_f", settings.get("rmf_temp_f"))
    if explicit is None and _header_temperature_f(well, temperature) is None:
        return None
    return f"LAS header ({resistivity})"


def _header_requirement(resistivity: str, temperature: str, what: str) -> _Requirement:
    # Requirement predicates are module-level functions so that steps, workflows
    # and run results can be pickled.
    return _Requirement(
        label=f"header {resistivity} and {temperature}",
        what=what,
        hint=f"the LAS file needs {resistivity} and {temperature} parameters, or pass the "
             "values explicitly",
        predicate=functools.partial(_header_present, resistivity, temperature),
    )


def _formation_temperature(ctx: _StepContext) -> np.ndarray:
    """Formation temperature in degF: TEMP if present, else a default profile."""
    from petrophys.interpretation.temperature import calculate_temperature_profile
    from petrophys.model.units import temperature_unit

    found = ctx.optional("TEMP")
    if found is not None:
        name, values = found
        temp_curve = ctx.well.get(name)
        scale = temperature_unit(temp_curve.unit if temp_curve is not None else None)
        if scale == "C":
            values = values * 9.0 / 5.0 + 32.0
        elif scale == "K":
            values = (values - 273.15) * 9.0 / 5.0 + 32.0
        ctx.decide("Corrected along the TEMP curve")
        return values
    result = calculate_temperature_profile(ctx.well)
    if result.source == "fallback_geothermal_gradient":
        ctx.warn(Advice("No TEMP curve or bottom-hole temperature; formation temperature from the "
                        "default 30 degC/km gradient. Add steps.temperature(bht_f=...) first.", ("bht_f",)))
    ctx.decide(f"No TEMP curve; formation temperature from {result.source.replace('_', ' ')}")
    return np.asarray(result.temperature, dtype=float)


def _resistivity(check: _Params, label: str, value: Any, header: str) -> Any:
    """A positive resistivity in ohm.m, or "header"."""
    if isinstance(value, str):
        if value.strip().lower() != "header":
            check.add(f"{label}={value!r} is not valid; give a resistivity in ohm.m or "
                      f"'header' to read {header} from each well's LAS header")
            return value
        return "header"
    return check.number(label, value, positive=True)


@_builtin
def formation_water(rw: Any = None, rw_temp_f: Optional[float] = None, *,
                    salinity_ppm: Optional[float] = None, name: Optional[str] = None) -> Step:
    """Formation water resistivity at formation temperature (``RW``).

    Give Rw with the temperature it was measured at, or a salinity, or
    ``rw="header"`` to read ``RW`` and ``RWT`` from each well's LAS header
    (useful when every well has its own value). Rw is corrected to the
    formation temperature of each sample with Arps' equation along the
    ``TEMP`` curve; without ``TEMP``, a profile is built from the header BHT.
    Uses :func:`~petrophys.interpretation.electrical.rw.correct_rw_temperature`
    and :func:`~petrophys.interpretation.electrical.rw.rw_from_salinity`.

    Parameters
    ----------
    rw : float or "header", optional
        Formation water resistivity, ohm.m, at ``rw_temp_f``.
    rw_temp_f : float, optional
        Temperature at which ``rw`` was measured, degF. Required with a
        numeric ``rw``; with ``rw="header"`` it overrides the header RWT.
    salinity_ppm : float, optional
        Formation water salinity, ppm NaCl equivalent, instead of ``rw``.
    name : str, optional
        Step name; defaults to ``"formation_water"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Produces ``RW`` (ohm.m). Reads ``TEMP`` when present.

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.formation_water(rw=0.07, rw_temp_f=230)
    steps.formation_water(rw=0.07, rw_temp_f=230)
    >>> steps.formation_water(rw=0.07)
    Traceback (most recent call last):
    ...
    ValueError: Invalid steps.formation_water() parameters:
      - rw_temp_f is required with rw: the temperature (degF) at which rw was measured, e.g. rw=0.07, rw_temp_f=230
    """
    check = _Params("steps.formation_water()")
    value = _resistivity(check, "rw", rw, "RW and RWT")
    values = {
        "rw": value,
        "rw_temp_f": check.number("rw_temp_f", rw_temp_f, minimum=-60, maximum=800),
        "salinity_ppm": check.number("salinity_ppm", salinity_ppm, positive=True,
                                     maximum=400000),
    }
    if rw is None and salinity_ppm is None:
        check.add("give rw= (ohm.m, with rw_temp_f=), salinity_ppm=, or rw='header' to read "
                  "RW and RWT from each well's LAS header")
    if rw is not None and salinity_ppm is not None:
        check.add("give rw or salinity_ppm, not both")
    if isinstance(value, float) and rw_temp_f is None:
        check.add("rw_temp_f is required with rw: the temperature (degF) at which rw was "
                  "measured, e.g. rw=0.07, rw_temp_f=230")
    if salinity_ppm is not None and rw_temp_f is not None:
        check.add("rw_temp_f is only used with rw")
    check.raise_if_any()
    requirements = []
    if value == "header":
        requirements.append(_header_requirement("RW", "RWT", "formation water resistivity"))
    return _make(
        "formation_water", name, values,
        runner=_run_formation_water,
        requirements=requirements,
        produces={"RW": "ohm.m"},
        description="Formation water resistivity at formation temperature",
    )


def _run_formation_water(ctx: _StepContext, *, rw: Any, rw_temp_f: Optional[float],
                         salinity_ppm: Optional[float]) -> None:
    from petrophys.interpretation.electrical.rw import correct_rw_temperature, rw_from_salinity

    temperature = _formation_temperature(ctx)
    if salinity_ppm is not None:
        values = rw_from_salinity(float(salinity_ppm), temperature, "F")
        ctx.trace.method = "salinity"
        ctx.decide(f"Rw from salinity {salinity_ppm:g} ppm NaCl at formation temperature")
    else:
        if rw == "header":
            reference = _required(_header_value(ctx.well, "RW"), "RW header value")
            measured = _required(rw_temp_f if rw_temp_f is not None
                                 else _header_temperature_f(ctx.well, "RWT"), "RWT header value")
            source = "LAS header RW" + (" and RWT" if rw_temp_f is None else "")
            ctx.trace.method = "header_rw_arps"
        else:
            reference, measured, source = float(rw), _required(rw_temp_f, "rw_temp_f"), "given"
            ctx.trace.method = "given_rw_arps"
        values = correct_rw_temperature(reference, measured, temperature, "F")
        ctx.trace.parameters.update({"rw_used": reference, "rw_temp_f_used": measured})
        ctx.decide(f"Rw {reference:g} ohm.m at {measured:g} degF ({source}), "
                   "corrected with Arps' equation")
    values = np.broadcast_to(np.asarray(values, dtype=float), (ctx.well.n_samples,))
    ctx.add_curve("RW", values, "ohm.m", "Formation water resistivity")
    ctx.trace.parameters.update({"rw_min": float(np.nanmin(values)),
                                 "rw_max": float(np.nanmax(values))})
    ctx.decide(f"RW {np.nanmin(values):.4f}-{np.nanmax(values):.4f} ohm.m")


@_builtin
def mud_filtrate(rmf: Any = None, rmf_temp_f: Optional[float] = None, *,
                 name: Optional[str] = None) -> Step:
    """Mud-filtrate resistivity at formation temperature (``RMF``).

    Needed for invaded-zone saturation (``SXO`` from shallow resistivity in
    :func:`saturation`) and for :func:`mhi`. Give Rmf with the temperature it
    was measured at, or ``rmf="header"`` to read ``RMF`` and ``RMFT`` from
    each well's LAS header. Corrected with Arps' equation along ``TEMP``
    (:func:`~petrophys.interpretation.temperature.correct_resistivity_temperature`).

    Parameters
    ----------
    rmf : float or "header"
        Mud-filtrate resistivity, ohm.m, at ``rmf_temp_f``.
    rmf_temp_f : float, optional
        Temperature at which ``rmf`` was measured, degF. Required with a
        numeric ``rmf``; with ``rmf="header"`` it overrides the header RMFT.
    name : str, optional
        Step name; defaults to ``"mud_filtrate"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Produces ``RMF`` (ohm.m). Reads ``TEMP`` when present.

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.mud_filtrate(rmf=0.3, rmf_temp_f=75).produces
    {'RMF': 'ohm.m'}
    """
    check = _Params("steps.mud_filtrate()")
    value = _resistivity(check, "rmf", rmf, "RMF and RMFT")
    values = {"rmf": value,
              "rmf_temp_f": check.number("rmf_temp_f", rmf_temp_f, minimum=-60, maximum=800)}
    if rmf is None:
        check.add("give rmf= (ohm.m, with rmf_temp_f=) or rmf='header'")
    elif isinstance(value, float) and rmf_temp_f is None:
        check.add("rmf_temp_f is required with rmf: the temperature (degF) at which rmf was "
                  "measured, e.g. rmf=0.3, rmf_temp_f=75")
    check.raise_if_any()
    requirements = []
    if value == "header":
        requirements.append(_header_requirement("RMF", "RMFT", "mud-filtrate resistivity"))
    return _make(
        "mud_filtrate", name, values,
        runner=_run_mud_filtrate,
        requirements=requirements,
        produces={"RMF": "ohm.m"},
        description="Mud-filtrate resistivity at formation temperature",
    )


def _run_mud_filtrate(ctx: _StepContext, *, rmf: Any, rmf_temp_f: Optional[float]) -> None:
    from petrophys.interpretation.temperature import correct_resistivity_temperature

    temperature = _formation_temperature(ctx)
    if rmf == "header":
        reference = _required(_header_value(ctx.well, "RMF"), "RMF header value")
        measured = _required(rmf_temp_f if rmf_temp_f is not None
                             else _header_temperature_f(ctx.well, "RMFT"), "RMFT header value")
        source = "LAS header"
    else:
        reference, measured, source = float(rmf), _required(rmf_temp_f, "rmf_temp_f"), "given"
    values = correct_resistivity_temperature(reference, measured, temperature, "F")
    values = np.broadcast_to(np.asarray(values, dtype=float), (ctx.well.n_samples,))
    ctx.trace.method = "arps"
    ctx.add_curve("RMF", values, "ohm.m", "Mud filtrate resistivity")
    ctx.trace.parameters.update({"rmf_used": reference, "rmf_temp_f_used": measured})
    ctx.decide(f"Rmf {reference:g} ohm.m at {measured:g} degF ({source}), corrected "
               f"with Arps' equation: {np.nanmin(values):.3f}-{np.nanmax(values):.3f} ohm.m")


_SW_MODELS = ("archie", "simandoux", "modified_simandoux", "indonesia", "dual_water", "waxman_smits", "north_sea",
              "nigeria_schlumberger", "auto")
_SW_MODEL_OPTIONS = {
    "archie": (),
    "simandoux": ("rcl",),
    "modified_simandoux": ("rcl",),
    "indonesia": ("rcl",),
    "north_sea": ("rcl", "vsh_exp"),
    "nigeria_schlumberger": ("rcl", "shale_exp"),
    "dual_water": ("rwb", "clay_porosity"),
    "waxman_smits": ("cec", "b_coeff", "m_star", "matrix_density"),
    "auto": ("rcl", "rwb", "clay_porosity", "cec", "b_coeff", "m_star", "matrix_density",
             "salinity_ppm", "vcl_cutoff", "lithology"),
}
_SHALY_MODELS = {"simandoux", "modified_simandoux", "indonesia", "north_sea", "nigeria_schlumberger", "dual_water",
                 "waxman_smits", "auto"}


@_builtin
def saturation(model: str = "archie", *, a: float = 1.0, m: float = 2.0, n: float = 2.0,
               rcl: Optional[float] = None, vsh_exp: Optional[float] = None,
               shale_exp: Optional[float] = None, rwb: Optional[float] = None,
               clay_porosity: Optional[float] = None, cec: Optional[float] = None,
               b_coeff: Optional[float] = None, m_star: Optional[float] = None,
               matrix_density: Optional[float] = None, salinity_ppm: Optional[float] = None,
               vcl_cutoff: Optional[float] = None, lithology: Optional[str] = None,
               phie_curve: str = "PHIE", phit_curve: str = "PHIT", vcl_curve: str = "VCL",
               name: Optional[str] = None) -> Step:
    """Water saturation (``SW``) and flushed-zone saturation (``SXO``).

    Reads deep resistivity (``RDEP`` or ``RT``), ``RW`` (from
    :func:`formation_water`) and effective porosity; shaly-sand models also
    read clay volume, and Dual Water and Waxman-Smits total porosity. Uses
    the same equations and routing as :func:`petrophys.interpret`: the
    published Indonesia equation (Poupon and Leveaux, 1971), the wet-zone
    Dual Water form of Clavier, Coates and Dumanoir (1984), and Waxman and
    Smits (1968) with its total-porosity saturation Swt reported on the
    effective basis, ``SW = 1 - (PHIT / PHIE) * (1 - Swt)``.

    ``SXO`` is calculated from shallow resistivity where ``RMF`` (from
    :func:`mud_filtrate`) and ``RSHA``/``RXO``/``RESS`` exist, and from the
    quick-look ``SW ** 0.2`` elsewhere.

    .. list-table::
       :header-rows: 1
       :widths: 30 70

       * - ``model``
         - Extra parameters (defaults)
       * - ``"archie"``
         - ``a``, ``m``, ``n``
       * - ``"simandoux"``, ``"modified_simandoux"``, ``"indonesia"``
         - ``rcl`` (median RT where VCL > 0.9, else 5 ohm.m)
       * - ``"north_sea"``
         - ``rcl``, ``vsh_exp`` (1.6)
       * - ``"nigeria_schlumberger"``
         - ``rcl``, ``shale_exp`` (1.6)
       * - ``"dual_water"``
         - ``rwb`` (required; not estimated), ``clay_porosity`` (0.15)
       * - ``"waxman_smits"``
         - ``cec`` and ``b_coeff`` (required), ``m_star`` (2.0),
           ``matrix_density`` (2.65); uses ``n``, not ``a`` and ``m``
       * - ``"auto"``
         - Archie where VCL < ``vcl_cutoff`` (0.3), else a shaly-sand model
           chosen per sample from ``lithology`` (clastic), ``salinity_ppm``,
           ``cec`` and ``b_coeff``; carbonate samples use Dual Water with
           ``rwb`` and the clastic routing without it

    A parameter that the chosen model does not use is an error.

    Parameters
    ----------
    model : str, default "archie"
        Saturation model; see the table.
    a, m, n : float, default 1, 2, 2
        Archie tortuosity factor, cementation and saturation exponents.
    rcl : float, optional
        Clay (shale) resistivity, ohm.m, for the shaly-sand models.
    vsh_exp : float, optional
        Shale exponent of the North Sea model.
    shale_exp : float, optional
        Shale exponent of the Nigeria-Schlumberger model.
    rwb : float, optional
        Bound-water resistivity of the Dual Water model, ohm.m, from lab data;
        it is not estimated.
    clay_porosity : float, optional
        Clay porosity of the Dual Water model, v/v.
    cec : float, optional
        Cation exchange capacity for Waxman-Smits, meq/100 g (0 to 150),
        converted to the meq/g of
        :func:`~petrophys.interpretation.saturation.waxman_smits`.
    b_coeff : float, optional
        Equivalent counter-ion conductance B for Waxman-Smits,
        (S/m)/(meq/cm3).
    m_star : float, optional
        Waxman-Smits cementation exponent.
    matrix_density : float, optional
        Grain density for the Waxman-Smits Qv, g/cm3.
    salinity_ppm : float, optional
        Water salinity for ``"auto"`` routing, ppm NaCl equivalent.
    vcl_cutoff : float, optional
        Clay volume below which ``"auto"`` uses Archie, v/v.
    lithology : {"clastic", "carbonate"}, optional
        Lithology family for ``"auto"`` routing.
    phie_curve : str, default "PHIE"
        Effective porosity curve, e.g. ``"PHIE_CC"`` for a core-corrected
        input.
    phit_curve : str, default "PHIT"
        Total porosity curve (Dual Water, Waxman-Smits).
    vcl_curve : str, default "VCL"
        Clay volume curve (shaly-sand models).
    name : str, optional
        Step name; defaults to ``"saturation"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Produces ``SW`` and ``SXO`` (v/v).

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.saturation(model="archie", a=1, m=2, n=2).requires
    ('PHIE', 'RDEP or RT', 'RW')
    >>> steps.saturation(model="archie", rcl=4)
    Traceback (most recent call last):
    ...
    ValueError: Invalid steps.saturation() parameters:
      - rcl is not used by model='archie'; remove it
    """
    from petrophys.interpretation.saturation.waxman_smits import MAX_CEC_MEQ_PER_100G

    check = _Params("steps.saturation()")
    model = check.choice("model", model, _SW_MODELS)
    options = {
        "rcl": check.number("rcl", rcl, positive=True),
        "vsh_exp": check.number("vsh_exp", vsh_exp, positive=True),
        "shale_exp": check.number("shale_exp", shale_exp, positive=True),
        "rwb": check.number("rwb", rwb, positive=True),
        "clay_porosity": check.number("clay_porosity", clay_porosity, fraction=True),
        "cec": check.number("cec", cec, minimum=0, maximum=MAX_CEC_MEQ_PER_100G),
        "b_coeff": check.number("b_coeff", b_coeff, positive=True),
        "m_star": check.number("m_star", m_star, positive=True),
        "matrix_density": check.number("matrix_density", matrix_density, minimum=1.5, maximum=4.0),
        "salinity_ppm": check.number("salinity_ppm", salinity_ppm, positive=True),
        "vcl_cutoff": check.number("vcl_cutoff", vcl_cutoff, fraction=True),
        "lithology": check.choice("lithology", lithology, ("clastic", "carbonate"), optional=True),
    }
    values = {
        "model": model,
        "a": check.number("a", a, optional=False, positive=True),
        "m": check.number("m", m, optional=False, positive=True),
        "n": check.number("n", n, optional=False, positive=True),
        **options,
        "phie_curve": check.required_curve("phie_curve", phie_curve),
        "phit_curve": check.required_curve("phit_curve", phit_curve),
        "vcl_curve": check.required_curve("vcl_curve", vcl_curve),
    }
    if model in _SW_MODEL_OPTIONS:
        _unused(check, f"model={model!r}", options, _SW_MODEL_OPTIONS[model])
    if model == "waxman_smits" and (cec is None or b_coeff is None):
        check.add("model='waxman_smits' needs cec= (meq/100 g) and b_coeff=")
    if model == "dual_water" and rwb is None:
        check.add("model='dual_water' needs rwb= (bound-water resistivity, ohm.m); "
                  "it is not estimated")
    check.raise_if_any()

    requirements = [
        _curve_requirement(values["phie_curve"], what="effective porosity"),
        _curve_requirement("RDEP or RT", _RT_CURVES, "deep resistivity"),
        _curve_requirement("RW", what="formation water resistivity"),
    ]
    if model in _SHALY_MODELS:
        requirements.append(_curve_requirement(values["vcl_curve"], what="clay volume"))
    if model in ("dual_water", "waxman_smits"):
        requirements.append(_curve_requirement(values["phit_curve"], what="total porosity"))
    return _make(
        "saturation", name, values,
        runner=_run_saturation,
        requirements=requirements,
        produces={"SW": "v/v", "SXO": "v/v"},
        description="Water saturation from resistivity, porosity and Rw",
    )


def _run_saturation(ctx: _StepContext, *, model: str, a: float, m: float, n: float,
                    rcl: Optional[float], vsh_exp: Optional[float], shale_exp: Optional[float],
                    rwb: Optional[float], clay_porosity: Optional[float], cec: Optional[float],
                    b_coeff: Optional[float], m_star: Optional[float],
                    matrix_density: Optional[float], salinity_ppm: Optional[float],
                    vcl_cutoff: Optional[float], lithology: Optional[str], phie_curve: str,
                    phit_curve: str, vcl_curve: str) -> None:
    from petrophys.interpretation.clay_volume import RclModule
    from petrophys.interpretation.saturation.sw_module import SaturationModule

    well = ctx.well
    phie = ctx.data(phie_curve)
    rw = ctx.data("RW")
    ctx.use(ctx.curve_name("RDEP or RT"))
    shaly = model in _SHALY_MODELS
    vcl = ctx.data(vcl_curve) if shaly else np.zeros(well.n_samples)
    phit = None
    if model in ("dual_water", "waxman_smits"):
        phit = ctx.data(phit_curve)
    elif model == "auto":
        found = ctx.optional(phit_curve)
        phit = found[1] if found is not None else None
    rcl_used = rcl
    if shaly and rcl is None:
        rcl_used = RclModule().calculate(well, vcl=vcl)
        shale = np.isfinite(vcl) & (vcl > 0.9)
        if not np.any(shale):
            ctx.warn("No samples with VCL > 0.9 to estimate clay resistivity; Rcl = 5 ohm.m "
                     "(pass rcl=)")
        else:
            ctx.decide(f"Rcl = {rcl_used:.2f} ohm.m (median RT where VCL > 0.9)")
    rmf = ctx.optional("RMF")
    if cec is not None and 0.0 < cec < 1.0:
        ctx.warn(Advice(f"cec={cec:g} meq/100 g is a clean-sand value; a meq/g value must be "
                        "multiplied by 100", ("cec",)))
    module = SaturationModule(a=a, m=m, n=n, model=model,
                              vcl_cutoff=vcl_cutoff if vcl_cutoff is not None else 0.3,
                              rwb_fallback=None, effective_waxman_smits=True)
    result = module.calculate(
        well, rw=rw, rcl=rcl_used, salinity=salinity_ppm,
        rmf=rmf[1] if rmf is not None else None, vcl=vcl, phie=phie, phit=phit,
        lithology=lithology or "clastic",
        # Lab CEC is reported in meq/100 g; waxman_smits() takes meq/g.
        cec=cec / 100.0 if cec is not None else None, b_coeff=b_coeff,
        rho_ma=matrix_density if matrix_density is not None else 2.65,
        m_star=m_star if m_star is not None else 2.0, rwb=rwb,
        phi_cl=clay_porosity if clay_porosity is not None else 0.15,
        vsh_exp=vsh_exp if vsh_exp is not None else 1.6,
        shale_exp=shale_exp if shale_exp is not None else 1.6,
    )
    ctx.trace.method = result.model
    ctx.trace.parameters.update({"rcl_used": rcl_used, "model_sample_counts": result.model_counts,
                                 "sxo_method": result.sxo_method})
    sw = np.asarray(result.sw, dtype=float)
    ctx.add_curve("SW", sw, "v/v", f"Water saturation ({result.model})")
    if result.sxo is not None:
        ctx.add_curve("SXO", result.sxo, "v/v", "Flushed-zone water saturation")
        ctx.decide(f"SXO: {result.sxo_method} ({result.sxo_reason})")
    if model == "auto" and result.model_counts:
        counts = ", ".join(f"{key}={value}" for key, value in result.model_counts.items())
        ctx.decide(f"Routed per sample: {counts}")
    else:
        ctx.decide(f"{result.model} with a={a:g}, m={m:g}, n={n:g}")
    names = {"indonesia": "Indonesia", "simandoux": "Simandoux",
             "modified_simandoux": "Modified Simandoux", "waxman_smits": "Waxman-Smits"}
    for route, count in (result.fallback_counts or {}).items():
        model_used = route.split("->", 1)[1]
        ctx.decide(f"{names.get(model_used, model_used)} for {count} shaly carbonate samples: "
                   "Dual Water needs rwb=, which is not estimated")
    if "waxman_smits" in (result.model_counts or {}) and cec is not None:
        ctx.decide(f"Waxman-Smits with CEC {cec:g} meq/100 g = {cec / 100.0:g} meq/g; "
                   "SW = 1 - (PHIT/PHIE) * (1 - Swt) on the effective basis")
        if result.waxman_smits_clipped_to_zero:
            ctx.decide(f"Waxman-Smits: {result.waxman_smits_clipped_to_zero} samples with Swt below "
                       "the bound-water fraction (PHIT - PHIE)/PHIT set to SW = 0")
    if np.any(np.isfinite(sw)):
        ctx.decide(f"SW {np.nanmin(sw):.3f}-{np.nanmax(sw):.3f}")
    else:
        ctx.warn("SW has no finite values; check the porosity, resistivity and RW inputs")


# --------------------------------------------------------------------------
# Permeability and reservoir quality
# --------------------------------------------------------------------------


@_builtin
def permeability(method: str = "coates", *, swirr: Optional[float] = None,
                 fallback_method: Optional[str] = None, min_core_points: Optional[int] = None,
                 phie_curve: str = "PHIE", name: Optional[str] = None) -> Step:
    """Permeability (``PERM``, mD) from an empirical transform or core.

    ==================  ===================================================
    ``method``          Calculation
    ==================  ===================================================
    ``"coates"``        ``k = 10000 PHIE^4 ((1 - Swirr) / Swirr)^2``
    ``"timur"``         ``k = 0.136 (100 PHIE)^4.4 / (100 Swirr)^2``
    ``"wyllie_rose"``   ``k = 62500 PHIE^6 / Swirr^2`` (Wyllie-Rose form
                        with the Morris-Biggs oil constant,
                        ``sqrt(k) = 250 PHIE^3 / Swirr``)
    ``"lith_vcl"``      Lithology- and clay-dependent transform
    ``"auto"``          Poro-perm transform fitted to same-well core
                        (``PHIC``/``KH_CORE``) where possible, the
                        ``fallback_method`` elsewhere
    ==================  ===================================================

    Without ``swirr``, irreducible water saturation comes from a ``SWIRR``
    curve, NMR curves, or the low tail of ``SW`` in the reservoir, and
    otherwise defaults to 0.20 (with a warning). Wraps
    :class:`~petrophys.interpretation.PermeabilityModule`; values are
    capped as in :func:`petrophys.interpret`.

    Parameters
    ----------
    method : str, default "coates"
        Transform; see the table.
    swirr : float, optional
        Irreducible water saturation, v/v (not used by ``"lith_vcl"``).
    fallback_method : str, optional
        Empirical method where core cannot calibrate (``"auto"`` only);
        default ``"coates"``.
    min_core_points : int, optional
        Minimum core samples to fit a transform (``"auto"`` only); default 3.
    phie_curve : str, default "PHIE"
        Porosity curve; with ``"auto"`` also the log porosity the core
        poro-perm model is fitted on.
    name : str, optional
        Step name; defaults to ``"permeability"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Produces ``PERM`` (mD).

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.permeability(method="timur", swirr=0.15)
    steps.permeability(method='timur', swirr=0.15)
    """
    from petrophys.interpretation.permeability.perm_module import (
        PERMEABILITY_METHOD_ALIASES,
        PERMEABILITY_METHODS,
    )

    check = _Params("steps.permeability()")
    if isinstance(method, str) and method.strip().lower() in PERMEABILITY_METHOD_ALIASES:
        method = PERMEABILITY_METHOD_ALIASES[method.strip().lower()]
    method = check.choice("method", method, PERMEABILITY_METHODS)
    empirical = tuple(item for item in PERMEABILITY_METHODS if item != "auto")
    values = {
        "method": method,
        "swirr": check.number("swirr", swirr, minimum=0.0, maximum=1.0),
        "fallback_method": check.choice("fallback_method", fallback_method, empirical,
                                        optional=True),
        "min_core_points": check.number("min_core_points", min_core_points, minimum=2),
        "phie_curve": check.required_curve("phie_curve", phie_curve),
    }
    if values["swirr"] is not None and not 0 < values["swirr"] < 1:
        check.add(f"swirr must be between 0 and 1 (exclusive); got {swirr!r}")
    if min_core_points is not None and not float(min_core_points).is_integer():
        check.add(f"min_core_points must be a whole number; got {min_core_points!r}")
    uses = {"auto": ("swirr", "fallback_method", "min_core_points"),
            "lith_vcl": ()}.get(method, ("swirr",))
    _unused(check, f"method={method!r}",
            {key: values[key] for key in ("swirr", "fallback_method", "min_core_points")}, uses)
    check.raise_if_any()
    return _make(
        "permeability", name, values,
        runner=_run_permeability,
        requirements=[_curve_requirement(values["phie_curve"], what="effective porosity")],
        produces={"PERM": "mD"},
        description="Permeability from an empirical transform or same-well core",
    )


def _run_permeability(ctx: _StepContext, *, method: str, swirr: Optional[float],
                      fallback_method: Optional[str], min_core_points: Optional[float],
                      phie_curve: str) -> None:
    from petrophys.interpretation.permeability import PermeabilityModule

    # The core poro-perm model is fitted on, and applied to, the step's porosity curve.
    module = PermeabilityModule(
        method=method, fallback_method=fallback_method or "coates",
        min_core_points=int(min_core_points) if min_core_points is not None else 3,
        core_porosity_curve=ctx.curve_name(phie_curve),
    )
    phi = ctx.data(phie_curve)
    result = module.calculate(ctx.well, swi=swirr, phi=phi)
    ctx.trace.method = result.method
    ctx.add_curve("PERM", result.perm, "mD", f"Permeability ({result.method})")
    for curve in result.metadata.get("input_curves") or []:
        if curve and curve != "PHIE" and ctx.well.get(curve) is not None:
            ctx.use(curve)
    swi: Any = result.swi_used
    if np.isscalar(swi) and swi is not None:
        swi = float(np.asarray(swi, dtype=float))
        ctx.trace.parameters["swirr_used"] = swi
        ctx.decide(f"Swirr = {swi:.3f} ({result.swi_source})")
        if result.swi_source == "default_0.20":
            ctx.warn("Swirr could not be estimated from the logs and defaulted to 0.20; "
                     "pass swirr=")
        elif swirr is None and swi < 0.05:
            ctx.warn(f"Swirr estimated from the logs as {swi:.3f}; values this low "
                     "inflate permeability. Pass swirr= from core or NMR data.")
    elif swi is not None:
        ctx.decide(f"Swirr from a curve ({result.swi_source})")
    models = result.metadata.get("core_models") or []
    if models:
        ctx.decide(f"{len(models)} core poro-perm model(s) fitted; "
                   f"{result.metadata.get('core_sections_count', 0)} samples core-calibrated")
    if result.metadata.get("perm_max_clip_md") is not None:
        ctx.trace.parameters["perm_max_clip_md"] = result.metadata["perm_max_clip_md"]
    perm = np.asarray(result.perm, dtype=float)
    if np.any(np.isfinite(perm)):
        ctx.decide(f"PERM {np.nanmin(perm):.3g}-{np.nanmax(perm):.3g} mD")


@_builtin
def rock_quality(*, perm_curve: str = "PERM", phie_curve: str = "PHIE",
                 name: Optional[str] = None) -> Step:
    """Reservoir quality index, normalized porosity and flow zone indicator.

    ``RQI = 0.0314 sqrt(PERM / PHIE)``, ``PHI_Z = PHIE / (1 - PHIE)`` and
    ``FZI = RQI / PHI_Z``, from
    :class:`~petrophys.evaluation.RockQualityModule`.

    Parameters
    ----------
    perm_curve : str, default "PERM"
        Permeability curve, mD.
    phie_curve : str, default "PHIE"
        Effective porosity curve, v/v.
    name : str, optional
        Step name; defaults to ``"rock_quality"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Produces ``RQI`` and ``FZI`` (microns) and ``PHI_Z`` (v/v).

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> list(steps.rock_quality().produces)
    ['RQI', 'FZI', 'PHI_Z']
    """
    check = _Params("steps.rock_quality()")
    values = {"perm_curve": check.required_curve("perm_curve", perm_curve),
              "phie_curve": check.required_curve("phie_curve", phie_curve)}
    check.raise_if_any()
    return _make(
        "rock_quality", name, values,
        runner=_run_rock_quality,
        requirements=[_curve_requirement(values["perm_curve"], what="permeability"),
                      _curve_requirement(values["phie_curve"], what="effective porosity")],
        produces={"RQI": "microns", "FZI": "microns", "PHI_Z": "v/v"},
        description="Reservoir quality index and flow zone indicator",
    )


def _run_rock_quality(ctx: _StepContext, *, perm_curve: str, phie_curve: str) -> None:
    from petrophys.evaluation.rock_quality import RockQualityModule

    result = RockQualityModule().calculate(ctx.well, perm=ctx.data(perm_curve),
                                           phi_e=ctx.data(phie_curve))
    ctx.trace.method = "rqi_fzi"
    ctx.add_curve("RQI", result.rqi, "microns", "Reservoir Quality Index")
    ctx.add_curve("FZI", result.fzi, "microns", "Flow Zone Indicator")
    ctx.add_curve("PHI_Z", result.phi_z, "v/v", "Normalized Porosity")
    ctx.decide(f"RQI = 0.0314 * sqrt({ctx.curve_name(perm_curve)} / {ctx.curve_name(phie_curve)}); "
               "FZI = RQI / PHI_Z")


@_builtin
def bvw(*, phie_curve: str = "PHIE", sw_curve: str = "SW", name: Optional[str] = None) -> Step:
    """Bulk volume water, ``BVW = PHIE * SW``.

    Uses :func:`~petrophys.evaluation.calculate_bvw`.

    Parameters
    ----------
    phie_curve : str, default "PHIE"
        Effective porosity curve.
    sw_curve : str, default "SW"
        Water saturation curve.
    name : str, optional
        Step name; defaults to ``"bvw"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Produces ``BVW`` (v/v).

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.bvw().requires
    ('PHIE', 'SW')
    """
    check = _Params("steps.bvw()")
    values = {"phie_curve": check.required_curve("phie_curve", phie_curve),
              "sw_curve": check.required_curve("sw_curve", sw_curve)}
    check.raise_if_any()
    return _make(
        "bvw", name, values,
        runner=_run_bvw,
        requirements=[_curve_requirement(values["phie_curve"], what="effective porosity"),
                      _curve_requirement(values["sw_curve"], what="water saturation")],
        produces={"BVW": "v/v"},
        description="Bulk volume water",
    )


def _run_bvw(ctx: _StepContext, *, phie_curve: str, sw_curve: str) -> None:
    from petrophys.evaluation import calculate_bvw

    ctx.trace.method = "phie_x_sw"
    ctx.add_curve("BVW", calculate_bvw(ctx.data(phie_curve), ctx.data(sw_curve)), "v/v",
                  "Bulk Volume Water")
    ctx.decide(f"BVW = {ctx.curve_name(phie_curve)} * {ctx.curve_name(sw_curve)}")


@_builtin
def mhi(*, sw_curve: str = "SW", sxo_curve: str = "SXO", name: Optional[str] = None) -> Step:
    """Moveable hydrocarbon index, ``MHI = SW / SXO``.

    Kept only where ``SXO`` has invaded-zone support (shallow resistivity
    and ``RMF``), so run :func:`mud_filtrate` before :func:`saturation`.
    Uses :class:`~petrophys.evaluation.MhiModule`.

    Parameters
    ----------
    sw_curve : str, default "SW"
        Water saturation curve.
    sxo_curve : str, default "SXO"
        Flushed-zone saturation curve.
    name : str, optional
        Step name; defaults to ``"mhi"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Produces ``MHI`` (unitless).

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.mhi().requires
    ('SW', 'SXO', 'RMF', 'RSHA or RXO or RESS')
    """
    check = _Params("steps.mhi()")
    values = {"sw_curve": check.required_curve("sw_curve", sw_curve),
              "sxo_curve": check.required_curve("sxo_curve", sxo_curve)}
    check.raise_if_any()
    return _make(
        "mhi", name, values,
        runner=_run_mhi,
        requirements=[_curve_requirement(values["sw_curve"], what="water saturation"),
                      _curve_requirement(values["sxo_curve"], what="flushed-zone saturation"),
                      _curve_requirement("RMF", what="mud-filtrate resistivity"),
                      _curve_requirement("RSHA or RXO or RESS", _RXO_CURVES,
                                         "shallow resistivity")],
        produces={"MHI": ""},
        description="Moveable hydrocarbon index",
    )


def _run_mhi(ctx: _StepContext, *, sw_curve: str, sxo_curve: str) -> None:
    from petrophys.evaluation import MhiModule

    ctx.use(ctx.curve_name("RMF"))
    ctx.use(ctx.curve_name("RSHA or RXO or RESS"))
    values = MhiModule().calculate(ctx.well, sxo=ctx.data(sxo_curve), sw=ctx.data(sw_curve))
    ctx.trace.method = "sw_over_sxo"
    ctx.add_curve("MHI", values, "", "Moveable Hydrocarbon Index")
    kept = int(np.count_nonzero(np.isfinite(values)))
    ctx.decide(f"MHI = SW / SXO on {kept} samples with invaded-zone SXO; NaN elsewhere")


@_builtin
def net_pay(*, vcl_cutoff: Optional[float] = None, phie_cutoff: Optional[float] = None,
            sw_cutoff: Optional[float] = None, lithology: str = "clastic",
            vcl_curve: Optional[str] = "VCL", phie_curve: str = "PHIE", sw_curve: str = "SW",
            exclude_non_reservoir: bool = True, name: Optional[str] = None) -> Step:
    """Net rock, net reservoir and net pay flags.

    ``NET_ROCK`` where ``VCL < vcl_cutoff``; ``NET_RES`` where also
    ``PHIE > phie_cutoff``; ``NET_PAY`` where also ``SW < sw_cutoff``.
    Cutoffs left as None take the standard values
    (:data:`petrophys.evaluation.STANDARD_CUTOFFS`: clastic 0.40 / 0.08 /
    0.60, carbonate 0.30 / 0.045 / 0.70) of each sample's lithology family,
    as :func:`petrophys.interpret` takes them: with a lithology code log
    (``LITHOLOGY_GEOLINK``, or another lithology-code curve as
    :func:`petrophys.interpret` reads it, decoded with the same built-in
    lithology code table), limestone, chalk, dolomite and marl codes are carbonate and the
    other codes clastic; samples without a code take ``lithology``. With
    ``exclude_non_reservoir``, the samples :func:`petrophys.interpret` keeps
    out of net reservoir and net pay leave them here too: evaporite and coal
    codes, halite recognised on the logs (RHOB 1.95-2.15 g/cm3, GR < 15 API
    and RT > 500 ohm.m), and ``BAD_HOLE`` = 1. Uses
    :class:`~petrophys.evaluation.NetPayModule`.

    Parameters
    ----------
    vcl_cutoff : float, optional
        Maximum clay volume of net rock, v/v.
    phie_cutoff : float, optional
        Minimum effective porosity of net reservoir, v/v.
    sw_cutoff : float, optional
        Maximum water saturation of net pay, v/v.
    lithology : {"clastic", "carbonate"}, default "clastic"
        Family whose standard cutoffs fill the ones not given, at samples
        without a lithology code.
    vcl_curve : str or None, default "VCL"
        Clay volume curve; None applies no clay cutoff.
    phie_curve : str, default "PHIE"
        Effective porosity curve.
    sw_curve : str, default "SW"
        Water saturation curve.
    exclude_non_reservoir : bool, default True
        Leave non-reservoir samples (evaporite, coal, halite on the logs, bad
        hole) out of net reservoir and net pay. False applies the cutoffs
        alone.
    name : str, optional
        Step name; defaults to ``"net_pay"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Produces ``NET_ROCK``, ``NET_RES`` and ``NET_PAY`` (1 or 0).

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.net_pay(vcl_cutoff=0.35, phie_cutoff=0.10, sw_cutoff=0.60)
    steps.net_pay(vcl_cutoff=0.35, phie_cutoff=0.1, sw_cutoff=0.6)
    """
    check = _Params("steps.net_pay()")
    values = {
        "vcl_cutoff": check.number("vcl_cutoff", vcl_cutoff, fraction=True),
        "phie_cutoff": check.number("phie_cutoff", phie_cutoff, fraction=True),
        "sw_cutoff": check.number("sw_cutoff", sw_cutoff, fraction=True),
        "lithology": check.choice("lithology", lithology, ("clastic", "carbonate")),
        "vcl_curve": check.curve("vcl_curve", vcl_curve, optional=True),
        "phie_curve": check.required_curve("phie_curve", phie_curve),
        "sw_curve": check.required_curve("sw_curve", sw_curve),
        "exclude_non_reservoir": check.flag("exclude_non_reservoir", exclude_non_reservoir),
    }
    if vcl_curve is None and vcl_cutoff is not None:
        check.add("vcl_cutoff needs a clay volume curve; remove it or set vcl_curve")
    check.raise_if_any()
    requirements = []
    clay_curve = values["vcl_curve"]
    if isinstance(clay_curve, str):
        requirements.append(_curve_requirement(clay_curve, what="clay volume"))
    requirements += [_curve_requirement(values["phie_curve"], what="effective porosity"),
                     _curve_requirement(values["sw_curve"], what="water saturation")]
    return _make(
        "net_pay", name, values,
        runner=_run_net_pay,
        requirements=requirements,
        produces={"NET_ROCK": "", "NET_RES": "", "NET_PAY": ""},
        description="Net rock, net reservoir and net pay flags from cutoffs",
    )


def _run_net_pay(ctx: _StepContext, *, vcl_cutoff: Optional[float], phie_cutoff: Optional[float],
                 sw_cutoff: Optional[float], lithology: str, vcl_curve: Optional[str],
                 phie_curve: str, sw_curve: str, exclude_non_reservoir: bool) -> None:
    from petrophys._internal.analysis_helpers import get_curve
    from petrophys.interpretation.lithology.matrix_inference import find_lithology_code_curve
    from petrophys.workflow import dynamic_workflow as workflow

    well = ctx.well
    # Per-sample lithology groups from the lithology code log, grouped as
    # petrophys.interpret groups them; without one, ``lithology`` throughout.
    code_curve = find_lithology_code_curve(well)
    codes = get_curve(well, [code_curve]) if code_curve is not None else None
    groups = lithology if codes is None else workflow._group_array_from_numeric_codes(
        np.asarray(codes.data, dtype=float), workflow._CURRENT_GROUND_TRUTH_CODE_TO_GROUP
    )
    if codes is not None and code_curve != "LITHOLOGY_GEOLINK":
        ctx.warn(
            f"Lithology families from {code_curve}, decoded with the built-in "
            "lithology code table."
        )
    excluded = {}
    if exclude_non_reservoir:
        excluded = workflow._non_reservoir_samples(well, groups)
    vcl = ctx.data(vcl_curve) if vcl_curve else np.zeros(well.n_samples)
    explicit = {"vcl_cutoff": vcl_cutoff, "phie_cutoff": phie_cutoff, "sw_cutoff": sw_cutoff}
    _, result, details = workflow._net_flags_by_family(
        well, workflow._cutoff_families(groups, lithology), vcl,
        ctx.data(phie_curve), ctx.data(sw_curve), explicit, excluded,
    )
    by_family = details["cutoffs_by_family"]
    ctx.trace.method = "cutoffs"
    ctx.trace.parameters.update({f"{key}_used": value for key, value in details["cutoffs"].items()})
    ctx.trace.parameters["cutoffs_by_family"] = by_family
    step = abs(float(well.step)) if well.step else 0.0
    flags = (("NET_ROCK", result.net_rock, "Net rock flag"),
             ("NET_RES", result.net_reservoir, "Net reservoir flag"),
             ("NET_PAY", result.net_pay, "Net pay flag"))
    cutoff = {key: workflow._family_cutoff_text(by_family, key) for key in explicit}
    tests = (f"{ctx.curve_name(vcl_curve)} < {cutoff['vcl_cutoff']}" if vcl_curve
             else "no clay cutoff",
             f"+ {ctx.curve_name(phie_curve)} > {cutoff['phie_cutoff']}",
             f"+ {ctx.curve_name(sw_curve)} < {cutoff['sw_cutoff']}")
    for (curve, flag, description), test in zip(flags, tests):
        ctx.add_curve(curve, np.asarray(flag, dtype=float), "", description)
        count = int(np.count_nonzero(flag))
        ctx.decide(f"{curve}: {test} -> {count} samples ({count * step:.1f} {well.depth_unit})")
    if exclude_non_reservoir:
        removed = {
            kind: {
                **item,
                "net_reservoir_removed": round(item["net_reservoir_samples_removed"] * step, 2),
                "net_pay_removed": round(item["net_pay_samples_removed"] * step, 2),
                "depth_unit": well.depth_unit,
            }
            for kind, item in details["excluded"].items()
        }
        ctx.trace.parameters["excluded_from_net_reservoir_and_pay"] = removed
        for line in workflow._net_pay_detail_decisions({}, by_family, removed):
            ctx.decide(line)
        if workflow.add_exclusion_curve(well, details):
            ctx.trace.output_curves.append("NET_EXCLUSION")


# --------------------------------------------------------------------------
# Specialist data, the prebuilt workflow and summaries
# --------------------------------------------------------------------------


_SPECTRAL_ELEMENTS = ("POTASSIUM", "THORIUM", "URANIUM")


def _spectral_gamma_present(well: Well, settings: Mapping[str, Any]) -> Optional[str]:
    from petrophys.interpretation.geochemistry import interpret_spectral_gamma

    result = interpret_spectral_gamma(well, add_curves=False)
    return f"well ({', '.join(result.input_curves)})" if result.sources else None


def _spectral_gamma_requirement() -> _Requirement:
    return _Requirement(
        label="spectral gamma curves",
        what="processed K, Th, U, SGR or CGR",
        hint="the well needs processed potassium, thorium, uranium or spectral GR curves",
        predicate=_spectral_gamma_present,
    )


@_builtin
def spectral_gamma(*, api_coefficients: Optional[Mapping[str, float]] = None,
                   uranium_fraction_threshold: float = 0.5, name: Optional[str] = None) -> Step:
    """Standardize processed spectral gamma ray (K, Th, U) and derive ratios.

    Recognizes vendor mnemonics (``POTA``, ``THOR``, ``URAN``, ``HSGR``,
    ``HCGR``, ...) and writes ``POTASSIUM`` (%), ``THORIUM`` and ``URANIUM``
    (ppm), the ``TH_K_RATIO``, ``U_TH_RATIO`` and ``U_K_RATIO`` ratios, the
    uranium-free ``CGR_CALC`` (API) and uranium diagnostics, as far as the
    inputs allow. ``CGR_CALC`` can feed ``clay_volume(gr_curve="CGR_CALC")``.
    Wraps :func:`petrophys.interpretation.geochemistry.interpret_spectral_gamma`,
    which :func:`petrophys.interpret` does not run.

    Parameters
    ----------
    api_coefficients : dict, optional
        API sensitivities by ``"POTASSIUM"``, ``"THORIUM"`` and
        ``"URANIUM"``; defaults 16.32 API/%, 3.93 API/ppm and 8.09 API/ppm.
    uranium_fraction_threshold : float, default 0.5
        Uranium fraction of total gamma ray above which
        ``SGR_URANIUM_EXCESS_FLAG`` is set.
    name : str, optional
        Step name; defaults to ``"spectral_gamma"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Requires at least one processed spectral gamma curve.

    Examples
    --------
    >>> import petrophys as pp
    >>> from petrophys.workflow import Workflow, steps
    >>> run = Workflow("sgr", [steps.spectral_gamma()]).run(pp.read(pp.sample("25_5-1")))
    >>> run.steps[0].output_curves
    ['THORIUM', 'URANIUM', 'U_TH_RATIO', 'SGR_URANIUM_API']
    """
    check = _Params("steps.spectral_gamma()")
    coefficients = None
    if api_coefficients is not None:
        if not isinstance(api_coefficients, Mapping):
            check.add(f"api_coefficients must be a dict such as {{'POTASSIUM': 16.32}}; "
                      f"got {api_coefficients!r}")
        else:
            coefficients = {}
            for key, value in api_coefficients.items():
                canonical = str(key).strip().upper()
                if canonical not in _SPECTRAL_ELEMENTS:
                    check.add(f"api_coefficients key {key!r} is not valid; use POTASSIUM, "
                              "THORIUM or URANIUM" + _suggestion(canonical, _SPECTRAL_ELEMENTS))
                    continue
                coefficients[canonical] = check.number(f"api_coefficients[{key!r}]", value,
                                                       minimum=0)
    values = {
        "api_coefficients": coefficients,
        "uranium_fraction_threshold": check.number("uranium_fraction_threshold",
                                                   uranium_fraction_threshold, optional=False,
                                                   fraction=True),
    }
    check.raise_if_any()
    return _make(
        "spectral_gamma", name, values,
        runner=_run_spectral_gamma,
        requirements=[_spectral_gamma_requirement()],
        description="Processed K-Th-U spectral gamma ray and ratios",
        planner=_plan_spectral_gamma,
        produces_text="K, Th, U, ratios, CGR_CALC (as available)",
    )


def _plan_spectral_gamma(well: Well, available: Set[str]) -> Iterable[str]:
    from petrophys.interpretation.geochemistry import interpret_spectral_gamma

    result = interpret_spectral_gamma(well, add_curves=False)
    return list(result.measurements) + list(result.derived)


def _run_spectral_gamma(ctx: _StepContext, *, api_coefficients: Optional[Mapping[str, float]],
                        uranium_fraction_threshold: float) -> None:
    from petrophys.interpretation.geochemistry import interpret_spectral_gamma

    result = interpret_spectral_gamma(ctx.well, add_curves=True, api_coefficients=api_coefficients,
                                      uranium_fraction_threshold=uranium_fraction_threshold)
    ctx.trace.method = "processed_k_th_u_spectral_gamma"
    for curve in result.input_curves:
        ctx.use(curve)
    for curve in result.output_curves:
        _tag(ctx.well, curve, ctx.step.name)
        ctx.output(curve)
    ctx.trace.parameters.update({"sources": result.sources, "conversions": result.conversions})
    if result.sources:
        ctx.decide("Resolved " + ", ".join(f"{key.lower()} from {value}"
                                           for key, value in result.sources.items()))
    if "CGR_CALC" in result.derived:
        ctx.decide("CGR_CALC = uranium-free gamma ray from potassium and thorium")
    for warning in result.warnings:
        ctx.warn(warning)


_INTERPRET_PRODUCTS = {"VCL": "v/v", "PHIT": "v/v", "PHIE": "v/v", "SW": "v/v", "PERM": "mD",
                       "NET_ROCK": "", "NET_RES": "", "NET_PAY": ""}
_INTERPRET_STEP_FOR = {"VCL": "vclay", "PHIT": "porosity", "PHIE": "porosity",
                       "SW": "saturation", "PERM": "permeability", "NET_ROCK": "net_pay",
                       "NET_RES": "net_pay", "NET_PAY": "net_pay"}


@_builtin
def interpret(*, name: Optional[str] = None, **settings: Any) -> Step:
    """Run the whole prebuilt workflow of :func:`petrophys.interpret` as one step.

    Use it to extend the automatic interpretation with your own steps before
    or after it: for example a despiking step before, or a custom cutoff or
    rock-typing step after. Settings are
    :class:`~petrophys.workflow.DynamicWorkflowConfig` fields, validated when
    the step is created. The full
    :class:`~petrophys.workflow.DynamicWorkflowResult` is kept in
    ``run.artifacts[name]``; its warnings and failed sub-steps are copied to
    the run's warnings.

    Parameters
    ----------
    name : str, optional
        Step name; defaults to ``"interpret"``.
    **settings
        :class:`~petrophys.workflow.DynamicWorkflowConfig` settings, e.g.
        ``rw=0.07, rw_meas_temp_f=230``. They must be JSON-serializable
        (pass files as paths).

    Returns
    -------
    ~petrophys.workflow.Step
        Requires nothing; declares ``VCL``, ``PHIT``, ``PHIE``, ``SW``,
        ``PERM`` and the net flags, which :func:`petrophys.interpret`
        produces when the well's data allow (a missing one is reported as a
        warning).

    Raises
    ------
    TypeError
        If a setting name is unknown.
    ValueError
        If a setting value is invalid.

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.interpret(rw=0.07, rw_meas_temp_f=230, sw_modle="archie")
    Traceback (most recent call last):
    ...
    TypeError: steps.interpret() got unexpected setting(s) 'sw_modle' (did you mean 'sw_model'?). See petrophys.DynamicWorkflowConfig for the available settings.
    """
    import dataclasses

    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig

    known = [field.name for field in dataclasses.fields(DynamicWorkflowConfig) if not field.name.startswith("_")]
    unknown = [key for key in settings if key not in known]
    if unknown:
        hints = ", ".join(f"{key!r}" + _suggestion(key, known, cutoff=0.5) for key in unknown)
        raise TypeError(f"steps.interpret() got unexpected setting(s) {hints}. See "
                        "petrophys.DynamicWorkflowConfig for the available settings.")
    values = {key: _json_param(value, f"steps.interpret({key})") for key, value in settings.items()}
    try:
        DynamicWorkflowConfig(**values)
    except ValueError as exc:
        raise ValueError(str(exc).replace("Invalid DynamicWorkflowConfig",
                                          "Invalid steps.interpret() settings")) from None
    return _make(
        "interpret", name, values,
        runner=_run_interpret,
        produces=_INTERPRET_PRODUCTS,
        description="The prebuilt petrophys.interpret workflow",
    )


def _run_interpret(ctx: _StepContext, **settings: Any) -> None:
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig, run_dynamic_workflow

    well = ctx.well
    before = {curve.upper() for curve in well.curves}
    result = run_dynamic_workflow(well, config=DynamicWorkflowConfig(**settings))
    ctx.artifact = result
    ctx.trace.method = "petrophys.interpret"
    ctx.trace.parameters["resolved"] = {"matrix_type": result.config.matrix_type,
                                        "lithology_domain": result.lithology_domain}
    for record in result.steps_executed:
        for curve in record.input_curves:
            if curve.upper() in before:
                ctx.use(curve)
        text = f"{record.step}: {record.method or 'done'}"
        if not record.success:
            text += f" (failed: {record.message})"
        ctx.decide(text)
    for step_name, reason in result.specific_steps_skipped.items():
        ctx.decide(f"skipped {step_name}: {reason}")
    for mnemonic in well.curves:
        if mnemonic.upper() not in before:
            _tag(well, mnemonic, ctx.step.name)
            ctx.output(mnemonic.upper())
    for warning in result.warnings:
        ctx.warn(warning)
    for error in result.errors:
        if error not in result.warnings:
            ctx.warn(f"sub-step error: {error}")
    for product in _INTERPRET_PRODUCTS:
        if product not in ctx.trace.output_curves:
            why = result.specific_steps_skipped.get(_INTERPRET_STEP_FOR[product])
            ctx.warn(f"{product} was not produced" + (f": {why}" if why else ""))
    if not result.has_usable_interpretation:
        raise RuntimeError("petrophys.interpret produced no interpretation curves: "
                           + ("; ".join(result.errors) or "no step could run"))


def _formations_present(well: Well, settings: Mapping[str, Any]) -> Optional[str]:
    if settings.get("formations") is not None:
        return "parameter formations"
    records = (well.get_parameter("_geology_context") or {}).get("formations")
    return "attached formation tops" if records else None


def _formations_requirement() -> _Requirement:
    return _Requirement(
        label="formation tops",
        hint="read the well with petrophys.read(path, formations=...), pass "
             "run_many(..., formations=...), or give formations= to the step",
        predicate=_formations_present,
        inputs=("formations",),
    )


@_builtin
def formation_summary(*, rank: Optional[str] = None, formations: Any = None,
                      name: Optional[str] = None) -> Step:
    """Per-formation summary table, stored in ``run.artifacts[name]``.

    Gross, net rock, net reservoir and net pay thickness per formation, with
    PHIE, SW, VCL and PERM averaged over net-pay samples: the table of
    :meth:`petrophys.workflow.WorkflowRunResult.formation_summary`, computed
    as part of the run so it is saved in the run's JSON export. Place it
    after the steps whose curves it summarizes.

    Parameters
    ----------
    rank : str, optional
        Level to summarize when the tops mix levels, e.g. ``"formation"``
        or ``"group"``.
    formations : str or list of dict, optional
        Formation tops file, or records with ``formation_name`` and
        ``top_depth`` (and ``bottom_depth``). Defaults to the formations
        attached to the well.
    name : str, optional
        Step name; defaults to ``"formation_summary"``.

    Returns
    -------
    ~petrophys.workflow.Step
        Requires formation tops; produces no curves.

    Examples
    --------
    >>> from petrophys.workflow import steps
    >>> steps.formation_summary(rank="formation").requires
    ('formation tops',)
    """
    check = _Params("steps.formation_summary()")
    values = {"rank": check.text("rank", rank), "formations": formations}
    if formations is not None and not isinstance(formations, (str, list, tuple)):
        check.add("formations must be a file path or a list of records; in-memory tables "
                  "cannot be saved with the workflow (attach them with petrophys.read instead)")
    check.raise_if_any()
    return _make(
        "formation_summary", name, values,
        runner=_run_formation_summary,
        requirements=[_formations_requirement()],
        description="Per-formation net pay and average properties (a table)",
        produces_text="table in run.artifacts",
    )


def _run_formation_summary(ctx: _StepContext, *, rank: Optional[str], formations: Any) -> None:
    from petrophys.workflow.summary import formation_table

    table = formation_table(ctx.well, formations, rank=rank)
    ctx.artifact = table
    ctx.trace.method = "formation_table"
    ctx.decide(f"{len(table)} formations summarized" + (f" at rank {rank!r}" if rank else ""))
    if "NET_PAY" not in {curve.upper() for curve in ctx.well.curves}:
        ctx.warn("No NET_PAY curve: net thickness and pay averages are empty; add "
                 "steps.net_pay() before this step")
