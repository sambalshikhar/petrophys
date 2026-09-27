"""
Dynamic petrophysics workflow.

This module provides a DynamicWorkflow class that automatically detects
available input curves and runs the appropriate modules in the correct
order. It supports both granular (single-module) and full pipeline
execution modes.

Pipeline execution order:
    Module 1: QC & Pre-Processing
    Module 2: Spectral Gamma and Elemental Spectroscopy
    Module 3: Volume of Clay, Clay Resistivity, Rmf, and HC preparation
    Module 4: Provisional Porosity
    Module 5: Lithology Determination
    Module 6: Porosity Refinement and Core Correction
    Module 7: Formation Water Resistivity
    Module 8: Water Saturation
    Module 9: Permeability
    Module 10: Rock Quality (RQI, FZI)
    Module 11: Net Pay
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path
import re
from petrophys.model.well import Well
from petrophys.model.data_types import DataType
from petrophys.model.units import temperature_to_fahrenheit
from petrophys._internal.analysis_helpers import (
    curve_source_mnemonic,
    get_curve,
    get_curve_data_depthwise,
)
import pandas as pd
from petrophys.io import ascii_reader, las_reader

DEFAULT_RESISTIVITY_REFERENCE_TEMP_F = 75.0
INPUT_INTERPRETATION_SOURCES = {"LAS", "DLIS", "CSV", "ASCII", "DATAFRAME"}
INTERPRETATION_DESCRIPTION_MARKERS = (
    "HILT",
    "LITHOLOGY COMPUTATION",
    "INTERPRET",
    "PROCESSED",
    "VENDOR",
    "SPECTROLITH",
)
APPARENT_POROSITY_METHODS = {
    "density_only": {
        "source": "density-derived",
        "basis": "PHID",
        "reason": "only density porosity (PHID) was available",
    },
    "neutron_only": {
        "source": "neutron-derived",
        "basis": "PHIN",
        "reason": "only neutron porosity (PHIN) was available",
    },
    "sonic_only": {
        "source": "sonic-derived",
        "basis": "PHIS",
        "reason": "only sonic porosity (PHIS) was available",
    },
}


def _apparent_porosity_info(phit_method: Optional[str]) -> Optional[Dict[str, str]]:
    """Return provenance for single-log porosity routes."""
    return APPARENT_POROSITY_METHODS.get(str(phit_method or "").strip().lower())


class WorkflowStep(Enum):
    """Individual workflow steps."""
    NORMALIZATION = "normalization"
    BAD_HOLE = "bad_hole"
    SPECTRAL_GAMMA = "spectral_gamma"
    ELEMENTAL_SPECTROSCOPY = "elemental_spectroscopy"
    TEMPERATURE = "temperature"
    VCLAY = "vclay"
    RCL = "rcl"
    LITHOLOGY = "lithology"
    RMF = "rmf"
    RW = "rw"
    INVASION = "invasion"
    HC_CORRECTION = "hc_correction"
    POROSITY = "porosity"
    POROSITY_REFINEMENT = "porosity_refinement"
    CORE_CORRECTION = "core_correction"
    SATURATION = "saturation"
    PERMEABILITY = "permeability"
    ROCK_QUALITY = "rock_quality"
    BVW = "bvw"
    MHI = "mhi"
    NET_PAY = "net_pay"


def _parameter_float(well: "Well", name: str) -> Optional[float]:
    """Fetch a numeric well parameter if present."""
    value = well.get_parameter(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parameter_text(well: "Well", name: str) -> str:
    """Fetch a text well parameter if present."""
    value = well.get_parameter(name)
    return str(value or "").strip()


def _curve_is_computed(curve: Any) -> bool:
    metadata = getattr(curve, "metadata", None)
    data_type = getattr(metadata, "data_type", "")
    value = getattr(data_type, "value", data_type)
    return str(value or "").strip().lower() == "computed"


def _curve_source_text(curve: Any) -> str:
    metadata = getattr(curve, "metadata", None)
    return str(getattr(metadata, "source", "") or "").strip()


def _curve_description_text(curve: Any) -> str:
    metadata = getattr(curve, "metadata", None)
    return str(getattr(metadata, "description", "") or "").strip()


def _is_prioritized_input_interpretation_curve(curve: Any, curve_type: str) -> bool:
    """Return True for imported interpretation curves that should stay active."""
    if curve is None or _curve_is_computed(curve):
        return False

    source = _curve_source_text(curve).upper()
    description = _curve_description_text(curve).upper()
    mnemonic = str(getattr(curve, "mnemonic", "") or "").upper()
    curve_type = curve_type.upper()

    if curve_type in {"VCL", "PHIE", "PHIT"} and source in INPUT_INTERPRETATION_SOURCES:
        return True
    return any(marker in description or marker in mnemonic for marker in INTERPRETATION_DESCRIPTION_MARKERS)


def _prioritized_curve_label(curve: Any) -> str:
    if curve is None:
        return ""
    source = _curve_source_text(curve) or "input"
    description = _curve_description_text(curve)
    if description:
        return f"{getattr(curve, 'mnemonic', '')} from {source}: {description}"
    return f"{getattr(curve, 'mnemonic', '')} from {source}"


def _constant_zoned_parameter(well: "Well", prefix: str) -> Optional[float]:
    """
    Return a unique zoned LAS parameter value, e.g. RW_Z1/RW_Z2 -> RW.

    If multiple distinct zoned values exist, return None so callers do not
    silently choose the wrong interval constant.
    """
    values = set()
    pattern = re.compile(rf"^{re.escape(prefix.upper())}_Z\d+$")

    for key, value in getattr(well, "_parameters", {}).items():
        if not isinstance(key, str) or "__" in key:
            continue
        if not pattern.match(key.upper()):
            continue
        try:
            values.add(float(value))
        except (TypeError, ValueError):
            continue

    if len(values) == 1:
        return next(iter(values))
    return None


def _parameter_temperature_f(well: "Well", name: str) -> Optional[float]:
    """Fetch a numeric temperature parameter and normalize it to Fahrenheit."""
    value = _parameter_float(well, name)
    unit = _parameter_text(well, f"{name}__UNIT")
    if value is not None:
        return temperature_to_fahrenheit(value, unit)

    value = _constant_zoned_parameter(well, name)
    unit = _parameter_text(well, f"{name}_Z1__UNIT")
    if value is not None:
        return temperature_to_fahrenheit(value, unit)

    return None


def _metadata_bht_f(well: "Well") -> Optional[float]:
    """Return BHT metadata in Fahrenheit."""
    if well.header.bht is not None:
        unit = _parameter_text(well, "BHT__UNIT") or getattr(well.header, "bht_unit", "F")
        return temperature_to_fahrenheit(float(well.header.bht), unit)

    value = _parameter_float(well, "BHT")
    unit = _parameter_text(well, "BHT__UNIT").lower()
    if value is not None:
        return temperature_to_fahrenheit(value, unit)

    value = _constant_zoned_parameter(well, "BHT")
    unit = _parameter_text(well, "BHT_Z1__UNIT").lower()
    if value is not None:
        return temperature_to_fahrenheit(value, unit)

    return None


GROUND_TRUTH_CODE_TO_GROUP = {
    1: "sandstone",
    2: "sandstone",
    3: "sandstone",
    4: "sandstone",
    5: "silt",
    6: "silt",
    7: "shale",
    8: "shale",
    9: "limestone",
    10: "sandstone",
    11: "limestone",
    12: "limestone",
    13: "limestone",
    14: "calcareous_shale",
    15: "dolomite",
    16: "limestone",
    17: "clastic_coarse",
    18: "coal",
    19: "igneous",
    20: "igneous",
    21: "halite",
    22: "anhydrite",
    23: "igneous",
    24: "igneous",
    25: "shale",
    26: "evaporite",
    27: "halite",
    28: "igneous",
    29: "chert",
    30: "sandstone",
    31: "clastic_coarse",
    32: "halite",
    33: "halite",
    34: "halite",
    35: "sandstone",
    36: "clastic_coarse",
}

COMPUTED_LITHO_CODE_TO_GROUP = {
    1: "sandstone",
    2: "limestone",
    3: "dolomite",
    4: "shale",
    5: "anhydrite",
    6: "halite",
    7: "coal",
}

LITHOLOGY_GROUP_TO_MATRIX = {
    "sandstone": "sandstone",
    "silt": "sandstone",
    "clastic_coarse": "sandstone",
    "igneous": "quartz",
    "chert": "quartz",
    "limestone": "limestone",
    "dolomite": "dolomite",
    "shale": "shale",
    "calcareous_shale": "shale",
    "anhydrite": "anhydrite",
    "evaporite": "gypsum",
    "halite": "halite",
    "coal": "coal",
}


def _representative_finite_value(values: Any) -> Optional[float]:
    """Return the modal finite value, falling back to median."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    rounded = np.round(finite, 4)
    unique, counts = np.unique(rounded, return_counts=True)
    if unique.size:
        return float(unique[np.argmax(counts)])
    return float(np.nanmedian(finite))


def _coerce_lithology_group(value: Any) -> Optional[str]:
    """Normalize lithology strings to workflow group names."""
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "-999.25"}:
        return None

    aliases = {
        "sand": "sandstone",
        "ss": "sandstone",
        "lime": "limestone",
        "ls": "limestone",
        "dolo": "dolomite",
        "salt": "halite",
    }
    return aliases.get(text, text)


def _group_array_from_numeric_codes(data: np.ndarray, mapping: Dict[int, str]) -> np.ndarray:
    groups = np.full(data.shape, None, dtype=object)
    finite = np.isfinite(data)
    if not np.any(finite):
        return groups

    rounded = np.rint(data[finite]).astype(int)
    grouped = [mapping.get(code) for code in rounded]
    groups[finite] = grouped
    return groups


def _finite_unique_numeric(values: Any) -> list[float]:
    """Return sorted finite unique numeric values rounded for stable summaries."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []
    return sorted(float(value) for value in np.unique(np.round(finite, 6)))


def _scalar_or_lithology_dependent(values: Any) -> Any:
    """Return a scalar when all values match, otherwise a compact label."""
    unique = _finite_unique_numeric(values)
    if len(unique) == 1:
        return unique[0]
    return "lithology_dependent"


def _is_shale_group(name: Any) -> bool:
    text = str(name or "").strip().lower()
    return "shale" in text


def _format_group_list(groups: List[str]) -> str:
    if not groups:
        return ""
    return ", ".join(sorted({str(group).upper() for group in groups if group}))


def _ordered_unique_values(values: List[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _trace_array_summary(value: np.ndarray) -> Dict[str, Any]:
    """Summarize array values without writing full curve payloads to logs."""
    data = np.asarray(value)
    payload: Dict[str, Any] = {
        "type": "ndarray",
        "shape": list(data.shape),
        "size": int(data.size),
    }
    if data.ndim == 0:
        payload["value"] = _json_safe_trace_value(data.item())
        return payload

    sample_count = min(5, int(data.size))
    payload["sample"] = [
        _json_safe_trace_value(item) for item in list(data.flat[:sample_count])
    ]

    if np.issubdtype(data.dtype, np.number):
        numeric = data.astype(float, copy=False)
        finite = numeric[np.isfinite(numeric)]
        payload.update({
            "finite_count": int(finite.size),
            "nan_count": int(np.isnan(numeric).sum()),
            "inf_count": int(np.isinf(numeric).sum()),
        })
        if finite.size:
            payload.update({
                "min": float(np.nanmin(finite)),
                "max": float(np.nanmax(finite)),
                "mean": float(np.nanmean(finite)),
                "p05": float(np.nanpercentile(finite, 5)),
                "p50": float(np.nanpercentile(finite, 50)),
                "p95": float(np.nanpercentile(finite, 95)),
            })

    return payload


def _json_safe_trace_value(value: Any) -> Any:
    """Convert trace metadata to JSON-safe, compact values."""
    if isinstance(value, np.ndarray):
        return _trace_array_summary(value)
    if isinstance(value, np.generic):
        return _json_safe_trace_value(value.item())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_trace_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_trace_value(item) for item in value]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _curve_summaries_for_trace(
    well: "Well",
    curve_names: Optional[List[str]],
) -> Dict[str, Any]:
    """Build compact per-curve summaries for workflow trace exports."""
    if not curve_names:
        return {}

    summaries: Dict[str, Any] = {}
    for name in curve_names:
        curve = well.get(name) if hasattr(well, "get") else None
        if curve is None:
            continue
        summaries[str(name).upper()] = {
            "unit": getattr(curve, "unit", ""),
            "description": getattr(getattr(curve, "metadata", None), "description", ""),
            "data": _trace_array_summary(np.asarray(curve.data)),
        }
    return summaries


@dataclass
class DynamicWorkflowConfig:
    """
    Settings for :func:`petrophys.interpret` and the dynamic workflow.

    Every setting is optional. The workflow inspects the available curves
    and runs each step it can; settings supply values that logs cannot
    (formation water resistivity, temperatures, mud properties) or override
    automatic choices. Values are validated when the config is created: an
    invalid choice raises ``ValueError`` listing the accepted values.

    Temperatures are in degrees Fahrenheit, resistivities in ohm.m, densities
    in g/cm3 and slownesses in us/ft, whatever the depth unit of the well.
    Fields left as ``None`` are read from the LAS header where available or
    derived from the data, and the trace records where each value came from.

    Examples
    --------
    >>> from petrophys import DynamicWorkflowConfig
    >>> config = DynamicWorkflowConfig(rw=0.05, bht_f=220, sw_model="indonesia")
    >>> config.matrix_type
    'auto'
    """

    # -- Input handling ------------------------------------------------------
    #: Standardize curve aliases and units before interpretation.
    standardize_input: bool = True
    #: Rename vendor mnemonics to canonical names instead of adding aliases.
    rename_input_curves: bool = False
    #: Allow a 0.05 ohm.m default Rw when no Rw source is found. When False,
    #: saturation is not computed without an Rw source.
    allow_default_rw: bool = False
    #: ``"auto"``, ``"vendor_benchmark"`` or ``"generic_inference"``. Auto
    #: uses the vendor-benchmark route when a validated vendor lithology
    #: curve is present.
    interpretation_profile: str = "auto"

    # -- Temperature ---------------------------------------------------------
    #: Surface (mean annual) temperature, degF.
    surface_temp_f: float = 68.0
    #: Bottom-hole temperature, degF. Defaults to the LAS header BHT.
    bht_f: Optional[float] = None
    #: Total depth for the BHT, in feet. Defaults to the deepest sample.
    td_ft: Optional[float] = None

    # -- Mud properties ------------------------------------------------------
    #: ``"WBM"`` (water-based) or ``"OBM"`` (oil-based) mud.
    mud_type: str = "WBM"
    #: The mud contains clay or KCl, which disables SP-based Rw estimation.
    mud_has_clay_or_kcl: bool = False
    #: Mud-filtrate resistivity Rmf at ``rmf_meas_temp_f``, ohm.m.
    rmf_at_meas_temp: Optional[float] = None
    #: Temperature at which ``rmf_at_meas_temp`` was measured, degF.
    rmf_meas_temp_f: float = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
    #: Mud-filtrate salinity, ppm NaCl equivalent (alternative to Rmf).
    rmf_salinity_ppm: Optional[float] = None
    #: Mud resistivity Rm at ``rm_meas_temp_f``, ohm.m (Rmf is derived from it).
    rm: Optional[float] = None
    #: Temperature at which ``rm`` was measured, degF.
    rm_meas_temp_f: float = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
    #: ``"fresh"`` or ``"salt"`` mud, which sets the Rm-to-Rmf factor.
    mud_water: str = "fresh"

    # -- Formation water -----------------------------------------------------
    #: Formation water resistivity Rw, ohm.m, at ``rw_meas_temp_f``.
    rw: Optional[float] = None
    #: Temperature of ``rw``, degF. When omitted, 75 degF is assumed and
    #: recorded in the trace.
    rw_meas_temp_f: Optional[float] = None
    #: Formation water salinity, ppm NaCl equivalent (used when Rw is absent).
    salinity_ppm: Optional[float] = None

    # -- Matrix properties ---------------------------------------------------
    #: ``"auto"`` (resolved from lithology evidence) or a mineral name such as
    #: ``"sandstone"``, ``"limestone"`` or ``"dolomite"``; see
    #: :meth:`petrophys.model.constants.MatrixProperties.names`.
    matrix_type: str = "auto"
    #: Matrix density, g/cm3. Defaults to the value for ``matrix_type``.
    rho_ma: Optional[float] = None
    #: Matrix neutron porosity, v/v. Defaults to the value for ``matrix_type``.
    nphi_ma: Optional[float] = None
    #: Pore-fluid density, g/cm3.
    rho_fl: float = 1.0
    #: Clay density, g/cm3.
    rho_cl: float = 2.45
    #: Matrix sonic slowness, us/ft. Defaults to the value for ``matrix_type``.
    dt_ma: Optional[float] = None
    #: Pore-fluid sonic slowness, us/ft.
    dt_fl: float = 189.0

    # -- Lithology -----------------------------------------------------------
    #: Cutoff family ``"auto"``, ``"clastic"``, ``"carbonate"`` or
    #: ``"evaporite"``, or a rock name with tabulated Archie coefficients
    #: (e.g. ``"sandstone"``, ``"shaly sand"``).
    lithology: str = "auto"
    #: Lithology intervals CSV (``top,bottom,lithology``).
    lithology_csv: Optional[str] = None
    #: Lithology intervals as a table or CSV/Excel path.
    lithology_intervals: Optional[Any] = None
    #: Lithology source: ``"auto"``, ``"external"`` (intervals),
    #: ``"vendor"`` (lithology curve), ``"computed"`` or ``"config"``.
    lithology_source: str = "auto"
    #: Raise on the first failed step instead of recording it and continuing.
    raise_on_error: bool = False

    # -- Archie parameters (None: lithology-dependent defaults) ---------------
    #: Tortuosity factor a.
    a: Optional[float] = None
    #: Cementation exponent m.
    m: Optional[float] = None
    #: Saturation exponent n.
    n: Optional[float] = None

    # -- Waxman-Smits parameters ---------------------------------------------
    #: Cation exchange capacity, meq/100 g.
    cec: Optional[float] = None
    #: Equivalent counter-ion conductance B.
    b_coeff: Optional[float] = None
    #: Waxman-Smits cementation exponent m*.
    m_star: float = 2.0

    # -- Dual-water parameters -----------------------------------------------
    #: Bound-water resistivity, ohm.m.
    rwb: Optional[float] = None
    #: Clay (shale) porosity, v/v.
    phi_cl: float = 0.15

    # -- Hydrocarbon correction ----------------------------------------------
    #: Apply the neutron-density hydrocarbon correction (opt-in).
    apply_hc_correction: bool = False
    #: Hydrocarbon type for the correction: ``"oil"`` or ``"gas"``.
    hc_type: str = "oil"

    # -- Net-pay cutoffs (None: lithology-family standards) --------------------
    #: Maximum clay volume for net rock, v/v.
    vcl_cutoff: Optional[float] = None
    #: Maximum water saturation for net pay, v/v.
    sw_cutoff: Optional[float] = None
    #: Minimum effective porosity for net reservoir, v/v.
    phie_cutoff: Optional[float] = None

    # -- Gamma-ray clay volume (None: picked from the log) -------------------
    #: Clean-sand gamma-ray value, API.
    gr_min: Optional[float] = None
    #: Shale gamma-ray value, API.
    gr_max: Optional[float] = None
    #: Transform: ``"linear"``, ``"larionov_tertiary"``, ``"larionov_older"``,
    #: ``"steiber"`` or ``"clavier"``.
    gr_method: str = "linear"

    # -- SP clay volume (None: picked from the log) --------------------------
    #: Clean (static) SP baseline, mV.
    sp_min: Optional[float] = None
    #: Shale SP baseline, mV.
    sp_max: Optional[float] = None

    # -- Permeability --------------------------------------------------------
    #: ``"auto"``, ``"timur"``, ``"coates"``, ``"wyllie_rose"`` or ``"lith_vcl"``.
    perm_method: str = "auto"
    #: Empirical method tried when core calibration is unavailable.
    perm_fallback_method: str = "coates"
    #: Irreducible water saturation, v/v. Defaults to curves or NMR data.
    swirr: Optional[float] = None
    #: Offset wells (``Well`` objects) whose core data calibrate permeability.
    peer_wells: Optional[List[Any]] = None
    #: Formation intervals (``FormationTops``) for per-formation core
    #: calibration. Defaults to formations attached by ``petrophys.read``.
    formation_tops: Optional[Any] = None
    #: Calibrate permeability with core data from ``peer_wells``.
    use_peer_core: bool = False

    # -- Water saturation ----------------------------------------------------
    #: ``"auto"`` (routed per sample by clay content) or one of
    #: ``"archie"``, ``"simandoux"``, ``"indonesia"``, ``"dual_water"``,
    #: ``"waxman_smits"``, ``"north_sea"``, ``"nigeria_schlumberger"``.
    sw_model: str = "auto"
    #: Shale resistivity, ohm.m. Defaults to the computed clay resistivity.
    rtsh: Optional[float] = None
    #: Shale exponent of the North Sea model.
    vsh_exp: float = 1.6
    #: Shale exponent of the Nigeria-Schlumberger model.
    shale_exp: float = 1.6

    # -- Step selection ------------------------------------------------------
    #: Workflow steps to skip, e.g. ``["permeability"]``; see
    #: :class:`WorkflowStep` for the names.
    skip_steps: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_workflow_config(self)

def _attached_formation_tops(records: List[Dict[str, Any]], terminal_depth: float) -> Any:
    """Build validated, non-overlapping FormationTops from attached records.

    Raises
    ------
    ValueError
        If the records cannot form bounded, non-overlapping intervals.
    """
    import pandas as pd
    from petrophys.model.formation_tops import FormationTops

    table = pd.DataFrame(records)
    if "bottom_depth" in table and table["bottom_depth"].isna().all():
        table = table.drop(columns="bottom_depth")
    tops = FormationTops.from_dataframe(
        table, name_col="formation_name", terminal_depth=terminal_depth
    )
    tops.validate()
    return tops


# --- DynamicWorkflowConfig validation ---------------------------------------

_INTERPRETATION_PROFILES = ("auto", "vendor_benchmark", "generic_inference")
_MUD_TYPES = ("WBM", "OBM")
_MUD_WATERS = ("fresh", "salt")
_LITHOLOGY_FAMILIES = ("auto", "clastic", "carbonate", "evaporite")
_LITHOLOGY_SOURCES = ("auto", "external", "vendor", "computed", "config")
_HC_TYPES = ("oil", "gas")
_BOOL_FIELDS = (
    "standardize_input", "rename_input_curves", "allow_default_rw",
    "mud_has_clay_or_kcl", "raise_on_error", "apply_hc_correction", "use_peer_core",
)
_POSITIVE_FIELDS = (
    "rw", "rmf_at_meas_temp", "rm", "rmf_salinity_ppm", "salinity_ppm", "td_ft",
    "rho_ma", "rho_fl", "rho_cl", "dt_ma", "dt_fl", "a", "m", "n", "m_star",
    "rwb", "rtsh", "vsh_exp", "shale_exp", "b_coeff",
)
_FRACTION_FIELDS = ("vcl_cutoff", "sw_cutoff", "phie_cutoff", "swirr", "phi_cl")
_FINITE_FIELDS = (
    "surface_temp_f", "bht_f", "rmf_meas_temp_f", "rm_meas_temp_f",
    "rw_meas_temp_f", "nphi_ma", "gr_min", "gr_max", "sp_min", "sp_max", "cec",
)


def _choice_error(name: str, value: Any, choices: Sequence[str]) -> str:
    import difflib

    message = f"{name}={value!r} is not valid; use one of: {', '.join(choices)}"
    close = difflib.get_close_matches(
        str(value).strip().lower(), [c.lower() for c in choices], n=1, cutoff=0.75
    )
    if close:
        match = next(c for c in choices if c.lower() == close[0])
        message += f" (did you mean {match!r}?)"
    return message


def _normalized_choice(
    config: "DynamicWorkflowConfig",
    name: str,
    choices: Sequence[str],
    errors: List[str],
    *,
    aliases: Sequence[str] = (),
) -> None:
    value = getattr(config, name)
    if not isinstance(value, str):
        errors.append(f"{name} must be a string; got {type(value).__name__}")
        return
    key = value.strip().lower()
    canonical = {choice.lower(): choice for choice in choices}
    if key in canonical:
        setattr(config, name, canonical[key])
    elif key in {alias.lower() for alias in aliases}:
        setattr(config, name, key)
    else:
        errors.append(_choice_error(name, value, choices))


def _validate_lithology(config: "DynamicWorkflowConfig", errors: List[str]) -> None:
    """Accept a cutoff family or any lithology with tabulated Archie coefficients."""
    from petrophys.model.constants import ArchieParameterDefaults

    value = config.lithology
    if not isinstance(value, str):
        errors.append(f"lithology must be a string; got {type(value).__name__}")
        return
    key = value.strip().lower()
    if key in _LITHOLOGY_FAMILIES:
        config.lithology = key
        return
    if ArchieParameterDefaults.normalize(value) in ArchieParameterDefaults.names():
        return
    message = _choice_error("lithology", value, _LITHOLOGY_FAMILIES + tuple(ArchieParameterDefaults.names()))
    errors.append(message)


def _validate_workflow_config(config: "DynamicWorkflowConfig") -> None:
    """Validate and normalize a :class:`DynamicWorkflowConfig` in place.

    Raises
    ------
    ValueError
        Listing every invalid field, with the accepted values.
    """
    from petrophys.interpretation.clay_volume.gr_method import GR_VCL_METHODS
    from petrophys.interpretation.permeability.perm_module import (
        PERMEABILITY_METHOD_ALIASES,
        PERMEABILITY_METHODS,
    )
    from petrophys.interpretation.saturation.sw_module import SwModel
    from petrophys.model.constants import MatrixProperties

    errors: List[str] = []
    _normalized_choice(config, "interpretation_profile", _INTERPRETATION_PROFILES, errors)
    _normalized_choice(config, "mud_type", _MUD_TYPES, errors)
    _normalized_choice(config, "mud_water", _MUD_WATERS, errors)
    _normalized_choice(config, "matrix_type", ("auto", *MatrixProperties.names()), errors)
    _validate_lithology(config, errors)
    _normalized_choice(config, "lithology_source", _LITHOLOGY_SOURCES, errors)
    _normalized_choice(config, "hc_type", _HC_TYPES, errors)
    _normalized_choice(config, "gr_method", GR_VCL_METHODS, errors)
    _normalized_choice(
        config, "perm_method", PERMEABILITY_METHODS, errors,
        aliases=tuple(PERMEABILITY_METHOD_ALIASES),
    )
    _normalized_choice(
        config, "perm_fallback_method",
        tuple(method for method in PERMEABILITY_METHODS if method != "auto"), errors,
        aliases=tuple(PERMEABILITY_METHOD_ALIASES),
    )
    _normalized_choice(config, "sw_model", ("auto", *(model.value for model in SwModel)), errors)

    step_names = tuple(step.value for step in WorkflowStep)
    if isinstance(config.skip_steps, str) or not isinstance(config.skip_steps, (list, tuple, set)):
        errors.append(
            f"skip_steps must be a list of step names; got {config.skip_steps!r}"
        )
    else:
        steps: List[str] = []
        for step in config.skip_steps:
            name = step.value if isinstance(step, WorkflowStep) else str(step).strip().lower()
            if name in step_names:
                steps.append(name)
            else:
                errors.append(_choice_error("skip_steps entry", step, step_names))
        config.skip_steps = steps

    for name in _BOOL_FIELDS:
        value = getattr(config, name)
        if not isinstance(value, (bool, np.bool_)):
            errors.append(f"{name} must be True or False; got {value!r}")

    def _number(name: str) -> Optional[float]:
        value = getattr(config, name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            errors.append(f"{name} must be a number; got {value!r}")
            return None
        if not np.isfinite(value):
            errors.append(f"{name} must be finite; got {value!r}")
            return None
        return float(value)

    for name in _FINITE_FIELDS:
        _number(name)
    for name in _POSITIVE_FIELDS:
        value = _number(name)
        if value is not None and value <= 0:
            errors.append(f"{name} must be greater than 0; got {value!r}")
    for name in _FRACTION_FIELDS:
        value = _number(name)
        if value is not None and not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be a fraction between 0 and 1; got {value!r}")
    if config.gr_min is not None and config.gr_max is not None and config.gr_min >= config.gr_max:
        errors.append(f"gr_min ({config.gr_min}) must be less than gr_max ({config.gr_max})")
    if config.lithology_csv is not None and config.lithology_intervals is not None:
        errors.append("Supply lithology_csv or lithology_intervals, not both.")

    if errors:
        raise ValueError(
            "Invalid DynamicWorkflowConfig:\n  - " + "\n  - ".join(errors)
        )


@dataclass
class StepResult:
    """
    Result of a single workflow step with full trace metadata.

    Attributes
    ----------
    step : str
        Step name (e.g., "vclay", "porosity").
    success : bool
        Whether the step completed without error.
    message : str
        Error message if failed, or summary if successful.
    method : str
        Algorithm/equation used (e.g., "larionov_tertiary", "simandoux").
    input_curves : list
        Log curves consumed by this step.
    output_curves : list
        Curves added/modified on the well.
    parameters : dict
        All numeric parameters, thresholds, and constants used.
    decisions : list
        Human-readable trace of decisions/logic branches taken.
    """
    step: str
    success: bool
    message: str = ""
    method: str = ""
    input_curves: List[str] = field(default_factory=list)
    output_curves: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    decisions: List[str] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation of this workflow step trace."""
        return {
            "step": self.step,
            "success": bool(self.success),
            "method": self.method,
            "input_curves": list(self.input_curves),
            "output_curves": list(self.output_curves),
            "parameters": _json_safe_trace_value(self.parameters),
            "decisions": list(self.decisions),
            "message": self.message,
            "data": _json_safe_trace_value(self.data or {}),
        }


@dataclass
class DynamicWorkflowResult:
    """Results from dynamic workflow execution."""
    well: "Well"
    config: DynamicWorkflowConfig
    steps_executed: List[StepResult] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    specific_steps_skipped: dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    interpretation_profile: str = "generic_inference"

    # Computed values
    rw_computed: Optional[Union[float, np.ndarray]] = None
    rmf_computed: Optional[Union[float, np.ndarray]] = None
    rcl_computed: Optional[float] = None
    temperature_gradient: Optional[float] = None

    @property
    def has_usable_interpretation(self) -> bool:
        """Return whether a successful interpretive step produced a curve."""
        non_interpretive_steps = {
            WorkflowStep.NORMALIZATION.value,
            WorkflowStep.BAD_HOLE.value,
            WorkflowStep.TEMPERATURE.value,
            "input_context",
        }
        return any(
            step.success and bool(step.output_curves)
            for step in self.steps_executed
            if step.step not in non_interpretive_steps
        )

    @property
    def status(self) -> str:
        """Return a compact execution status for API and trace consumers."""
        if not self.success:
            return "failed"
        if self.has_usable_interpretation:
            return "interpreted"
        return "no_interpretation"

    @property
    def success(self) -> bool:
        # Honest success: no hard errors AND no executed step reported failure.
        # Previously this only checked `errors`, which `run()` never populates,
        # so a workflow in which every step threw still reported success=True.
        step_failed = any(
            getattr(step, "success", True) is False
            for step in self.steps_executed
        )
        return len(self.errors) == 0 and not step_failed

    def raise_for_errors(self) -> "DynamicWorkflowResult":
        """Raise when execution failed; return this result for chaining."""
        if not self.success:
            details = self.errors or [
                step.message or step.step
                for step in self.steps_executed if not step.success
            ]
            raise RuntimeError("Workflow failed: " + "; ".join(details))
        return self

    def plot(self, **options):
        """Plot the interpreted well using the standard log viewer."""
        from petrophys.visualization import create_log_viewer
        return create_log_viewer(self.well, **options)

    def print_trace(self) -> None:
        """Print a comprehensive human-readable trace of all steps."""
        print(f"\n{'=' * 90}")
        print(f"WORKFLOW TRACE: {self.well.name or 'Unknown Well'}")
        print(f"{'=' * 90}")
        print(f"Overall: {'SUCCESS' if self.success else 'FAILED'} | "
              f"Executed: {len(self.steps_executed)} | Skipped: {len(self.steps_skipped)}")
        print(f"Interpretation profile: {self.interpretation_profile}")
        if self.rw_computed is not None:
            def _summary(value):
                arr = np.asarray(value, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return "NaN"
                if finite.size == 1:
                    return f"{finite[0]:.4f}"
                return f"{np.nanmin(finite):.4f}-{np.nanmax(finite):.4f}"
            print(f"Rw = {_summary(self.rw_computed)} ohm.m | Rmf = {_summary(self.rmf_computed) if self.rmf_computed is not None else 'None'} | "
                  f"Rcl = {self.rcl_computed}")
        print()

        for sr in self.steps_executed:
            status = "OK" if sr.success else "FAIL"
            print(f"[{status:4s}] {sr.step}")
            if sr.method:
                print(f"       Method: {sr.method}")
            if sr.input_curves:
                print(f"       Inputs: {', '.join(sr.input_curves)}")
            if sr.output_curves:
                print(f"       Outputs: {', '.join(sr.output_curves)}")
            if sr.parameters:
                for k, v in sr.parameters.items():
                    if isinstance(v, float):
                        print(f"       {k}: {v:.6g}")
                    else:
                        print(f"       {k}: {v}")
            for d in sr.decisions:
                print(f"       -> {d}")
            if sr.message and not sr.success:
                print(f"       ERROR: {sr.message}")
            print()

        if self.steps_skipped:
            print("Skipped:")
            for name in self.steps_skipped:
                reason = self.specific_steps_skipped.get(name)
                print(f"  - {name}: {reason}" if reason else f"  - {name}")
        if self.warnings:
            print(f"\nWarnings:")
            for w in self.warnings:
                print(f"  - {w}")
        print(f"{'=' * 90}")

    def to_dataframe(self):
        """Return the step trace as a DataFrame, one row per executed step.

        For the interpreted log curves use ``result.well.to_dataframe()``.
        """
        import pandas as pd
        rows = []
        for sr in self.steps_executed:
            rows.append({
                "Step": sr.step,
                "Status": "OK" if sr.success else "FAIL",
                "Method": sr.method,
                "Input Curves": ", ".join(sr.input_curves),
                "Output Curves": ", ".join(sr.output_curves),
                "Parameters": str(sr.parameters) if sr.parameters else "",
                "Decisions": " | ".join(sr.decisions),
                "Message": sr.message,
            })
        return pd.DataFrame(rows)

    def to_trace_dict(
        self,
        include_curves: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Export the full workflow route as a JSON-safe dictionary.

        The trace is recorded at the workflow module boundary, so each step
        shows why that module ran, which method was selected, which curves were
        consumed, which curves were written, and the parameters used.
        """
        try:
            well_name = self.well.name
        except Exception:
            header = getattr(self.well, "header", None)
            well_name = getattr(header, "well_name", "") if header is not None else ""
            if not well_name and isinstance(header, dict):
                well_name = header.get("WELL") or header.get("well_name") or ""

        well_summary = {
            "name": well_name or "Unknown Well",
            "depth_unit": getattr(self.well, "depth_unit", ""),
            "top": _json_safe_trace_value(getattr(self.well, "top", None)),
            "bottom": _json_safe_trace_value(getattr(self.well, "bottom", None)),
            "step": _json_safe_trace_value(getattr(self.well, "step", None)),
            "n_samples": int(getattr(self.well, "n_samples", 0) or 0),
            "curve_count": int(getattr(self.well, "n_curves", 0) or 0),
            "available_curves": sorted(str(curve) for curve in getattr(self.well, "curves", [])),
        }

        from petrophys.version import __version__

        return {
            "schema": "dynamic_workflow_trace",
            "schema_version": 1,
            "petrophys_version": __version__,
            "well": well_summary,
            "success": bool(self.success),
            "status": self.status,
            "has_usable_interpretation": self.has_usable_interpretation,
            "interpretation_profile": self.interpretation_profile,
            "executed_steps": [step.step for step in self.steps_executed],
            "skipped_steps": list(self.steps_skipped),
            "specific_steps_skipped": _json_safe_trace_value(self.specific_steps_skipped),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "computed": {
                "rw": _json_safe_trace_value(self.rw_computed),
                "rmf": _json_safe_trace_value(self.rmf_computed),
                "rcl": _json_safe_trace_value(self.rcl_computed),
                "temperature_gradient": _json_safe_trace_value(self.temperature_gradient),
            },
            "config": _json_safe_trace_value(vars(self.config)),
            "steps": [step.to_dict() for step in self.steps_executed],
            "curve_summaries": _curve_summaries_for_trace(self.well, include_curves),
        }

    def to_json(
        self,
        path: Optional[Union[str, Path]] = None,
        include_curves: Optional[List[str]] = None,
        indent: int = 2,
    ) -> Union[str, Path]:
        """
        Export the workflow route as JSON.

        If ``path`` is provided, the JSON file is written and the output path is
        returned. Otherwise the JSON string is returned.
        """
        text = json.dumps(
            self.to_trace_dict(include_curves=include_curves),
            indent=indent,
            sort_keys=False,
            allow_nan=False,
        )
        if path is None:
            return text

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        return output_path

    def summary(
        self, top: Optional[float] = None, bottom: Optional[float] = None
    ) -> Dict[str, Any]:
        """Summarize the interpretation as compact, JSON-serializable data.

        Designed for reports and LLM agents: everything needed to judge the
        result without walking the full trace.

        Parameters
        ----------
        top, bottom : float, optional
            Depth window (well depth unit) for ``key_results`` and
            ``net_thickness``; defaults to the whole well.

        Returns
        -------
        dict
            ``status``, ``resolved`` matrix and lithology, ``settings_used``
            (non-default settings), ``key_results`` (statistics of VCL, PHIT,
            PHIE, SW, PERM), ``net_thickness``, ``executed_steps``,
            ``skipped_steps`` with reasons, ``warnings`` and ``errors``.
            Use :meth:`to_trace_dict` for every step's full record.

        Examples
        --------
        >>> import petrophys as pp
        >>> result = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07)
        >>> result.summary(top=2841, bottom=2958)["status"]
        'interpreted'
        """
        from petrophys.workflow.summary import summarize_result

        return summarize_result(self, top, bottom)

    def provenance_parameters(self) -> List[Dict[str, Any]]:
        """Summarize the settings this interpretation used, as LAS parameters.

        Returns
        -------
        list of dict
            ``PP_*`` entries (``mnemonic``, ``value``, ``unit``, ``descr``):
            package version, status, profile, resolved matrix and lithology,
            Rw input and its temperature-corrected range, Sw model and Archie
            exponents.
        """
        from petrophys.version import __version__

        steps = {step.step: step for step in self.steps_executed if step.success}
        items: List[Dict[str, Any]] = [
            {"mnemonic": "PP_VERSION", "value": __version__, "descr": "petrophys version"},
            {"mnemonic": "PP_STATUS", "value": self.status, "descr": "Interpretation status"},
            {"mnemonic": "PP_PROFILE", "value": self.interpretation_profile,
             "descr": "Interpretation profile"},
            {"mnemonic": "PP_MATRIX", "value": self.config.matrix_type, "descr": "Matrix used"},
            {"mnemonic": "PP_LITH", "value": self.config.lithology, "descr": "Lithology family used"},
        ]
        rw_step = steps.get(WorkflowStep.RW.value)
        if rw_step is not None:
            parameters = rw_step.parameters
            if parameters.get("rw") is not None:
                reference = parameters.get("reference_temp_f")
                descr = f"Rw input ({rw_step.method})"
                if reference is not None:
                    descr = f"Rw input at {reference:g} degF ({rw_step.method})"
                items.append({"mnemonic": "PP_RW", "value": parameters["rw"], "unit": "ohm.m", "descr": descr})
            for key, mnemonic, descr in (
                ("rw_min", "PP_RWTMIN", "Minimum Rw at formation temperature"),
                ("rw_max", "PP_RWTMAX", "Maximum Rw at formation temperature"),
            ):
                if parameters.get(key) is not None:
                    items.append({"mnemonic": mnemonic, "value": parameters[key], "unit": "ohm.m", "descr": descr})
        sw_step = steps.get(WorkflowStep.SATURATION.value)
        if sw_step is not None:
            parameters = sw_step.parameters
            items.append({"mnemonic": "PP_SWMODEL", "value": sw_step.method, "descr": "Water saturation model(s)"})
            for key in ("a", "m", "n"):
                if parameters.get(key) is not None:
                    items.append({"mnemonic": f"PP_{key.upper()}", "value": parameters[key],
                                  "descr": f"Archie {key}"})
        return items

    def to_las(self, path: Union[str, Path], **options: Any) -> Path:
        """Write the interpreted well to LAS with the settings used.

        The ``~Params`` section keeps the source file's header values and adds
        :meth:`provenance_parameters` (``PP_*``); ``~Other`` lists executed and
        skipped steps. Use :meth:`to_json` for the complete trace.

        Parameters
        ----------
        path : str or Path
            Output LAS file.
        **options
            Passed to :func:`petrophys.io.export_las` (e.g. ``curves=[...]``).

        Returns
        -------
        Path
            The written file.
        """
        from petrophys.io.export import export_las
        from petrophys.version import __version__

        lines = [f"Interpreted with petrophys {__version__}; status {self.status}."]
        lines.append("Executed steps: " + ", ".join(step.step for step in self.steps_executed))
        for name in self.steps_skipped:
            reason = self.specific_steps_skipped.get(name)
            lines.append(f"Skipped {name}" + (f": {reason}" if reason else ""))
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        export_las(
            self.well,
            output,
            extra_parameters=self.provenance_parameters(),
            other="\n".join(lines),
            **options,
        )
        return output

    def to_markdown(
        self,
        path: Optional[Union[str, Path]] = None,
        include_curves: Optional[List[str]] = None,
    ) -> Union[str, Path]:
        """
        Export the workflow route as a human-readable Markdown report.

        This mirrors ``to_json`` but is easier to scan when checking per-module
        routing decisions for a single well.
        """
        trace = self.to_trace_dict(include_curves=include_curves)
        well = trace["well"]
        lines = [
            "# Dynamic Workflow Trace",
            "",
            f"- Well: `{well['name']}`",
            f"- Depth: `{well['top']} - {well['bottom']} {well['depth_unit']}`",
            f"- Samples: `{well['n_samples']}`",
            f"- Overall success: `{trace['success']}`",
            f"- Executed steps: `{', '.join(trace['executed_steps']) or 'none'}`",
            f"- Skipped steps: `{', '.join(trace['skipped_steps']) or 'none'}`",
        ]

        if trace["warnings"]:
            lines.append(f"- Warnings: `{' ; '.join(trace['warnings'])}`")
        if trace["errors"]:
            lines.append(f"- Errors: `{' ; '.join(trace['errors'])}`")

        computed = trace["computed"]
        lines.extend([
            "",
            "## Computed Values",
            f"- Rw: `{json.dumps(computed['rw'])}`",
            f"- Rmf: `{json.dumps(computed['rmf'])}`",
            f"- Rcl: `{json.dumps(computed['rcl'])}`",
            f"- Temperature gradient: `{json.dumps(computed['temperature_gradient'])}`",
        ])

        for step in trace["steps"]:
            lines.extend([
                "",
                f"## {step['step']}",
                f"- Success: `{step['success']}`",
                f"- Method: `{step['method'] or 'none'}`",
                f"- Inputs: `{', '.join(step['input_curves']) or 'none'}`",
                f"- Outputs: `{', '.join(step['output_curves']) or 'none'}`",
                f"- Parameters: `{json.dumps(step['parameters'], sort_keys=True)}`",
            ])
            if step["decisions"]:
                lines.append("- Decisions:")
                lines.extend(f"  - {decision}" for decision in step["decisions"])
            if step["message"]:
                lines.append(f"- Message: `{step['message']}`")
            if step["data"]:
                lines.append(f"- Data: `{json.dumps(step['data'], sort_keys=True)}`")

        if trace["curve_summaries"]:
            lines.extend(["", "## Curve Summaries"])
            for name, summary in trace["curve_summaries"].items():
                lines.append(f"- {name}: `{json.dumps(summary, sort_keys=True)}`")

        text = "\n".join(lines) + "\n"
        if path is None:
            return text

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return output_path


class DynamicWorkflow:
    """
    Dynamic petrophysics workflow that auto-detects available inputs.

    This workflow inspects the well's available curves and the provided
    configuration to determine which processing steps can be executed.
    It runs modules in the correct pipeline order and passes results
    between dependent steps.

    Examples
    --------
    >>> config = DynamicWorkflowConfig(
    ...     bht_f=220, td_ft=8500, rw=0.05,
    ...     lithology='clastic', salinity_ppm=25000
    ... )
    >>> workflow = DynamicWorkflow(config)
    >>> result = workflow.run(well)
    >>> print(result.steps_executed)
    """

    def __init__(self, config: Optional[DynamicWorkflowConfig] = None):
        self.config = config or DynamicWorkflowConfig()
        self._result: Optional[DynamicWorkflowResult] = None
        self._lithology_log: Optional[np.ndarray] = None
        self._lithology_source: str = "config"
        self._computed_lithology_written = False
        self._last_phit_nd_average_mask: Optional[np.ndarray] = None
        self._last_phi_cl_info: Optional[Dict[str, Any]] = None
        self._auto_matrix = False
        self._auto_lith = False
        self._lithology_pending = False
        self._interpretation_profile = "generic_inference"
        self._vendor_lithology_resolution = None
        self._selected_lithology_source = "auto"
        self._external_lithology = None

    def _prepare_input_context(self, well: "Well") -> None:
        from petrophys.io import read_lithology_intervals
        from petrophys.interpretation.lithology.matrix_inference import (
            resolve_from_lithology_log,
        )

        policy = str(self.config.lithology_source).strip().lower()
        if policy not in {"auto", "external", "vendor", "computed", "config"}:
            raise ValueError("lithology_source must be auto, external, vendor, computed, or config")
        if self.config.lithology_csv is not None and self.config.lithology_intervals is not None:
            raise ValueError("Supply lithology_csv or lithology_intervals, not both.")
        context = dict(well.get_parameter("_geology_context") or {})
        source = self.config.lithology_intervals
        if source is None:
            source = self.config.lithology_csv
        if source is None:
            source = context.get("lithology")
        vendor_available = self._has_curve(well, "LITHOLOGY_GEOLINK") or resolve_from_lithology_log(well, warn=False) is not None
        selected = policy
        if policy == "auto":
            selected = "external" if source is not None else ("vendor" if vendor_available else "computed")
        if selected == "external" and source is None:
            raise ValueError("lithology_source='external' requires lithology_intervals or lithology_csv.")
        if selected == "vendor" and not vendor_available:
            raise ValueError("lithology_source='vendor' requires a recognized vendor lithology curve.")
        if selected == "config" and (
            self.config.matrix_type == "auto" or self.config.lithology == "auto"
        ):
            raise ValueError("lithology_source='config' requires explicit matrix_type and lithology.")
        self._selected_lithology_source = selected
        self._external_lithology = None
        report = {"requested_source": policy, "selected_source": selected,
                  "external_available": source is not None, "vendor_available": vendor_available}
        output_curves = []
        if selected == "external":
            table = read_lithology_intervals(source, well=well)
            self._external_lithology = table
            labels = list(dict.fromkeys(table["lithology"]))
            codes = np.full(well.depth.shape, np.nan)
            groups = np.full(well.depth.shape, None, dtype=object)
            aliases = {
                "sand": "sandstone", "sst": "sandstone", "ss": "sandstone",
                "clastic": "sandstone", "clastics": "sandstone",
                "siltstone": "silt", "mudstone": "shale", "claystone": "shale",
                "chalk": "limestone", "carbonate": "limestone", "carb": "limestone",
                "marl": "limestone", "lime": "limestone", "ls": "limestone",
                "dolo": "dolomite", "salt": "halite",
            }
            unknown = set()
            for row in table.itertuples():
                key = str(row.lithology).strip().lower()
                group = aliases.get(key, key)
                if group not in LITHOLOGY_GROUP_TO_MATRIX:
                    unknown.add(row.lithology)
                    group = None
                mask = (well.depth >= row.top) & (well.depth < row.bottom)
                if row.bottom == table["bottom"].max():
                    mask |= well.depth == row.bottom
                codes[mask] = labels.index(row.lithology) + 1
                groups[mask] = group
            well.add_curve("LITHO_CLASS", codes, "", "External lithology categories")
            well["LITHO_CLASS"].metadata.custom.update({
                "category_labels": {index + 1: label for index, label in enumerate(labels)},
                "source": "external", "code_scheme": "local_categories",
            })
            self._lithology_log = groups
            self._lithology_source = "external_lithology"
            context["lithology"] = table.to_dict("records")
            report.update({"import_report": table.attrs.get("import_report", {}),
                           "unmapped_labels": sorted(unknown),
                           "covered_samples": int(np.isfinite(codes).sum()),
                           "uncovered_samples": int(np.isnan(codes).sum())})
            output_curves = ["LITHO_CLASS"]
            if unknown or np.isnan(codes).any():
                self._result.warnings.append(
                    f"External lithology: unmapped labels {sorted(unknown)}, "
                    f"{np.isnan(codes).sum()} uncovered samples. Labels/gaps are preserved; "
                    "unclassified samples use configured calculation fallbacks, not inferred rock labels."
                )
        decisions = [f"Selected {selected} lithology; alternate sources were retained"]
        decisions.extend(self._select_formation_tops(well, context, report))
        context["selected_lithology_source"] = selected
        context["lithology_selection"] = report
        well.set_parameter("_geology_context", context)
        self._result.steps_executed.append(StepResult(
            step="input_context", success=True, method="validated_geology_selection",
            output_curves=output_curves, parameters=report,
            decisions=decisions,
        ))

    def _select_formation_tops(
        self, well: "Well", context: Dict[str, Any], report: Dict[str, Any]
    ) -> List[str]:
        """Use formations attached by :func:`petrophys.read` when unambiguous.

        Formation tops drive formation-specific core permeability calibration.
        An explicit ``DynamicWorkflowConfig.formation_tops`` always wins;
        otherwise attached formations are used when they form a single,
        non-overlapping level.
        """
        if self.config.formation_tops is not None:
            report["formations"] = {"source": "config", "used": True}
            return []
        records = context.get("formations")
        if not records:
            return []
        try:
            tops = _attached_formation_tops(records, float(np.nanmax(well.depth)))
        except ValueError as exc:
            lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
            reason = " ".join(lines[:2])
            if not reason.endswith("."):
                reason += "."
            report["formations"] = {"source": "attached", "used": False, "reason": reason}
            return [
                "Attached formations are used for plotting and reporting only: "
                f"{reason} Pass one stratigraphic level as "
                "DynamicWorkflowConfig(formation_tops=...) to calibrate core "
                "permeability per formation."
            ]
        self.config.formation_tops = tops
        names = [interval.name for interval in tops.intervals]
        report["formations"] = {"source": "attached", "used": True, "intervals": names}
        return [
            f"Using {len(names)} attached formation intervals for formation-specific "
            "core permeability calibration"
        ]

    @property
    def _uses_vendor_benchmark(self) -> bool:
        return self._interpretation_profile == "vendor_benchmark"

    def _resolve_interpretation_profile(self, well: "Well") -> str:
        """Select the benchmark or generic route from authoritative inputs."""
        from petrophys.interpretation.lithology.matrix_inference import (
            resolve_from_lithology_log,
        )

        requested = str(self.config.interpretation_profile or "auto").strip().lower()
        aliases = {
            "auto": "auto",
            "vendor": "vendor_benchmark",
            "benchmark": "vendor_benchmark",
            "vendor_benchmark": "vendor_benchmark",
            "legacy_benchmark": "vendor_benchmark",
            "generic": "generic_inference",
            "generic_inference": "generic_inference",
        }
        if requested not in aliases:
            allowed = "auto, vendor_benchmark, generic_inference"
            raise ValueError(
                f"Unknown interpretation_profile={requested!r}; expected one of {allowed}"
            )

        requested = aliases[requested]
        self._vendor_lithology_resolution = (
            resolve_from_lithology_log(well, warn=False)
            if self._selected_lithology_source in {"auto", "vendor"} else None
        )
        has_vendor_lithology = self._vendor_lithology_resolution is not None

        if requested == "auto":
            resolved = "vendor_benchmark" if has_vendor_lithology else "generic_inference"
        elif requested == "vendor_benchmark" and not (
            has_vendor_lithology or self._selected_lithology_source == "external"
        ):
            resolved = "generic_inference"
            if self._result is not None:
                self._result.warnings.append(
                    "vendor_benchmark requested without a validated vendor lithology "
                    "curve or lithology CSV; generic_inference was used instead"
                )
        else:
            resolved = requested

        self._interpretation_profile = resolved
        if self._result is not None:
            self._result.interpretation_profile = resolved
        return resolved

    def _set_auto_lithology(self, res) -> list:
        """Write an auto-resolved lithology/matrix to the (copied) run config."""
        applied = []
        if self._auto_matrix:
            self.config.matrix_type = res.matrix
            applied.append(f"matrix_type={res.matrix}")
        if self._auto_lith:
            self.config.lithology = res.lithology
            applied.append(f"lithology={res.lithology}")
        return applied

    def _apply_lithology_resolution(self, res, iterations: int = 0) -> None:
        """Apply a resolution and record it (single-shot: vendor log / default)."""
        applied = self._set_auto_lithology(res)
        self._lithology_pending = False
        if self._result is not None and applied:
            iter_note = f", {iterations} iteration(s)" if iterations else ""
            self._result.warnings.append(
                f"Auto lithology resolved via {res.method} "
                f"({res.confidence:.0%} of {res.n} samples{iter_note}): " + ", ".join(applied)
            )

    def _converge_auto_matrix(self, well, max_iters: int = 3) -> None:
        """Close the VCL <-> multi-mineral-solver loop by iterating to a fixed point.

        The N-D VCL clean line needs a matrix; the solver-derived matrix needs VCL.
        Re-run VCLAY -> POROSITY -> LITHOLOGY with each newly-resolved matrix until
        the solver's dominant matrix stops changing (or max_iters is hit).
        """
        from petrophys.interpretation.lithology.matrix_inference import resolve_from_minerals

        last_matrix = None
        final_res = None
        iterations = 0
        for _ in range(max_iters):
            res = resolve_from_minerals(well, warn=False)
            if res is None:
                return  # can't resolve; leave pending -> end-of-run default
            final_res = res
            if res.matrix == last_matrix:
                break  # converged
            last_matrix = res.matrix
            self._set_auto_lithology(res)
            iterations += 1
            # recompute the VCL -> porosity -> lithology cycle with the new matrix
            self._step_vclay(well)
            self._step_porosity(well)
            self._step_lithology(well)
        if final_res is not None:
            self._apply_lithology_resolution(final_res, iterations=iterations)

    def _resolved_cutoff(self, key: str):
        """Config cutoff if set, else the lithology-family standard."""
        value = getattr(self.config, key)
        if value is not None:
            return value
        if self._uses_vendor_benchmark:
            historical = {"vcl_cutoff": 0.3, "sw_cutoff": 0.7}
            if key in historical:
                return historical[key]
        from petrophys.evaluation.net_pay.cutoffs import standard_cutoffs
        return standard_cutoffs(self.config.lithology)[key]

    def _cutoff_argument(self, key: str):
        """Value passed to a module, preserving historical None semantics."""
        value = getattr(self.config, key)
        if value is not None or not self._uses_vendor_benchmark:
            return value
        if key == "vcl_cutoff":
            return 0.3
        if key == "sw_cutoff":
            return 0.7
        return None

    def _archie_coefficients_for_lithology(
        self,
        lithology: Union[str, np.ndarray],
        n_samples: int,
    ) -> tuple[Any, Any, Any, Dict[str, Dict[str, Any]]]:
        """Build lithology-dependent Archie coefficients with config overrides."""
        from petrophys.model.constants import ArchieParameterDefaults

        if isinstance(lithology, np.ndarray):
            lithology_values = np.asarray(lithology, dtype=object)
            if lithology_values.shape != (n_samples,):
                # Pad/truncate to n_samples; never cyclically tile (np.resize), which
                # would mis-assign per-depth Archie/matrix endpoints.
                fixed = np.full(n_samples, self.config.lithology, dtype=object)
                k = min(lithology_values.size, n_samples)
                fixed[:k] = lithology_values.ravel()[:k]
                lithology_values = fixed
        else:
            lithology_values = np.full(n_samples, lithology or self.config.lithology, dtype=object)

        fallback = self.config.lithology or "generic_clean"
        a_values = np.full(n_samples, np.nan, dtype=float)
        m_values = np.full(n_samples, np.nan, dtype=float)
        n_values = np.full(n_samples, np.nan, dtype=float)
        summary: Dict[str, Dict[str, Any]] = {}

        for idx, raw_lithology in enumerate(lithology_values):
            lithology_name = _coerce_lithology_group(raw_lithology) or fallback
            coeffs = ArchieParameterDefaults.get(lithology_name)
            a_values[idx] = float(self.config.a) if self.config.a is not None else float(coeffs.a)
            m_values[idx] = float(self.config.m) if self.config.m is not None else float(coeffs.m)
            n_values[idx] = float(self.config.n) if self.config.n is not None else float(coeffs.n)

            key = str(coeffs.lithology or lithology_name).upper()
            item = summary.setdefault(
                key,
                {
                    "a": a_values[idx],
                    "m": m_values[idx],
                    "n": n_values[idx],
                    "closest_lithology": coeffs.closest_lithology,
                    "samples": 0,
                },
            )
            item["samples"] += 1

        a_out = float(a_values[0]) if len(_finite_unique_numeric(a_values)) == 1 else a_values
        m_out = float(m_values[0]) if len(_finite_unique_numeric(m_values)) == 1 else m_values
        n_out = float(n_values[0]) if len(_finite_unique_numeric(n_values)) == 1 else n_values
        return a_out, m_out, n_out, summary

    def get_pipeline(self):
        return [
            (WorkflowStep.NORMALIZATION, self._step_normalization, self._can_normalization),
            (WorkflowStep.TEMPERATURE, self._step_temperature, self._can_temperature),
            (WorkflowStep.BAD_HOLE, self._step_bad_hole, self._can_bad_hole),
            (WorkflowStep.SPECTRAL_GAMMA, self._step_spectral_gamma, self._can_spectral_gamma),
            (WorkflowStep.ELEMENTAL_SPECTROSCOPY, self._step_elemental_spectroscopy, self._can_elemental_spectroscopy),
            (WorkflowStep.VCLAY, self._step_vclay, self._can_vclay),
            (WorkflowStep.RCL, self._step_rcl, self._can_rcl),
            (WorkflowStep.RMF, self._step_rmf, self._can_rmf),
            (WorkflowStep.HC_CORRECTION, self._step_hc_correction, self._can_hc_correction),
            (WorkflowStep.POROSITY, self._step_porosity, self._can_porosity),
            (WorkflowStep.LITHOLOGY, self._step_lithology, self._can_lithology),
            (WorkflowStep.POROSITY_REFINEMENT, self._step_porosity_refinement, self._can_porosity_refinement),
            (WorkflowStep.CORE_CORRECTION, self._step_core_correction, self._can_core_correction),
            (WorkflowStep.RW, self._step_rw, self._can_rw),
            (WorkflowStep.INVASION, self._step_invasion, self._can_invasion),
            (WorkflowStep.SATURATION, self._step_saturation, self._can_saturation),
            (WorkflowStep.PERMEABILITY, self._step_permeability, self._can_permeability),
            (WorkflowStep.ROCK_QUALITY, self._step_rock_quality, self._can_rock_quality),
            (WorkflowStep.BVW, self._step_bvw, self._can_bvw),
            (WorkflowStep.MHI, self._step_mhi, self._can_mhi),
            (WorkflowStep.NET_PAY, self._step_net_pay, self._can_net_pay),
        ]

    def run_selected_steps(self, well: "Well", steps: list[WorkflowStep]) -> DynamicWorkflowResult:
        """
        Execute only the specified workflow steps, in pipeline order.
        """
        self._result = DynamicWorkflowResult(well=well, config=self.config)
        self._lithology_log = None
        self._lithology_source = "config"
        self._computed_lithology_written = False
        self._last_phit_nd_average_mask = None
        self._last_phi_cl_info = None
        self._interpretation_profile = "generic_inference"
        self._vendor_lithology_resolution = None
        try:
            self._prepare_input_context(well)
            self._resolve_interpretation_profile(well)
        except (ValueError, FileNotFoundError) as exc:
            if self.config.raise_on_error:
                raise
            self._result.errors.append(f"input_context: {exc}")
            return self._result

        allowed = set(steps)

        for step_enum, step_fn, _ in self.get_pipeline():
            if step_enum not in allowed:
                continue

            step_name = step_enum.value
            try:
                step_trace = step_fn(well)
                if step_trace is None:
                    step_trace = StepResult(step=step_name, success=True)
                step_trace.step = step_name
                # Record a StepResult (same type as run()) for consistent consumers.
                self._result.steps_executed.append(step_trace)
                if not step_trace.success:
                    self._result.errors.append(f"{step_name}: {step_trace.message}")
            except Exception as e:
                if self.config.raise_on_error:
                    raise
                self._result.errors.append(f"{step_name}: {e}")
                self._result.specific_steps_skipped[step_enum.name] = str(e)
                self._result.steps_executed.append(
                    StepResult(step=step_name, success=False, message=str(e))
                )

        return self._result

    def run(self, well: "Well") -> DynamicWorkflowResult:
        """
        Run the dynamic workflow on a well.

        Automatically detects available curves and executes
        applicable processing steps in pipeline order.

        Parameters
        ----------
        well : Well
            Well object with raw log data

        Returns
        -------
        DynamicWorkflowResult
            ``result.config`` is the configuration used for this run, with any
            ``"auto"`` matrix/lithology replaced by the resolved values.

        Notes
        -----
        Each run works on its own copy of :attr:`config`, so a workflow can be
        reused across wells: values resolved for one well (e.g. an ``"auto"``
        matrix) never carry over to the next, and :attr:`config` itself is
        left unchanged.
        """
        requested_config = self.config
        self.config = dataclasses.replace(requested_config)
        try:
            return self._run(well)
        finally:
            self.config = requested_config

    def _run(self, well: "Well") -> DynamicWorkflowResult:
        self._auto_matrix = str(self.config.matrix_type).strip().lower() == "auto"
        self._auto_lith = str(self.config.lithology).strip().lower() == "auto"

        self._result = DynamicWorkflowResult(well=well, config=self.config)
        self._lithology_log = None
        self._lithology_source = "config"
        self._computed_lithology_written = False
        self._last_phit_nd_average_mask = None
        self._last_phi_cl_info = None
        self._interpretation_profile = "generic_inference"
        self._vendor_lithology_resolution = None

        input_error = self._input_validation_error(well)
        if input_error is not None:
            self._result.errors.append(input_error)
            return self._result

        # Normalize before any auto-lithology routing examines the inputs. The
        # pipeline still exposes this step for discovery and selected-step runs.
        normalization_done = False
        if (
            self.config.standardize_input
            and WorkflowStep.NORMALIZATION.value not in self.config.skip_steps
        ):
            try:
                self._result.steps_executed.append(self._step_normalization(well))
                normalization_done = True
            except Exception as exc:
                if self.config.raise_on_error:
                    raise
                message = f"normalization: {exc}"
                self._result.errors.append(message)
                self._result.steps_executed.append(
                    StepResult(step="normalization", success=False, message=str(exc))
                )
                return self._result

        try:
            self._prepare_input_context(well)
            self._resolve_interpretation_profile(well)
        except (ValueError, FileNotFoundError) as exc:
            if self.config.raise_on_error:
                raise
            self._result.errors.append(str(exc))
            return self._result

        # Tiers 0-1: resolve auto lithology from raw-input sources up front (a vendor
        # lithology log, else core-porosity apparent matrix density), so it drives the
        # whole pipeline (N-D VCL clean line, porosity matrix, Sw, cutoffs). Both are
        # independent of VCL/porosity, so there is no circularity here.
        self._lithology_pending = self._auto_matrix or self._auto_lith
        if self._lithology_pending:
            from petrophys.interpretation.lithology.matrix_inference import (
                resolve_from_lithology_log, resolve_from_core_matrix,
            )
            rho_hc = {"oil": 0.8, "gas": 0.25, "condensate": 0.5}.get(
                str(self.config.hc_type).lower(), 0.8
            )
            res = None
            if self._selected_lithology_source == "external":
                from petrophys.interpretation.lithology.matrix_inference import _finalize
                groups = self._lithology_log
                clastic = int(sum(value in {"sandstone", "shale", "silt", "clastic_coarse", "calcareous_shale"} for value in groups))
                carbonate = int(sum(value == "limestone" for value in groups))
                dolomite = int(sum(value == "dolomite" for value in groups))
                res = _finalize(clastic, carbonate, dolomite, "external_lithology", clastic + carbonate + dolomite, False)
            elif self._selected_lithology_source == "vendor":
                res = resolve_from_lithology_log(well, warn=False)
            elif self._selected_lithology_source != "config":
                res = resolve_from_core_matrix(well, rho_hc=rho_hc, warn=False)
            if res is not None:
                self._apply_lithology_resolution(res)
            elif self._selected_lithology_source == "config":
                raise ValueError("lithology_source='config' requires explicit lithology and matrix_type, not 'auto'.")

        # Detect available curves
        available = self._detect_available_curves(well)

        # Execute pipeline in order
        pipeline = self.get_pipeline()

        for step_enum, step_fn, can_fn in pipeline:
            step_name = step_enum.value

            if step_enum is WorkflowStep.NORMALIZATION and normalization_done:
                continue

            if step_name in self.config.skip_steps:
                self._result.steps_skipped.append(step_name)
                self._result.specific_steps_skipped[step_name] = "Disabled in skip_steps"
                continue

            if can_fn(well):
                try:
                    step_trace = step_fn(well)
                    if step_trace is None:
                        step_trace = StepResult(step=step_name, success=True)
                    step_trace.step = step_name
                    self._result.steps_executed.append(step_trace)
                    if not step_trace.success:
                        self._result.errors.append(f"{step_name}: {step_trace.message}")
                        if self.config.raise_on_error:
                            raise RuntimeError(step_trace.message)
                    # Tier 1: once the multi-mineral solver has run, iterate the
                    # VCL <-> solver loop to a fixed point, resolving any still-pending
                    # auto lithology before the later saturation / net-pay steps.
                    if (step_name == WorkflowStep.LITHOLOGY.value
                            and self._lithology_pending
                            and not self._uses_vendor_benchmark):
                        self._converge_auto_matrix(well)
                except Exception as e:
                    if self.config.raise_on_error:
                        raise
                    msg = f"{step_name}: {str(e)}"
                    self._result.warnings.append(msg)
                    self._result.errors.append(msg)
                    self._result.steps_executed.append(
                        StepResult(step=step_name, success=False, message=str(e))
                    )
            else:
                self._result.steps_skipped.append(step_name)
                self._result.specific_steps_skipped[step_name] = self._skip_reason(step_enum, well)

        # The May context benchmark finalized apparent Swirr after NET_RES had
        # been written, then recalculated permeability from that final reservoir
        # mask.  Preserve the sequence explicitly instead of relying on the old
        # accidental second workflow pass.  Generic runs keep the single-pass
        # dependency order above.
        permeability_method = str(self.config.perm_method or "").strip().lower().replace("-", "_")
        benchmark_finalize = (
            self._uses_vendor_benchmark
            and WorkflowStep.PERMEABILITY.value not in self.config.skip_steps
            and self.config.swirr is None
            and permeability_method not in {"lith_vcl", "lithology_vcl", "dunite", "dunite_basic"}
            and self._can_permeability(well)
            and any(
                step.step == WorkflowStep.NET_PAY.value and step.success
                for step in self._result.steps_executed
            )
        )
        if benchmark_finalize:
            try:
                final_perm_trace = self._step_permeability(well)
                final_perm_trace.step = "permeability_benchmark_finalize"
                final_perm_trace.parameters["interpretation_profile"] = self._interpretation_profile
                final_perm_trace.decisions.insert(
                    0,
                    "Benchmark finalization: recalculated Swirr and permeability from final RESFLAG",
                )
                self._result.steps_executed.append(final_perm_trace)

                if self._can_rock_quality(well):
                    final_quality_trace = self._step_rock_quality(well)
                    final_quality_trace.step = "rock_quality_benchmark_finalize"
                    final_quality_trace.parameters["interpretation_profile"] = self._interpretation_profile
                    final_quality_trace.decisions.insert(
                        0,
                        "Benchmark finalization: refreshed rock-quality curves after permeability",
                    )
                    self._result.steps_executed.append(final_quality_trace)
            except Exception as exc:
                if self.config.raise_on_error:
                    raise
                self._result.errors.append(f"permeability_benchmark_finalize: {exc}")
                self._result.warnings.append(f"permeability_benchmark_finalize: {exc}")
                self._result.steps_executed.append(
                    StepResult(
                        step="permeability_benchmark_finalize",
                        success=False,
                        message=str(exc),
                    )
                )

        # Fallback: if auto lithology is still unresolved (no vendor log, solver
        # skipped), apply the default (sandstone/clastic) with a warning.
        if self._lithology_pending:
            from petrophys.interpretation.lithology.matrix_inference import (
                _default, resolve_from_minerals,
            )
            self._apply_lithology_resolution(
                resolve_from_minerals(well, warn=False) or _default(warn=False)
            )

        return self._result

    @staticmethod
    def _input_validation_error(well: "Well") -> Optional[str]:
        """Return a user-facing error when a well cannot support a workflow."""
        if well is None:
            return "A Well instance is required"
        depth = getattr(well, "depth", None)
        if depth is None:
            return "Well has no depth samples"
        try:
            depth_values = np.asarray(depth, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return "Well depth must be numeric"
        if depth_values.size == 0 or not np.any(np.isfinite(depth_values)):
            return "Well has no finite depth samples"

        for mnemonic in getattr(well, "curves", []):
            if str(mnemonic).strip().upper() in {
                "DEPTH", "DEPT", "MD", "TDEP", "TVD", "TVDSS",
            }:
                continue
            curve = well.get(mnemonic)
            if curve is None:
                continue
            try:
                values = np.asarray(curve.data, dtype=float)
            except (TypeError, ValueError):
                continue
            if values.size and np.any(np.isfinite(values)):
                return None
        return "Well has no finite log-curve samples"

    def _detect_available_curves(self, well: "Well") -> set:
        """Detect which curves are available in the well."""
        available = set()
        for name in ["GR", "SP", "NPHI", "RHOB", "DTC", "DTS", "RDEP", "RT",
                      "RESS", "RESM", "RSHA", "RXO", "PEF", "CALI", "CAL", "BS", "DRHO",
                      "VCLAY", "PHIE", "PHIT", "SWT-LOG", "SXO-LOG", "PERM", "TEMP"]:
            if get_curve(well, [name]) is not None:
                available.add(name)
        return available

    def _missing(self, well: "Well", *requirements: tuple) -> List[str]:
        """Return the labels of unmet ``(label, curve names...)`` requirements."""
        return [label for label, *names in requirements if not self._has_curve(well, *names)]

    def _skip_reason(self, step: "WorkflowStep", well: "Well") -> str:
        """Explain, in user terms, why a pipeline step cannot run on ``well``."""
        porosity = ("porosity (PHIE/PHIT)", "PHIE_CC", "PHIE", "PHIT")
        saturation = ("water saturation (SW)", "SW", "SWA", "SWT-LOG")
        deep_res = ("deep resistivity (RDEP/RT)", "RDEP", "RT")
        vendor_only = "Not part of the vendor_benchmark interpretation profile"
        missing: List[str] = []
        if step is WorkflowStep.NORMALIZATION:
            return "standardize_input=False"
        if step is WorkflowStep.TEMPERATURE:
            return "The well has no depth index"
        if step is WorkflowStep.BAD_HOLE:
            return "Needs a caliper (CALI) or density-correction (DRHO) curve"
        if step is WorkflowStep.VCLAY:
            return "Needs a gamma-ray curve, an SP curve, or neutron and density curves"
        if step is WorkflowStep.RCL:
            missing = self._missing(well, deep_res, ("clay volume (VCL)", "VCL", "VCLAY"))
        elif step is WorkflowStep.LITHOLOGY:
            discriminator = (
                ("neutron porosity (NPHI)", "NPHI", "PHIN") if self._uses_vendor_benchmark
                else ("a neutron, PEF or sonic curve", "NPHI", "PHIN", "PEF", "DTC", "DT")
            )
            missing = self._missing(
                well, ("bulk density (RHOB)", "RHOB"), discriminator, ("clay volume (VCL)", "VCL", "VCLAY")
            )
        elif step in (WorkflowStep.SPECTRAL_GAMMA, WorkflowStep.ELEMENTAL_SPECTROSCOPY):
            if self._uses_vendor_benchmark:
                return vendor_only
            if step is WorkflowStep.SPECTRAL_GAMMA:
                return "Needs processed spectral gamma curves (potassium, thorium or uranium)"
            return "Needs processed elemental concentration or yield curves"
        elif step is WorkflowStep.RMF:
            return (
                "No mud-filtrate resistivity source: set mud_type='OBM', rmf_at_meas_temp, "
                "rmf_salinity_ppm or rm, or provide RMF/RM in the LAS header"
            )
        elif step is WorkflowStep.INVASION:
            missing = self._missing(
                well,
                ("shallow or medium resistivity (RXO/RSHA/RMED)", "RSHA", "RESS", "RXO", "RMED", "RESM"),
                deep_res,
            )
        elif step is WorkflowStep.HC_CORRECTION:
            if not self.config.apply_hc_correction:
                return "apply_hc_correction=False (hydrocarbon correction is opt-in)"
            missing = self._missing(well, ("neutron porosity (NPHI)", "NPHI"), ("bulk density (RHOB)", "RHOB"))
        elif step is WorkflowStep.POROSITY:
            return "Needs a density, neutron or sonic curve, or an input porosity curve"
        elif step is WorkflowStep.POROSITY_REFINEMENT:
            if WorkflowStep.POROSITY.value in self.config.skip_steps:
                return "The porosity step is disabled in skip_steps"
            if self._uses_vendor_benchmark:
                return vendor_only
            if not self._computed_lithology_written:
                return "No computed lithology log (LITHO) was produced to refine porosity by rock type"
            missing = self._missing(well, ("computed lithology (LITHO)", "LITHO"))
        elif step is WorkflowStep.CORE_CORRECTION:
            missing = self._missing(
                well,
                porosity,
                ("core porosity (PHIC/CPOR)", "PHIC", "CPOR"),
                ("core permeability (KH_CORE/CPERM)", "KH_CORE", "CPERM", "K_CORE", "PERM_CORE"),
            )
        elif step is WorkflowStep.SATURATION:
            missing = self._missing(well, ("effective porosity (PHIE)", "PHIE_CC", "PHIE"), deep_res)
            if self._result is not None and self._result.rw_computed is None:
                missing.append("formation water resistivity (Rw)")
        elif step is WorkflowStep.PERMEABILITY:
            missing = self._missing(well, porosity)
        elif step is WorkflowStep.ROCK_QUALITY:
            missing = self._missing(well, ("permeability (PERM)", "PERM_CC", "PERM"), porosity)
        elif step in (WorkflowStep.BVW, WorkflowStep.NET_PAY):
            missing = self._missing(well, porosity, saturation)
        elif step is WorkflowStep.MHI:
            missing = self._missing(well, saturation)
            if not self._has_sxo_curve(well):
                missing.append("flushed-zone saturation (SXO)")
        if missing:
            return "Missing " + ", ".join(missing)
        return "Required usable curves or parameters are unavailable"

    def _can_normalization(self, well: "Well") -> bool:
        return bool(self.config.standardize_input)

    def _step_normalization(self, well: "Well") -> StepResult:
        """Standardize aliases and physical units before interpretation."""
        from petrophys.preprocessing.standardization import (
            add_standard_curve_aliases,
            standardize_curves,
        )

        before = {
            name: {
                "unit": str(getattr(well.get(name), "unit", "") or ""),
                "valid_count": int(well.get(name).valid_count()),
            }
            for name in well.curves
            if well.get(name) is not None
        }
        standardize_curves(
            well,
            rename_curves=self.config.rename_input_curves,
            convert_units=True,
        )
        rename_map = dict(well.get_parameter("_rename_map") or {})
        working_aliases = {}
        if not self.config.rename_input_curves:
            working_aliases = add_standard_curve_aliases(well)
        unit_changes = {}
        for original, details in before.items():
            target = rename_map.get(original, original)
            curve = well.get(target) or well.get(original)
            if curve is None:
                continue
            after_unit = str(getattr(curve, "unit", "") or "")
            if after_unit != details["unit"]:
                unit_changes[original] = {
                    "from": details["unit"],
                    "to": after_unit,
                }

        from petrophys.preprocessing.standardization.curve_standardizer import (
            RANGE_SCREENING_WARNING_FRACTION,
        )

        range_screening = {
            name: dict(curve.metadata.custom["range_screening"])
            for name in well.curves
            for curve in [well.get(name)]
            if "range_screening" in curve.metadata.custom
        }
        for name, item in range_screening.items():
            if item["fraction"] >= RANGE_SCREENING_WARNING_FRACTION:
                message = (
                    f"{name}: {item['masked']} of {item['finite']} samples were outside the "
                    f"plausible {item['standard_curve']} range {item['low']:g}-{item['high']:g} "
                    "and were removed"
                )
                if item["hint"]:
                    message += f"; {item['hint']}, check the unit label {item['source_unit']!r}"
                self._result.warnings.append(message + ".")

        return StepResult(
            step="normalization",
            success=True,
            method="curve_alias_and_unit_standardization",
            input_curves=sorted(before),
            output_curves=sorted(well.curves),
            parameters={
                "rename_curves": self.config.rename_input_curves,
                "convert_units": True,
                "rename_map": rename_map,
                "working_aliases": working_aliases,
                "unit_changes": unit_changes,
                "core_saturation_normalization": {
                    name: curve.metadata.custom["core_saturation_normalization"]
                    for name in well.curves
                    for curve in [well.get(name)]
                    if "core_saturation_normalization" in curve.metadata.custom
                },
                "range_screening": range_screening,
            },
            decisions=[
                "Standardized curve units, nulls, and plausible ranges before routing",
                (
                    "Added canonical working aliases while retaining vendor mnemonics"
                    if working_aliases
                    else "No canonical working aliases were required"
                ),
            ],
        )

    def _has_curve(self, well: "Well", *names: str) -> bool:
        """Check if any named curve has finite usable numeric samples."""
        return get_curve(well, names) is not None

    def _has_nmr_swirr_route(self, well: "Well") -> bool:
        """Return True when NMR Swirr can be read or derived."""
        if self._has_curve(well, "SWIRR_NMR", "NMR_SWIRR", "SWIRR_CMR", "CMR_SWIRR"):
            return True
        has_total = self._has_curve(
            well,
            "NMR_PHIT",
            "MPHI",
            "TCMR",
            "CMRPHI",
            "NMRP",
            "PHINMR",
        )
        has_bvi_or_ffi = self._has_curve(
            well,
            "NMR_BVI",
            "MBVI",
            "CBVI",
            "CMR_BVI",
            "BVI",
            "NMR_FFI",
            "MFFI",
            "CMFF",
            "CMR_FFI",
            "FFI",
        )
        return bool(has_total and has_bvi_or_ffi)

    # ---- CAN-RUN checks ----

    def _can_temperature(self, well: "Well") -> bool:
        return (
            well.depth is not None
            or self._has_curve(well, "DEPT", "DEPTH")
        )

    def _can_bad_hole(self, well: "Well") -> bool:
        return self._has_curve(well, "CALI", "CAL") or self._has_curve(well, "DRHO")

    def _can_vclay(self, well: "Well") -> bool:
        # Accept the GR family: a CGR/SGR-only well still supports GR-based VCL.
        # The step also supports SP-only and neutron-density-only wells, so a
        # GR-less well must not be silently left without any clay volume
        # (downstream Sw would otherwise treat every sample as clean sand).
        return (
            self._has_curve(
                well, "GR", "CGR", "CGR_CALC", "SGR", "SGR_CALC",
                "HSGR", "HCGR", "GRKT", "GRTO", "GRSL"
            )
            or self._has_curve(well, "SP")
            or (self._has_curve(well, "NPHI", "PHIN") and self._has_curve(well, "RHOB"))
        )

    def _can_rcl(self, well: "Well") -> bool:
        return self._has_curve(well, "RDEP", "RT") and self._has_curve(well, "VCL", "VCLAY")

    def _can_lithology(self, well: "Well") -> bool:
        if self._uses_vendor_benchmark:
            # The historical inversion requires neutron porosity.  Vendor
            # lithology remains authoritative when it is absent, so skip only
            # the optional mineral-volume solve instead of failing the run.
            return (
                self._has_curve(well, "RHOB")
                and self._has_curve(well, "NPHI", "PHIN")
                and self._has_curve(well, "VCL", "VCLAY")
            )
        has_discriminator = self._has_curve(well, "NPHI", "PHIN", "PEF", "DTC", "DT")
        return (
            self._has_curve(well, "RHOB")
            and has_discriminator
            and self._has_curve(well, "VCL", "VCLAY")
        )

    def _can_spectral_gamma(self, well: "Well") -> bool:
        if self._uses_vendor_benchmark:
            return False
        from petrophys.interpretation.geochemistry import has_spectral_gamma_inputs
        return has_spectral_gamma_inputs(well)

    def _can_elemental_spectroscopy(self, well: "Well") -> bool:
        if self._uses_vendor_benchmark:
            return False
        from petrophys.interpretation.geochemistry import has_elemental_spectroscopy_inputs
        return has_elemental_spectroscopy_inputs(well)

    def _can_rmf(self, well: "Well") -> bool:
        return (
            str(self.config.mud_type).upper() == "OBM"
            or self.config.rmf_at_meas_temp is not None
            or self.config.rmf_salinity_ppm is not None
            or self.config.rm is not None
            or well.header.rmf is not None
            or well.header.rm is not None
            or _parameter_float(well, "RMF") is not None
            or _constant_zoned_parameter(well, "RMF") is not None
            or _parameter_float(well, "RM") is not None
            or _constant_zoned_parameter(well, "RM") is not None
        )

    def _can_rw(self, well: "Well") -> bool:
        # Always run the Rw step so the workflow can fall back through
        # metadata, salinity, log-based heuristics, and finally the built-in
        # default when no explicit source is available.
        return True

    def _can_invasion(self, well: "Well") -> bool:
        return (self._has_curve(well, "RSHA", "RESS", "RXO", "RMED", "RESM") and
                self._has_curve(well, "RDEP", "RT"))

    def _can_hc_correction(self, well: "Well") -> bool:
        return (
            self.config.apply_hc_correction
            and self._has_curve(well, "NPHI")
            and self._has_curve(well, "RHOB")
        )

    def _can_porosity(self, well: "Well") -> bool:
        return self._has_curve(well, "RHOB", "NPHI", "DTC", "PHIN", "PHIT", "PHID", "PHIS")

    def _can_porosity_refinement(self, well: "Well") -> bool:
        return (
            WorkflowStep.POROSITY.value not in self.config.skip_steps
            and not self._uses_vendor_benchmark
            and self._computed_lithology_written
            and self._can_porosity(well)
            and self._has_curve(well, "LITHO")
        )

    def _can_core_correction(self, well: "Well") -> bool:
        return (
            self._has_curve(well, "PHIE", "PHIT")
            and self._has_curve(well, "PHIC", "CPOR")
            and self._has_curve(well, "KH_CORE", "CPERM", "K_CORE", "PERM_CORE")
        )

    def _can_saturation(self, well: "Well") -> bool:
        has_porosity = self._has_curve(well, "PHIE_CC", "PHIE")
        can_compute_sw = self._has_curve(well, "RDEP", "RT") and self._result.rw_computed is not None
        can_compute_sxo = self._has_curve(well, "RSHA", "RXO", "RESS") and self._result.rmf_computed is not None
        return has_porosity and (can_compute_sw or can_compute_sxo)

    def _can_permeability(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE") or self._has_curve(well, "PHIT")

    def _can_rock_quality(self, well: "Well") -> bool:
        return (
            self._has_curve(well, "PERM_CC", "PERM")
            and self._has_curve(well, "PHIE_CC", "PHIE", "PHIT")
        )

    def _can_bvw(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE_CC", "PHIE", "PHIT") and self._has_curve(well, "SW", "SWA", "SWT-LOG")

    def _can_mhi(self, well: "Well") -> bool:
        return self._has_curve(well, "SW", "SWA", "SWT-LOG") and self._has_sxo_curve(well)

    def _can_net_pay(self, well: "Well") -> bool:
        return self._has_curve(well, "PHIE_CC", "PHIE", "PHIT") and self._has_curve(well, "SW", "SWA", "SWT-LOG")

    # ---- STEP implementations ----

    def _step_temperature(self, well: "Well") -> StepResult:
        from petrophys.interpretation.temperature import (
            calculate_temperature_profile,
            gradient_to_celsius_per_km,
        )
        from petrophys.interpretation.temperature.temp_profile import (
            store_temperature_parameters,
        )

        trace = StepResult(step="temperature", success=True, method="temperature_profile")
        surface_temp_f = (
            self.config.surface_temp_f
            if self.config.surface_temp_f is not None
            else 68.0
        )
        bht_f = self.config.bht_f
        td_ft = self.config.td_ft
        td_unit = "ft" if td_ft is not None else None

        if bht_f is None:
            bht_f = _metadata_bht_f(well)

        if td_ft is None and well.header.td is not None:
            td_ft = float(well.header.td)
            td_unit = well.depth_unit
        if td_ft is None and well.bottom is not None:
            td_ft = float(well.bottom)
            td_unit = well.depth_unit

        trace.parameters = {
            "surface_temp_f": surface_temp_f,
            "bht_f": bht_f,
            "td_ft": td_ft,
            "td_unit": td_unit,
        }

        try:
            result = calculate_temperature_profile(
                well,
                bht=bht_f,
                surface_temp=surface_temp_f,
                bht_unit="F",
                surface_temp_unit="F",
                td=td_ft,
                td_unit=td_unit,
                depth_unit=well.depth_unit,
            )
        except Exception as exc:
            trace.success = False
            trace.message = str(exc)
            self._result.warnings.append(f"Temperature profile failed: {exc}")
            return trace

        well.add_curve("TEMP", result.temperature, "degF", "Formation Temperature")
        store_temperature_parameters(well, result)

        self._result.temperature_gradient = result.gradient
        trace.method = result.source
        trace.output_curves = ["TEMP"]
        trace.parameters["gradient_f_per_100ft"] = result.gradient
        trace.parameters["gradient_c_per_km"] = gradient_to_celsius_per_km(result.gradient)
        if result.reference_temperature is not None:
            trace.parameters["reference_temperature_f"] = result.reference_temperature
        if result.reference_depth is not None:
            trace.parameters["reference_depth_ft"] = result.reference_depth

        if result.source == "fallback_geothermal_gradient":
            trace.decisions.append("No BHT temperature anchor found -> using fallback 30 C/km geothermal gradient")
        else:
            trace.decisions.append("Temperature profile anchored to BHT metadata/input")
        trace.decisions.append(
            f"Gradient = {result.gradient:.4f} F/100ft ({trace.parameters['gradient_c_per_km']:.2f} C/km)"
        )
        trace.decisions.append(
            f"Temp range: {np.nanmin(result.temperature):.1f} - {np.nanmax(result.temperature):.1f} F"
        )
        return trace

    def _step_bad_hole(self, well: "Well") -> StepResult:
        from petrophys.preprocessing.quality.bad_hole.flag_generator import generate_bad_hole_flags

        trace = StepResult(step="bad_hole", success=True, method="caliper_drho_flags")
        inputs = []

        cali = get_curve(well, ["CALI", "CAL"])
        bs_curve = get_curve(well, ["BS"])
        drho = get_curve(well, ["DRHO"])
        rhob = get_curve(well, ["RHOB"])
        pef = get_curve(well, ["PEF"])

        if bs_curve is not None:
            bs = np.asarray(bs_curve.data, dtype=float)
            bit_size = bs
            representative_bit_size = _representative_finite_value(bs)
            bit_size_source = "BS_curve_depth_by_depth"
            finite_bs = bs[np.isfinite(bs)]
            bit_size_unique_values = sorted({float(value) for value in np.round(finite_bs, 4)}) if finite_bs.size else []
        else:
            header_bit_size = getattr(well.header, "bit_size", None)
            representative_bit_size = float(header_bit_size) if header_bit_size is not None else None
            bit_size = representative_bit_size
            bit_size_source = "well_header" if header_bit_size is not None else None
            bit_size_unique_values = [float(representative_bit_size)] if representative_bit_size is not None else []
        if cali is not None: inputs.append(curve_source_mnemonic(cali) or cali.mnemonic)
        if bs_curve is not None: inputs.append(curve_source_mnemonic(bs_curve) or "BS")
        if drho is not None: inputs.append(curve_source_mnemonic(drho) or "DRHO")
        if rhob is not None: inputs.append(curve_source_mnemonic(rhob) or "RHOB")
        if pef is not None: inputs.append(curve_source_mnemonic(pef) or "PEF")
        trace.input_curves = inputs

        flags = generate_bad_hole_flags(
            well,
            bit_size=bit_size
        )
        well.add_curve("BAD_HOLE", flags["bad_hole"].astype(float), "", "Bad Hole Flag")

        n_bad = int(np.sum(flags["bad_hole"]))
        trace.output_curves = ["BAD_HOLE"]
        trace.parameters["bit_size_used"] = representative_bit_size
        trace.parameters["bit_size_source"] = bit_size_source
        trace.parameters["bit_size_unique_values"] = bit_size_unique_values
        trace.parameters["bit_size_depth_dependent"] = bs_curve is not None and len(bit_size_unique_values) > 1
        if bit_size is None:
            trace.decisions.append("No BS curve or header bit size found; CALI-BS and large-hole criteria skipped")
        trace.decisions.append(f"{n_bad} bad-hole samples flagged out of {len(flags['bad_hole'])}")
        return trace

    def _step_spectral_gamma(self, well: "Well") -> StepResult:
        """Standardize processed K-Th-U logs and derive spectral diagnostics."""
        from petrophys.interpretation.geochemistry import interpret_spectral_gamma

        result = interpret_spectral_gamma(well)
        if result.warnings:
            self._result.warnings.extend(
                f"spectral_gamma: {warning}" for warning in result.warnings
            )

        measurements = sorted(result.measurements)
        decisions = [
            "Used processed concentration/gamma curves; raw detector spectra were not inverted",
        ]
        if measurements:
            decisions.insert(0, f"Resolved processed products: {', '.join(measurements)}")
        if "CGR_CALC" in result.derived:
            decisions.append("Reconstructed uranium-free gamma ray from potassium and thorium")
        if result.warnings:
            decisions.append(
                f"Completed with {len(result.warnings)} unit/provenance warning(s)"
            )

        return StepResult(
            step="spectral_gamma",
            success=True,
            method="processed_k_th_u_spectral_gamma",
            input_curves=result.input_curves,
            output_curves=result.output_curves,
            parameters={
                "sources": result.sources,
                "conversions": result.conversions,
                "measurements": measurements,
                "derived": sorted(result.derived),
            },
            decisions=decisions,
            message="; ".join(result.warnings),
        )

    def _step_elemental_spectroscopy(self, well: "Well") -> StepResult:
        """Resolve processed elemental dry weights and relative yields."""
        from petrophys.interpretation.geochemistry import interpret_elemental_spectroscopy

        result = interpret_elemental_spectroscopy(well)
        if result.warnings:
            self._result.warnings.extend(
                f"elemental_spectroscopy: {warning}" for warning in result.warnings
            )

        dry_elements = sorted(result.dry_weight_fractions)
        yield_products = sorted(result.relative_yields)
        decisions = [
            "Kept relative yields separate from dry-weight fractions",
            "Did not invert raw spectra or apply vendor sensitivities implicitly",
        ]
        if dry_elements:
            decisions.insert(0, f"Resolved dry-weight elements: {', '.join(dry_elements)}")
        if yield_products:
            decisions.append(f"Resolved relative-yield products: {', '.join(yield_products)}")
        if result.warnings:
            decisions.append(
                f"Completed with {len(result.warnings)} unit/provenance warning(s)"
            )

        return StepResult(
            step="elemental_spectroscopy",
            success=True,
            method="processed_elemental_spectroscopy",
            input_curves=result.input_curves,
            output_curves=result.output_curves,
            parameters={
                "sources": result.sources,
                "conversions": result.conversions,
                "dry_weight_elements": dry_elements,
                "relative_yields": yield_products,
                "derived": sorted(result.derived),
            },
            decisions=decisions,
            message="; ".join(result.warnings),
        )

    def _get_temperature_profile(self, well: "Well") -> np.ndarray:
        """Return a formation-temperature array in Fahrenheit."""
        temp_curve = well.get("TEMP")
        if temp_curve is not None:
            temp = np.asarray(temp_curve.data, dtype=float)
            if temp.size == well.n_samples:
                return temp

        if well.depth is not None:
            from petrophys.interpretation.temperature import calculate_temperature_profile

            try:
                surface_temp_f = (
                    self.config.surface_temp_f
                    if self.config.surface_temp_f is not None
                    else 68.0
                )
                result = calculate_temperature_profile(
                    well,
                    bht=self.config.bht_f or _metadata_bht_f(well),
                    surface_temp=surface_temp_f,
                    bht_unit="F",
                    surface_temp_unit="F",
                    td=self.config.td_ft,
                    td_unit="ft" if self.config.td_ft is not None else None,
                    depth_unit=well.depth_unit,
                )
                return np.asarray(result.temperature, dtype=float)
            except Exception:
                pass

        avg_temp = self._get_avg_formation_temp(well)
        return np.full(well.n_samples, avg_temp, dtype=float)

    def _reference_temperature(self, temp_profile: np.ndarray) -> float:
        """Choose a reference temperature for scalar resistivity inputs."""
        finite = temp_profile[np.isfinite(temp_profile)]
        if finite.size == 0:
            return 150.0
        return float(np.nanmedian(finite))

    def _step_vclay(self, well: "Well") -> StepResult:
        from petrophys.interpretation.clay_volume.vcl_module import VclModule
        from petrophys.interpretation.clay_volume.nd_method import nd_clean_line

        trace = StepResult(step="vclay", success=True)
        decisions = []
        existing_vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        preserve_existing_vcl = _is_prioritized_input_interpretation_curve(
            existing_vcl_curve,
            "VCL",
        ) and not self._uses_vendor_benchmark

        gr_min = self.config.gr_min
        gr_max = self.config.gr_max
        gr_min_source = "user" if gr_min is not None else "auto"
        gr_max_source = "user" if gr_max is not None else "auto"

        if self._uses_vendor_benchmark:
            gr_candidates = ["GR", "CGR", "SGR", "HSGR", "HCGR"]
        else:
            # Uranium-free spectral gamma is preferred where U is decoupled
            # from clay; the historical benchmark retains its GR-first route.
            gr_candidates = [
                "CGR", "CGR_CALC", "GRKT", "GR", "SGR", "SGR_CALC",
                "GRTO", "GRSL", "HSGR", "HCGR",
            ]
        gr_curve = get_curve(well, gr_candidates)
        gr_curve_name = gr_curve.mnemonic if gr_curve is not None else None

        if gr_min is None or gr_max is None:
            if gr_curve is not None:
                valid_gr = gr_curve.data[~np.isnan(gr_curve.data)]
                if len(valid_gr) > 0:
                    if gr_min is None:
                        gr_min = float(np.percentile(valid_gr, 5))
                    if gr_max is None:
                        gr_max = float(np.percentile(valid_gr, 95))

        # Use `is None` (not `or`) so a legitimately computed gr_min of 0.0 is kept.
        if gr_min is None:
            gr_min = 20.0
        if gr_max is None:
            gr_max = 100.0

        use_sp = (self._has_curve(well, "SP") and
                  not self.config.mud_has_clay_or_kcl)
        use_nd = self._has_curve(well, "NPHI") and self._has_curve(well, "RHOB")
        sp_min = self.config.sp_min
        sp_max = self.config.sp_max
        sp_min_source = "user" if sp_min is not None else "auto"
        sp_max_source = "user" if sp_max is not None else "auto"

        if use_sp:
            sp_curve = get_curve(well, ["SP"])
            if sp_curve is not None and (sp_min is None or sp_max is None):
                valid_sp = sp_curve.data[~np.isnan(sp_curve.data)]
                if len(valid_sp) > 0:
                    if sp_max is None:
                        sp_max = float(np.percentile(valid_sp, 5))
                    if sp_min is None:
                        sp_min = float(np.percentile(valid_sp, 95))

        decisions.append(f"GR_min = {gr_min:.1f} ({gr_min_source}, P5 of GR)")
        decisions.append(f"GR_max = {gr_max:.1f} ({gr_max_source}, P95 of GR)")
        if gr_curve is not None:
            decisions.append(f"Gamma-ray clay input: {gr_curve_name}")
            if not self._uses_vendor_benchmark and gr_curve_name in {
                "CGR", "CGR_CALC", "GRKT"
            }:
                decisions.append(
                    "Preferred uranium-free spectral gamma over total gamma ray for VCL"
                )
        decisions.append(f"GR method: {self.config.gr_method}")
        decisions.append(f"SP method: {'enabled' if use_sp else 'disabled'}"
                        + (f" (mud has clay/KCl)" if self.config.mud_has_clay_or_kcl else ""))
        decisions.append(f"N-D method: {'enabled' if use_nd else 'disabled (missing NPHI or RHOB)'}")
        if use_sp and sp_max is not None and sp_min is not None:
            decisions.append(f"SP_clean = {sp_max:.1f} ({sp_max_source}, P5 of SP)")
            decisions.append(f"SP_shale = {sp_min:.1f} ({sp_min_source}, P95 of SP)")

        # Select the lowest valid clay estimate from the available methods at
        # each depth, consistent with the workflow's VCL convention.
        vcl_combine = "min"
        # The historical context benchmark used the fixed limestone N-D clean
        # line even though the downstream porosity matrix remained configurable.
        # Keep that quirk isolated to the dated benchmark profile.
        vcl_matrix = "limestone" if self._uses_vendor_benchmark else self.config.matrix_type
        vcl_module = VclModule(
            gr_min=gr_min, gr_max=gr_max,
            gr_method=self.config.gr_method,
            use_sp=use_sp, use_nd=use_nd,
            sp_min=sp_min, sp_max=sp_max,
            matrix=vcl_matrix,
            combine=vcl_combine,
            rho_fl=self.config.rho_fl,
        )
        curve_mapping = {"gr": gr_curve_name} if gr_curve_name else None
        result = vcl_module.calculate(well, curves=curve_mapping)

        inputs = []
        if gr_curve is not None:
            inputs.append(curve_source_mnemonic(gr_curve) or gr_curve_name)
        outputs = []
        if use_sp: inputs.append("SP")
        if use_nd: inputs.extend(["NPHI", "RHOB"])
        if preserve_existing_vcl:
            well.add_curve(
                "VCL_CALC",
                result.vcl,
                unit="v/v",
                description="Calculated clay volume; input VCL preserved as active",
                data_type=DataType.COMPUTED,
            )
            well.add_curve(
                "VCLAY_CALC",
                result.vcl,
                unit="v/v",
                description="Calculated clay volume; input VCLAY/VCL preserved as active",
                data_type=DataType.COMPUTED,
            )
            outputs.extend(["VCL_CALC", "VCLAY_CALC"])
            decisions.append(
                "Existing interpreted clay-volume curve preserved as active: "
                + _prioritized_curve_label(existing_vcl_curve)
            )
            decisions.append("Workflow clay-volume calculation written to VCL_CALC/VCLAY_CALC")
        else:
            well.add_curve("VCL", result.vcl, unit="v/v", description="Clay Volume")
            well.add_curve("VCLAY", result.vcl, unit="v/v", description="Clay Volume")
            outputs.extend(["VCL", "VCLAY"])

        if result.vcl_gr is not None:
            well.add_curve("VCL_GR", result.vcl_gr, unit="v/v", description="VCL from GR")
            well.add_curve("VCLGR", result.vcl_gr, unit="v/v", description="VCL from GR")
            outputs.extend(["VCL_GR", "VCLGR"])
        if result.vcl_sp is not None:
            well.add_curve("VCL_SP", result.vcl_sp, unit="v/v", description="VCL from SP")
            well.add_curve("VCLSP", result.vcl_sp, unit="v/v", description="VCL from SP")
            outputs.extend(["VCL_SP", "VCLSP"])
        if result.vcl_nd is not None:
            well.add_curve("VCL_ND", result.vcl_nd, unit="v/v", description="VCL from N-D")
            well.add_curve("VCLND", result.vcl_nd, unit="v/v", description="VCL from N-D")
            outputs.extend(["VCL_ND", "VCLND"])

        decisions.append("Final VCL = min(available methods), range: "
                         f"{np.nanmin(result.vcl):.3f} - {np.nanmax(result.vcl):.3f}")
        if self._uses_vendor_benchmark:
            decisions.append(
                "Vendor benchmark profile: May 2026 limestone N-D endpoints retained"
            )

        nd_clean1_density, nd_clean1_neutron, nd_clean2_density, nd_clean2_neutron = (
            nd_clean_line(vcl_matrix)
        )

        trace.method = self.config.gr_method
        trace.input_curves = inputs
        trace.output_curves = outputs
        trace.parameters = {
            "gr_min": gr_min, "gr_max": gr_max,
            "gr_curve": gr_curve_name,
            "gr_method": self.config.gr_method,
            "use_sp": use_sp, "use_nd": use_nd,
            "sp_min": sp_min, "sp_max": sp_max,
            "sp_min_source": sp_min_source if use_sp else None,
            "sp_max_source": sp_max_source if use_sp else None,
            "nd_matrix": vcl_matrix,
            "nd_clean1_density": nd_clean1_density,
            "nd_clean1_neutron": nd_clean1_neutron,
            "nd_clean2_density": nd_clean2_density,
            "nd_clean2_neutron": nd_clean2_neutron,
            "nd_clay_density": 2.62,
            "nd_clay_neutron": 0.38,
            "active_vcl_source": "input_interpretation" if preserve_existing_vcl else "workflow_calculation",
            "computed_vcl_curve": "VCL_CALC" if preserve_existing_vcl else "VCL",
            "interpretation_profile": self._interpretation_profile,
        }
        trace.decisions = decisions
        return trace

    def _step_rcl(self, well: "Well") -> StepResult:
        from petrophys.interpretation.clay_volume.rcl import RclModule

        trace = StepResult(step="rcl", success=True, method="median_Rt_in_shale_zones")
        rdep_curve = get_curve(well, ["RDEP", "RT"])
        vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        trace.input_curves = [
            curve.mnemonic for curve in (rdep_curve, vcl_curve) if curve is not None
        ]
        trace.parameters = {"vcl_threshold": 0.9, "default_rcl": 5.0, "statistic": "median"}

        lithology = self._build_lithology_log(well)
        shale_mask = None
        shale_groups: List[str] = []
        if isinstance(lithology, np.ndarray):
            shale_mask = np.asarray([_is_shale_group(item) for item in lithology], dtype=bool)
            shale_groups = _ordered_unique_values(
                str(item).upper() for item in lithology.tolist() if _is_shale_group(item)
            )

        rcl_module = RclModule(statistic="median")
        self._result.rcl_computed = rcl_module.calculate(
            well,
            vcl=vcl_curve.data if vcl_curve is not None else None,
            shale_mask=shale_mask,
        )
        shale_sample_count = 0
        source = "default"
        if rdep_curve is not None:
            rdep = np.asarray(rdep_curve.data, dtype=float)
            if shale_mask is not None:
                valid = shale_mask & np.isfinite(rdep)
                if np.any(valid):
                    shale_sample_count = int(np.count_nonzero(valid))
                    source = "lithology_shale_median_rt"
            if source == "default" and vcl_curve is not None:
                vcl = np.asarray(vcl_curve.data, dtype=float)
                valid = np.isfinite(rdep) & np.isfinite(vcl) & (vcl > 0.9)
                if np.any(valid):
                    shale_sample_count = int(np.count_nonzero(valid))
                    source = "vcl_gt_0.9_median_rt"

        trace.method = source
        trace.parameters["rcl_computed"] = self._result.rcl_computed
        trace.parameters["source"] = source
        trace.parameters["shale_sample_count"] = shale_sample_count
        trace.parameters["shale_lithology_groups"] = shale_groups
        if source == "lithology_shale_median_rt":
            trace.decisions.append(
                f"Rcl = {self._result.rcl_computed:.2f} ohm.m (median RT in shale lithology samples, n={shale_sample_count})"
            )
        elif source == "vcl_gt_0.9_median_rt":
            trace.decisions.append(
                f"Rcl = {self._result.rcl_computed:.2f} ohm.m (median RT where VCL > 0.9, n={shale_sample_count})"
            )
        else:
            trace.decisions.append(
                f"Rcl = {self._result.rcl_computed:.2f} ohm.m (default because no shale lithology or VCL > 0.9 samples were available)"
            )
        return trace

    def _step_lithology(self, well: "Well") -> StepResult:
        from petrophys.interpretation.lithology.lithology_module import LithologyModule

        trace = StepResult(step="lithology", success=True, method="multi_mineral_solver")
        trace.input_curves = ["RHOB", "VCL"]
        nphi_curve = get_curve(well, ["NPHI", "PHIN"])
        if nphi_curve is not None:
            trace.input_curves.append(nphi_curve.mnemonic)
        if self._has_curve(well, "PHIE"):
            trace.input_curves.append("PHIE")
        elif self._has_curve(well, "PHIT"):
            trace.input_curves.append("PHIT")
        if self._has_curve(well, "PEF"):
            trace.input_curves.append("PEF")
        if self._has_curve(well, "DTC"):
            trace.input_curves.append("DTC")

        mineral_family = (
            "carbonate" if (
                not self._auto_lith and str(self.config.lithology).lower() == "carbonate"
            ) or (
                not self._auto_matrix
                and str(self.config.matrix_type).lower() in {"limestone", "dolomite"}
            )
            else "auto"
        )
        litho_module = LithologyModule(lithology=mineral_family)
        if self._uses_vendor_benchmark:
            if nphi_curve is None:
                raise ValueError(
                    "May 2026 vendor benchmark mineral inversion requires NPHI/PHIN"
                )
            from petrophys.workflow._vendor_benchmark import (
                BENCHMARK_PROFILE_VERSION,
                May2026BenchmarkMultiMineralSolver,
            )

            litho_module.solver = May2026BenchmarkMultiMineralSolver()
            trace.method = "may_2026_multi_mineral_solver"
            trace.parameters["benchmark_version"] = BENCHMARK_PROFILE_VERSION
        result = litho_module.calculate(well)
        input_lithology_available = self._selected_lithology_source == "external" or (
            self._selected_lithology_source == "vendor"
            and self._has_curve(well, "LITHOLOGY_GEOLINK")
        )
        if self._selected_lithology_source == "auto":
            input_lithology_available = self._has_curve(well, "LITHOLOGY_GEOLINK") or self.config.lithology_csv is not None
        write_lithology_code = not input_lithology_available
        if self._uses_vendor_benchmark:
            well.add_curve("VQUAR", result.v_quartz, "v/v", "Quartz Volume")
            well.add_curve("VCALC", result.v_calcite, "v/v", "Calcite Volume")
            well.add_curve("VDOLO", result.v_dolomite, "v/v", "Dolomite Volume")
            well.add_curve("VANHY", result.v_anhydrite, "v/v", "Anhydrite Volume")
            if write_lithology_code:
                well.add_curve(
                    "LITHO",
                    result.lithology_code.astype(float),
                    "",
                    "Lithology Code",
                )
        else:
            litho_module.add_to_well(
                well,
                result,
                include_lithology_code=write_lithology_code,
            )
        if write_lithology_code:
            self._lithology_log = None
            self._lithology_source = "config"
            self._computed_lithology_written = True

        trace.output_curves = ["VQUAR", "VCALC", "VDOLO", "VANHY"]
        if not self._uses_vendor_benchmark:
            trace.output_curves.append("MMQC")
        if write_lithology_code:
            trace.output_curves.append("LITHO")
        trace.parameters["lithology_curve_written"] = write_lithology_code
        trace.parameters["lithology_source_priority"] = (
            "multi_mineral_fallback" if write_lithology_code else "input_lithology"
        )
        trace.parameters["multi_mineral_solved_samples"] = int(np.count_nonzero(result.solved))
        trace.parameters["reduced_quartz_calcite_samples"] = int(
            np.count_nonzero(result.reduced_qc_fallback)
            if result.reduced_qc_fallback is not None else 0
        )
        trace.parameters["multi_mineral_qc_flagged_samples"] = int(np.count_nonzero(result.qc_flag))
        trace.parameters["multi_mineral_total_samples"] = int(well.n_samples)
        if not self._uses_vendor_benchmark:
            trace.parameters["mineral_selection"] = result.mineral_selection
        trace.parameters["interpretation_profile"] = self._interpretation_profile
        if write_lithology_code:
            trace.decisions.append(
                "No GeoLink or CSV lithology was available -> inferred LITHO from mineral volumes"
            )
        else:
            trace.decisions.append(
                "Input lithology was available -> mineral volumes computed, LITHO not inferred or overwritten"
            )
        if not self._uses_vendor_benchmark:
            trace.decisions.append(
                "Mineral volumes use measured responses and VCL; porosity is solved "
                "jointly or constrained by available core porosity"
            )
        trace.decisions.append("PEF screened for plausibility; DTC used when PEF is absent or invalid")
        if result.reduced_qc_fallback is not None and np.any(result.reduced_qc_fallback):
            trace.decisions.append(
                "Underdetermined samples used fixed PHIE with a reduced quartz-calcite inversion"
            )
        if self._uses_vendor_benchmark:
            trace.decisions.append(
                "Vendor benchmark profile retained the May 2026 equation set and closure gate"
            )
        trace.decisions.append(
            f"Multi-mineral solved {int(np.count_nonzero(result.solved))}/{well.n_samples} samples; "
            f"QC flagged {int(np.count_nonzero(result.qc_flag))}"
        )
        return trace

    def _step_rmf(self, well: "Well") -> StepResult:
        from petrophys.interpretation.electrical.rmf.rmf_module import RmfModule
        from petrophys.interpretation.electrical.rw.salinity_rw import rw_from_salinity
        from petrophys.interpretation.temperature import correct_resistivity_temperature

        trace = StepResult(step="rmf", success=True)
        decisions = []

        temp_profile = self._get_temperature_profile(well)
        reference_temp = self._reference_temperature(temp_profile)

        def _finalize_rmf(
            method: str,
            values: Union[float, np.ndarray],
            parameters: Dict[str, Any],
            output_curves: Optional[List[str]] = None,
        ) -> StepResult:
            rmf_curve = np.asarray(values, dtype=float)
            if rmf_curve.ndim == 0:
                rmf_curve = np.full(well.n_samples, float(rmf_curve), dtype=float)
            self._result.rmf_computed = rmf_curve
            well.add_curve("RMF", rmf_curve, "ohm.m", "Mud Filtrate Resistivity")
            trace.method = method
            trace.output_curves = output_curves or ["RMF"]
            trace.parameters = {
                **parameters,
                "formation_temp_min_f": float(np.nanmin(temp_profile)),
                "formation_temp_max_f": float(np.nanmax(temp_profile)),
                "rmf_min": float(np.nanmin(rmf_curve)),
                "rmf_max": float(np.nanmax(rmf_curve)),
            }
            trace.decisions = decisions
            return trace

        if self.config.mud_type == "OBM":
            decisions.append("OBM detected -> Rmf set to non-conductive default and written as a constant curve")
            return _finalize_rmf(
                "OBM_default",
                np.full(well.n_samples, 200.0, dtype=float),
                {"rmf": 200.0, "mud_type": "OBM"},
            )

        rmf_meas_temp_f = (
            self.config.rmf_meas_temp_f
            if self.config.rmf_meas_temp_f is not None
            else DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
        )
        if rmf_meas_temp_f == DEFAULT_RESISTIVITY_REFERENCE_TEMP_F and well.header.rm_temp is not None:
            rmf_meas_temp_f = float(well.header.rm_temp)
        metadata_rmf_temp_f = _parameter_temperature_f(well, "RMFT")
        if metadata_rmf_temp_f is not None and rmf_meas_temp_f == DEFAULT_RESISTIVITY_REFERENCE_TEMP_F:
            rmf_meas_temp_f = metadata_rmf_temp_f

        las_rmf = well.header.rmf
        if las_rmf is None:
            las_rmf = _parameter_float(well, "RMF")
        if las_rmf is None:
            las_rmf = _constant_zoned_parameter(well, "RMF")

        rm_meas_temp_f = (
            self.config.rm_meas_temp_f
            if self.config.rm_meas_temp_f is not None
            else DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
        )
        if rm_meas_temp_f == DEFAULT_RESISTIVITY_REFERENCE_TEMP_F and well.header.rm_temp is not None:
            rm_meas_temp_f = float(well.header.rm_temp)
        metadata_rm_temp_f = _parameter_temperature_f(well, "RMT")
        if metadata_rm_temp_f is not None and rm_meas_temp_f == DEFAULT_RESISTIVITY_REFERENCE_TEMP_F:
            rm_meas_temp_f = metadata_rm_temp_f

        las_rm = well.header.rm
        if las_rm is None:
            las_rm = _parameter_float(well, "RM")
        if las_rm is None:
            las_rm = _constant_zoned_parameter(well, "RM")

        if self.config.rmf_at_meas_temp is not None:
            rmf_module = RmfModule()
            result = rmf_module.calculate(
                rmf_surface=self.config.rmf_at_meas_temp,
                temp_measured=rmf_meas_temp_f,
                temp_formation=temp_profile
            )
            decisions.append(f"Rmf corrected from {self.config.rmf_at_meas_temp:.3f} ohm.m "
                           f"@ {rmf_meas_temp_f:.0f}F across the temperature profile")
            return _finalize_rmf(
                "arps_temperature_profile_correction",
                result.rmf,
                {
                    "rmf_at_surface": self.config.rmf_at_meas_temp,
                    "measured_temp_f": rmf_meas_temp_f,
                },
            )

        if las_rmf is not None:
            rmf_module = RmfModule()
            if metadata_rmf_temp_f is not None or well.header.rm_temp is not None:
                result = rmf_module.calculate(
                    rmf_surface=float(las_rmf),
                    temp_measured=rmf_meas_temp_f,
                    temp_formation=temp_profile,
                )
                decisions.append(
                    f"Rmf corrected from LAS metadata: {float(las_rmf):.4f} ohm.m "
                    f"@ {rmf_meas_temp_f:.0f}F across the temperature profile"
                )
                return _finalize_rmf(
                    "las_metadata_profile_temperature_correction",
                    result.rmf,
                    {
                        "rmf_at_measured_temp": float(las_rmf),
                        "measured_temp_f": rmf_meas_temp_f,
                    },
                )
            else:
                assumed_temp_f = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
                result = rmf_module.calculate(
                    rmf_surface=float(las_rmf),
                    temp_measured=assumed_temp_f,
                    temp_formation=temp_profile,
                )
                decisions.append(
                    f"Rmf from LAS metadata used assumed reference temperature {assumed_temp_f:.1f}F and was corrected along the temperature profile"
                )
                return _finalize_rmf(
                    "las_metadata_assumed_reference_temperature",
                    result.rmf,
                    {
                        "rmf_at_reference_temp": float(las_rmf),
                        "reference_temp_f": assumed_temp_f,
                    },
                )

        if self.config.rmf_salinity_ppm is not None:
            rmf_at_meas_temp = float(rw_from_salinity(self.config.rmf_salinity_ppm, rmf_meas_temp_f))
            rmf_module = RmfModule()
            result = rmf_module.calculate(
                rmf_surface=rmf_at_meas_temp,
                temp_measured=rmf_meas_temp_f,
                temp_formation=temp_profile,
            )
            decisions.append(
                f"Rmf from CLFL salinity: {self.config.rmf_salinity_ppm:.0f} ppm -> "
                f"{rmf_at_meas_temp:.4f} ohm.m @ {rmf_meas_temp_f:.0f}F and then along the temperature profile"
            )
            return _finalize_rmf(
                "mud_filtrate_salinity_conversion",
                result.rmf,
                {
                    "mud_filtrate_salinity_ppm": self.config.rmf_salinity_ppm,
                    "rmf_at_measured_temp": rmf_at_meas_temp,
                    "measured_temp_f": rmf_meas_temp_f,
                },
            )
        rm_input = self.config.rm if self.config.rm is not None else las_rm
        if rm_input is not None:
            k = 0.75 if self.config.mud_water == "fresh" else 0.85
            rm_curve = correct_resistivity_temperature(
                float(rm_input),
                rm_meas_temp_f,
                temp_profile,
                "F",
            )
            rmf_curve = np.asarray(rm_curve, dtype=float) * k
            well.add_curve("RM", rm_curve, "ohm.m", "Mud Resistivity")
            rm_source = "config.rm" if self.config.rm is not None else "LAS metadata"
            decisions.append(
                f"Rm from {rm_source} corrected from {float(rm_input):.4f} ohm.m "
                f"@ {rm_meas_temp_f:.0f}F across the temperature profile; Rmf = Rm(T) * {k}"
            )
            return _finalize_rmf(
                f"rm_to_rmf_conversion_factor_{k}",
                rmf_curve,
                {
                    "rm_at_measured_temp": float(rm_input),
                    "rm_measured_temp_f": rm_meas_temp_f,
                    "rm_min": float(np.nanmin(rm_curve)),
                    "rm_max": float(np.nanmax(rm_curve)),
                    "conversion_factor": k,
                    "mud_water": self.config.mud_water,
                    "source_priority": rm_source,
                },
                ["RM", "RMF"],
            )

        trace.decisions = decisions
        return trace

    def _step_rw(self, well: "Well") -> StepResult:
        from petrophys.interpretation.electrical.rw.rw_module import RwModule
        from petrophys.interpretation.electrical.rw.salinity_rw import rw_from_salinity
        from petrophys.interpretation.electrical.rw.sp_rw import estimate_sp_lines, rw_from_sp, rw_from_rxo_rt
        from petrophys.interpretation.electrical.rw.temp_correction import correct_rw_temperature

        trace = StepResult(step="rw", success=True)
        decisions = []
        temp_profile = self._get_temperature_profile(well)
        reference_temp = self._reference_temperature(temp_profile)
        las_rw_temp_f = _parameter_temperature_f(well, "RWT")

        def _finalize_rw(
            method: str,
            value: Union[float, np.ndarray],
            parameters: Dict[str, Any],
            inputs: Optional[List[str]] = None,
        ) -> StepResult:
            rw_curve = np.asarray(value, dtype=float)
            if rw_curve.ndim == 0:
                rw_curve = np.full(well.n_samples, float(rw_curve), dtype=float)
            self._result.rw_computed = rw_curve
            well.add_curve("RW", rw_curve, "ohm.m", "Formation Water Resistivity")
            trace.method = method
            trace.parameters = {
                **parameters,
                "rw_min": float(np.nanmin(rw_curve)),
                "rw_max": float(np.nanmax(rw_curve)),
                "formation_temp_min_f": float(np.nanmin(temp_profile)),
                "formation_temp_max_f": float(np.nanmax(temp_profile)),
            }
            trace.input_curves = inputs or []
            trace.output_curves = ["RW"]
            trace.decisions = decisions
            return trace

        if self.config.rw is not None:
            if self.config.rw_meas_temp_f is not None:
                rw_curve = correct_rw_temperature(self.config.rw, self.config.rw_meas_temp_f, temp_profile, "F")
                decisions.append(
                    f"Rw directly corrected from {self.config.rw:.4f} ohm.m @ {self.config.rw_meas_temp_f:.0f}F "
                    f"with Arps along the temperature profile"
                )
                return _finalize_rw(
                    "user_provided_temperature_profile_corrected",
                    rw_curve,
                    {
                        "rw_at_measured_temp": self.config.rw,
                        "measured_temp_f": self.config.rw_meas_temp_f,
                        "source_priority": "config.rw",
                    },
                )
            assumed_temp_f = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
            rw_curve = correct_rw_temperature(self.config.rw, assumed_temp_f, temp_profile, "F")
            decisions.append(
                f"Rw input {self.config.rw:.4f} ohm.m used assumed reference temperature {assumed_temp_f:.1f}F "
                "and was directly corrected with Arps along the temperature profile"
            )
            return _finalize_rw(
                "user_provided_assumed_reference_temperature",
                rw_curve,
                {
                    "rw": float(self.config.rw),
                    "reference_temp_f": assumed_temp_f,
                    "source_priority": "config.rw",
                },
            )

        las_rw = _parameter_float(well, "RW")
        if las_rw is None:
            las_rw = _constant_zoned_parameter(well, "RW")
        if las_rw is not None:
            if las_rw_temp_f is not None:
                rw_curve = correct_rw_temperature(float(las_rw), las_rw_temp_f, temp_profile, "F")
                decisions.append(
                    f"Rw directly corrected from LAS metadata: {float(las_rw):.4f} ohm.m @ {las_rw_temp_f:.0f}F "
                    f"with Arps along the temperature profile"
                )
                return _finalize_rw(
                    "las_metadata_temperature_profile_corrected",
                    rw_curve,
                    {
                        "rw_at_measured_temp": float(las_rw),
                        "measured_temp_f": las_rw_temp_f,
                        "source_priority": "las.RW",
                    },
                )
            assumed_temp_f = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
            rw_curve = correct_rw_temperature(float(las_rw), assumed_temp_f, temp_profile, "F")
            decisions.append(
                f"Rw from LAS metadata used assumed reference temperature {assumed_temp_f:.1f}F "
                "and was directly corrected with Arps along the temperature profile"
            )
            return _finalize_rw(
                "las_metadata_assumed_reference_temperature",
                rw_curve,
                {
                    "rw": float(las_rw),
                    "reference_temp_f": assumed_temp_f,
                    "source_priority": "las.RW",
                },
            )

        if self.config.salinity_ppm is not None:
            rw_curve = rw_from_salinity(self.config.salinity_ppm, temp_profile)
            decisions.append(
                f"Rw from salinity {self.config.salinity_ppm:.0f} ppm across the temperature profile"
            )
            return _finalize_rw(
                "salinity_conversion",
                rw_curve,
                {
                    "salinity_ppm": self.config.salinity_ppm,
                    "source_priority": "salinity_ppm",
                },
            )

        if self._result.rmf_computed is not None:
            rmf_curve = np.asarray(self._result.rmf_computed, dtype=float)
            if rmf_curve.ndim == 0:
                rmf_curve = np.full(well.n_samples, float(rmf_curve), dtype=float)
            decisions.append(
                "Rw direct source unavailable -> using available RMF curve for heuristic estimation"
            )

            if self._has_curve(well, "SP"):
                if self.config.mud_has_clay_or_kcl:
                    decisions.append("SP method skipped because mud_has_clay_or_kcl=True")
                else:
                    try:
                        sp_module = RwModule(method="sp")
                        sp_curve = get_curve(well, ["SP"])
                        sp_data = np.asarray(sp_curve.data, dtype=float)
                        clean_mask = sp_module._build_clean_water_mask(well)
                        shale_mask = sp_module._build_shale_mask(well)
                        clean_sp = sp_data[clean_mask] if clean_mask is not None else sp_data
                        clean_sp = clean_sp[np.isfinite(clean_sp)]
                        if clean_sp.size > 0:
                            sp_clean, _ = estimate_sp_lines(clean_sp)
                        else:
                            sp_clean, _ = estimate_sp_lines(sp_data)

                        if shale_mask is not None:
                            shale_sp = sp_data[shale_mask]
                            shale_sp = shale_sp[np.isfinite(shale_sp)]
                            if shale_sp.size > 0:
                                sp_shale = float(np.nanmedian(shale_sp))
                            else:
                                _, sp_shale = estimate_sp_lines(sp_data)
                        else:
                            _, sp_shale = estimate_sp_lines(sp_data)

                        rw_curve = rw_from_sp(sp_clean, sp_shale, rmf_curve, temp_profile)
                        decisions.append(
                            f"Rw via SP clean-water method with SPclean={sp_clean:.2f} mV and SPshale={sp_shale:.2f} mV across the temperature profile"
                        )
                        return _finalize_rw(
                            "sp_clean_water",
                            rw_curve,
                            {
                                "rmf_min": float(np.nanmin(rmf_curve)),
                                "rmf_max": float(np.nanmax(rmf_curve)),
                                "sp_clean_mv": float(sp_clean),
                                "sp_shale_mv": float(sp_shale),
                                "ssp_mv": float(sp_clean - sp_shale),
                                "source_priority": "rmf -> SP",
                            },
                            ["SP"],
                        )
                    except Exception as exc:
                        message = f"Rw SP heuristic failed: {exc}"
                        self._result.warnings.append(message)
                        decisions.append(message)
            else:
                decisions.append("SP method skipped because SP curve is unavailable")

            try:
                from petrophys.interpretation.electrical.rw.sp_rw import rw_from_rxo_rt
                rxo = get_curve(well, ["RSHA", "RXO", "RESS"])
                rt = get_curve(well, ["RDEP", "RT"])
                if rxo is None or rt is None:
                    raise ValueError("Rxo/Rt curves unavailable")
                rxo_data = np.asarray(rxo.data, dtype=float)
                rt_data = np.asarray(rt.data, dtype=float)
                clean_mask = RwModule()._build_clean_water_mask(well)
                if clean_mask is not None:
                    valid = clean_mask & np.isfinite(rxo_data) & np.isfinite(rt_data) & np.isfinite(rmf_curve)
                else:
                    valid = np.isfinite(rxo_data) & np.isfinite(rt_data) & np.isfinite(rmf_curve)
                if not np.any(valid):
                    raise ValueError("No valid clean-water Rxo/Rt samples")
                rw_clean = rw_from_rxo_rt(rxo_data[valid], rt_data[valid], rmf_curve[valid])
                rw_reference_temp = float(np.nanmedian(temp_profile[valid]))
                rw_at_reference = float(np.nanmedian(correct_rw_temperature(rw_clean, temp_profile[valid], rw_reference_temp, "F")))
                rw_curve = correct_rw_temperature(rw_at_reference, rw_reference_temp, temp_profile, "F")
                decisions.append(
                    f"Rw via Rxo/Rt clean-water method using {int(np.count_nonzero(valid))} valid samples "
                    "and direct Arps temperature correction"
                )
                return _finalize_rw(
                    "rxo_rt_clean_water",
                    rw_curve,
                    {
                        "rmf_min": float(np.nanmin(rmf_curve)),
                        "rmf_max": float(np.nanmax(rmf_curve)),
                        "rw_at_reference_temp": rw_at_reference,
                        "rw_reference_temp_f": rw_reference_temp,
                        "n_points": int(np.count_nonzero(valid)),
                        "source_priority": "rmf -> rxo_rt",
                    },
                    ["RSHA", "RDEP"],
                )
            except Exception as exc:
                message = f"Rw log heuristic failed: {exc}"
                self._result.warnings.append(message)
                decisions.append(message)
        else:
            decisions.append("Rw heuristic methods skipped because Rmf is unavailable")

        rt_curve = get_curve(well, ["RDEP", "RT"])
        phie_curve = get_curve(well, ["PHIE_CC", "PHIE"])
        vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        if rt_curve is not None and phie_curve is not None and vcl_curve is not None:
            rdep = np.asarray(rt_curve.data, dtype=float)
            phie = np.asarray(phie_curve.data, dtype=float)
            vcl = np.asarray(vcl_curve.data, dtype=float)
            lithology = self._build_lithology_log(well)
            archie_a, archie_m, _, archie_summary = self._archie_coefficients_for_lithology(
                lithology,
                well.n_samples,
            )
            # Exclude washed-out hole: there RDEP/PHIE are unreliable and would
            # corrupt Rwa.
            bad_hole, _, _ = get_curve_data_depthwise(well, ["BAD_HOLE"])
            good_hole = (
                ~(np.asarray(bad_hole, dtype=float) > 0.5)
                if bad_hole is not None else np.ones_like(rdep, dtype=bool)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                rwa = rdep * np.power(phie, archie_m) / archie_a
            # Prefer a tight cleanliness cut (cleaner Rwa); relax if too sparse.
            base = np.isfinite(rdep) & np.isfinite(phie) & np.isfinite(vcl) & np.isfinite(rwa) & (rwa > 0)
            tight = base & good_hole & (vcl < 0.2) & (phie > 0.08)
            loose = base & (vcl < 0.3) & (phie > 0.05)
            if int(np.count_nonzero(tight)) >= 100:
                valid, vcl_cut, phie_cut = tight, 0.2, 0.08
            else:
                valid, vcl_cut, phie_cut = loose, 0.3, 0.05
            if np.any(valid):
                rwa_reference_temp = float(np.nanmedian(temp_profile[valid]))
                rwa_reference_values = correct_rw_temperature(rwa[valid], temp_profile[valid], rwa_reference_temp, "F")
                # Adaptive percentile from the meaning of Rwa = Rt*phi^m/a:
                # it equals Rw in 100% water and EXCEEDS Rw where hydrocarbons
                # are present. If the clean-sand Rwa spread is narrow the section
                # is essentially all-wet and its median estimates Rw; if it is
                # broad a hydrocarbon leg has inflated the upper part, so the low
                # tail (P10) estimates the water-leg Rw. Using a fixed low
                # percentile (P5) biases Rw low in all-water wells -> Archie then
                # false-flags pay; using the median biases it high in HC wells
                # -> Archie misses pay. The spread test resolves both.
                p5 = float(np.percentile(rwa_reference_values, 5))
                p10 = float(np.percentile(rwa_reference_values, 10))
                p50 = float(np.percentile(rwa_reference_values, 50))
                spread = p50 / p5 if p5 > 0 else np.inf
                if spread < 3.0:
                    rw_value, rwa_pct, regime = p50, 50.0, "narrow_spread_all_wet_median"
                else:
                    rw_value, rwa_pct, regime = p10, 10.0, "wide_spread_hc_leg_low_tail"
                rw_curve = correct_rw_temperature(rw_value, rwa_reference_temp, temp_profile, "F")
                decisions.append(
                    f"Rw from Rwa clean-wet fallback: adaptive P{rwa_pct:.0f} "
                    f"({regime}, P50/P5 spread={spread:.1f}) of "
                    f"{rt_curve.mnemonic} * {phie_curve.mnemonic}^m / a "
                    f"where VCL < {vcl_cut} and {phie_curve.mnemonic} > {phie_cut} "
                    f"(n={int(np.count_nonzero(valid))}), lithology-dependent Archie "
                    "coefficients, direct Arps temperature correction"
                )
                return _finalize_rw(
                    "rwa_clean_wet_adaptive",
                    rw_curve,
                    {
                        "rwa_percentile": rwa_pct,
                        "rwa_regime": regime,
                        "rwa_spread_p50_p5": float(spread),
                        "rwa_clean_wet_samples": int(np.count_nonzero(valid)),
                        "rwa_min": float(np.nanmin(rwa[valid])),
                        "rwa_max": float(np.nanmax(rwa[valid])),
                        "rw_from_rwa": rw_value,
                        "archie_a": _scalar_or_lithology_dependent(archie_a),
                        "archie_m": _scalar_or_lithology_dependent(archie_m),
                        "archie_coefficients": archie_summary,
                        "archie_lithology_source": self._lithology_source,
                        "rwa_reference_temp_f": rwa_reference_temp,
                        "vcl_cutoff": vcl_cut,
                        "phie_cutoff": phie_cut,
                        "source_priority": "Rwa clean-wet fallback",
                    },
                    [rt_curve.mnemonic, phie_curve.mnemonic, vcl_curve.mnemonic],
                )
            decisions.append("Rwa clean-wet fallback skipped because no valid clean-wet samples were available")
        else:
            decisions.append("Rwa clean-wet fallback skipped because RDEP/RT, effective porosity, or VCL is unavailable")

        if not self.config.allow_default_rw:
            message = (
                "Rw could not be determined from configuration, metadata, salinity, "
                "SP, Rxo/Rt, or clean-wet Rwa; saturation was not computed"
            )
            self._result.warnings.append(message)
            decisions.append(message)
            trace.method = "unavailable"
            trace.message = message
            trace.decisions = decisions
            return trace

        self._result.warnings.append(
            "Rw could not be determined. Using explicitly enabled default 0.05 ohm.m"
        )
        decisions.append(
            "All Rw priority sources failed; allow_default_rw=True enabled the 0.05 ohm.m fallback"
        )
        assumed_temp_f = DEFAULT_RESISTIVITY_REFERENCE_TEMP_F
        rw_curve = correct_rw_temperature(0.05, assumed_temp_f, temp_profile, "F")
        return _finalize_rw(
            "default_fallback_profile",
            rw_curve,
            {
                "rw_default": 0.05,
                "reference_temp_f": assumed_temp_f,
                "source_priority": "default",
            },
        )

    def _step_invasion(self, well: "Well") -> StepResult:
        from petrophys.interpretation.electrical.invasion.invasion_module import InvasionModule

        trace = StepResult(step="invasion", success=True, method="rxo_rt_ratio")
        invasion_module = InvasionModule()
        result = invasion_module.analyze(well)
        well.add_curve("INVASION_RATIO", result.ratio, "", "Invasion Ratio Rxo/Rt")
        trace.input_curves = [
            name for name in (
                result.statistics.get("rt_curve"),
                result.statistics.get("rxo_curve"),
            )
            if name
        ]
        trace.output_curves = ["INVASION_RATIO"]
        trace.parameters = dict(result.statistics)
        rejected = int(result.statistics.get("nan_samples") or 0)
        total = int(result.statistics.get("total_samples") or len(result.ratio))
        if result.statistics.get("fallback_proxy_used"):
            trace.decisions.append(
                "INVASION_RATIO fallback medium/deep proxy = RMED/RDEP because RXO/RSHA/RESS were unavailable or non-finite at those samples"
            )
        if rejected:
            trace.decisions.append(
                "INVASION_RATIO samples written as NaN where shallow/deep resistivity inputs were invalid, "
                f"RT/RDEP was below {result.statistics.get('rt_min_used', 0.2)}, "
                f"or RXO/RT exceeded {result.statistics.get('max_ratio_used', 100.0)} "
                f"({rejected} of {total} samples rejected)"
            )
        else:
            trace.decisions.append(f"INVASION_RATIO valid for all {total} samples")
        return trace

    def _step_hc_correction(self, well: "Well") -> StepResult:
        from petrophys.interpretation.porosity.hydrocarbon_correction.hc_correction_module import HcCorrectionModule
        from petrophys.model.constants import MatrixProperties

        trace = StepResult(step="hc_correction", success=True, method=f"{self.config.hc_type}_correction")
        decisions = []

        rho_ma = self.config.rho_ma
        rho_ma_source = "user"
        if rho_ma is None:
            try:
                rho_ma = MatrixProperties.get(self.config.matrix_type).density
                rho_ma_source = f"MatrixProperties[{self.config.matrix_type}]"
            except KeyError:
                rho_ma = 2.65
                rho_ma_source = "default"
        decisions.append(f"rho_ma = {rho_ma} g/cm3 (source: {rho_ma_source})")

        hc_module = HcCorrectionModule(
            hc_type=self.config.hc_type, rho_ma=rho_ma,
            rho_fl=self.config.rho_fl, phi_cl=self.config.phi_cl,
            rho_cl=self.config.rho_cl
        )
        result = hc_module.calculate(well)
        hc_module.add_to_well(well, result)

        trace.input_curves = ["NPHI", "RHOB"]
        trace.output_curves = ["NPHI_HC", "RHOB_HC"]
        trace.parameters = {"hc_type": self.config.hc_type, "rho_ma": rho_ma,
                           "rho_fl": self.config.rho_fl, "phi_cl": self.config.phi_cl,
                           "rho_cl": self.config.rho_cl}
        trace.decisions = decisions
        return trace

    def _resolve_matrix_arrays(self, well: "Well") -> Dict[str, Any]:
        from petrophys.model.constants import MatrixProperties

        try:
            fallback_props = MatrixProperties.get(self.config.matrix_type)
        except KeyError:
            fallback_props = MatrixProperties.get("sandstone")

        fallback_rho_ma = float(self.config.rho_ma if self.config.rho_ma is not None else fallback_props.density)
        fallback_dt_ma = float(self.config.dt_ma if self.config.dt_ma is not None else fallback_props.slowness)
        fallback_nphi_ma = float(self.config.nphi_ma if self.config.nphi_ma is not None else fallback_props.nphi)

        rho_ma = np.full(well.n_samples, fallback_rho_ma, dtype=float)
        dt_ma = np.full(well.n_samples, fallback_dt_ma, dtype=float)
        nphi_ma = np.full(well.n_samples, fallback_nphi_ma, dtype=float)

        matrix_endpoints: List[Dict[str, Any]] = []
        matrix_groups_used: List[str] = []
        matrix_mode = "constant"
        matrix_source = f"config_or_default[{self.config.matrix_type}]"

        lithology = self._build_lithology_log(well)
        if isinstance(lithology, np.ndarray):
            matrix_mode = "lithology_dependent"
            matrix_source = self._lithology_source
            valid_groups = _ordered_unique_values(
                group for group in lithology.tolist()
                if _coerce_lithology_group(group) is not None
            )
            for raw_group in valid_groups:
                group = _coerce_lithology_group(raw_group)
                matrix_key = LITHOLOGY_GROUP_TO_MATRIX.get(group or "", self.config.matrix_type)
                try:
                    props = MatrixProperties.get(matrix_key)
                except KeyError:
                    props = fallback_props
                    matrix_key = self.config.matrix_type

                mask = np.asarray([item == raw_group for item in lithology], dtype=bool)
                if not np.any(mask):
                    continue

                rho_ma[mask] = float(props.density)
                dt_ma[mask] = float(props.slowness)
                nphi_ma[mask] = float(props.nphi)
                matrix_groups_used.append(str(raw_group).upper())
                matrix_endpoints.append(
                    {
                        "lithology_group": str(raw_group).upper(),
                        "matrix_key": matrix_key,
                        "rho_ma": float(props.density),
                        "dt_ma": float(props.slowness),
                        "nphi_ma": float(props.nphi),
                    }
                )

        return {
            "rho_ma": rho_ma,
            "dt_ma": dt_ma,
            "nphi_ma": nphi_ma,
            "matrix_mode": matrix_mode,
            "matrix_source": matrix_source,
            "matrix_groups_used": matrix_groups_used,
            "matrix_endpoints": matrix_endpoints,
            "fallback_rho_ma": fallback_rho_ma,
            "fallback_dt_ma": fallback_dt_ma,
            "fallback_nphi_ma": fallback_nphi_ma,
        }

    def _resolve_phi_cl(
        self,
        well: "Well",
        vcl_curve,
        porosity_basis: Optional[np.ndarray] = None,
        porosity_basis_name: Optional[str] = None,
        porosity_basis_valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        explicit_phi_cl = _parameter_float(well, "PNSH")
        explicit_source = "LAS PNSH parameter"
        if explicit_phi_cl is None:
            explicit_phi_cl = _constant_zoned_parameter(well, "PNSH")
            explicit_source = "zoned LAS PNSH fallback"

        if explicit_phi_cl is not None:
            return {
                "phi_cl": float(explicit_phi_cl),
                "phi_cl_source": explicit_source,
                "phi_cl_basis_count": None,
                "phi_cl_basis": "explicit_parameter",
            }

        def shale_control_mask() -> tuple[np.ndarray, str]:
            shale_mask = np.zeros(well.n_samples, dtype=bool)
            lithology = self._build_lithology_log(well)
            if isinstance(lithology, np.ndarray):
                shale_mask = np.asarray([_is_shale_group(item) for item in lithology], dtype=bool)
            basis = "lithology_shale"
            if not np.any(shale_mask) and vcl_curve is not None:
                vcl = np.asarray(vcl_curve.data, dtype=float)
                shale_mask = np.isfinite(vcl) & (vcl > 0.9)
                basis = "vcl_gt_0.9"
            return shale_mask, basis

        fallback_reason = ""

        def from_porosity(
            values: Any,
            label: str,
            valid_source_mask: Optional[np.ndarray] = None,
        ) -> Optional[Dict[str, Any]]:
            nonlocal fallback_reason
            if values is None:
                return None
            porosity = np.asarray(values, dtype=float)
            shale_mask, basis = shale_control_mask()
            valid_mask = shale_mask & np.isfinite(porosity)
            if valid_source_mask is None:
                fallback_reason = "PHIT clay-porosity control requires neutron-density averaged PHIT samples"
                return None
            source_mask = np.asarray(valid_source_mask, dtype=bool)
            if source_mask.shape != valid_mask.shape:
                fallback_reason = "PHIT source mask was unavailable for clay-porosity control"
                return None
            valid_mask &= source_mask
            if not np.any(valid_mask):
                fallback_reason = "no shale-control PHIT samples were calculated from neutron-density averaging"
                return None
            phi_cl = float(np.clip(np.nanmean(porosity[valid_mask]), 0.0, 0.6))
            label_clean = str(label or "porosity").lower()
            source = f"{basis}_average_{label_clean}_nd"
            return {
                "phi_cl": phi_cl,
                "phi_cl_source": source,
                "phi_cl_basis_count": int(np.count_nonzero(valid_mask)),
                "phi_cl_basis": source,
                "phi_cl_fallback_reason": None,
            }

        preferred = from_porosity(
            porosity_basis,
            porosity_basis_name or "porosity",
            porosity_basis_valid_mask,
        )
        if preferred is not None:
            return preferred

        return {
            "phi_cl": float(self.config.phi_cl),
            "phi_cl_source": "config_or_default",
            "phi_cl_basis_count": None,
            "phi_cl_basis": "config_or_default",
            "phi_cl_fallback_reason": fallback_reason or "no usable neutron-density averaged PHIT shale-control samples",
        }

    def _step_porosity_refinement(self, well: "Well") -> StepResult:
        self._lithology_log = None
        self._lithology_source = "config"
        return self._step_porosity(well, refinement=True)

    def _step_porosity(self, well: "Well", refinement: bool = False) -> StepResult:
        from petrophys.interpretation.porosity.density_porosity import density_porosity
        from petrophys.interpretation.porosity.effective_porosity import effective_porosity
        from petrophys.interpretation.porosity.neutron_porosity import neutron_porosity
        from petrophys.interpretation.porosity.sonic_porosity import secondary_porosity, wyllie_time_average
        from petrophys.interpretation.porosity.total_porosity import total_porosity

        trace = StepResult(
            step="porosity_refinement" if refinement else "porosity",
            success=True,
            method="component_based_lithology_refined" if refinement else "component_based",
        )
        decisions = []
        inputs = []
        outputs = []

        nphi_curve = get_curve(well, ["NPHI"])
        rhob_curve = get_curve(well, ["RHOB"])
        dtc_curve = get_curve(well, ["DTC"])
        existing_phin_curve = get_curve(well, ["PHIN"])
        existing_phid_curve = get_curve(well, ["PHID"])
        existing_phis_curve = get_curve(well, ["PHIS"])
        existing_phit_curve = get_curve(well, ["PHIT"])
        existing_phie_curve = get_curve(well, ["PHIE"])
        vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        preserve_existing_phit = _is_prioritized_input_interpretation_curve(
            existing_phit_curve,
            "PHIT",
        )
        preserve_existing_phie = _is_prioritized_input_interpretation_curve(
            existing_phie_curve,
            "PHIE",
        )

        matrix_info = self._resolve_matrix_arrays(well)
        rho_ma = matrix_info["rho_ma"]
        dt_ma = matrix_info["dt_ma"]
        nphi_ma = matrix_info["nphi_ma"]
        if refinement:
            decisions.append(
                "Refinement pass after inferred LITHO: porosity curves recomputed using computed lithology matrix endpoints"
            )
        elif matrix_info["matrix_mode"] == "lithology_dependent":
            decisions.append(
                "Initial porosity pass used an available lithology log for matrix endpoints"
            )
        else:
            decisions.append(
                "Initial porosity pass used configured matrix endpoints; it will be refined if the workflow infers LITHO"
            )

        phin = None
        phid = None
        phis = None
        phit = None
        phie_basis = None
        phie_basis_name = None
        phit_method = None
        phie = None
        phit_source_counts: Dict[str, int] = {}
        phit_nd_average_mask: Optional[np.ndarray] = None

        if nphi_curve is not None:
            inputs.append("NPHI")
            phin = neutron_porosity(nphi_curve.data, nphi_matrix=nphi_ma)
            well.add_curve(
                "PHIN",
                phin,
                "v/v",
                "Neutron Porosity refined with computed lithology" if refinement else "Neutron Porosity",
            )
            outputs.append("PHIN")
            if matrix_info["matrix_mode"] == "lithology_dependent":
                decisions.append(
                    "PHIN = clip(NPHI - nphi_ma(depth), 0, 0.6) using lithology-specific matrix neutron endpoints"
                )
            else:
                decisions.append(
                    f"PHIN = clip(NPHI - nphi_ma, 0, 0.6) using nphi_ma={matrix_info['fallback_nphi_ma']:.3f}"
                )
        elif existing_phin_curve is not None:
            phin = existing_phin_curve.data
            inputs.append("PHIN")
            decisions.append("PHIN reused from existing well curve")

        if rhob_curve is not None:
            inputs.append("RHOB")
            phid = density_porosity(rhob_curve.data, rho_ma, self.config.rho_fl)
            well.add_curve(
                "PHID",
                phid,
                "v/v",
                "Density Porosity refined with computed lithology" if refinement else "Density Porosity",
            )
            outputs.append("PHID")
            if matrix_info["matrix_mode"] == "lithology_dependent":
                decisions.append("PHID = (rho_ma(depth) - RHOB) / (rho_ma(depth) - rho_fl)")
            else:
                decisions.append("PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)")
        elif existing_phid_curve is not None:
            phid = existing_phid_curve.data
            inputs.append("PHID")
            decisions.append("PHID reused from existing well curve")

        if dtc_curve is not None:
            inputs.append("DTC")
            phis = wyllie_time_average(dtc_curve.data, dt_ma, self.config.dt_fl)
            well.add_curve(
                "PHIS",
                phis,
                "v/v",
                "Sonic Porosity refined with computed lithology" if refinement else "Sonic Porosity",
            )
            outputs.append("PHIS")
            if matrix_info["matrix_mode"] == "lithology_dependent":
                decisions.append("PHIS from Wyllie time-average equation using dt_ma(depth)")
            else:
                decisions.append("PHIS from Wyllie time-average equation")
        elif existing_phis_curve is not None:
            phis = existing_phis_curve.data
            inputs.append("PHIS")
            decisions.append("PHIS reused from existing well curve")

        if phid is not None and phin is not None:
            phit = total_porosity(phin, phid)
            phie_basis = phit
            phie_basis_name = "PHIT"
            phit_method = "adaptive_nd"
            phin_arr = np.asarray(phin, dtype=float)
            phid_arr = np.asarray(phid, dtype=float)
            both = np.isfinite(phin_arr) & np.isfinite(phid_arr)
            density_missing_phin = ~np.isfinite(phin_arr) & np.isfinite(phid_arr)
            neutron_missing_phid = np.isfinite(phin_arr) & ~np.isfinite(phid_arr)
            density_gas = both & (phid_arr > phin_arr + 0.05)
            weighted_nd = both & ~density_gas
            phit_nd_average_mask = weighted_nd
            phit_arr = np.asarray(phit, dtype=float)
            phit_source_counts = {
                "density_from_missing_phin": int(np.count_nonzero(density_missing_phin)),
                "density_from_gas_crossover": int(np.count_nonzero(density_gas)),
                "weighted_nd": int(np.count_nonzero(weighted_nd)),
                "neutron_from_missing_phid": int(np.count_nonzero(neutron_missing_phid)),
                "nan": int(np.count_nonzero(~np.isfinite(phit_arr))),
            }
            decisions.append(
                "PHIT = adaptive(PHIN, PHID) with per-sample fallback: "
                + ", ".join(f"{name}={count}" for name, count in phit_source_counts.items())
            )
        elif phid is not None:
            phit = phid
            phie_basis = phit
            phie_basis_name = "PHIT"
            phit_method = "density_only"
            phit_source_counts = {
                "density_only": int(np.count_nonzero(np.isfinite(np.asarray(phit, dtype=float)))),
                "nan": int(np.count_nonzero(~np.isfinite(np.asarray(phit, dtype=float)))),
            }
            decisions.append("Density-only porosity: PHIT = PHID because PHIN was unavailable")
        elif phin is not None:
            phit = phin
            phie_basis = phit
            phie_basis_name = "PHIT"
            phit_method = "neutron_only"
            phit_source_counts = {
                "neutron_only": int(np.count_nonzero(np.isfinite(np.asarray(phit, dtype=float)))),
                "nan": int(np.count_nonzero(~np.isfinite(np.asarray(phit, dtype=float)))),
            }
            decisions.append("Neutron-only porosity: PHIT = PHIN because PHID was unavailable")
        elif phis is not None:
            phit = phis
            phie_basis = phit
            phie_basis_name = "PHIT"
            phit_method = "sonic_only"
            phit_source_counts = {
                "sonic_only": int(np.count_nonzero(np.isfinite(np.asarray(phit, dtype=float)))),
                "nan": int(np.count_nonzero(~np.isfinite(np.asarray(phit, dtype=float)))),
            }
            decisions.append("Sonic-only porosity: PHIT = PHIS because PHIN and PHID were unavailable")
        else:
            if existing_phit_curve is not None:
                phit = existing_phit_curve.data
                phie_basis = phit
                phie_basis_name = "PHIT"
                phit_method = "existing_phit"
                phit_source_counts = {
                    "existing_phit": int(np.count_nonzero(np.isfinite(np.asarray(phit, dtype=float)))),
                    "nan": int(np.count_nonzero(~np.isfinite(np.asarray(phit, dtype=float)))),
                }
                inputs.append("PHIT")
                decisions.append("PHIT reused from existing well curve")

        if phit is not None:
            apparent_info = _apparent_porosity_info(phit_method)
            phit_output = "PHIT_CALC" if preserve_existing_phit else "PHIT"
            if apparent_info is not None:
                phit_description = (
                    f"Apparent total porosity, {apparent_info['source']} "
                    f"from {apparent_info['basis']}"
                )
                if preserve_existing_phit:
                    phit_description += "; input PHIT preserved as active"
                elif refinement:
                    phit_description += " with computed lithology matrix endpoints"
            else:
                phit_description = (
                    "Calculated total porosity; input PHIT preserved as active"
                    if preserve_existing_phit
                    else (
                        "Total Porosity refined with computed lithology"
                        if refinement
                        else "Total Porosity"
                    )
                )
            phit_curve = well.add_curve(
                phit_output,
                phit,
                "v/v",
                phit_description,
                data_type=DataType.COMPUTED,
            )
            phit_curve.metadata.custom.update(
                {
                    "porosity_role": "total",
                    "porosity_method": phit_method,
                    "porosity_apparent": apparent_info is not None,
                    "porosity_basis": apparent_info["basis"] if apparent_info else phie_basis_name,
                    "porosity_source": apparent_info["source"] if apparent_info else "workflow_calculation",
                }
            )
            if phit_output not in outputs:
                outputs.append(phit_output)
            if apparent_info is not None:
                decisions.append(
                    f"{phit_output} written as apparent total porosity "
                    f"({apparent_info['source']}) because {apparent_info['reason']}"
                )
            if preserve_existing_phit:
                decisions.append(
                    "Existing interpreted total porosity curve preserved as active: "
                    + _prioritized_curve_label(existing_phit_curve)
                )
                decisions.append("Workflow total-porosity calculation written to PHIT_CALC")

        phi_cl_info = self._resolve_phi_cl(
            well,
            vcl_curve,
            porosity_basis=phie_basis,
            porosity_basis_name=phie_basis_name,
            porosity_basis_valid_mask=phit_nd_average_mask if phie_basis_name == "PHIT" else None,
        )
        phi_cl = float(phi_cl_info["phi_cl"])

        if phie_basis is not None:
            if vcl_curve is not None:
                inputs.append(vcl_curve.mnemonic)
                vcl = vcl_curve.data
            else:
                vcl = np.zeros(well.n_samples, dtype=float)

            phie = effective_porosity(phie_basis, vcl, phi_cl)
            phie_output = "PHIE_CALC" if preserve_existing_phie else "PHIE"
            apparent_info = _apparent_porosity_info(phit_method)
            if apparent_info is not None:
                correction_text = (
                    f"{vcl_curve.mnemonic} correction"
                    if vcl_curve is not None
                    else "no clay-curve correction"
                )
                phie_description = (
                    f"Apparent effective porosity from {apparent_info['source']} "
                    f"PHIT and {correction_text}"
                )
                if preserve_existing_phie:
                    phie_description += "; input PHIE preserved as active"
                elif refinement:
                    phie_description += " with computed lithology matrix endpoints"
            else:
                phie_description = (
                    "Calculated effective porosity; input PHIE preserved as active"
                    if preserve_existing_phie
                    else (
                        "Effective Porosity refined with computed lithology"
                        if refinement
                        else "Effective Porosity"
                    )
                )
            phie_curve = well.add_curve(
                phie_output,
                phie,
                "v/v",
                phie_description,
                data_type=DataType.COMPUTED,
            )
            phie_curve.metadata.custom.update(
                {
                    "porosity_role": "effective",
                    "porosity_method": phit_method,
                    "porosity_apparent": apparent_info is not None,
                    "porosity_basis": phie_basis_name,
                    "porosity_source": apparent_info["source"] if apparent_info else "workflow_calculation",
                }
            )
            outputs.append(phie_output)
            if apparent_info is not None:
                decisions.append(
                    f"{phie_output} written as apparent effective porosity from "
                    f"{apparent_info['source']} {phie_basis_name}"
                )
            if vcl_curve is not None:
                decisions.append(
                    f"{phie_output} = {phie_basis_name} - {vcl_curve.mnemonic} * phi_cl ({phi_cl:.3f})"
                )
            else:
                decisions.append(f"{phie_output} = {phie_basis_name} (no clay curve available)")
            if preserve_existing_phie:
                decisions.append(
                    "Existing interpreted effective porosity curve preserved as active: "
                    + _prioritized_curve_label(existing_phie_curve)
                )
                decisions.append("Workflow effective-porosity calculation written to PHIE_CALC")
            if phi_cl_info.get("phi_cl_source") == "config_or_default" and phi_cl_info.get("phi_cl_fallback_reason"):
                decisions.append(
                    "phi_cl fallback used because "
                    + str(phi_cl_info["phi_cl_fallback_reason"])
                )

        if phis is not None and phit is not None:
            phi_sec = secondary_porosity(phis, phit)
            well.add_curve("PHI_SEC", phi_sec, "v/v", "Secondary Porosity Index")
            outputs.append("PHI_SEC")
            decisions.append("PHI_SEC = PHIT - PHIS")

        trace.input_curves = inputs
        trace.output_curves = outputs
        trace.parameters = {
            "matrix_type": self.config.matrix_type,
            "matrix_mode": matrix_info["matrix_mode"],
            "matrix_source": matrix_info["matrix_source"],
            "porosity_pass": "refined_after_computed_lithology" if refinement else "initial",
            "uses_computed_lithology": matrix_info["matrix_source"] == "computed_lithology_curve",
            "matrix_groups_used": matrix_info["matrix_groups_used"],
            "matrix_endpoints": matrix_info["matrix_endpoints"],
            "nphi_ma": matrix_info["fallback_nphi_ma"] if matrix_info["matrix_mode"] == "constant" else None,
            "nphi_ma_min": float(np.nanmin(nphi_ma)),
            "nphi_ma_max": float(np.nanmax(nphi_ma)),
            "rho_ma": matrix_info["fallback_rho_ma"] if matrix_info["matrix_mode"] == "constant" else None,
            "rho_ma_min": float(np.nanmin(rho_ma)),
            "rho_ma_max": float(np.nanmax(rho_ma)),
            "rho_fl": self.config.rho_fl,
            "dt_ma": matrix_info["fallback_dt_ma"] if matrix_info["matrix_mode"] == "constant" else None,
            "dt_ma_min": float(np.nanmin(dt_ma)),
            "dt_ma_max": float(np.nanmax(dt_ma)),
            "dt_fl": self.config.dt_fl,
            "phi_cl": phi_cl,
            "phi_cl_source": phi_cl_info["phi_cl_source"],
            "phi_cl_basis": phi_cl_info["phi_cl_basis"],
            "phi_cl_basis_count": phi_cl_info["phi_cl_basis_count"],
            "phi_cl_fallback_reason": phi_cl_info.get("phi_cl_fallback_reason"),
            "phie_basis": phie_basis_name,
            "phit_method": phit_method,
            "phit_source_counts": phit_source_counts,
            "phit_nd_average_samples": int(np.count_nonzero(phit_nd_average_mask)) if phit_nd_average_mask is not None else 0,
            "active_phit_source": "input_interpretation" if preserve_existing_phit else "workflow_calculation",
            "active_phie_source": "input_interpretation" if preserve_existing_phie else "workflow_calculation",
            "computed_phit_curve": "PHIT_CALC" if preserve_existing_phit else "PHIT",
            "computed_phie_curve": "PHIE_CALC" if preserve_existing_phie else "PHIE",
            "phit_apparent": _apparent_porosity_info(phit_method) is not None,
            "phie_apparent": _apparent_porosity_info(phit_method) is not None and phie is not None,
            "apparent_porosity_source": (
                _apparent_porosity_info(phit_method)["source"]
                if _apparent_porosity_info(phit_method) is not None
                else None
            ),
        }
        trace.decisions = decisions
        self._last_phit_nd_average_mask = phit_nd_average_mask
        self._last_phi_cl_info = phi_cl_info
        return trace

    def _step_core_correction(self, well: "Well") -> StepResult:
        """Fit same-well core correction before saturation uses porosity."""
        from petrophys.interpretation.permeability.perm_module import PermeabilityModule

        trace = StepResult(step="core_correction", success=True, method="same_well_core_correction")
        decisions: list[str] = []
        outputs: list[str] = []

        stale_core_curves = [name for name in ("PHIE_CC", "PERM_CC") if well.get(name) is not None]
        for curve_name in stale_core_curves:
            well.remove_curve(curve_name)
        if stale_core_curves:
            decisions.append(
                "Existing core-corrected curves cleared before pre-saturation core correction"
            )

        perm_module = PermeabilityModule(
            method=self.config.perm_method,
            fallback_method=self.config.perm_fallback_method,
            use_peer_core=self.config.use_peer_core,
        )
        result = perm_module.calculate_core_correction(
            well,
            peer_wells=self.config.peer_wells,
            formation_tops=self.config.formation_tops,
        )

        phi_corecal = result.metadata.get("phi_corecal")
        perm_core = result.metadata.get("perm_core")
        if phi_corecal is not None:
            well.add_curve(
                "PHIE_CC",
                phi_corecal,
                "v/v",
                "Core-corrected effective porosity",
            )
            outputs.append("PHIE_CC")
            decisions.append("Core-corrected porosity stored as PHIE_CC before saturation")
        if perm_core is not None:
            well.add_curve(
                "PERM_CC",
                perm_core,
                "mD",
                "Permeability from core-calibrated poro-perm transform",
            )
            outputs.append("PERM_CC")
            decisions.append("Core-corrected permeability stored as PERM_CC")

        trace.input_curves = [result.metadata.get("porosity_curve", "PHIE")]
        trace.output_curves = outputs
        trace.parameters = {
            "strategy": result.metadata.get("strategy"),
            "same_well_core_models": result.metadata.get("same_well_core_models"),
            "peer_core_models": result.metadata.get("peer_core_models"),
            "peer_wells_considered": result.metadata.get("peer_wells_considered"),
            "peer_core_enabled": result.metadata.get("peer_core_enabled"),
            "core_min_points": result.metadata.get("core_min_points"),
            "core_models": result.metadata.get("core_models"),
            "core_rejected_models": result.metadata.get("core_rejected_models"),
            "core_sections_count": result.metadata.get("core_sections_count"),
            "perm_max_clip_md": result.metadata.get("perm_max_clip_md"),
        }
        if not outputs:
            decisions.append("No PHIE_CC/PERM_CC written because no same-well core model was fitted")
        for model in result.metadata.get("core_models", []):
            decisions.append(
                "Core model fitted "
                f"(source={model.get('source_type')}, n={model.get('n_core_points')}, "
                f"core_phi={model.get('core_porosity_curve') or 'unknown'}, "
                f"core_perm={model.get('core_permeability_curve') or 'unknown'})"
            )
        trace.decisions = decisions
        return trace

    def _step_saturation(self, well: "Well") -> StepResult:
        from petrophys.interpretation.saturation.sw_module import SaturationModule

        trace = StepResult(step="saturation", success=True)
        lithology = self._build_lithology_log(well)
        a_value, m_value, n_value, archie_summary = self._archie_coefficients_for_lithology(
            lithology,
            well.n_samples,
        )

        vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        phit_curve_for_phi_cl = get_curve(well, ["PHIT"])
        phi_cl_info = self._resolve_phi_cl(
            well,
            vcl_curve,
            porosity_basis=phit_curve_for_phi_cl.data if phit_curve_for_phi_cl is not None else None,
            porosity_basis_name="PHIT" if phit_curve_for_phi_cl is not None else None,
            porosity_basis_valid_mask=(
                self._last_phit_nd_average_mask
                if phit_curve_for_phi_cl is not None
                and getattr(self, "_last_phit_nd_average_mask", None) is not None
                else None
            ),
        )
        phi_cl = float(phi_cl_info["phi_cl"])

        saturation_module_class = SaturationModule
        if self._uses_vendor_benchmark:
            from petrophys.workflow._vendor_benchmark import (
                May2026BenchmarkSaturationModule,
            )

            saturation_module_class = May2026BenchmarkSaturationModule

        sw_module = saturation_module_class(
            a=a_value, m=m_value, n=n_value,
            model=self.config.sw_model, vcl_cutoff=self._resolved_cutoff("vcl_cutoff")
        )

        result = sw_module.calculate(
            well, rw=self._result.rw_computed, rcl=self._result.rcl_computed,
            salinity=self.config.salinity_ppm, rmf=self._result.rmf_computed,
            lithology=lithology, cec=self.config.cec, b_coeff=self.config.b_coeff,
            rho_ma=self.config.rho_ma or 2.65, m_star=self.config.m_star,
            rwb=self.config.rwb, phi_cl=phi_cl,
            rtsh=self.config.rtsh, vsh_exp=self.config.vsh_exp,
            shale_exp=self.config.shale_exp,
        )

        computed_sw = np.asarray(result.sw, dtype=float)
        if self._uses_vendor_benchmark:
            source_sw, source_sw_source, source_sw_meta = None, None, {}
        else:
            source_sw, source_sw_source, source_sw_meta = get_curve_data_depthwise(
                well, ["SW_INPUT", "SW_RAW", "SW_SOURCE"]
            )
        source_sw_mask = (
            np.isfinite(source_sw)
            & (source_sw >= 0.0)
            & (source_sw <= 1.0)
            if source_sw is not None
            else np.zeros(well.n_samples, dtype=bool)
        )
        final_sw = computed_sw.copy()
        if source_sw is not None:
            final_sw[source_sw_mask] = source_sw[source_sw_mask]

        saturation_unit = "v/v" if self._uses_vendor_benchmark else "frac"
        saturation_description = (
            f"Water Saturation ({result.model})"
            if self._uses_vendor_benchmark
            else f"Water Saturation ({result.model}; source input priority)"
        )
        well.add_curve("SW", final_sw, saturation_unit, saturation_description)
        outputs = ["SW"]
        if result.sxo is not None:
            well.add_curve("SXO", result.sxo, saturation_unit, "Flushed Zone Water Saturation")
            outputs.append("SXO")

        trace.method = result.model
        _, rt_source, _ = get_curve_data_depthwise(well, ["RDEP", "RT"])
        _, phie_source, _ = get_curve_data_depthwise(well, ["PHIE_CC", "PHIE", "PHIT"])
        _, vcl_source, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY"])
        trace.input_curves = [
            source for source in (source_sw_source, rt_source, phie_source, vcl_source) if source
        ]
        trace.output_curves = outputs
        trace.parameters = {
            "sw_model": result.model,
            "sw_model_requested": self.config.sw_model,
            "sw_model_sample_counts": result.model_counts or {},
            "a": _scalar_or_lithology_dependent(a_value),
            "m": _scalar_or_lithology_dependent(m_value),
            "n": _scalar_or_lithology_dependent(n_value),
            "archie_coefficients": archie_summary,
            "archie_lithology_source": self._lithology_source,
            "archie_a_source": "config_or_las" if self.config.a is not None else "ArchieParameterDefaults",
            "archie_m_source": "config_or_las" if self.config.m is not None else "ArchieParameterDefaults",
            "archie_n_source": "config_or_las" if self.config.n is not None else "ArchieParameterDefaults",
            "rw_min": (
                float(np.nanmin(np.asarray(self._result.rw_computed, dtype=float)))
                if self._result.rw_computed is not None
                else None
            ),
            "rw_max": (
                float(np.nanmax(np.asarray(self._result.rw_computed, dtype=float)))
                if self._result.rw_computed is not None
                else None
            ),
            "rcl": self._result.rcl_computed,
            "salinity_ppm": self.config.salinity_ppm,
            "cec": self.config.cec,
            "cec_present": self.config.cec is not None,
            "b_coeff": self.config.b_coeff,
            "b_coeff_present": self.config.b_coeff is not None,
            "m_star": self.config.m_star,
            "rwb": self.config.rwb,
            "phi_cl": phi_cl,
            "phi_cl_source": phi_cl_info["phi_cl_source"],
            "phi_cl_basis": phi_cl_info["phi_cl_basis"],
            "phi_cl_basis_count": phi_cl_info["phi_cl_basis_count"],
            "phi_cl_fallback_reason": phi_cl_info.get("phi_cl_fallback_reason"),
            "rtsh": self.config.rtsh if self.config.rtsh is not None else self._result.rcl_computed,
            "vsh_exp": self.config.vsh_exp,
            "shale_exp": self.config.shale_exp,
            "vcl_cutoff_for_routing": self._resolved_cutoff("vcl_cutoff"),
            "lithology_type": self.config.lithology if isinstance(lithology, str) else "per-depth array",
            "sxo_method": result.sxo_method,
            "sxo_reason": result.sxo_reason,
            "sxo_method_counts": result.sxo_method_counts or {},
            "sxo_input_sources": result.sxo_input_sources or {},
            "source_sw_curve": source_sw_source,
            "source_sw_samples_used": int(np.count_nonzero(source_sw_mask)),
            "archie_sw_samples_used": int(np.count_nonzero(~source_sw_mask & np.isfinite(computed_sw))),
            "source_sw_input_sources": source_sw_meta.get("source_counts", {}) if source_sw_meta else {},
            "interpretation_profile": self._interpretation_profile,
        }
        if self._result.rmf_computed is not None:
            trace.parameters["rmf_min"] = float(np.nanmin(np.asarray(self._result.rmf_computed, dtype=float)))
            trace.parameters["rmf_max"] = float(np.nanmax(np.asarray(self._result.rmf_computed, dtype=float)))
        trace.decisions.append(f"Model selected: {result.model}")
        if self._uses_vendor_benchmark:
            trace.decisions.append(
                "Vendor benchmark profile retained the May 2026 Indonesia and Dual Water equations"
            )
        if self.config.sw_model == "auto":
            trace.decisions.append("Auto-routing: depth-by-depth model selection based on VCLAY and lithology")
            if result.model_counts:
                trace.decisions.append(
                    "Depth routing counts: "
                    + ", ".join(f"{model}={count}" for model, count in result.model_counts.items())
                )
        trace.decisions.append(
            "Archie coefficients from lithology"
            + (f" ({self._lithology_source})" if self._lithology_source else "")
        )
        if np.any(source_sw_mask):
            trace.decisions.append(
                f"SW uses source input {source_sw_source} for {int(np.count_nonzero(source_sw_mask))} finite samples; "
                "Archie fills remaining samples where inputs are finite"
            )
        else:
            trace.decisions.append("No finite source SW input was available; SW is Archie-computed where inputs are finite")
        trace.decisions.append(
            f"Dual Water phi_cl = {phi_cl:.3f} from {phi_cl_info['phi_cl_source']}"
        )
        if phi_cl_info.get("phi_cl_source") == "config_or_default" and phi_cl_info.get("phi_cl_fallback_reason"):
            trace.decisions.append(
                "Dual Water phi_cl fallback used because "
                + str(phi_cl_info["phi_cl_fallback_reason"])
            )
        if trace.parameters["rw_min"] is not None and trace.parameters["rw_max"] is not None:
            trace.decisions.append(
                f"Rw range = {trace.parameters['rw_min']:.4f} to {trace.parameters['rw_max']:.4f}, Rcl = {self._result.rcl_computed}"
            )
        else:
            trace.decisions.append(
                f"Rw unavailable for SW; Rcl = {self._result.rcl_computed}"
            )
        if result.sxo_method == "archie_rxo_rmf":
            trace.decisions.append("SXO from Archie invaded-zone equation using RSHA/RXO/RESS and RMF")
        elif result.sxo_method == "archie_rxo_rmf+fifth_root_quicklook":
            counts = result.sxo_method_counts or {}
            count_text = (
                " ("
                + ", ".join(f"{method}={count}" for method, count in counts.items())
                + ")"
                if counts else ""
            )
            trace.decisions.append(
                "SXO from invaded-zone Archie where RSHA/RXO/RESS and RMF were finite; "
                "SXO = SW^(1/5) elsewhere"
                + count_text
            )
        elif result.sxo_method == "fifth_root_quicklook":
            trace.decisions.append(
                f"SXO fallback: SXO = SW^(1/5) because {result.sxo_reason}"
            )
        if np.any(np.isfinite(final_sw)):
            trace.decisions.append(f"Sw range: {np.nanmin(final_sw):.3f} - {np.nanmax(final_sw):.3f}")
        else:
            trace.decisions.append("Sw range: all values are NaN")
        return trace

    def _step_permeability(self, well: "Well") -> StepResult:
        from petrophys.interpretation.permeability.perm_module import PermeabilityModule

        trace = StepResult(step="permeability", success=True, method=self.config.perm_method)
        decisions = []
        resflag_info = None
        perm_method_norm = str(self.config.perm_method or "").strip().lower().replace("-", "_")
        perm_method_uses_swirr = perm_method_norm not in {"lith_vcl", "lithology_vcl", "dunite", "dunite_basic"}
        if (
            perm_method_uses_swirr
            and self.config.swirr is None
            and not self._has_curve(well, "SWIRR", "SWI", "SWIR")
            and not self._has_nmr_swirr_route(well)
        ):
            resflag_info = self._ensure_resflag_for_swirr(well)
        outputs = []

        if resflag_info is not None:
            if resflag_info.get("created"):
                outputs.append("RESFLAG")
                if resflag_info.get("source_curve") == "NET_RES":
                    decisions.append(
                        "RESFLAG copied from existing NET_RES before permeability "
                        f"({resflag_info['reservoir_samples']} reservoir samples)"
                    )
                else:
                    decisions.append(
                        "RESFLAG created before permeability from NetPayModule net_reservoir "
                        f"using {resflag_info['vcl_curve']}, {resflag_info['phie_curve']}, "
                        f"and {resflag_info['sw_curve']} ({resflag_info['reservoir_samples']} samples)"
                    )
            else:
                decisions.append(
                    f"{resflag_info['source_curve']} already present before permeability "
                    "and available for Swirr routing"
                )

        stale_core_curves = [name for name in ("PHIE_CC", "PERM_CC") if well.get(name) is not None]
        for curve_name in stale_core_curves:
            well.remove_curve(curve_name)
        if stale_core_curves:
            decisions.append(
                "Existing core-corrected curves cleared before permeability; "
                "PHIE_CC/PERM_CC are rewritten only when the current same-well core model is fitted"
            )

        perm_module = PermeabilityModule(
            method=self.config.perm_method,
            fallback_method=self.config.perm_fallback_method,
            use_peer_core=self.config.use_peer_core,
        )
        lithology = None
        if not self._has_curve(well, "LITHOLOGY_GEOLINK", "LITHO_CLASS", "LITHO"):
            lithology = self._build_lithology_log(well)

        result = perm_module.calculate(
            well,
            swi=self.config.swirr,
            lithology=lithology,
            peer_wells=self.config.peer_wells,
            formation_tops=self.config.formation_tops,
        )
        empirical_perm = result.metadata.get("empirical_perm")
        empirical_method = result.metadata.get("empirical_method") or self.config.perm_fallback_method
        perm_output = empirical_perm if empirical_perm is not None else result.perm
        perm_desc = empirical_method if empirical_perm is not None else result.method
        well.add_curve("PERM", perm_output, "mD", f"Permeability ({perm_desc})")

        outputs.append("PERM")
        phi_corecal = result.metadata.get("phi_corecal")
        perm_core = result.metadata.get("perm_core")
        if phi_corecal is not None:
            well.add_curve(
                "PHIE_CC",
                phi_corecal,
                "v/v",
                "Permeability input porosity from core calibration",
            )
            outputs.append("PHIE_CC")
        if perm_core is not None:
            well.add_curve(
                "PERM_CC",
                perm_core,
                "mD",
                "Permeability from core-calibrated poro-perm transform",
            )
            outputs.append("PERM_CC")
        swirr_metadata = dict(result.swi_metadata)
        swirr_app = swirr_metadata.pop("swirr_app", None)
        if swirr_app is not None and empirical_method != "lith_vcl":
            well.add_curve(
                "SWIRR_APP",
                swirr_app,
                "v/v",
                "Apparent irreducible water saturation from hydrocarbon-mask P5 method",
            )
            outputs.append("SWIRR_APP")
        method_metadata = result.metadata.get("method_metadata") or {}
        method_input_curves = result.metadata.get("input_curves") or []
        input_curves = (
            list(method_input_curves)
            if method_input_curves
            else [result.metadata.get("porosity_curve", "PHIE")]
        )
        if empirical_method != "lith_vcl" and result.swi_source in {"hc_mask_p5", "default_0.20"}:
            input_curves.append("SW")
            rt_curve = swirr_metadata.get("rt_curve")
            if rt_curve:
                input_curves.append(rt_curve)
            if swirr_metadata.get("resflag_source"):
                input_curves.append(swirr_metadata["resflag_source"])
        trace.input_curves = list(dict.fromkeys(input_curves))
        trace.output_curves = outputs
        trace.parameters = {
            "method": self.config.perm_method,
            "fallback_method": self.config.perm_fallback_method,
            "swirr_source": result.swi_source,
            "swirr_app_method": swirr_metadata.get("swirr_app_method"),
            "bvw_irr": result.bvw_irr,
            "swirr_metadata": swirr_metadata,
            "strategy": result.metadata.get("strategy"),
            "same_well_core_models": result.metadata.get("same_well_core_models"),
            "peer_core_models": result.metadata.get("peer_core_models"),
            "peer_wells_considered": result.metadata.get("peer_wells_considered"),
            "peer_core_enabled": result.metadata.get("peer_core_enabled"),
            "core_min_points": result.metadata.get("core_min_points"),
            "core_models": result.metadata.get("core_models"),
            "core_rejected_models": result.metadata.get("core_rejected_models"),
            "empirical_method": empirical_method,
            "empirical_error": result.metadata.get("empirical_error"),
            "empirical_attempted_methods": result.metadata.get("empirical_attempted_methods"),
            "empirical_method_metadata": method_metadata,
            "perm_max_clip_md": result.metadata.get("perm_max_clip_md"),
        }
        if resflag_info is not None:
            trace.parameters["pre_permeability_resflag"] = resflag_info
        if result.metadata.get("core_sections_count"):
            decisions.append(
                f"Core-calibrated sections: {result.metadata['core_sections_count']} samples"
            )
        for model in result.metadata.get("core_models", []):
            source = model.get("source_well") or "unknown"
            formation = model.get("source_formation") or "all_depths"
            source_type = model.get("source_type") or "core"
            decisions.append(
                f"{source_type}: {source} / {formation} "
                f"(n={model.get('n_core_points')}, "
                f"core_phi={model.get('core_porosity_curve') or 'unknown'}, "
                f"core_perm={model.get('core_permeability_curve') or 'unknown'})"
            )
        if np.isscalar(result.swi_used):
            trace.parameters["swirr"] = float(result.swi_used)
            if result.swi_source == "hc_mask_p5":
                n_samples = swirr_metadata.get("hc_mask_samples")
                rt_curve = swirr_metadata.get("rt_curve", "RDEP")
                decisions.append(
                    f"SWIRR_APP = P5(SW where resflag and {rt_curve} > 5) = {float(result.swi_used):.3f} (n={n_samples})"
                )
            elif result.swi_source == "default_0.20":
                n_samples = swirr_metadata.get("hc_mask_samples")
                reason = swirr_metadata.get("reason")
                if n_samples is not None:
                    decisions.append(f"SWIRR_APP = 0.200 fallback because hydrocarbon mask had {n_samples} samples")
                elif reason:
                    decisions.append(f"SWIRR_APP = 0.200 fallback because {reason}")
                else:
                    decisions.append("SWIRR_APP = 0.200 fallback")
            else:
                decisions.append(f"Swirr = {float(result.swi_used):.3f} ({result.swi_source})")
        elif isinstance(result.swi_used, np.ndarray):
            decisions.append(f"Swirr resolved as curve ({result.swi_source})")
        elif result.swi_source == "not_required" and empirical_method == "lith_vcl":
            decisions.append(
                "Swirr not used because lith_vcl permeability uses PHIT, VCL, and lithology"
            )
        else:
            decisions.append("Swirr not used because permeability came from core calibration")
        if empirical_method == "lith_vcl":
            decisions.append(
                "lith_vcl empirical permeability: "
                "10 ** min(4.5, slope * PHIT * (1 - VCL * 1.2) + intercept)"
            )
            groups = method_metadata.get("lithology_parameter_groups") or []
            if groups:
                decisions.append("lith_vcl parameter groups used: " + ", ".join(groups))
        elif empirical_method == "coates":
            decisions.append(
                "Coates-Denoo empirical permeability: "
                "k = 10000 * PHIE^4 * ((1 - SWIRR) / SWIRR)^2"
            )
        elif empirical_method == "timur":
            decisions.append(
                "Timur empirical permeability: "
                "k = 0.136 * (100 * PHIE)^4.4 / (100 * SWIRR)^2"
            )
        if result.metadata.get("empirical_error") and empirical_method == "lith_vcl":
            decisions.append(f"Earlier empirical method failed before lith_vcl: {result.metadata['empirical_error']}")
        if result.metadata.get("perm_max_clip_md") is not None:
            decisions.append(f"Permeability clipped to <= {result.metadata['perm_max_clip_md']:.0f} mD")
        decisions.append(f"Empirical permeability curve stored as PERM ({perm_desc})")
        if perm_core is not None:
            decisions.append("Core-corrected permeability stored as PERM_CC")
        decisions.append(f"Permeability strategy: {result.method}")
        decisions.append(f"Perm range: {np.nanmin(perm_output):.3f} - {np.nanmax(perm_output):.1f} mD")
        trace.decisions = decisions
        return trace

    def _ensure_resflag_for_swirr(self, well: "Well") -> Optional[Dict[str, Any]]:
        """Create RESFLAG before permeability so Swirr uses workflow net-reservoir logic."""
        resflag_curve = get_curve(well, ["RESFLAG"])
        if resflag_curve is not None:
            data = np.asarray(resflag_curve.data, dtype=float)
            return {
                "source_curve": "RESFLAG",
                "created": False,
                "reservoir_samples": int(np.nansum(data > 0.5)),
            }

        net_res_curve = get_curve(well, ["NET_RES"])
        if net_res_curve is not None:
            data = np.asarray(net_res_curve.data, dtype=float)
            well.add_curve("RESFLAG", data, "", "Reservoir Flag")
            return {
                "source_curve": "NET_RES",
                "created": True,
                "vcl_curve": "existing_NET_RES",
                "phie_curve": "existing_NET_RES",
                "sw_curve": "existing_NET_RES",
                "reservoir_samples": int(np.nansum(data > 0.5)),
            }

        if not self._has_curve(well, "PHIE") or not self._has_curve(well, "SW", "SWA", "SWT-LOG"):
            return None

        from petrophys.evaluation.net_pay.net_pay_module import NetPayModule

        phie_curve = self._first_curve(well, "PHIE")
        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        vcl_curve = get_curve(well, ["VCL", "VCLAY"])
        vcl = np.asarray(vcl_curve.data, dtype=float) if vcl_curve is not None else None

        lithology = self._build_lithology_log(well)
        np_module = NetPayModule(
            vcl_cutoff=self._cutoff_argument("vcl_cutoff"),
            phie_cutoff=self._cutoff_argument("phie_cutoff"),
            sw_cutoff=self._cutoff_argument("sw_cutoff"),
            lithology=lithology,
        )
        result = np_module.calculate(
            well,
            vcl=vcl,
            phie=np.asarray(phie_curve.data, dtype=float),
            sw=np.asarray(sw_curve.data, dtype=float),
        )
        resflag = result.net_reservoir.astype(float)
        well.add_curve("RESFLAG", resflag, "", "Reservoir Flag")

        return {
            "source_curve": "NetPayModule.net_reservoir",
            "created": True,
            "vcl_curve": vcl_curve.mnemonic if vcl_curve is not None else "implicit_0",
            "phie_curve": phie_curve.mnemonic,
            "sw_curve": sw_curve.mnemonic,
            "vcl_cutoff": self._resolved_cutoff("vcl_cutoff"),
            "phie_cutoff": self._resolved_cutoff("phie_cutoff"),
            "sw_cutoff": self._resolved_cutoff("sw_cutoff"),
            "cutoff_source": (
                "config"
                if self.config.vcl_cutoff is not None
                else "may_2026_benchmark"
                if self._uses_vendor_benchmark
                else "standard(" + str(self.config.lithology) + ")"
            ),
            "reservoir_samples": int(np.nansum(resflag > 0.5)),
        }

    def _step_rock_quality(self, well: "Well") -> StepResult:
        from petrophys.evaluation.rock_quality.rock_quality_module import RockQualityModule

        trace = StepResult(step="rock_quality", success=True, method="RQI_FZI_phi_z")
        rq_module = RockQualityModule()
        perm, perm_source, _ = get_curve_data_depthwise(well, ["PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"])
        phie, phie_source, _ = get_curve_data_depthwise(well, ["PHIE_CC", "PHIE", "PHIT"])
        result = rq_module.calculate(well, perm=perm, phi_e=phie)
        rq_module.add_to_well(well, result)
        trace.input_curves = [source for source in (perm_source, phie_source) if source]
        trace.output_curves = ["RQI", "FZI", "PHI_Z"]
        trace.decisions.append(
            f"RQI = 0.0314*sqrt({perm_source}/{phie_source}); "
            f"PHI_Z = {phie_source}/(1-{phie_source}); FZI = RQI/PHI_Z"
        )
        return trace

    def _step_bvw(self, well: "Well") -> StepResult:
        from petrophys.evaluation.bvw.bvw_module import calculate_bvw

        phie, phie_source, _ = get_curve_data_depthwise(well, ["PHIE_CC", "PHIE", "PHIT"])
        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        sw = sw_curve.data
        bvw = calculate_bvw(phie, sw)
        well.add_curve("BVW", bvw, "v/v", "Bulk Volume Water")

        trace = StepResult(step="bvw", success=True, method="PHIE_x_SW")
        trace.input_curves = [phie_source, sw_curve.mnemonic]
        trace.output_curves = ["BVW"]
        trace.decisions.append(f"BVW = {phie_source} * {sw_curve.mnemonic}")
        return trace

    def _step_mhi(self, well: "Well") -> StepResult:
        from petrophys.evaluation.mhi.mhi_module import MhiModule

        sw_curve = self._first_curve(well, "SW", "SWA", "SWT-LOG")
        sxo_curve = self._first_sxo_curve(well)
        mhi_module = MhiModule()
        trace = StepResult(step="mhi", success=True, method="moveable_HC_index")
        trace.input_curves = [sw_curve.mnemonic, sxo_curve.mnemonic]
        diagnostic_mask = mhi_module.diagnostic_sxo_mask(
            well,
            sxo=sxo_curve.data,
            sw=sw_curve.data,
        )
        diagnostic_samples = int(np.count_nonzero(diagnostic_mask))
        trace.parameters = {
            "diagnostic_sxo_samples": diagnostic_samples,
            "total_samples": int(well.n_samples),
            "rule": "MHI only where SXO has invaded-zone support from RSHA/RXO/RESS plus RMF",
        }
        if diagnostic_samples == 0:
            well.remove_curve("MHI")
            trace.output_curves = []
            trace.decisions.append(
                "MHI not computed because SXO was only SW^(1/5) fallback or lacked invaded-zone support"
            )
            return trace

        mhi = mhi_module.calculate(well, sxo=sxo_curve.data, sw=sw_curve.data)
        well.add_curve("MHI", mhi, "", "Moveable Hydrocarbon Index")

        trace.output_curves = ["MHI"]
        trace.decisions.append(
            f"MHI = {sw_curve.mnemonic} / {sxo_curve.mnemonic} for {diagnostic_samples}/{well.n_samples} "
            "samples with diagnostic SXO; fallback-SXO rows were set to NaN"
        )
        return trace

    def _step_net_pay(self, well: "Well") -> StepResult:
        from petrophys.evaluation.diagnostics import HydrocarbonAnalysis
        from petrophys.evaluation.net_pay.net_pay_module import NetPayModule

        trace = StepResult(step="net_pay", success=True, method="cutoff_flags")
        lithology = self._build_lithology_log(well)

        np_module = NetPayModule(
            vcl_cutoff=self._cutoff_argument("vcl_cutoff"),
            phie_cutoff=self._cutoff_argument("phie_cutoff"),
            sw_cutoff=self._cutoff_argument("sw_cutoff"),
            lithology=lithology
        )
        vcl, vcl_source, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY", "VSH"])
        phie, phie_source, _ = get_curve_data_depthwise(well, ["PHIE_CC", "PHIE", "PHIT"])
        sw, sw_source, _ = get_curve_data_depthwise(well, ["SW", "SWA", "SWT-LOG"])
        # Exclude washed-out (bad-hole) intervals from net reservoir / pay: there
        # RHOB and NPHI are unreliable, so PHIE and Sw cannot certify pay.
        bad_hole, _, _ = get_curve_data_depthwise(well, ["BAD_HOLE"])
        quality_mask = None
        n_excluded = 0
        if bad_hole is not None and not self._uses_vendor_benchmark:
            quality_mask = ~(np.asarray(bad_hole, dtype=float) > 0.5)
            n_excluded = int(np.count_nonzero(~quality_mask))
        result = np_module.calculate(well, vcl=vcl, phie=phie, sw=sw, quality_mask=quality_mask)
        np_module.add_to_well(well, result)

        well.add_curve("RESFLAG", result.net_reservoir.astype(float), "", "Reservoir Flag")

        if vcl is None:
            vcl_data = np.zeros(well.n_samples, dtype=float)
            vcl_source = "VCL default 0"
        else:
            vcl_data = np.asarray(vcl, dtype=float)
        seal_flag = np.full(vcl_data.shape, np.nan, dtype=float)
        valid_vcl = np.isfinite(vcl_data)
        seal_flag[valid_vcl] = (vcl_data[valid_vcl] >= 0.8).astype(float)
        well.add_curve("SEALFLAG", seal_flag, "", "Seal Flag")

        hc_result = HydrocarbonAnalysis().analyze(well, add_curves=True)

        trace.input_curves = [source for source in (vcl_source, phie_source, sw_source) if source]
        trace.output_curves = ["NET_ROCK", "NET_RES", "NET_PAY", "RESFLAG", "SEALFLAG"]
        if well.get("CROSSOVER_FLAG") is not None:
            trace.output_curves.append("CROSSOVER_FLAG")
        trace.parameters = {
            "vcl_cutoff": self._resolved_cutoff("vcl_cutoff"),
            "phie_cutoff": self._resolved_cutoff("phie_cutoff"),
            "sw_cutoff": self._resolved_cutoff("sw_cutoff"),
            "cutoff_source": (
                "config"
                if self.config.vcl_cutoff is not None
                else "may_2026_benchmark"
                if self._uses_vendor_benchmark
                else "standard(" + str(self.config.lithology) + ")"
            ),
            "lithology": self.config.lithology if isinstance(lithology, str) else "per-depth",
            "seal_vcl_cutoff": 0.8,
            "interpretation_profile": self._interpretation_profile,
        }
        n_rock = int(np.nansum(well.get("NET_ROCK").data > 0.5)) if well.get("NET_ROCK") else 0
        n_res = int(np.nansum(well.get("NET_RES").data > 0.5)) if well.get("NET_RES") else 0
        n_pay = int(np.nansum(well.get("NET_PAY").data > 0.5)) if well.get("NET_PAY") else 0
        n_seal = int(np.nansum(seal_flag > 0.5))
        step_m = well.step or 0.15
        trace.decisions.append(f"NET_ROCK: VCL < {self._resolved_cutoff('vcl_cutoff')} -> {n_rock} samples ({n_rock*step_m:.1f} m)")
        trace.decisions.append(f"NET_RES:  + {phie_source} > {self._resolved_cutoff('phie_cutoff')} -> {n_res} samples ({n_res*step_m:.1f} m)")
        trace.decisions.append(f"NET_PAY:  + {sw_source} < {self._resolved_cutoff('sw_cutoff')} -> {n_pay} samples ({n_pay*step_m:.1f} m)")
        if n_excluded:
            trace.decisions.append(
                f"Bad-hole exclusion: {n_excluded} washed-out samples removed from net reservoir/pay"
            )
        trace.decisions.append(f"SEALFLAG: VCL >= 0.8 -> {n_seal} samples ({n_seal*step_m:.1f} m)")
        if well.get("CROSSOVER_FLAG") is not None:
            n_crossover = int(np.nansum(np.asarray(well.get("CROSSOVER_FLAG").data, dtype=float) > 0.5))
            trace.decisions.append(
                f"CROSSOVER_FLAG from hydrocarbon analysis -> {n_crossover} samples ({n_crossover*step_m:.1f} m)"
            )
        trace.decisions.extend(
            f"Hydrocarbon analysis: {section_name}"
            for section_name in hc_result.sections.keys()
        )
        return trace

    # ---- Helpers ----

    def _first_curve(self, well: "Well", *names: str):
        curve = get_curve(well, names)
        if curve is not None:
            return curve
        raise ValueError(f"None of the requested curves are available: {', '.join(names)}")

    def _has_sxo_curve(self, well: "Well") -> bool:
        if self._has_curve(well, "SXO", "SWXOA", "SXO-LOG"):
            return True
        return any(
            name.upper().startswith("SXO-")
            and get_curve(well, [name]) is not None
            for name in well.curves
        )

    def _first_sxo_curve(self, well: "Well"):
        for name in ("SXO", "SWXOA", "SXO-LOG"):
            curve = get_curve(well, [name])
            if curve is not None:
                return curve
        for name in well.curves:
            if name.upper().startswith("SXO-"):
                curve = get_curve(well, [name])
                if curve is not None:
                    return curve
        raise ValueError("No SXO curve is available")

    def _build_lithology_log(self, well: "Well") -> Union[str, np.ndarray]:
        """Build per-depth lithology from well curves, CSV, or global config."""
        if self._lithology_log is not None:
            return self._lithology_log

        if self._selected_lithology_source == "config":
            self._lithology_source = "config"
            return self.config.lithology
        gt_curve = get_curve(well, ["LITHOLOGY_GEOLINK"])
        if self._selected_lithology_source not in {"auto", "vendor"}:
            gt_curve = None
        if gt_curve is not None:
            try:
                gt_data = np.asarray(gt_curve.data, dtype=float)
                self._lithology_log = _group_array_from_numeric_codes(gt_data, GROUND_TRUTH_CODE_TO_GROUP)
                self._lithology_source = "ground_truth_lithology_geolink"
                return self._lithology_log
            except (TypeError, ValueError):
                pass

        computed_curve = get_curve(well, ["LITHO"])
        if computed_curve is not None and not self._uses_vendor_benchmark:
            try:
                computed_data = np.asarray(computed_curve.data, dtype=float)
                self._lithology_log = _group_array_from_numeric_codes(computed_data, COMPUTED_LITHO_CODE_TO_GROUP)
                self._lithology_source = "computed_lithology_curve"
                return self._lithology_log
            except (TypeError, ValueError):
                pass

        self._lithology_source = "config"
        return self.config.lithology

    def _get_avg_formation_temp(self, well: "Well") -> float:
        """Get average formation temperature."""
        temp = get_curve(well, ["TEMP"])
        if temp is not None:
            return float(np.nanmean(temp.data))

        surface_temp_f = (
            self.config.surface_temp_f
            if self.config.surface_temp_f is not None
            else 68.0
        )
        bht_f = self.config.bht_f if self.config.bht_f is not None else _metadata_bht_f(well)
        if bht_f is not None:
            return (surface_temp_f + float(bht_f)) / 2.0

        if well.depth is not None:
            from petrophys.interpretation.temperature import calculate_temperature_profile

            try:
                result = calculate_temperature_profile(
                    well,
                    surface_temp=surface_temp_f,
                    surface_temp_unit="F",
                    td=self.config.td_ft,
                    td_unit="ft" if self.config.td_ft is not None else None,
                    depth_unit=well.depth_unit,
                )
                return float(np.nanmean(result.temperature))
            except Exception:
                pass

        return 150.0  # Last-resort estimate when depth is unavailable


def run_dynamic_workflow(
    well: "Well",
    config: Optional[DynamicWorkflowConfig] = None,
    **kwargs
) -> DynamicWorkflowResult:
    """
    Run the dynamic workflow on a well, adding calculated curves to it.

    This is the lower-level form of :func:`petrophys.interpret`. It works on
    ``well`` itself: calculated curves are added to the object you pass in.
    Use ``petrophys.interpret(well)`` to leave the input unchanged.

    Parameters
    ----------
    well : Well
        Well object with raw log data. Modified in place.
    config : DynamicWorkflowConfig, optional
        Workflow configuration. When omitted, one is built from ``kwargs``.
    **kwargs
        DynamicWorkflowConfig settings; with ``config`` they override its
        fields on a copy.

    Returns
    -------
    DynamicWorkflowResult
        ``result.well`` is ``well``.

    Examples
    --------
    >>> result = run_dynamic_workflow(well, rw=0.05, lithology='clastic')
    >>> print(f"Steps executed: {[s.step for s in result.steps_executed]}")
    >>> print(f"Steps skipped: {result.steps_skipped}")
    """
    if config is None:
        config = DynamicWorkflowConfig(**kwargs)
    else:
        config = dataclasses.replace(config, **kwargs)

    workflow = DynamicWorkflow(config)
    return workflow.run(well)
