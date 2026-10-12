"""
Zone-aware batch processing helpers for LAS enrichment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import copy
import csv
import re
from warnings import warn as _warn
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from petrophys._internal.analysis_helpers import get_curve
from petrophys.interpretation.electrical.rw.salinity_rw import NACL_PER_CHLORIDE, brine_density
from petrophys.interpretation.lithology.matrix_inference import (
    GEOLINK_LITHOLOGY_CURVE,
    interpreted_lithology_code_curve,
)
from petrophys.io import export_las, read_las
from petrophys.model.units import temperature_to_fahrenheit
from petrophys.preprocessing.standardization import standardize_curves
from petrophys.workflow.dynamic_workflow import (
    _BENCHMARK_PROFILES,
    DynamicWorkflowConfig,
    DynamicWorkflowResult,
    _left_at_default,
    _rerun_calculations,
    _uses_rerun_calculations,
    run_dynamic_workflow,
)

if TYPE_CHECKING:
    from petrophys.model.well import Well


ZONE_VALUE_RE = re.compile(r"^(?P<base>.+)_Z(?P<index>\d+)$", re.IGNORECASE)
DEPTH_TOP_KEYS = {"ZTOP", "TOP", "TOP_DEPTH"}
DEPTH_BOTTOM_KEYS = {"ZBOT", "BOT", "BOTTOM", "BASE", "ZBASE"}

PARAMETER_ALIASES: dict[str, str] = {
    "A": "a",
    "M": "m",
    "N": "n",
    "RW": "rw",
    "RWT": "rw_meas_temp_f",
    "RMF": "rmf_at_meas_temp",
    "RMFT": "rmf_meas_temp_f",
    "MFST": "rmf_meas_temp_f",
    "RM": "rm",
    "RMT": "rm_meas_temp_f",
    "MST": "rm_meas_temp_f",
    "CLFL": "rmf_chlorides_mg_l",
    "BHT": "bht_f",
    "TD": "td_ft",
    "SAL": "salinity_ppm",
    "SALINITY": "salinity_ppm",
    "SALINITY_PPM": "salinity_ppm",
    "WATSAL": "salinity_ppm",
    "WSAL": "salinity_ppm",
    "WATER_SALINITY": "salinity_ppm",
    "WATER_SALINITY_PPM": "salinity_ppm",
    "VSHE": "vsh_exp",
    "VSH_EXP": "vsh_exp",
    "SHALE_EXP": "shale_exp",
    "RTSH": "rtsh",
    "CEC": "cec",
    "B_COEFF": "b_coeff",
    "BCOEFF": "b_coeff",
    "RWB": "rwb",
    "MUDTYPE": "mud_type",
    "MUD_TYPE": "mud_type",
    "MUD": "mud_type",
    "MUDWATER": "mud_water",
    "LITH": "lithology_domain",
    "LITHOLOGY": "lithology_domain",
    "LITHOLOGY_DOMAIN": "lithology_domain",
    "PP_DOMAIN": "lithology_domain",
    "MATRIX": "matrix_type",
    "MATRIX_TYPE": "matrix_type",
    "APPLY_HC_CORRECTION": "apply_hc_correction",
    "HC_CORRECTION": "apply_hc_correction",
    "HCCORRECTION": "apply_hc_correction",
    "HC_TYPE": "hc_type",
    "PHI_CL": "phi_cl",
    "PHICL": "phi_cl",
    "RHO_FL": "rho_fl",
    "RHO_MA": "rho_ma",
    "RHMA": "rho_ma",
    "PNMA": "nphi_ma",
    "RBFL": "rho_fl",
    "RBSH": "rho_cl",
    "GR_METHOD": "gr_method",
    "GR_MIN": "gr_min",
    "GR_MAX": "gr_max",
    "VCL_CUTOFF": "vcl_cutoff",
    "PHIE_CUTOFF": "phie_cutoff",
    "SW_CUTOFF": "sw_cutoff",
}

#: Header parameters the benchmark profile reads differently: CLFL (mud-filtrate
#: chlorides) as NaCl-equivalent salinity, the mud-sample temperatures MFST and
#: MST not at all, and PNSH as the clay porosity phi_cl.
_BENCHMARK_PARAMETER_ALIASES: dict[str, Optional[str]] = {
    "CLFL": "rmf_salinity_ppm",
    "MFST": None,
    "MST": None,
    "PNSH": "phi_cl",
}

#: Header parameters the "current" calculations read from the well itself, not
#: as settings. PNSH ("Neutron Shale") is the neutron log's reading in shale:
#: DynamicWorkflow turns it, with RBSH ("Shale Density"), into a clay porosity.
_READ_FROM_WELL_PARAMETERS = {"PNSH"}

#: Header parameters used only when their setting has no other source: the
#: mud-sample temperatures MFST and MST yield to RMFT and RMT, as in
#: DynamicWorkflow, whatever the header order.
_FALLBACK_PARAMETERS = {"MFST", "MST"}

DIRECT_INTERVAL_METADATA: set[str] = set()
AUDIT_TARGET_CURVES = [
    "TEMP",
    "RMF",
    "RW",
    "BAD_HOLE",
    "VCL",
    "PHIN",
    "PHID",
    "PHIS",
    "PHIT",
    "PHIE",
    "PHI_SEC",
    "SW",
    "PERM",
    "PHIE_CC",
    "PERM_CC",
    "RQI",
    "FZI",
    "PHI_Z",
    "SWIRR_APP",
    "INVASION_RATIO",
    "RESFLAG",
    "SEALFLAG",
    "CROSSOVER_FLAG",
]
CURVE_TO_STEP = {
    "TEMP": "temperature",
    "RMF": "rmf",
    "RW": "rw",
    "BAD_HOLE": "bad_hole",
    "VCL": "vclay",
    "PHIN": "porosity",
    "PHID": "porosity",
    "PHIS": "porosity",
    "PHIT": "porosity",
    "PHIE": "porosity",
    "PHI_SEC": "porosity",
    "SW": "saturation",
    "PERM": "permeability",
    "PHIE_CC": "permeability",
    "PERM_CC": "permeability",
    "RQI": "rock_quality",
    "FZI": "rock_quality",
    "PHI_Z": "rock_quality",
    "SWIRR_APP": "permeability",
    "INVASION_RATIO": "invasion",
    "RESFLAG": "net_pay",
    "SEALFLAG": "net_pay",
    "CROSSOVER_FLAG": "net_pay",
}


@dataclass
class ZonedParameterInterval:
    """One depth zone defined by ``*_Z<n>`` LAS header parameters.

    ``config_overrides`` are the DynamicWorkflowConfig fields the zone sets,
    ``injected_parameters`` the well parameters it provides, and
    ``unmapped_parameters`` the zone parameters that were not recognized.
    """

    index: int
    top: float
    bottom: float
    top_unit: str = ""
    bottom_unit: str = ""
    raw_parameters: Dict[str, Any] = field(default_factory=dict)
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    injected_parameters: Dict[str, Any] = field(default_factory=dict)
    unmapped_parameters: List[str] = field(default_factory=list)


@dataclass
class ZonedWorkflowRunResult:
    """Result of :func:`run_dynamic_workflow_with_zones`.

    ``base_result`` is the whole-well run; ``zone_results`` holds one run per
    zone whose parameters differ, merged into ``well`` over the zone depths.
    """

    well: "Well"
    base_result: DynamicWorkflowResult
    zone_results: List[DynamicWorkflowResult] = field(default_factory=list)
    zones: List[ZonedParameterInterval] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    global_overrides: Dict[str, Any] = field(default_factory=dict)
    global_unmapped: List[str] = field(default_factory=list)
    zoned_default_overrides: Dict[str, Any] = field(default_factory=dict)
    zoned_default_injected: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchLasResult:
    """Outcome for one file of :func:`process_las_directory_with_workflow`."""

    source_file: str
    output_file: str
    success: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    input_curves: List[str] = field(default_factory=list)
    output_curves: List[str] = field(default_factory=list)
    computed_curves: List[str] = field(default_factory=list)
    zones_applied: int = 0
    json_log_file: str = ""
    result: Optional[ZonedWorkflowRunResult] = None


def _parameter_text(well: "Well", name: str) -> str:
    value = well.get_parameter(name)
    return str(value or "").strip()


def _parameter_float(well: "Well", name: str) -> Optional[float]:
    value = well.get_parameter(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def lithology_domain_from_metadata(well: "Well") -> Optional[str]:
    """Return the broad interpretation domain declared in LAS metadata."""
    for name in ("PP_DOMAIN", "LITHOLOGY_DOMAIN"):
        value = well.get_parameter(name)
        if value is None or not str(value).strip():
            continue
        domain = str(value).strip().lower()
        if domain not in {"clastic", "carbonate", "evaporite", "auto"}:
            raise ValueError(
                f"LAS parameter {name} must be clastic, carbonate, evaporite, or auto; "
                f"got {value!r}."
            )
        return domain
    return None


def _clone_config(config: Optional[DynamicWorkflowConfig]) -> DynamicWorkflowConfig:
    return copy.deepcopy(config) if config is not None else DynamicWorkflowConfig()


def _reproduces_benchmark(config: Optional[DynamicWorkflowConfig]) -> bool:
    """True when this batch runs the benchmark profile."""
    return getattr(config, "calculation_profile", "current") in _BENCHMARK_PROFILES


def _prepare_batch_config(config: Optional[DynamicWorkflowConfig]) -> DynamicWorkflowConfig:
    prepared = _clone_config(config)
    skip_steps = list(prepared.skip_steps)
    if not prepared.apply_hc_correction and "hc_correction" not in skip_steps:
        skip_steps.append("hc_correction")
    prepared.skip_steps = skip_steps
    return prepared


def _depth_unit_is_metric(unit: str) -> bool:
    text = (unit or "").strip().lower()
    return text in {"m", "meter", "meters", "metre", "metres"}


def _depth_to_ft(value: float, unit: str) -> float:
    if _depth_unit_is_metric(unit):
        return float(value) * 3.28084
    return float(value)


def _convert_depth_between_units(value: float, from_unit: str, to_unit: str) -> float:
    if not from_unit or not to_unit or from_unit.lower() == to_unit.lower():
        return float(value)
    from_is_metric = _depth_unit_is_metric(from_unit)
    to_is_metric = _depth_unit_is_metric(to_unit)
    if from_is_metric and not to_is_metric:
        return float(value) * 3.28084
    if not from_is_metric and to_is_metric:
        return float(value) / 3.28084
    return float(value)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _parse_bool(value: Any) -> bool:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int, np.floating, float)) and not isinstance(value, bool):
        return bool(float(value))
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y", "on"}


def _salinity_to_ppm(value: float, unit: str) -> float:
    text = (unit or "").strip().lower()
    if not text:
        return float(value)
    if "kppm" in text or "ppk" in text:
        return float(value) * 1000.0
    if "mg/l" in text or "mg\\l" in text or "ppm" in text:
        return float(value)
    if "g/l" in text:
        return float(value) * 1000.0
    if "%" in text and "/" not in text:
        return float(value) * 10000.0
    return float(value)


def _chlorides_to_mg_per_l(value: float, unit: str) -> float:
    """Chloride concentration in mg/l from a header value such as CLFL.

    Mud reports give filtrate chlorides in mg/l (API RP 13B-1), which is how a
    value without a unit is read. Per-weight units (ppm, kppm, %) are converted
    with the density of the NaCl solution that holds that much chloride.
    """
    text = (unit or "").strip().lower()
    if "kppm" in text or "ppk" in text:
        chlorides_ppm = float(value) * 1000.0
    elif "ppm" in text:
        chlorides_ppm = float(value)
    elif "%" in text and "/" not in text:
        chlorides_ppm = float(value) * 10000.0
    elif "g/l" in text and "mg" not in text:
        return float(value) * 1000.0
    else:
        return float(value)
    return chlorides_ppm * brine_density(NACL_PER_CHLORIDE * chlorides_ppm)


def _cec_to_meq_per_100g(value: float, unit: str) -> float:
    """A header CEC in meq/100 g, the unit of the ``cec`` setting.

    A value labelled meq/g is multiplied by 100; meq/100 g, the unit labs
    report, and a value without a unit are taken as they stand.
    """
    text = "".join((unit or "").lower().split())
    if text in {"meq/g", "meq/gm"}:
        return float(value) * 100.0
    return float(value)


def _set_config_value(config: DynamicWorkflowConfig, field_name: str, value: Any) -> None:
    if value is None or not hasattr(config, field_name):
        return
    if field_name in {"lithology", "lithology_domain"}:
        domain = str(value).strip().lower()
        config.lithology_domain = domain
        config.lithology = domain
        return
    setattr(config, field_name, value)


def _value_signature(value: Any) -> tuple[str, Any]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return ("number", round(float(value), 8))
    return ("text", str(value).strip().lower())


def _select_common_or_z1(values: List[Any], z1_value: Any = None) -> Any:
    if not values:
        return z1_value

    counts: Dict[tuple[str, Any], int] = {}
    originals: Dict[tuple[str, Any], Any] = {}
    for value in values:
        key = _value_signature(value)
        counts[key] = counts.get(key, 0) + 1
        originals.setdefault(key, value)

    max_count = max(counts.values())
    winners = [key for key, count in counts.items() if count == max_count]

    if max_count > 1 and len(winners) == 1:
        return originals[winners[0]]
    if z1_value is not None:
        return z1_value
    return originals[winners[0]]


def _derive_default_values_from_zones(
    zones: List[ZonedParameterInterval],
) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    if not zones:
        return {}, {}, []

    z1 = zones[0]
    default_overrides: Dict[str, Any] = {}
    default_injected: Dict[str, Any] = {}
    notes: List[str] = []

    override_fields = sorted({key for zone in zones for key in zone.config_overrides})
    injected_fields = sorted({key for zone in zones for key in zone.injected_parameters})

    for field_name in override_fields:
        values = [zone.config_overrides[field_name] for zone in zones if field_name in zone.config_overrides]
        z1_value = z1.config_overrides.get(field_name)
        selected = _select_common_or_z1(values, z1_value)
        if selected is not None:
            default_overrides[field_name] = selected

    for field_name in injected_fields:
        values = [zone.injected_parameters[field_name] for zone in zones if field_name in zone.injected_parameters]
        z1_value = z1.injected_parameters.get(field_name)
        selected = _select_common_or_z1(values, z1_value)
        if selected is not None:
            default_injected[field_name] = selected

    if default_overrides:
        notes.append(
            "Uncovered intervals use zoned fallback config: "
            + ", ".join(f"{key}={value}" for key, value in sorted(default_overrides.items()))
        )
    if default_injected:
        notes.append(
            "Uncovered intervals use zoned fallback metadata: "
            + ", ".join(f"{key}={value}" for key, value in sorted(default_injected.items()))
        )

    return default_overrides, default_injected, notes


def _merge_curve_into_well(parent_well: "Well", interval_well: "Well", top: float, bottom: float) -> None:
    if parent_well.depth is None or interval_well.depth is None:
        return

    mask = (parent_well.depth >= top) & (parent_well.depth <= bottom)
    expected = int(np.count_nonzero(mask))
    if expected == 0:
        return

    for mnemonic in interval_well.curves:
        source_curve = interval_well.get(mnemonic)
        if source_curve is None:
            continue

        source_data = np.asarray(source_curve.data)
        if source_data.shape[0] != expected:
            continue

        target_curve = parent_well.get(mnemonic)
        if target_curve is None:
            if np.issubdtype(source_data.dtype, np.number):
                full_data = np.full(parent_well.n_samples, np.nan, dtype=float)
            else:
                full_data = np.full(parent_well.n_samples, None, dtype=object)
            parent_well.add_curve(
                mnemonic,
                full_data,
                unit=source_curve.unit,
                description=source_curve.metadata.description,
                data_type=source_curve.metadata.data_type,
            )
            target_curve = parent_well.get(mnemonic)

        target_curve._data[mask] = source_data


def _apply_interval_metadata(interval_well: "Well", zone: ZonedParameterInterval) -> None:
    for name, value in zone.injected_parameters.items():
        interval_well.set_parameter(name, value)
        unit_name = f"{name}__UNIT"
        original_unit = _parameter_text(interval_well, unit_name)
        if original_unit:
            interval_well.set_parameter(unit_name, original_unit)


def _derive_sw_model(overrides: Dict[str, Any], config: DynamicWorkflowConfig) -> None:
    if config.sw_model != "auto":
        return
    if overrides.get("vsh_exp") is not None:
        config.sw_model = "north_sea"


def _finalize_export_curves(
    well: "Well",
    original_curves: set[str],
    *,
    retain_computed_lithology: bool = False,
    drop_redundant_porosity: bool = False,
) -> None:
    if well.get("SW") is None:
        for alias in ("SWA", "SWT-LOG"):
            curve = well.get(alias)
            if curve is not None:
                well.add_curve("SW", curve.data, "frac", "Water Saturation")
                break

    if well.get("SXO") is None:
        sxo_curve = None
        for alias in ("SWXOA", "SXO-LOG"):
            sxo_curve = well.get(alias)
            if sxo_curve is not None:
                break
        if sxo_curve is None:
            for curve_name in well.curves:
                if curve_name.upper().startswith("SXO-"):
                    sxo_curve = well.get(curve_name)
                    break
        if sxo_curve is not None:
            well.add_curve("SXO", sxo_curve.data, "frac", "Flushed Zone Water Saturation")

    well.remove_curve("SWA")
    well.remove_curve("SWXOA")
    for curve_name in list(well.curves):
        if curve_name.upper().startswith("SXO-"):
            well.remove_curve(curve_name)

    vcl_method_curves = [
        "VCL_GR",
        "VCLGR",
        "VCL_SP",
        "VCLSP",
        "VCL_ND",
        "VCLND",
    ]
    available = []
    for name in ["VCL", "VCLAY", *vcl_method_curves]:
        curve = well.get(name)
        if curve is None:
            continue
        if getattr(curve.metadata, "custom", {}).get("in_vcl") is False:
            # An indicator the workflow reported without letting it set VCL.
            continue
        data = np.asarray(curve.data, dtype=float)
        if data.shape[0] != well.n_samples:
            continue
        available.append(data)

    if available:
        stack = np.vstack(available)
        valid_mask = np.isfinite(stack)
        final_vcl = np.full(well.n_samples, np.nan, dtype=float)
        any_valid = np.any(valid_mask, axis=0)
        if np.any(any_valid):
            with np.errstate(all="ignore"):
                final_vcl[any_valid] = np.nanmin(stack[:, any_valid], axis=0)
        if well.get("VCL") is None:
            well.add_curve("VCL", final_vcl, "v/v", "Clay Volume")
        else:
            well.get("VCL")._data = final_vcl
            well.get("VCL").unit = "v/v"

    well.remove_curve("VCLAY")
    for name in vcl_method_curves:
        if name not in original_curves:
            well.remove_curve(name)

    for name in ("NPHI_HC", "RHOB_HC"):
        if name not in original_curves:
            well.remove_curve(name)

    if (
        "LITHO" not in original_curves
        and (
            not retain_computed_lithology
            or well.get(interpreted_lithology_code_curve(well)) is not None
        )
    ):
        well.remove_curve("LITHO")

    for name in ("PRIME_GAS_TARGET", "PRIME_OIL_TARGET"):
        if name not in original_curves:
            well.remove_curve(name)

    if drop_redundant_porosity:
        # A sonic-only porosity pass leaves PHIT equal to PHIS and PHI_SEC all
        # zero; the eval package's interpreted LAS carried neither.
        phis = well.get("PHIS")
        phit = well.get("PHIT")
        if phis is not None and phit is not None and "PHIT" not in original_curves:
            a = np.asarray(phit.data, dtype=float)
            b = np.asarray(phis.data, dtype=float)
            if np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99)):
                well.remove_curve("PHIT")
        secondary = well.get("PHI_SEC")
        if secondary is not None and "PHI_SEC" not in original_curves:
            values = np.asarray(secondary.data, dtype=float)
            finite = np.isfinite(values)
            if finite.any() and not np.any(values[finite]):
                well.remove_curve("PHI_SEC")


def _apply_parameter_alias(
    base_name: str,
    value: Any,
    unit: str,
    overrides: Dict[str, Any],
    injected_parameters: Dict[str, Any],
    unmapped_parameters: List[str],
    *,
    calculation_profile: str = "current",
) -> None:
    key = base_name.upper()
    target = PARAMETER_ALIASES.get(key)
    if calculation_profile in _BENCHMARK_PROFILES:
        target = _BENCHMARK_PARAMETER_ALIASES.get(key, target)
    elif key in _READ_FROM_WELL_PARAMETERS:
        return

    if key in DIRECT_INTERVAL_METADATA:
        try:
            injected_parameters[key] = float(value)
        except (TypeError, ValueError):
            unmapped_parameters.append(base_name)
        return

    if target is None:
        if key not in DEPTH_TOP_KEYS and key not in DEPTH_BOTTOM_KEYS:
            unmapped_parameters.append(base_name)
        return

    if key in _FALLBACK_PARAMETERS and target in overrides:
        return

    if target in {
        "mud_type", "mud_water", "lithology", "lithology_domain",
        "matrix_type", "hc_type", "gr_method",
    }:
        overrides[target] = _normalize_text(value)
        return

    if target in {"apply_hc_correction"}:
        overrides[target] = _parse_bool(value)
        return

    if target in {"gr_min", "gr_max"} and _normalize_text(value).lower() == "auto":
        overrides[target] = None
        return

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        unmapped_parameters.append(base_name)
        return

    if target in {"bht_f", "rmf_meas_temp_f", "rm_meas_temp_f", "rw_meas_temp_f"}:
        overrides[target] = temperature_to_fahrenheit(numeric_value, unit)
    elif target in {"td_ft"}:
        overrides[target] = _depth_to_ft(numeric_value, unit)
    elif target in {"salinity_ppm", "rmf_salinity_ppm"}:
        overrides[target] = _salinity_to_ppm(numeric_value, unit)
    elif target == "rmf_chlorides_mg_l":
        overrides[target] = _chlorides_to_mg_per_l(numeric_value, unit)
    elif target == "cec" and calculation_profile not in _BENCHMARK_PROFILES:
        overrides[target] = _cec_to_meq_per_100g(numeric_value, unit)
    else:
        overrides[target] = numeric_value


def _extract_global_parameter_overrides(
    well: "Well", *, calculation_profile: str = "current"
) -> tuple[Dict[str, Any], List[str]]:
    overrides: Dict[str, Any] = {}
    unmapped: List[str] = []

    for key, value in getattr(well, "_parameters", {}).items():
        if not isinstance(key, str) or "__" in key:
            continue
        if key.startswith("_"):
            continue
        if ZONE_VALUE_RE.match(key):
            continue
        unit = _parameter_text(well, f"{key}__UNIT")
        _apply_parameter_alias(
            key, value, unit, overrides, {}, unmapped,
            calculation_profile=calculation_profile,
        )

    return overrides, sorted(set(unmapped))


def extract_zoned_parameters(
    well: "Well", *, calculation_profile: str = "current"
) -> List[ZonedParameterInterval]:
    """Read depth zones from ``<NAME>_Z<n>`` LAS header parameters.

    A zone needs a top (``TOP_Z1``, ``ZTOP_Z1`` or ``TOP_DEPTH_Z1``) and a
    bottom (``BOT_Z1``, ``BASE_Z1``, ...); its other ``*_Z<n>`` parameters,
    such as ``RW_Z1`` or ``BHT_Z1``, become zone-specific settings.

    Parameters
    ----------
    well : Well
        Well whose LAS header parameters define the zones.
    calculation_profile : str, default "current"
        :attr:`DynamicWorkflowConfig.calculation_profile` the settings are
        for; the benchmark profile reads CLFL as NaCl-equivalent salinity,
        ignores MFST and MST, and takes a CEC in meq/g without conversion.

    Returns
    -------
    list of ZonedParameterInterval
        Zones in index order; zones without both depths are ignored.
    """
    grouped: Dict[int, Dict[str, Any]] = {}

    for key, value in getattr(well, "_parameters", {}).items():
        if not isinstance(key, str) or "__" in key:
            continue
        if key.startswith("_"):
            continue
        match = ZONE_VALUE_RE.match(key)
        if not match:
            continue

        zone_index = int(match.group("index"))
        zone_bucket = grouped.setdefault(zone_index, {})
        zone_bucket[match.group("base").upper()] = value

    zones: List[ZonedParameterInterval] = []
    for zone_index in sorted(grouped):
        params = grouped[zone_index]

        top_key = next((name for name in DEPTH_TOP_KEYS if name in params), None)
        bottom_key = next((name for name in DEPTH_BOTTOM_KEYS if name in params), None)
        if top_key is None or bottom_key is None:
            continue

        top_unit = _parameter_text(well, f"{top_key}_Z{zone_index}__UNIT") or well.depth_unit
        bottom_unit = _parameter_text(well, f"{bottom_key}_Z{zone_index}__UNIT") or well.depth_unit

        try:
            top_value = float(params[top_key])
            bottom_value = float(params[bottom_key])
        except (TypeError, ValueError):
            continue

        top = _convert_depth_between_units(top_value, top_unit, well.depth_unit)
        bottom = _convert_depth_between_units(bottom_value, bottom_unit, well.depth_unit)

        overrides: Dict[str, Any] = {}
        injected_parameters: Dict[str, Any] = {}
        unmapped: List[str] = []
        for base_name, raw_value in params.items():
            unit = _parameter_text(well, f"{base_name}_Z{zone_index}__UNIT")
            _apply_parameter_alias(
                base_name,
                raw_value,
                unit,
                overrides,
                injected_parameters,
                unmapped,
                calculation_profile=calculation_profile,
            )

        zones.append(
            ZonedParameterInterval(
                index=zone_index,
                top=top,
                bottom=bottom,
                top_unit=top_unit,
                bottom_unit=bottom_unit,
                raw_parameters=params,
                config_overrides=overrides,
                injected_parameters=injected_parameters,
                unmapped_parameters=sorted(set(unmapped)),
            )
        )

    return zones


def run_dynamic_workflow_with_zones(
    well: "Well",
    config: Optional[DynamicWorkflowConfig] = None,
) -> ZonedWorkflowRunResult:
    """Run the dynamic workflow with depth-varying LAS header parameters.

    Whole-well header parameters (e.g. ``RW``, ``BHT``, ``RHO_FL``) override
    settings left at their defaults; a value set explicitly in the config wins
    and the conflict is noted in ``warnings``. Zones from
    :func:`extract_zoned_parameters` are then re-run with their own settings
    and merged over their depth ranges.

    Parameters
    ----------
    well : Well
        Well to interpret. Calculated curves are added to it.
    config : DynamicWorkflowConfig, optional
        Base settings for every zone.

    Returns
    -------
    ZonedWorkflowRunResult
        Whole-well and per-zone results, zones, and parameter mapping notes.
    """
    profile = getattr(config, "calculation_profile", "current")
    zones = extract_zoned_parameters(well, calculation_profile=profile)
    global_overrides, global_unmapped = _extract_global_parameter_overrides(
        well, calculation_profile=profile
    )
    zoned_default_overrides, zoned_default_injected, zoned_default_notes = _derive_default_values_from_zones(zones)

    base_config = _prepare_batch_config(config)
    # Header-derived parameters override settings left at their defaults; an
    # explicitly configured value wins (e.g. a documented Rw over an
    # unreconciled header RW).
    override_notes: List[str] = []
    for field_name, value in global_overrides.items():
        if _left_at_default(base_config, field_name):
            _set_config_value(base_config, field_name, value)
        else:
            override_notes.append(
                f"Header parameter maps to {field_name}={value!r} but the "
                "config sets it explicitly; config value kept."
            )
    for field_name, value in zoned_default_overrides.items():
        if field_name not in global_overrides and _left_at_default(base_config, field_name):
            _set_config_value(base_config, field_name, value)
    _derive_sw_model(global_overrides, base_config)
    if "vsh_exp" not in global_overrides:
        _derive_sw_model(zoned_default_overrides, base_config)

    for name, value in zoned_default_injected.items():
        if well.get_parameter(name) is None:
            well.set_parameter(name, value)

    if base_config.td_ft is None and well.bottom is not None:
        base_config.td_ft = _depth_to_ft(well.bottom, well.depth_unit)

    base_result = run_dynamic_workflow(well, base_config)
    well_nd_clay = _well_nd_clay_point(base_config, base_result)
    well_code_curve = _well_lithology_code_curve(base_config, well)
    zone_results: List[DynamicWorkflowResult] = []
    warnings = list(zoned_default_notes)
    warnings.extend(override_notes)
    if global_unmapped:
        warnings.append(f"Unmapped global LAS parameters: {', '.join(global_unmapped)}")

    for zone in zones:
        interval_well = well.depth_slice(top=zone.top, bottom=zone.bottom)
        interval_config = _clone_config(base_config)

        for field_name, value in zone.config_overrides.items():
            _set_config_value(interval_config, field_name, value)
        _derive_sw_model(zone.config_overrides, interval_config)
        if well_nd_clay is not None and interval_config.nd_clay is None:
            interval_config.nd_clay = well_nd_clay
        if (
            well_code_curve is not None
            and interval_config.lithology_curve is None
            and get_curve(interval_well, [well_code_curve]) is not None
        ):
            interval_config.lithology_curve = well_code_curve

        if interval_config.td_ft is None:
            interval_config.td_ft = _depth_to_ft(zone.bottom, well.depth_unit)

        _apply_interval_metadata(interval_well, zone)
        zone_result = run_dynamic_workflow(interval_well, interval_config)
        _merge_curve_into_well(well, interval_well, zone.top, zone.bottom)
        zone_results.append(zone_result)

        if zone.unmapped_parameters:
            warnings.append(
                f"Zone Z{zone.index} unmapped LAS parameters: {', '.join(zone.unmapped_parameters)}"
            )

    return ZonedWorkflowRunResult(
        well=well,
        base_result=base_result,
        zone_results=zone_results,
        zones=zones,
        warnings=warnings,
        global_overrides=global_overrides,
        global_unmapped=global_unmapped,
        zoned_default_overrides=zoned_default_overrides,
        zoned_default_injected=zoned_default_injected,
    )


def _well_nd_clay_point(
    config: DynamicWorkflowConfig, result: DynamicWorkflowResult
) -> Optional[tuple]:
    """The whole-well N-D clay point that the zone runs reuse (``"current"`` only).

    Header zones often bracket a reservoir with few shales, where a pick of
    its own would fall back to the default point or rest on a handful of
    samples; the zones take the base run's pick instead, valid or not, so
    their VCL_ND follows the whole-well decision. None when the config sets
    ``nd_clay`` (the zones inherit it) or the base run picked no point.
    """
    if _reproduces_benchmark(config) or config.nd_clay is not None:
        return None
    step = next((item for item in result.steps_executed if item.step == "vclay"), None)
    picked = step.parameters.get("nd_clay_picked") if step is not None else None
    return tuple(picked) if picked else None


def _well_lithology_code_curve(config: DynamicWorkflowConfig, well: "Well") -> Optional[str]:
    """The base run's lithology code curve that the zone runs read by name (``"current"`` only).

    Under ``"current"`` a curve other than ``LITHOLOGY_GEOLINK`` is read as
    the lithology code log once its codes pass the code-log checks on the
    whole well; a zone may hold too few samples to pass them on its own, so
    the zones name the base run's curve and the well keeps one code curve.
    None under the benchmark profile, when the config names the curve (the
    zones inherit it), and for ``LITHOLOGY_GEOLINK``, which every run reads.
    """
    if _reproduces_benchmark(config) or config.lithology_curve is not None:
        return None
    name = interpreted_lithology_code_curve(well)
    return None if name == GEOLINK_LITHOLOGY_CURVE else name


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_safe(value.item())
        if np.issubdtype(value.dtype, np.number):
            data = np.asarray(value, dtype=float)
            finite = data[np.isfinite(data)]
            summary: Dict[str, Any] = {
                "type": "ndarray",
                "shape": list(value.shape),
                "finite_count": int(finite.size),
            }
            if finite.size:
                summary.update(
                    {
                        "min": float(np.nanmin(finite)),
                        "max": float(np.nanmax(finite)),
                        "mean": float(np.nanmean(finite)),
                    }
                )
            return summary
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "sample": [_json_safe(item) for item in value.flat[:10]],
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return None
        return numeric
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _workflow_runs(
    run_result: Optional[ZonedWorkflowRunResult],
) -> List[tuple[str, Optional[ZonedParameterInterval], DynamicWorkflowResult]]:
    if run_result is None:
        return []

    runs: List[tuple[str, Optional[ZonedParameterInterval], DynamicWorkflowResult]] = [
        ("base", None, run_result.base_result)
    ]
    for zone, zone_result in zip(run_result.zones, run_result.zone_results):
        runs.append((f"zone_z{zone.index}", zone, zone_result))
    return runs


def _extract_step_value(
    step_name: str,
    step,
    workflow_result: DynamicWorkflowResult,
    candidate_keys: List[str],
) -> Optional[str]:
    def _numeric_summary(value: Any) -> Optional[str]:
        try:
            data = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None

        if data.ndim == 0:
            numeric = float(data)
            return f"{numeric:.6g}" if np.isfinite(numeric) else None

        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return None
        min_value = float(np.nanmin(finite))
        max_value = float(np.nanmax(finite))
        if np.isclose(min_value, max_value):
            return f"{min_value:.6g}"
        return f"{min_value:.6g}-{max_value:.6g}"

    if step_name == "rw" and workflow_result.rw_computed is not None:
        return _numeric_summary(workflow_result.rw_computed)
    if step_name == "rmf" and workflow_result.rmf_computed is not None:
        return _numeric_summary(workflow_result.rmf_computed)

    for key in candidate_keys:
        summary = _numeric_summary(step.parameters.get(key))
        if summary is not None:
            return summary
    return None


def _collect_step_summaries(
    run_result: Optional[ZonedWorkflowRunResult],
    step_name: str,
    value_keys: List[str],
) -> tuple[List[str], List[str]]:
    methods: List[str] = []
    values: List[str] = []

    for scope, zone, workflow_result in _workflow_runs(run_result):
        prefix = scope if zone is None else f"Z{zone.index}"
        for step in workflow_result.steps_executed:
            if step.step != step_name:
                continue
            if step.method:
                methods.append(f"{prefix}:{step.method}")
            summary = _extract_step_value(step_name, step, workflow_result, value_keys)
            if summary is not None:
                values.append(f"{prefix}:{summary}")

    unique_methods = list(dict.fromkeys(methods))
    unique_values = list(dict.fromkeys(values))
    return unique_methods, unique_values


def _serialize_step_result(step) -> Dict[str, Any]:
    return _serialize_step_result_with_well(step, None)


def _curve_summary_from_array(
    data: Any,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    try:
        values = np.asarray(data, dtype=float)
    except (TypeError, ValueError):
        return {"present": False, "numeric": False}

    if mask is not None:
        if values.shape[0] != mask.shape[0]:
            return {"present": True, "numeric": True, "error": "mask/data length mismatch"}
        values = values[mask]

    finite = values[np.isfinite(values)]
    sample_count = int(values.size)
    summary: Dict[str, Any] = {
        "present": True,
        "numeric": True,
        "sample_count": sample_count,
        "finite_count": int(finite.size),
        "nan_count": int(np.count_nonzero(~np.isfinite(values))),
        "finite_percent": round(float(finite.size) / max(sample_count, 1) * 100.0, 4),
    }
    if finite.size:
        summary.update(
            {
                "min": round(float(np.nanmin(finite)), 4),
                "p05": round(float(np.nanpercentile(finite, 5)), 4),
                "p50": round(float(np.nanpercentile(finite, 50)), 4),
                "p95": round(float(np.nanpercentile(finite, 95)), 4),
                "max": round(float(np.nanmax(finite)), 4),
            }
        )
    return summary


def _curve_summary_for_log(
    well: Optional["Well"],
    curve_name: str,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if well is None:
        return {"present": False}
    curve = well.get(curve_name)
    if curve is None:
        return {"present": False}
    summary = _curve_summary_from_array(curve.data, mask=mask)
    summary["curve_name"] = curve.mnemonic
    summary["unit"] = curve.unit or ""
    description = getattr(curve.metadata, "description", "") if curve.metadata else ""
    summary["description"] = description
    return summary


def _curve_summaries_for_log(
    well: Optional["Well"],
    curve_names: Iterable[str],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for name in dict.fromkeys(str(curve) for curve in curve_names if curve):
        summaries[name] = _curve_summary_for_log(well, name, mask=mask)
    return summaries


def _mask_depth_intervals(well: "Well", mask: np.ndarray) -> List[str]:
    if getattr(well, "depth", None) is None or len(well.depth) != len(mask):
        return []
    depth_values = np.asarray(well.depth, dtype=float)
    intervals: List[str] = []
    start_idx: Optional[int] = None
    previous_idx: Optional[int] = None
    for idx, flag in enumerate(mask):
        if bool(flag) and start_idx is None:
            start_idx = idx
        if not bool(flag) and start_idx is not None and previous_idx is not None:
            intervals.append(
                f"{depth_values[start_idx]:.1f}-{depth_values[previous_idx]:.1f} {well.depth_unit or ''}".strip()
            )
            start_idx = None
        previous_idx = idx
    if start_idx is not None and previous_idx is not None:
        intervals.append(
            f"{depth_values[start_idx]:.1f}-{depth_values[previous_idx]:.1f} {well.depth_unit or ''}".strip()
        )
    return intervals


def _lithology_group_summaries_for_log(
    well: "Well",
    curve_names: Iterable[str],
) -> List[Dict[str, Any]]:
    lith_curve = well.get(interpreted_lithology_code_curve(well)) or well.get("LITHO")
    if lith_curve is None:
        return []

    values = np.asarray(lith_curve.data, dtype=object)
    summaries: List[Dict[str, Any]] = []
    for raw_value in sorted(set(values.tolist()), key=lambda item: str(item)):
        if raw_value is None:
            continue
        text = str(raw_value)
        if text.lower() in {"nan", "none", ""}:
            continue
        mask = values == raw_value
        if not np.any(mask):
            continue
        summaries.append(
            {
                "lithology_value": _json_safe(raw_value),
                "sample_count": int(np.count_nonzero(mask)),
                "depth_intervals": _mask_depth_intervals(well, mask)[:50],
                "curve_summaries": _curve_summaries_for_log(well, curve_names, mask=mask),
            }
        )
    return summaries


def _serialize_step_result_with_well(step, well: Optional["Well"]) -> Dict[str, Any]:
    input_curves = list(step.input_curves or [])
    output_curves = list(step.output_curves or [])
    return {
        "step": step.step,
        "success": step.success,
        "message": step.message,
        "method": step.method,
        "input_curves": _json_safe(input_curves),
        "output_curves": _json_safe(output_curves),
        "parameters": _json_safe(step.parameters),
        "decisions": _json_safe(step.decisions),
        "data": _json_safe(step.data),
        "input_curve_summaries": _curve_summaries_for_log(well, input_curves),
        "output_curve_summaries": _curve_summaries_for_log(well, output_curves),
    }


def _build_derived_log_notes(item: BatchLasResult) -> List[str]:
    notes: List[str] = [
        "ANALYSIS STARTED: Zone-aware dynamic workflow executed on standardized LAS data."
    ]

    run_result = item.result
    if run_result is not None:
        if run_result.global_overrides:
            notes.append(
                "GLOBAL LAS PARAMETERS USED: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(run_result.global_overrides.items())
                )
            )
        if run_result.zoned_default_overrides:
            notes.append(
                "UNCOVERED INTERVAL DEFAULTS FROM ZONES: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(run_result.zoned_default_overrides.items())
                )
            )
        if run_result.zoned_default_injected:
            notes.append(
                "UNCOVERED INTERVAL METADATA FROM ZONES: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(run_result.zoned_default_injected.items())
                )
            )
        if run_result.global_unmapped:
            notes.append(
                "UNMAPPED GLOBAL LAS PARAMETERS: "
                + ", ".join(run_result.global_unmapped)
            )

        for scope, zone, workflow_result in _workflow_runs(run_result):
            if zone is None:
                prefix = "BASE"
            else:
                prefix = f"ZONE Z{zone.index} {zone.top:.2f}-{zone.bottom:.2f} {run_result.well.depth_unit}"
                if zone.config_overrides:
                    notes.append(
                        f"{prefix}: config overrides "
                        + ", ".join(
                            f"{key}={value}" for key, value in sorted(zone.config_overrides.items())
                        )
                    )
                if zone.injected_parameters:
                    notes.append(
                        f"{prefix}: injected metadata "
                        + ", ".join(
                            f"{key}={value}" for key, value in sorted(zone.injected_parameters.items())
                        )
                    )
                if zone.unmapped_parameters:
                    notes.append(
                        f"{prefix}: unmapped LAS parameters "
                        + ", ".join(zone.unmapped_parameters)
                    )

            for step in workflow_result.steps_executed:
                header = f"{prefix} {step.step.upper()}"
                if step.method:
                    notes.append(f"{header}: method={step.method}")
                for decision in step.decisions:
                    notes.append(f"{header}: {decision}")
                if step.message and not step.success:
                    notes.append(f"{header}: error={step.message}")

            if workflow_result.steps_skipped:
                notes.append(
                    f"{prefix}: skipped steps = {', '.join(workflow_result.steps_skipped)}"
                )
            for warning in workflow_result.warnings:
                notes.append(f"{prefix} WARNING: {warning}")
            for error in workflow_result.errors:
                notes.append(f"{prefix} ERROR: {error}")

    for warning in item.warnings:
        notes.append(f"BATCH WARNING: {warning}")
    for error in item.errors:
        notes.append(f"BATCH ERROR: {error}")

    notes.append(
        "ANALYSIS COMPLETED: "
        f"success={item.success}, computed_curves={', '.join(item.computed_curves) or 'none'}."
    )

    return list(dict.fromkeys(notes))


def _write_json_log(
    item: BatchLasResult,
    relative_path: Path,
    output_root: Path,
) -> str:
    json_path = output_root / "workflow_json" / relative_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    run_result = item.result
    payload: Dict[str, Any] = {
        "source_file": item.source_file,
        "output_file": item.output_file,
        "success": item.success,
        "zones_applied": item.zones_applied,
        "input_curves": item.input_curves,
        "output_curves": item.output_curves,
        "computed_curves": item.computed_curves,
        "warnings": item.warnings,
        "errors": item.errors,
        "derived_log_computation_notes": _build_derived_log_notes(item),
    }

    if run_result is not None:
        final_summary_curves = sorted(
            set(AUDIT_TARGET_CURVES)
            | set(item.computed_curves)
            | {"SW", "SXO", "PHIE_CC", "PHIE", "PERM_CC", "PERM", "RW", "RMF", "VCL"}
        )
        payload.update(
            {
                "well_name": run_result.well.name,
                "global_parameter_overrides": _json_safe(run_result.global_overrides),
                "global_unmapped_parameters": _json_safe(run_result.global_unmapped),
                "uncovered_interval_defaults": _json_safe(run_result.zoned_default_overrides),
                "uncovered_interval_metadata": _json_safe(run_result.zoned_default_injected),
                "zones": [
                    {
                        "index": zone.index,
                        "top": zone.top,
                        "bottom": zone.bottom,
                        "top_unit": zone.top_unit,
                        "bottom_unit": zone.bottom_unit,
                        "raw_parameters": _json_safe(zone.raw_parameters),
                        "config_overrides": _json_safe(zone.config_overrides),
                        "injected_parameters": _json_safe(zone.injected_parameters),
                        "unmapped_parameters": _json_safe(zone.unmapped_parameters),
                    }
                    for zone in run_result.zones
                ],
                "workflow_runs": [
                    {
                        "scope": scope,
                        "zone_index": zone.index if zone is not None else None,
                        "zone_top": zone.top if zone is not None else None,
                        "zone_bottom": zone.bottom if zone is not None else None,
                        "success": workflow_result.success,
                        "rw_computed": _json_safe(workflow_result.rw_computed),
                        "rmf_computed": _json_safe(workflow_result.rmf_computed),
                        "warnings": _json_safe(workflow_result.warnings),
                        "errors": _json_safe(workflow_result.errors),
                        "steps_skipped": _json_safe(workflow_result.steps_skipped),
                        "step_results": [
                            _serialize_step_result_with_well(step, workflow_result.well)
                            for step in workflow_result.steps_executed
                        ],
                    }
                    for scope, zone, workflow_result in _workflow_runs(run_result)
                ],
                "final_curve_summaries": _curve_summaries_for_log(
                    run_result.well,
                    final_summary_curves,
                ),
                "lithology_group_summaries": _lithology_group_summaries_for_log(
                    run_result.well,
                    final_summary_curves,
                ),
            }
        )

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)

    return str(json_path)


def _interpret_las_file(
    file_path: Path, config: Optional[DynamicWorkflowConfig]
) -> tuple[ZonedWorkflowRunResult, set[str]]:
    """Read, standardize and interpret one LAS file of a batch, ready for export.

    Under the benchmark profile the file gets the calculations of the eval
    package's re-run when its UWI, WELL header or file name names one of the
    package's re-run wells, and otherwise those of its main run (see
    :attr:`DynamicWorkflowConfig.calculation_profile`), once for the file.

    Returns
    -------
    tuple
        The zoned workflow result and the curve names the standardized input
        held.
    """
    well = read_las(file_path)
    rerun = _uses_rerun_calculations(well, file_stem=file_path.stem)
    main_run = _reproduces_benchmark(config) and not rerun
    well = standardize_curves(
        well,
        rename_curves=True,
        convert_units=True,
        # The main run's interpreted LAS kept the source's own label where the
        # standard unit is equivalent (e.g. m3/m3, gAPI).
        relabel_equivalent_units=not main_run,
    )
    input_curves = set(well.curves)
    with _rerun_calculations(rerun):
        workflow_result = run_dynamic_workflow_with_zones(well, config=config)
    _finalize_export_curves(
        workflow_result.well,
        input_curves,
        drop_redundant_porosity=main_run,
    )
    return workflow_result, input_curves


def process_las_directory_with_workflow(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.las",
    recursive: bool = False,
    config: Optional[DynamicWorkflowConfig] = None,
    well_names: Optional[Sequence[str]] = None,
) -> List[BatchLasResult]:
    """Interpret every LAS file in a folder with zone-aware settings.

    Each file is run through :func:`run_dynamic_workflow_with_zones`; the
    interpreted LAS and a JSON log are written under ``output_dir`` with the
    input folder structure. Files that fail are reported, not raised.

    Parameters
    ----------
    input_dir : str or Path
        Folder containing the LAS files.
    output_dir : str or Path
        Folder for interpreted LAS files and JSON logs (created if needed).
    pattern : str, optional
        Glob pattern selecting input files (default ``"*.las"``).
    recursive : bool, default False
        Also search subfolders.
    config : DynamicWorkflowConfig, optional
        Base settings for every well. Under ``calculation_profile="benchmark"``
        the eval package's two re-run wells, recognized by their UWI, WELL
        header or file name, get the re-run's calculations.
    well_names : sequence of str, optional
        Only interpret files whose name (without extension) matches one of
        these well names.

    Returns
    -------
    list of BatchLasResult
        One entry per input file. Summarize with :func:`write_batch_report`
        and :func:`write_curve_audit_report`.

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` is not an existing folder.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"LAS input folder not found: {input_path}")
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))
    if well_names:
        wanted = {str(name).strip().lower().replace("/", "_") for name in well_names}
        files = [item for item in files if item.stem.strip().lower() in wanted]
    if not files:
        _warn(
            f"No files matching {pattern!r} in {input_path}"
            + ("" if recursive else "; pass recursive=True to search subfolders"),
            UserWarning,
            stacklevel=2,
        )

    results: List[BatchLasResult] = []
    for file_path in files:
        relative_path = file_path.relative_to(input_path)
        export_path = output_path / relative_path
        export_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            workflow_result, input_curves = _interpret_las_file(file_path, config)
            export_las(workflow_result.well, export_path)

            computed_curves = sorted(set(workflow_result.well.curves) - input_curves)
            warnings = list(workflow_result.warnings)
            warnings.extend(workflow_result.base_result.warnings)
            for zone_result in workflow_result.zone_results:
                warnings.extend(zone_result.warnings)

            overall_success = workflow_result.base_result.success and all(
                zone_result.success for zone_result in workflow_result.zone_results
            )
            json_log_file = _write_json_log(
                BatchLasResult(
                    source_file=str(file_path),
                    output_file=str(export_path),
                    success=overall_success,
                    warnings=warnings,
                    input_curves=sorted(input_curves),
                    output_curves=sorted(workflow_result.well.curves),
                    computed_curves=computed_curves,
                    zones_applied=len(workflow_result.zones),
                    result=workflow_result,
                ),
                relative_path,
                output_path,
            )

            results.append(
                BatchLasResult(
                    source_file=str(file_path),
                    output_file=str(export_path),
                    success=overall_success,
                    warnings=warnings,
                    input_curves=sorted(input_curves),
                    output_curves=sorted(workflow_result.well.curves),
                    computed_curves=computed_curves,
                    zones_applied=len(workflow_result.zones),
                    json_log_file=json_log_file,
                    result=workflow_result,
                )
            )
        except Exception as exc:
            failure_item = BatchLasResult(
                source_file=str(file_path),
                output_file=str(export_path),
                success=False,
                errors=[str(exc)],
            )
            failure_item.json_log_file = _write_json_log(failure_item, relative_path, output_path)
            results.append(
                failure_item
            )

    return results


def write_workflow_json_traces_directory(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    pattern: str = "*.las",
    recursive: bool = False,
    config: Optional[DynamicWorkflowConfig] = None,
    well_names: Optional[Sequence[str]] = None,
) -> List[BatchLasResult]:
    """Run the workflow and write fresh per-well workflow_json traces only.

    This does not export LAS files or reports. It is intended for audit/training
    traces after processed LAS/context generation, where the user wants one
    JSON per well with full step decisions, parameters, inputs, outputs, and
    lithology-group summaries.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir is not None else input_path
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))
    if well_names:
        wanted = {str(name).lower() for name in well_names}
        files = [
            path for path in files
            if path.stem.lower() in wanted or path.name.lower() in wanted
        ]

    results: List[BatchLasResult] = []
    for file_path in files:
        relative_path = file_path.relative_to(input_path)
        try:
            workflow_result, input_curves = _interpret_las_file(file_path, config)

            computed_curves = sorted(set(workflow_result.well.curves) - input_curves)
            warnings = list(workflow_result.warnings)
            warnings.extend(workflow_result.base_result.warnings)
            for zone_result in workflow_result.zone_results:
                warnings.extend(zone_result.warnings)
            errors = list(workflow_result.base_result.errors)
            for zone_result in workflow_result.zone_results:
                errors.extend(zone_result.errors)

            overall_success = workflow_result.base_result.success and all(
                zone_result.success for zone_result in workflow_result.zone_results
            )
            item = BatchLasResult(
                source_file=str(file_path),
                output_file=str(output_path / relative_path),
                success=overall_success,
                warnings=warnings,
                errors=errors,
                input_curves=sorted(input_curves),
                output_curves=sorted(workflow_result.well.curves),
                computed_curves=computed_curves,
                zones_applied=len(workflow_result.zones),
                result=workflow_result,
            )
            item.json_log_file = _write_json_log(item, relative_path, output_path)
            results.append(item)
        except Exception as exc:
            failure_item = BatchLasResult(
                source_file=str(file_path),
                output_file=str(output_path / relative_path),
                success=False,
                errors=[str(exc)],
            )
            failure_item.json_log_file = _write_json_log(failure_item, relative_path, output_path)
            results.append(failure_item)

    return results


def _resolve_report_path(report_path: str | Path) -> Path:
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with report_file.open("w", newline="", encoding="utf-8"):
            pass
        return report_file
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = report_file.with_name(f"{report_file.stem}_{timestamp}{report_file.suffix}")
        with fallback.open("w", newline="", encoding="utf-8"):
            pass
        return fallback


def write_batch_report(results: Iterable[BatchLasResult], report_path: str | Path) -> Path:
    """Write one CSV row per batch file: status, zones, Rw/Rmf, curves, messages.

    Parameters
    ----------
    results : iterable of BatchLasResult
        Output of :func:`process_las_directory_with_workflow`.
    report_path : str or Path
        CSV file to write (a folder gets a default file name).

    Returns
    -------
    Path
        The written report.
    """
    report_file = _resolve_report_path(report_path)

    with report_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_file",
                "output_file",
                "success",
                "zones_applied",
                "rw_methods",
                "rw_values",
                "rmf_methods",
                "rmf_values",
                "computed_curves",
                "json_log_file",
                "warnings",
                "errors",
            ]
        )
        for item in results:
            rw_methods, rw_values = _collect_step_summaries(item.result, "rw", ["rw_at_formation", "rw_computed", "rw", "rw_default"])
            rmf_methods, rmf_values = _collect_step_summaries(
                item.result,
                "rmf",
                ["rmf_at_formation", "rmf", "rmf_at_surface"],
            )
            writer.writerow(
                [
                    item.source_file,
                    item.output_file,
                    item.success,
                    item.zones_applied,
                    ";".join(rw_methods),
                    ";".join(rw_values),
                    ";".join(rmf_methods),
                    ";".join(rmf_values),
                    ";".join(item.computed_curves),
                    item.json_log_file,
                    " | ".join(item.warnings),
                    " | ".join(item.errors),
                ]
            )

    return report_file


def write_curve_audit_report(
    results: Iterable[BatchLasResult],
    report_path: str | Path,
    target_curves: Optional[List[str]] = None,
) -> Path:
    """Write a CSV stating, per file and curve, whether it was computed and why not.

    Parameters
    ----------
    results : iterable of BatchLasResult
        Output of :func:`process_las_directory_with_workflow`.
    report_path : str or Path
        CSV file to write (a folder gets a default file name).
    target_curves : list of str, optional
        Curves to audit. Defaults to the standard interpretation products.

    Returns
    -------
    Path
        The written report.
    """
    report_file = _resolve_report_path(report_path)
    curves = target_curves or AUDIT_TARGET_CURVES

    with report_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_file",
                "output_file",
                "curve",
                "status",
                "reason",
            ]
        )

        for item in results:
            input_curves = set(item.input_curves)
            output_curves = set(item.output_curves)
            computed_curves = set(item.computed_curves)
            reason = " | ".join(item.warnings + item.errors)

            step_results = {}
            skipped_steps = set()
            if item.result is not None:
                all_results = [item.result.base_result, *item.result.zone_results]
                for workflow_result in all_results:
                    skipped_steps.update(workflow_result.steps_skipped)
                    for step in workflow_result.steps_executed:
                        if step.step not in step_results or step.success:
                            step_results[step.step] = step

            def _curve_reason(curve: str) -> str:
                step_name = CURVE_TO_STEP.get(curve)
                step = step_results.get(step_name) if step_name else None

                if step_name and step_name in skipped_steps:
                    return f"{step_name} step skipped"
                if step is not None and not step.success:
                    return step.message or f"{step_name} step failed"
                if curve in {"PHIE_CC", "PERM_CC"} and step is not None:
                    same_well_models = step.parameters.get("same_well_core_models")
                    peer_models = step.parameters.get("peer_core_models")
                    strategy = step.parameters.get("strategy") or step.parameters.get("method")
                    if (same_well_models or 0) == 0 and (peer_models or 0) == 0:
                        return f"no usable core permeability model; strategy={strategy}"
                if curve == "SWIRR_APP" and step is not None:
                    swirr_source = step.parameters.get("swirr_source")
                    if swirr_source:
                        return f"Swirr source was {swirr_source}"
                if curve == "SW" and step_name == "saturation" and step is None:
                    return "saturation step did not run"
                if curve == "PERM" and step_name == "permeability" and step is None:
                    return "permeability step did not run"
                return reason

            for curve in curves:
                if curve in computed_curves:
                    status = "computed"
                    curve_reason = ""
                elif curve in output_curves and curve in input_curves:
                    status = "already_present"
                    curve_reason = ""
                elif curve in output_curves:
                    status = "present"
                    curve_reason = ""
                else:
                    status = "missing"
                    curve_reason = _curve_reason(curve)

                writer.writerow(
                    [
                        item.source_file,
                        item.output_file,
                        curve,
                        status,
                        curve_reason,
                    ]
                )

    return report_file
