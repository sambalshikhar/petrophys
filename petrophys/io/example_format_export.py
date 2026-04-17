"""
Export helpers for writing delivery LAS files and paired JSON summaries.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import csv
import json
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from petrophys.analysis import HydrocarbonAnalysis, WellboreQC
from petrophys.io.las_reader import read_las
from petrophys.qaqc.standardization import standardize_curves
from petrophys.utils.analysis_helpers import CategoryAnalysisResult, build_section


NULL_VALUE = -9999.0

PREFERRED_CURVE_ORDER = [
    "DEPT",
    "CALI",
    "BS",
    "DRHO",
    "DTC",
    "DTE",
    "GR",
    "LITHOLOGY_GEOLINK",
    "NPHI",
    "PEF",
    "RHOB",
    "RDEP",
    "RMED",
    "RSHA",
    "RXO",
    "SP",
    "TMPC",
    "PHIC",
    "GDEN",
    "KH_CORE",
    "KV_CORE",
    "SWC",
    "SOC",
    "PHID",
    "PHIN",
    "PHIS",
    "PHIT",
    "PHIE",
    "PHIE_CORECAL",
    "PHI_SEC",
    "RW",
    "RMF",
    "RWA",
    "SAL",
    "SALAPP",
    "SALMUDAPP",
    "SW",
    "SXO",
    "PERM",
    "PERM_CC",
    "VCL",
    "VQUAR",
    "VCALC",
    "VDOLO",
    "VANHY",
    "RQI",
    "PHI_Z",
    "FZI",
    "SWIRR_APP",
    "INVASION_RATIO",
]

FAMILY_LABELS = {
    "DEPT": "Measured Depth",
    "CALI": "Caliper",
    "BS": "Bit Size",
    "DRHO": "Bulk Density Correction",
    "DTC": "Compressional Slowness",
    "DTE": "Shear Slowness",
    "GR": "Gamma Ray",
    "LITHOLOGY_GEOLINK": "Lithology",
    "NPHI": "Neutron Porosity",
    "PEF": "Photoelectric Factor",
    "RHOB": "Bulk Density",
    "RDEP": "Resistivity - Deep",
    "RMED": "Resistivity - Medium",
    "RSHA": "Resistivity - Shallow",
    "RXO": "Resistivity - Flushed Zone",
    "SP": "Spontaneous Potential",
    "TMPC": "Temperature",
    "PHIC": "Core Porosity",
    "GDEN": "Grain Density",
    "KH_CORE": "Core Permeability",
    "KV_CORE": "Core Permeability",
    "SWC": "Core Water Saturation",
    "SOC": "Core Oil Saturation",
    "PHID": "Density Porosity",
    "PHIN": "Neutron Porosity",
    "PHIS": "Sonic Porosity",
    "PHIT": "Total Porosity",
    "PHIE": "Effective Porosity",
    "PHIE_CORECAL": "Core-Calibrated Porosity",
    "PHI_SEC": "Secondary Porosity",
    "RW": "Formation Water Resistivity",
    "RMF": "Mud Filtrate Resistivity",
    "RWA": "Apparent Water Resistivity",
    "SAL": "Formation Water Salinity",
    "SALAPP": "Apparent Water Salinity",
    "SALMUDAPP": "Apparent Mud Filtrate Salinity",
    "SW": "Water Saturation",
    "SXO": "Flushed Zone Water Saturation",
    "PERM": "Permeability",
    "PERM_CC": "Core-Calibrated Permeability",
    "VCL": "Clay Volume",
    "VQUAR": "Quartz Volume",
    "VCALC": "Calcite Volume",
    "VDOLO": "Dolomite Volume",
    "VANHY": "Anhydrite Volume",
    "RQI": "Reservoir Quality Index",
    "PHI_Z": "Normalized Porosity",
    "FZI": "Flow Zone Indicator",
    "SWIRR_APP": "Apparent Irreducible Water Saturation",
    "INVASION_RATIO": "Invasion Ratio Rxo/Rt",
}

ALIASED_SOURCE_CURVES = {
    "DTS",
    "TEMP",
    "SWA",
    "SWXOA",
    "VCLAY",
    "VSH",
    "VDOL",
}

EXCLUDED_LAS_CURVES = {
    "BAD_HOLE",
    "BADHOLEFLAG",
    "CALI_SMOOTH",
    "CALI_ROUGHNESS",
    "COMPOSITE_CLAY",
    "CONSISTENCY_SCORE",
    "CROSSOVER_FLAG",
    "GT_LITHOLOGY_GROUP_CODE",
    "IMPERM_FLAG",
    "LIMESTONE_INDICATOR",
    "DOLOMITE_INDICATOR",
    "CARBONATE_INDICATOR",
    "GR_CLAY_INDICATOR",
    "NPHI_CLAY_INDICATOR",
    "PEF_CLAY_INDICATOR",
    "MHI",
    "NET_PAY",
    "NET_RES",
    "NET_ROCK",
    "PAY_FLAG_PETROPHYSICAL",
    "PRIME_GAS_TARGET",
    "PRIME_OIL_TARGET",
    "RESFLAG",
    "SEALFLAG",
    "SWA",
    "SWXOA",
    "SXO-ARCHIE",
    "SXO-INDONESIA",
    "SXO-NIGERIA_SCHLUMBERGER",
    "SXO-NORTH_SEA",
    "TEMP",
    "VCLAY",
    "VCLGR",
    "VCLSP",
    "VCLND",
    "VCL_GR",
    "VCL_SP",
    "VCL_ND",
    "VDOL",
    "VLIM",
    "VSH",
    "VSHGR",
    "VSHND",
    "VP_VS_RATIO",
    "VSAND",
}

DUPLICATE_RAW_CURVES = {
    "DCAL",
    "DT",
}

PRIMARY_LOGS_FOR_COMPLETENESS = [
    "CALI",
    "BS",
    "DRHO",
    "GR",
    "SP",
    "NPHI",
    "RHOB",
    "RDEP",
    "RMED",
    "RSHA",
    "RXO",
    "DTC",
    "DTS",
    "PEF",
    "TMPC",
    "RMF",
    "RW",
    "PHIC",
    "KH_CORE",
    "KV_CORE",
    "LITHOLOGY_GEOLINK",
]


@dataclass
class ExportCurveSpec:
    mnemonic: str
    unit: str
    description: str
    family: str
    data: np.ndarray
    source_curves: List[str] = field(default_factory=list)


@dataclass
class ExampleFormatExportResult:
    source_file: str
    output_file: str
    success: bool
    json_file: str = ""
    curves_exported: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _normalize_numeric_array(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)

    output = np.full(arr.shape, np.nan, dtype=float)
    for idx, value in np.ndenumerate(arr):
        try:
            output[idx] = float(value)
        except (TypeError, ValueError):
            output[idx] = np.nan
    return output


def _curve_description(mnemonic: str, curve) -> str:
    if curve is not None and curve.metadata.description:
        return str(curve.metadata.description)
    return FAMILY_LABELS.get(mnemonic, mnemonic.replace("_", " ").title())


def _curve_family(mnemonic: str, description: str) -> str:
    return FAMILY_LABELS.get(mnemonic, description or mnemonic.replace("_", " ").title())


def _convert_temperature_to_celsius(data: np.ndarray, unit: str) -> np.ndarray:
    text = (unit or "").strip().lower()
    numeric = _normalize_numeric_array(data)
    if "f" in text and "c" not in text:
        return (numeric - 32.0) * 5.0 / 9.0
    return numeric


def _first_curve(well: "Well", candidates: Sequence[str]):
    for mnemonic in candidates:
        curve = well.get(mnemonic)
        if curve is not None:
            return curve
    return None


def _first_sxo_curve(well: "Well"):
    direct = _first_curve(well, ["SXO", "SWXOA"])
    if direct is not None:
        return direct
    for mnemonic in well.curves:
        if mnemonic.upper().startswith("SXO-"):
            curve = well.get(mnemonic)
            if curve is not None:
                return curve
    return None


def _build_alias_specs(well: "Well") -> List[ExportCurveSpec]:
    specs: List[ExportCurveSpec] = []

    dte = _first_curve(well, ["DTE", "DTS"])
    if dte is not None:
        specs.append(
            ExportCurveSpec(
                mnemonic="DTE",
                unit=dte.unit or "us/ft",
                description=_curve_description("DTE", dte),
                family=FAMILY_LABELS["DTE"],
                data=_normalize_numeric_array(dte.data),
                source_curves=[dte.mnemonic],
            )
        )

    sw = _first_curve(well, ["SW", "SWA"])
    if sw is not None:
        specs.append(
            ExportCurveSpec(
                mnemonic="SW",
                unit="v/v",
                description="Water Saturation",
                family=FAMILY_LABELS["SW"],
                data=_normalize_numeric_array(sw.data),
                source_curves=[sw.mnemonic],
            )
        )

    sxo = _first_sxo_curve(well)
    if sxo is not None:
        specs.append(
            ExportCurveSpec(
                mnemonic="SXO",
                unit="v/v",
                description="Flushed Zone Water Saturation",
                family=FAMILY_LABELS["SXO"],
                data=_normalize_numeric_array(sxo.data),
                source_curves=[sxo.mnemonic],
            )
        )

    temp = _first_curve(well, ["TMPC", "TEMP"])
    if temp is not None:
        specs.append(
            ExportCurveSpec(
                mnemonic="TMPC",
                unit="degC",
                description="Temperature",
                family=FAMILY_LABELS["TMPC"],
                data=_convert_temperature_to_celsius(temp.data, temp.unit),
                source_curves=[temp.mnemonic],
            )
        )

    vcl = _first_curve(well, ["VCL", "VCLAY", "VSH"])
    if vcl is not None:
        specs.append(
            ExportCurveSpec(
                mnemonic="VCL",
                unit="v/v",
                description="Clay Volume",
                family=FAMILY_LABELS["VCL"],
                data=_normalize_numeric_array(vcl.data),
                source_curves=[vcl.mnemonic],
            )
        )

    return specs


def _direct_curve_spec(well: "Well", mnemonic: str) -> Optional[ExportCurveSpec]:
    curve = well.get(mnemonic)
    if curve is None:
        return None
    description = _curve_description(mnemonic, curve)
    return ExportCurveSpec(
        mnemonic=mnemonic,
        unit=curve.unit or "",
        description=description,
        family=_curve_family(mnemonic, description),
        data=_normalize_numeric_array(curve.data),
        source_curves=[curve.mnemonic],
    )


def _should_exclude_curve(mnemonic: str) -> bool:
    upper = mnemonic.upper()
    if upper in EXCLUDED_LAS_CURVES:
        return True
    if upper.startswith("SXO-"):
        return True
    return False


def _collect_export_curves(
    well: "Well",
    original_curves: Optional[set[str]] = None,
) -> List[ExportCurveSpec]:
    emitted: set[str] = set()
    consumed: set[str] = set()
    specs_by_name: dict[str, ExportCurveSpec] = {}

    raw_curves = {name.upper() for name in (original_curves or set(well.curves))}

    for spec in _build_alias_specs(well):
        specs_by_name[spec.mnemonic] = spec
        consumed.update(source.upper() for source in spec.source_curves)

    results: List[ExportCurveSpec] = []
    for mnemonic in PREFERRED_CURVE_ORDER:
        spec = specs_by_name.get(mnemonic)
        if spec is None:
            spec = _direct_curve_spec(well, mnemonic)
        if spec is None or spec.mnemonic in emitted or _should_exclude_curve(spec.mnemonic):
            continue
        results.append(spec)
        emitted.add(spec.mnemonic)

    for mnemonic in well.curves:
        upper = mnemonic.upper()
        if upper not in raw_curves:
            continue
        if upper in emitted or _should_exclude_curve(upper):
            continue
        if upper in consumed and upper in ALIASED_SOURCE_CURVES:
            continue
        if upper in DUPLICATE_RAW_CURVES:
            canonical = "CALI" if upper == "DCAL" else "DTC"
            if well.get(canonical) is not None:
                continue

        spec = _direct_curve_spec(well, upper)
        if spec is None:
            continue
        results.append(spec)
        emitted.add(spec.mnemonic)

    return results


def _format_header_line(name: str, unit: str, value: str, description: str) -> str:
    return f"{name:<18}.{unit:<8} {value:<30} : {description}"


def _format_curve_line(mnemonic: str, unit: str, description: str) -> str:
    return f"{mnemonic:<18}.{unit:<8} {'':<30} : {description}"


def _format_value(value) -> str:
    if value is None:
        return f"{NULL_VALUE:g}"

    if isinstance(value, (np.floating, float, np.integer, int)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return f"{NULL_VALUE:g}"
        return f"{numeric:.6f}".rstrip("0").rstrip(".")

    text = str(value).strip()
    return text if text else f"{NULL_VALUE:g}"


def _well_header_value(value: Optional[str]) -> str:
    return str(value or "").strip()


def export_example_format_las(
    well: "Well",
    filepath: str | Path,
    set_name: Optional[str] = None,
    original_curves: Optional[set[str]] = None,
) -> Path:
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    curve_specs = _collect_export_curves(well, original_curves=original_curves)
    date_value = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    set_value = set_name or f"{well.name} Petrophys Export"

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("~Version\n")
        handle.write(_format_header_line("VERS", "", "2.0", "CWLS LOG ASCII STANDARD - VERSION 2.0") + "\n")
        handle.write(_format_header_line("WRAP", "", "NO", "ONE LINE PER DEPTH STEP") + "\n")
        handle.write(_format_header_line("DLM", "", "SPACE", "DELIMITING CHARACTER(SPACE TAB OR COMMA)") + "\n")

        handle.write("~Well Information\n")
        handle.write("#_______________________________________________________________________________\n")
        handle.write("#\n")
        handle.write("#PARAMETER_NAME    .UNIT     VALUE                         : DESCRIPTION\n")
        handle.write("#_______________________________________________________________________________\n")
        handle.write(_format_header_line("STRT", well.depth_unit, _format_value(well.top), "First reference value") + "\n")
        handle.write(_format_header_line("STOP", well.depth_unit, _format_value(well.bottom), "Last reference value") + "\n")
        handle.write(_format_header_line("STEP", well.depth_unit, _format_value(well.step), "Step increment") + "\n")
        handle.write(_format_header_line("NULL", "", f"{NULL_VALUE:g}", "Missing value") + "\n")
        handle.write(_format_header_line("COMP", "", _well_header_value(well.header.company), "Company") + "\n")
        handle.write(_format_header_line("WELL", "", _well_header_value(well.name), "Well name") + "\n")
        handle.write(_format_header_line("FLD", "", _well_header_value(well.header.field), "Field") + "\n")
        handle.write(_format_header_line("LOC", "", _well_header_value(well.header.location), "Location") + "\n")
        handle.write(_format_header_line("PROV", "", "", "Province") + "\n")
        handle.write(_format_header_line("SRVC", "", "", "Service Company") + "\n")
        handle.write(_format_header_line("DATE", "", date_value, "Date") + "\n")
        handle.write(_format_header_line("UWI", "", _well_header_value(well.header.uwi or well.name), "Unique Well ID") + "\n")
        handle.write(_format_header_line("API", "", _well_header_value(well.header.api), "API NUMBER") + "\n")
        handle.write(_format_header_line("ORIGINALWELLNAME", "", _well_header_value(well.name), "") + "\n")

        handle.write("~Parameter Information Block\n")
        handle.write("#_______________________________________________________________________________\n")
        handle.write("#\n")
        handle.write("#PARAMETER_NAME    .UNIT     VALUE                         : DESCRIPTION\n")
        handle.write("#_______________________________________________________________________________\n")
        handle.write(_format_header_line("SET", "", set_value, "") + "\n")
        handle.write(_format_header_line("TLFamily_DEPT", "", FAMILY_LABELS["DEPT"], "") + "\n")
        for spec in curve_specs:
            handle.write(_format_header_line(f"TLFamily_{spec.mnemonic}", "", spec.family, "") + "\n")

        handle.write("~Curve Information\n")
        handle.write("#_______________________________________________________________________________\n")
        handle.write("#\n")
        handle.write("#LOGNAME           .UNIT     LOG_ID                        : DESCRIPTION\n")
        handle.write("#_______________________________________________________________________________\n")
        handle.write(_format_curve_line("DEPT", well.depth_unit, "DEPTH") + "\n")
        for spec in curve_specs:
            handle.write(_format_curve_line(spec.mnemonic, spec.unit, spec.description) + "\n")

        handle.write("~Ascii\n")
        depth = _normalize_numeric_array(well.depth)
        for row_index in range(well.n_samples):
            row = [_format_value(depth[row_index])]
            for spec in curve_specs:
                row.append(_format_value(spec.data[row_index]))
            handle.write(" ".join(row) + "\n")

    return output_path


def _binary_flag_section(
    well: "Well",
    curve_names: Sequence[str],
    section_name: str,
    description: str,
    methodology: str,
    calculation_thresholds: str,
    positive_label: str,
    negative_label: str,
) -> Optional[tuple[str, Any]]:
    curve = _first_curve(well, list(curve_names))
    if curve is None:
        return None

    values = _normalize_numeric_array(curve.data)
    labels = np.full(well.n_samples, "NO_DATA", dtype=object)
    valid = np.isfinite(values)
    labels[valid & (values > 0.5)] = positive_label
    labels[valid & (values <= 0.5)] = negative_label

    section = build_section(
        well,
        labels,
        description=description,
        methodology=methodology,
        calculation_thresholds=calculation_thresholds,
    )
    return section_name, section


def _workflow_flag_analysis(well: "Well") -> Optional[CategoryAnalysisResult]:
    sections = {}

    reservoir = _binary_flag_section(
        well,
        ["RESFLAG", "NET_RES"],
        "reservoir_flag_screening",
        "Reservoir screening from the workflow reservoir flag.",
        "Uses the workflow's reservoir flag derived from VCL and PHIE cutoffs.",
        "RESERVOIR_FLAG when reservoir flag > 0.5, otherwise NON_RESERVOIR_FLAG.",
        "RESERVOIR_FLAG",
        "NON_RESERVOIR_FLAG",
    )
    if reservoir is not None:
        name, section = reservoir
        sections[name] = section

    seal = _binary_flag_section(
        well,
        ["SEALFLAG"],
        "seal_flag_screening",
        "Seal screening from the workflow seal flag.",
        "Uses the workflow seal flag generated from high clay volume intervals.",
        "SEAL_FLAG when VCL >= 0.8, otherwise NON_SEAL_FLAG.",
        "SEAL_FLAG",
        "NON_SEAL_FLAG",
    )
    if seal is not None:
        name, section = seal
        sections[name] = section

    crossover = _binary_flag_section(
        well,
        ["CROSSOVER_FLAG"],
        "crossover_flag_screening",
        "Gas crossover screening from neutron-density separation.",
        "Uses the workflow crossover flag where PHID - PHIN > 0.01.",
        "CROSSOVER_FLAG when PHID - PHIN > 0.01, otherwise NO_CROSSOVER.",
        "CROSSOVER_FLAG",
        "NO_CROSSOVER",
    )
    if crossover is not None:
        name, section = crossover
        sections[name] = section

    if not sections:
        return None

    return CategoryAnalysisResult(
        category_key="WORKFLOW_FLAG_ANALYSIS",
        sections=sections,
    )


def _json_safe(value: Any) -> Any:
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


def _ordered_unique(values: Sequence[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _all_step_results(workflow_result, step_name: Optional[str] = None) -> List[Any]:
    step_results: List[Any] = []
    all_runs = [workflow_result.base_result, *workflow_result.zone_results]
    for run in all_runs:
        for step in run.steps_executed:
            if step_name is not None and step.step != step_name:
                continue
            step_results.append(step)
    return step_results


def _friendly_join(items: Sequence[str]) -> str:
    cleaned = [item for item in items if item]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _format_numeric(value: float, decimals: int = 4) -> str:
    text = f"{float(value):.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _format_numeric_summary(values: Sequence[Any], decimals: int = 4, unit: str = "") -> str:
    numerics: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            numerics.append(numeric)
    if not numerics:
        return ""

    rounded = _ordered_unique(round(value, decimals) for value in numerics)
    suffix = f" {unit}".rstrip()
    if len(rounded) == 1:
        return f"{_format_numeric(rounded[0], decimals)}{suffix}"
    return (
        f"{_format_numeric(min(numerics), decimals)} to "
        f"{_format_numeric(max(numerics), decimals)}{suffix}"
    )


def _step_parameter_values(step_results: Sequence[Any], key: str) -> List[Any]:
    values: List[Any] = []
    for step in step_results:
        value = step.parameters.get(key)
        if value is None:
            continue
        values.append(value)
    return values


def _method_label(name: str) -> str:
    mapping = {
        "mud_filtrate_salinity_conversion": "CLFL-to-RMF salinity conversion",
        "las_metadata": "direct LAS metadata",
        "las_metadata_arps_temperature_correction": "LAS RMF with RMFT temperature correction",
        "arps_temperature_correction": "explicit RMF temperature correction",
        "rm_to_rmf_conversion_factor_0.75": "RM-to-RMF conversion (fresh mud factor 0.75)",
        "rm_to_rmf_conversion_factor_0.85": "RM-to-RMF conversion (salt mud factor 0.85)",
        "OBM_default": "OBM non-conductive default",
        "user_provided": "direct RW value",
        "user_provided_temperature_corrected": "direct RW with temperature correction",
        "las_metadata_temperature_corrected": "LAS RW with RWT temperature correction",
        "salinity_conversion": "salinity-to-RW conversion",
        "sp_clean_water": "SP clean-water method",
        "rxo_rt_clean_water": "Rxo/Rt clean-water method",
        "default_fallback": "default RW fallback (0.05 ohm.m)",
        "component_based": "component-based porosity workflow",
        "linear_gradient_from_BHT": "linear temperature gradient from BHT",
        "constant_BHT_metadata": "constant BHT metadata temperature",
        "caliper_drho_flags": "caliper roughness with DRHO fallback",
        "linear": "linear VCL method",
        "auto": "auto permeability workflow",
        "moveable_HC_index": "MHI from SW and SXO",
        "cutoff_flags": "cutoff-based flagging",
    }
    return mapping.get(name, name.replace("_", " "))


def _model_reason(model: str, vcl_cutoff: Optional[float]) -> str:
    cutoff_text = f"{float(vcl_cutoff):.2f}" if vcl_cutoff is not None else "configured cutoff"
    mapping = {
        "archie": f"Archie at depth samples where VCL < {cutoff_text}",
        "indonesia": f"Indonesia at shaly clastic samples where VCL >= {cutoff_text} and no higher-salinity shaly model input was available",
        "simandoux": f"Simandoux at shaly clastic samples where VCL >= {cutoff_text} and saline shaly routing was selected",
        "north_sea": f"North Sea shaly-sand equation at samples where shale-exponent inputs were available",
        "dual_water": f"Dual Water at shaly carbonate samples where VCL >= {cutoff_text}",
        "waxman_smits": f"Waxman-Smits at shaly samples where shale-conductivity inputs were available",
        "nigeria_schlumberger": "Nigeria Schlumberger shaly-sand equation where selected",
    }
    return mapping.get(model, model.replace("_", " "))


def _config_source(workflow_result, field_name: str) -> str:
    global_overrides = getattr(workflow_result, "global_overrides", {}) or {}
    zoned_defaults = getattr(workflow_result, "zoned_default_overrides", {}) or {}
    if field_name in global_overrides:
        return "LAS metadata"
    if field_name in zoned_defaults:
        return "zoned LAS fallback"
    return "workflow default"


def _build_delivery_computation_notes(well: "Well", workflow_result) -> List[str]:
    notes: List[str] = []

    bad_hole_steps = _all_step_results(workflow_result, "bad_hole")
    if bad_hole_steps:
        note = "BAD HOLE QC: Bad-hole screening used caliper roughness with DRHO fallback."
        notes.append(note)

    vcl_steps = _all_step_results(workflow_result, "vclay")
    if vcl_steps:
        note = "VCL: Final VCL is the row-wise minimum of the available clay-volume methods at each depth sample."
        gr_method = _ordered_unique(
            str(step.parameters.get("gr_method"))
            for step in vcl_steps
            if step.parameters.get("gr_method")
        )
        gr_min = _format_numeric_summary(_step_parameter_values(vcl_steps, "gr_min"), 1, "API")
        gr_max = _format_numeric_summary(_step_parameter_values(vcl_steps, "gr_max"), 1, "API")
        if gr_method:
            note += (
                f" GR method: {_friendly_join(gr_method)}, using VCL_GR = clip((GR - GRmin)/(GRmax - GRmin), 0, 1)"
            )
            if gr_min and gr_max:
                note += f" with GRmin {gr_min} and GRmax {gr_max}"
            note += "."
        if any(step.parameters.get("use_sp") for step in vcl_steps):
            sp_clean = _format_numeric_summary(_step_parameter_values(vcl_steps, "sp_max"), 1, "mV")
            sp_shale = _format_numeric_summary(_step_parameter_values(vcl_steps, "sp_min"), 1, "mV")
            note += (
                " SP method: VCL_SP = clip((SPclean - SP)/(SPclean - SPshale), 0, 1)"
            )
            if sp_clean and sp_shale:
                note += f" with SPclean {sp_clean} and SPshale {sp_shale}"
            note += "."
        if any(step.parameters.get("use_nd") for step in vcl_steps):
            note += (
                " Neutron-density method: 3-point crossplot using clean points "
                "(RHOB=2.71 g/cc, NPHI=0.00) and (RHOB=2.03 g/cc, NPHI=0.40), "
                "and clay point (RHOB=2.62 g/cc, NPHI=0.38)."
            )
        notes.append(note)

    rcl_steps = _all_step_results(workflow_result, "rcl")
    if rcl_steps:
        rcl_values = _format_numeric_summary(_step_parameter_values(rcl_steps, "rcl_computed"), 4, "ohm.m")
        threshold = _format_numeric_summary(_step_parameter_values(rcl_steps, "vcl_threshold"), 2)
        note = "RCL: RCL is shale resistivity, not clay porosity."
        note += " It was estimated as the mean deep resistivity in shale samples"
        if threshold:
            note += f" where VCL > {threshold}"
        if rcl_values:
            note += f"; resulting RCL values used in this well: {rcl_values}"
        note += "."
        notes.append(note)

    rmf_steps = _all_step_results(workflow_result, "rmf")
    if rmf_steps:
        methods = _ordered_unique(step.method for step in rmf_steps if step.method)
        if methods == ["mud_filtrate_salinity_conversion"]:
            salinity = _format_numeric_summary(_step_parameter_values(rmf_steps, "mud_filtrate_salinity_ppm"), 0, "ppm")
            rmf_measured = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_measured_temp"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "measured_temp_f"), 1, "F")
            rmf_formation = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_formation"), 4, "ohm.m")
            formation_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "formation_temp_f"), 1, "F")
            note = "RMF: For this well, RMF was calculated from CLFL salinity conversion."
            if salinity:
                note += f" CLFL used: {salinity}."
            if rmf_measured and measured_temp:
                note += f" Converted RMF at measured temperature: {rmf_measured} at {measured_temp}."
            if rmf_formation:
                note += f" Final RMF at formation temperature: {rmf_formation}"
                if formation_temp:
                    note += f" at {formation_temp}"
                note += "."
            notes.append(note)
        elif methods == ["las_metadata_arps_temperature_correction"]:
            rmf_measured = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_measured_temp"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "measured_temp_f"), 1, "F")
            rmf_formation = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_formation"), 4, "ohm.m")
            formation_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "formation_temp_f"), 1, "F")
            note = "RMF: For this well, RMF was taken from LAS metadata and temperature-corrected using RMFT."
            if rmf_measured and measured_temp:
                note += f" Input RMF: {rmf_measured} at {measured_temp}."
            if rmf_formation:
                note += f" Final RMF at formation temperature: {rmf_formation}"
                if formation_temp:
                    note += f" at {formation_temp}"
                note += "."
            notes.append(note)
        elif methods == ["las_metadata"]:
            rmf_value = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf"), 4, "ohm.m")
            notes.append(f"RMF: For this well, RMF was taken directly from LAS metadata{f' as {rmf_value}' if rmf_value else ''}.")
        elif methods == ["arps_temperature_correction"]:
            rmf_surface = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_surface"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "surface_temp_f"), 1, "F")
            rmf_formation = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_formation"), 4, "ohm.m")
            formation_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "formation_temp_f"), 1, "F")
            note = "RMF: For this well, RMF was temperature-corrected from an explicit measured RMF value."
            if rmf_surface and measured_temp:
                note += f" Input RMF: {rmf_surface} at {measured_temp}."
            if rmf_formation:
                note += f" Final RMF at formation temperature: {rmf_formation}"
                if formation_temp:
                    note += f" at {formation_temp}"
                note += "."
            notes.append(note)
        elif all(name.startswith("rm_to_rmf_conversion_factor_") for name in methods):
            rm_value = _format_numeric_summary(_step_parameter_values(rmf_steps, "rm"), 4, "ohm.m")
            rmf_formation = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_formation"), 4, "ohm.m")
            notes.append(
                "RMF: For this well, RMF was calculated from RM using the library's RM-to-RMF conversion factor "
                f"workflow. Input RM: {rm_value}. Final RMF: {rmf_formation}."
            )
        elif methods == ["OBM_default"]:
            notes.append("RMF: For this well, OBM was indicated, so the non-conductive OBM default RMF was used.")
        else:
            labels = _ordered_unique(_method_label(name) for name in methods)
            notes.append(
                f"RMF: Multiple RMF methods were used across this well: {_friendly_join(labels)}."
            )

    rw_steps = _all_step_results(workflow_result, "rw")
    if rw_steps:
        methods = _ordered_unique(step.method for step in rw_steps if step.method)
        rw_values = _format_numeric_summary(_step_parameter_values(rw_steps, "rw_computed"), 4, "ohm.m")
        if not rw_values:
            rw_values = _format_numeric_summary(_step_parameter_values(rw_steps, "rw_at_formation"), 4, "ohm.m")

        if methods == ["sp_clean_water"]:
            rmf_used = _format_numeric_summary(_step_parameter_values(rw_steps, "rmf"), 4, "ohm.m")
            notes.append(
                "RW: For this well, RW was calculated with the SP clean-water method."
                + (f" RMF used in the SP calculation: {rmf_used}." if rmf_used else "")
                + (f" Final RW: {rw_values}." if rw_values else "")
            )
        elif methods == ["rxo_rt_clean_water"]:
            rmf_used = _format_numeric_summary(_step_parameter_values(rw_steps, "rmf"), 4, "ohm.m")
            notes.append(
                "RW: For this well, RW was calculated with the Rxo/Rt clean-water method."
                + (f" RMF used in the calculation: {rmf_used}." if rmf_used else "")
                + (f" Final RW: {rw_values}." if rw_values else "")
            )
        elif methods == ["salinity_conversion"]:
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm"), 0, "ppm")
            notes.append(
                "RW: For this well, RW was calculated directly from formation-water salinity."
                + (f" Salinity used: {salinity}." if salinity else "")
                + (f" Final RW: {rw_values}." if rw_values else "")
            )
        elif methods == ["las_metadata"]:
            notes.append(
                "RW: For this well, RW was taken directly from LAS metadata."
                + (f" Final RW: {rw_values}." if rw_values else "")
            )
        elif methods == ["las_metadata_temperature_corrected"]:
            measured_rw = _format_numeric_summary(_step_parameter_values(rw_steps, "rw_at_measured_temp"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "measured_temp_f"), 1, "F")
            notes.append(
                "RW: For this well, RW was taken from LAS metadata and temperature-corrected using RWT."
                + (f" Input RW: {measured_rw} at {measured_temp}." if measured_rw and measured_temp else "")
                + (f" Final RW: {rw_values}." if rw_values else "")
            )
        elif methods == ["user_provided"]:
            notes.append(
                "RW: For this well, RW was supplied directly as a workflow input."
                + (f" Final RW: {rw_values}." if rw_values else "")
            )
        elif methods == ["user_provided_temperature_corrected"]:
            measured_rw = _format_numeric_summary(_step_parameter_values(rw_steps, "rw_at_measured_temp"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "measured_temp_f"), 1, "F")
            notes.append(
                "RW: For this well, RW was supplied directly and temperature-corrected."
                + (f" Input RW: {measured_rw} at {measured_temp}." if measured_rw and measured_temp else "")
                + (f" Final RW: {rw_values}." if rw_values else "")
            )
        elif methods == ["default_fallback"]:
            notes.append(
                "RW: For this well, no usable direct RW, salinity, RMF+SP, or RMF+Rxo/Rt source was available, "
                f"so the default RW was used{f' ({rw_values})' if rw_values else ''}."
            )
        else:
            labels = _ordered_unique(_method_label(name) for name in methods)
            note = f"RW: Multiple RW methods were used across this well: {_friendly_join(labels)}."
            if rw_values:
                note += f" Final RW values span {rw_values}."
            if "default_fallback" in methods:
                note += " Default RW was used only where the higher-priority source for that interval was not usable."
            notes.append(note)

    porosity_steps = _all_step_results(workflow_result, "porosity")
    if porosity_steps:
        decisions = _ordered_unique(
            decision
            for step in porosity_steps
            for decision in step.decisions
        )
        nphi_ma = _format_numeric_summary(_step_parameter_values(porosity_steps, "nphi_ma"), 3)
        phi_cl_values = _format_numeric_summary(_step_parameter_values(porosity_steps, "phi_clay"), 3)
        rho_ma = _format_numeric_summary(_step_parameter_values(porosity_steps, "rho_ma"), 3)
        rho_fl = _format_numeric_summary(_step_parameter_values(porosity_steps, "rho_fl"), 3)
        dt_fl = _format_numeric_summary(_step_parameter_values(porosity_steps, "dt_fl"), 1)

        details: List[str] = []
        if any("PHIN from NPHI" in decision for decision in decisions):
            detail = "PHIN = clip(NPHI - nphi_ma, 0, 0.6)"
            if nphi_ma:
                detail += f" with nphi_ma={nphi_ma}"
            details.append(detail + ".")
        if any("PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)" in decision for decision in decisions):
            detail = "PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)"
            if rho_ma and rho_fl:
                detail += f" with rho_ma={rho_ma} and rho_fl={rho_fl}"
            details.append(detail + ".")
        if any("PHIS from Wyllie time-average equation" in decision for decision in decisions):
            detail = "PHIS from the Wyllie time-average equation"
            if dt_fl:
                detail += f" with dt_fl={dt_fl}"
            details.append(detail + ".")
        if any("PHIT = adaptive(PHIN, PHID)" in decision for decision in decisions):
            details.append(
                "PHIT from the library's adaptive neutron-density combination: "
                "PHIT = PHID where PHID > PHIN + 0.05, otherwise PHIT = 0.6*PHID + 0.4*PHIN."
            )
        elif any("Density-only porosity" in decision for decision in decisions):
            details.append("PHIT was not written; density-only porosity remained as PHID.")
        elif any("Neutron-only porosity" in decision for decision in decisions):
            details.append("PHIT was not written; neutron-only porosity remained as PHIN.")
        elif any("Sonic-only porosity" in decision for decision in decisions):
            details.append("PHIT was not written; sonic-only porosity remained as PHIS.")
        elif any("PHIT reused from existing well curve" in decision for decision in decisions):
            details.append("PHIT was taken directly from the existing PHIT curve in the well.")
        if any("PHIE =" in decision for decision in decisions):
            phi_cl_source = _config_source(workflow_result, "phi_cl")
            if phi_cl_values:
                details.append(
                    f"PHIE = selected porosity basis - VCL * phi_clay, with phi_clay={phi_cl_values} v/v "
                    f"from {phi_cl_source}."
                )
            else:
                details.append("PHIE was calculated from the selected porosity basis and VCL clay correction.")
        if well.get("PHI_SEC") is not None:
            details.append("PHI_SEC = max(PHIT - PHIS, 0).")
        if well.get("PHIE_CORECAL") is not None:
            details.append("PHIE_CORECAL was written separately where a valid core-calibration model existed.")
        notes.append("POROSITY: " + " ".join(details))

    saturation_steps = _all_step_results(workflow_result, "saturation")
    if saturation_steps:
        raw_models: List[str] = []
        vcl_cutoffs = []
        archie_params = set()
        for step in saturation_steps:
            method = step.method or str(step.parameters.get("sw_model") or "")
            for part in str(method).split("+"):
                part = part.strip()
                if part:
                    raw_models.append(part)
            cutoff = step.parameters.get("vcl_cutoff_for_routing")
            if cutoff is not None:
                try:
                    vcl_cutoffs.append(float(cutoff))
                except (TypeError, ValueError):
                    pass
            archie_params.add(
                (
                    step.parameters.get("a"),
                    step.parameters.get("m"),
                    step.parameters.get("n"),
                )
            )
        models = _ordered_unique(raw_models)
        cutoff = vcl_cutoffs[0] if vcl_cutoffs else None
        reasons = _ordered_unique(_model_reason(model, cutoff) for model in models)
        note = f"SW: For this well, SW was calculated with {_friendly_join(models)}."
        if cutoff is not None:
            note += f" Auto-routing was applied sample-by-sample with VCL cutoff {float(cutoff):.2f}."
        if reasons:
            note += f" Model usage: {_friendly_join(reasons)}."
        archie_params = [triple for triple in archie_params if all(value is not None for value in triple)]
        if archie_params:
            a_value, m_value, n_value = archie_params[0]
            note += f" Archie parameters used: a={a_value}, m={m_value}, n={n_value}."
        rw_used = _format_numeric_summary(_step_parameter_values(saturation_steps, "rw"), 4, "ohm.m")
        if rw_used:
            note += f" RW used in saturation calculations: {rw_used}."
        rcl_used = _format_numeric_summary(_step_parameter_values(saturation_steps, "rcl"), 4, "ohm.m")
        if any(model in {"north_sea", "indonesia", "simandoux", "nigeria_schlumberger"} for model in models) and rcl_used:
            note += f" RCL used in shaly-sand saturation calculations: {rcl_used}."
        rtsh_used = _format_numeric_summary(_step_parameter_values(saturation_steps, "rtsh"), 4, "ohm.m")
        if any(model in {"north_sea", "indonesia", "simandoux", "nigeria_schlumberger"} for model in models) and rtsh_used:
            note += f" Shale resistivity / RTSH used: {rtsh_used}."
        vsh_exp_used = _format_numeric_summary(_step_parameter_values(saturation_steps, "vsh_exp"), 3)
        if "north_sea" in models and vsh_exp_used:
            note += f" North Sea shale exponent used: {vsh_exp_used}."
        if well.get("SXO") is not None:
            note += " SXO was calculated as the final flushed-zone saturation curve from RXO/RSHA and RMF."
        notes.append(note)

    lithology_steps = _all_step_results(workflow_result, "lithology")
    if lithology_steps and any(well.get(curve) is not None for curve in ("VQUAR", "VCALC", "VDOLO", "VANHY")):
        pef_available = well.get("PEF") is not None
        dtc_available = well.get("DTC") is not None
        note = (
            "MINERAL VOLUMES: VQUAR, VCALC, VDOLO, and VANHY were solved with the multi-mineral solver after removing "
            "the clay and porosity terms from the mixing equations. The solver uses "
            "sum(Vi) = 1 - VCL - PHIE, density, neutron, and"
        )
        if pef_available:
            note += " U = PEF * RHOB"
            if dtc_available:
                note += ", with DTC as an additional discriminator when available"
        elif dtc_available:
            note += " DTC"
        else:
            note += " the reduced RHOB+NPHI system"
        note += "."
        notes.append(note)

    permeability_steps = _all_step_results(workflow_result, "permeability")
    if permeability_steps:
        empirical = _ordered_unique(
            step.parameters.get("empirical_method")
            for step in permeability_steps
            if step.parameters.get("empirical_method")
        )
        strategies = _ordered_unique(
            step.parameters.get("strategy")
            for step in permeability_steps
            if step.parameters.get("strategy")
        )
        swirr_sources = _ordered_unique(
            step.parameters.get("swirr_source")
            for step in permeability_steps
            if step.parameters.get("swirr_source")
        )
        bvw_irr = _format_numeric_summary(_step_parameter_values(permeability_steps, "bvw_irr"), 4, "v/v")
        note = "PERMEABILITY: "
        if empirical:
            note += f"PERM was calculated with the {_friendly_join(empirical)} empirical permeability method."
            if empirical == ["timur"]:
                note += " Timur equation: k = 8581 * PHIE^4.4 / Swirr^2."
        else:
            note += "PERM stores the empirical permeability curve."
        if well.get("PERM_CC") is not None:
            note += " PERM_CC was written separately where a valid core-calibrated permeability transform was available."
        else:
            note += " No valid core-calibrated permeability curve was available for this well, so only empirical PERM is retained."
        if strategies:
            note += f" Workflow strategy used: {_friendly_join(strategies)}."
        if swirr_sources:
            note += f" Swirr source used: {_friendly_join(swirr_sources)}."
            if "apparent_buckles" in swirr_sources and bvw_irr:
                note += f" Apparent Buckles BVWirr used: {bvw_irr}, with SWIRR_APP = BVWirr / PHIE."
        notes.append(note)

    if any(well.get(curve) is not None for curve in ("RQI", "FZI", "PHI_Z")):
        notes.append(
            "ROCK QUALITY: RQI = 0.0314 * sqrt(PERM / PHIE), PHI_Z = PHIE / (1 - PHIE), and FZI = RQI / PHI_Z."
        )

    if well.get("BVW") is not None:
        notes.append("BVW: BVW = PHIE * SW.")

    return notes


def _build_analysis_summary(well: "Well") -> dict[str, Any]:
    payload: dict[str, Any] = {}

    hydrocarbon = HydrocarbonAnalysis().analyze(well, add_curves=False)
    payload.update(hydrocarbon.to_dict())

    qc = WellboreQC().analyze(well, add_curves=False)
    payload.update(qc.to_dict())

    workflow_flags = _workflow_flag_analysis(well)
    if workflow_flags is not None:
        payload.update(workflow_flags.to_dict())

    return payload


def _reference_well_name(source_file: Path, well: "Well") -> str:
    return source_file.stem or (well.name or "UNKNOWN_WELL")


def _depth_summary(well: "Well") -> dict[str, Any]:
    depth = np.asarray(well.depth, dtype=float)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return {
            "min_depth": None,
            "max_depth": None,
            "thickness": 0.0,
            "data_points": 0,
            "median_dz": None,
        }

    dz = np.diff(finite)
    finite_dz = np.abs(dz[np.isfinite(dz) & (np.abs(dz) > 0)])
    median_dz = float(np.nanmedian(finite_dz)) if finite_dz.size else None
    return {
        "min_depth": round(float(np.nanmin(finite)), 4),
        "max_depth": round(float(np.nanmax(finite)), 4),
        "thickness": round(float(np.nanmax(finite) - np.nanmin(finite)), 4),
        "data_points": int(finite.size),
        "median_dz": round(float(median_dz), 4) if median_dz is not None else None,
    }


def _is_null_like_lithology(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.upper() in {"NAN", "NONE", "NULL"}


def _is_sentinel_lithology(value: Any) -> bool:
    if _is_null_like_lithology(value):
        return False
    text = str(value).strip().upper()
    return text in {"-999.25", "-9999", "-9999.0", "UNKNOWN", "UNDEFINED", "SENTINEL"}


def _valid_lithology_labels(well: "Well") -> List[str]:
    curve = well.get("LITHOLOGY_GEOLINK")
    if curve is None:
        return []
    labels: List[str] = []
    for value in np.asarray(curve.data, dtype=object):
        if _is_null_like_lithology(value) or _is_sentinel_lithology(value):
            continue
        labels.append(str(value).strip())
    return labels


def _dominant_lithology(well: "Well") -> str:
    labels = _valid_lithology_labels(well)
    if not labels:
        return "UNKNOWN"
    unique, counts = np.unique(np.asarray(labels, dtype=object), return_counts=True)
    return str(unique[int(np.argmax(counts))])


def _build_reference_lithology_info(well: "Well") -> dict[str, Any]:
    curve = well.get("LITHOLOGY_GEOLINK")
    total_rows = int(well.n_samples)
    if curve is None:
        return {
            "column_present": False,
            "total_rows": total_rows,
            "valid_lithology_count": 0,
            "nan_count": total_rows,
            "sentinel_count": 0,
            "valid_lithology_percentage": 0.0,
            "nan_percentage": 100.0 if total_rows else 0.0,
            "has_valid_lithology": False,
            "unique_lithologies": [],
            "unique_lithology_count": 0,
        }

    values = np.asarray(curve.data, dtype=object)
    valid_labels: List[str] = []
    nan_count = 0
    sentinel_count = 0
    for value in values:
        if _is_null_like_lithology(value):
            nan_count += 1
            continue
        if _is_sentinel_lithology(value):
            sentinel_count += 1
            continue
        valid_labels.append(str(value).strip())

    valid_count = len(valid_labels)
    unique_lithologies = sorted(_ordered_unique(valid_labels))
    return {
        "column_present": True,
        "total_rows": total_rows,
        "valid_lithology_count": valid_count,
        "nan_count": nan_count,
        "sentinel_count": sentinel_count,
        "valid_lithology_percentage": round((valid_count / total_rows) * 100.0, 4) if total_rows else 0.0,
        "nan_percentage": round((nan_count / total_rows) * 100.0, 4) if total_rows else 0.0,
        "has_valid_lithology": valid_count > 0,
        "unique_lithologies": unique_lithologies,
        "unique_lithology_count": len(unique_lithologies),
    }


def _has_finite_curve(well: "Well", mnemonic: str) -> bool:
    curve = well.get(mnemonic)
    if curve is None:
        return False
    try:
        data = _normalize_numeric_array(curve.data)
    except Exception:
        return False
    return bool(np.any(np.isfinite(data)))


def _round_stat(value: float, decimals: int = 4) -> float:
    return round(float(value), decimals)


def _curve_statistics_payload(data: np.ndarray) -> Optional[dict[str, float]]:
    arr = _normalize_numeric_array(data)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None

    positive = finite[finite > 0]
    rounded = np.round(finite, 4)
    unique, counts = np.unique(rounded, return_counts=True)
    mode_value = float(unique[int(np.argmax(counts))]) if unique.size else float(finite[0])

    if positive.size:
        geometric_mean = float(np.exp(np.nanmean(np.log(positive))))
        harmonic_mean = float(positive.size / np.sum(1.0 / positive))
    else:
        geometric_mean = 0.0
        harmonic_mean = 0.0

    third_power_average = float(np.cbrt(np.nanmean(np.power(finite, 3))))

    return {
        "minimum": _round_stat(np.nanmin(finite)),
        "maximum": _round_stat(np.nanmax(finite)),
        "arithmetic_mean": _round_stat(np.nanmean(finite)),
        "geometric_mean": _round_stat(geometric_mean),
        "harmonic_mean": _round_stat(harmonic_mean),
        "third_power_average": _round_stat(third_power_average),
        "median": _round_stat(np.nanmedian(finite)),
        "mode": _round_stat(mode_value),
        "standard_deviation": _round_stat(np.nanstd(finite)),
    }


def _build_reference_statistics(well: "Well") -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for mnemonic in well.curves:
        if mnemonic == "DEPT":
            continue
        curve = well.get(mnemonic)
        if curve is None:
            continue
        try:
            payload = _curve_statistics_payload(curve.data)
        except Exception:
            payload = None
        if payload is None:
            continue
        stats[f"{mnemonic}_overall"] = payload
    return stats


def _build_reference_bad_hole_qc(well: "Well") -> dict[str, Any]:
    analyzer = WellboreQC()
    qc = analyzer.analyze(well, add_curves=False)
    qc_summary = qc.metadata.get("qc_summary", {})
    badhole_curve = qc.derived_curves.get("BADHOLEFLAG")
    badhole_data = _normalize_numeric_array(badhole_curve.data) if badhole_curve is not None else np.full(well.n_samples, np.nan)
    bad_mask = np.isfinite(badhole_data) & (badhole_data > 0.5)
    bad_points = int(np.count_nonzero(bad_mask))
    total_points = int(well.n_samples)
    bad_percentage = round((bad_points / total_points) * 100.0, 4) if total_points else 0.0

    return {
        "total_points": total_points,
        "bad_hole_points": bad_points,
        "bad_hole_percentage": bad_percentage,
        "bad_hole_sequences": qc_summary.get("bad_hole_intervals", []),
        "log_availability": {
            "CALI": well.get("CALI") is not None,
            "BS": well.get("BS") is not None,
            "DRHO": well.get("DRHO") is not None,
        },
        "qc_method": "caliper_roughness_with_drho_fallback",
        "bitsize_analysis": qc_summary.get("bit_size_exclusion_zones", []),
        "summary_text": (
            f"{bad_points} of {total_points} samples were flagged as bad hole "
            f"({bad_percentage:.2f}%)."
        ),
        "parameters_used": {
            "cali_variation_threshold": analyzer.cali_variation_threshold,
            "drho_threshold": analyzer.drho_threshold,
            "washout_threshold": analyzer.washout_threshold,
            "bitsize_exclusion_distance": analyzer.bitsize_exclusion_distance,
        },
    }


def _build_reference_json_payload(
    well: "Well",
    workflow_result,
    source_file: Path,
    original_curves: set[str],
) -> dict[str, Any]:
    well_name = _reference_well_name(source_file, well)
    depth_info = _depth_summary(well)
    lithology_info = _build_reference_lithology_info(well)
    dominant_lithology = _dominant_lithology(well)
    derived_logs = sorted(set(well.curves) - set(original_curves))
    computation_notes = _build_delivery_computation_notes(well, workflow_result)

    hydrocarbon_summary = HydrocarbonAnalysis().analyze(well, add_curves=False).to_dict()

    return {
        "formation_info": {
            "name": "WHOLE_WELL",
            "min_depth": depth_info["min_depth"],
            "max_depth": depth_info["max_depth"],
            "thickness": depth_info["thickness"],
        },
        "analysis_results": {
            "well_name": well_name,
            "formation_name": "WHOLE_WELL",
            "lithology group": "WHOLE_WELL",
            "lithology": dominant_lithology,
            "lithology_group_info": {
                "start_depth": depth_info["min_depth"],
                "end_depth": depth_info["max_depth"],
                "thickness": depth_info["thickness"],
                "data_points": depth_info["data_points"],
                "median_dz": depth_info["median_dz"],
            },
            "ground_truth_lithology_present": lithology_info["has_valid_lithology"],
            "core_phi_present": _has_finite_curve(well, "PHIC"),
            "core_k_present": _has_finite_curve(well, "KH_CORE") or _has_finite_curve(well, "KV_CORE"),
            "derived_logs_computed": derived_logs,
            "statistics": _build_reference_statistics(well),
            "analysis_summary": hydrocarbon_summary,
            "data_completeness": {
                "missing_primary_logs": [name for name in PRIMARY_LOGS_FOR_COMPLETENESS if name not in original_curves],
                "derived_log_computation_notes": computation_notes,
            },
            "lithology_info": lithology_info,
            "input_logs_present": sorted(original_curves),
            "BADHOLE_QC_CHECK": _build_reference_bad_hole_qc(well),
        },
        "raw_log_traces_in_segment": {},
        "include_raw_traces_in_context": False,
    }


def _write_reference_analysis_json(
    well: "Well",
    workflow_result,
    source_file: Path,
    las_output: Path,
    original_curves: set[str],
) -> Path:
    json_path = las_output.with_suffix(".json")
    payload = _build_reference_json_payload(well, workflow_result, source_file, original_curves)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)
    return json_path


def _write_analysis_json(
    well: "Well",
    workflow_result,
    source_file: Path,
    las_output: Path,
    original_curves: set[str],
    warnings: List[str],
    errors: List[str],
) -> Path:
    json_path = las_output.with_suffix(".json")

    payload = {
        "well_name": well.name,
        "analysis_summary": _build_analysis_summary(well),
        "derived_log_computation_notes": _build_delivery_computation_notes(well, workflow_result),
    }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)

    return json_path


def export_example_format_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.las",
    recursive: bool = False,
) -> List[ExampleFormatExportResult]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))

    results: List[ExampleFormatExportResult] = []
    for file_path in files:
        relative_path = file_path.relative_to(input_path)
        export_path = output_path / relative_path
        export_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            well = read_las(file_path)
            curve_count = len(_collect_export_curves(well))
            written = export_example_format_las(well, export_path)
            results.append(
                ExampleFormatExportResult(
                    source_file=str(file_path),
                    output_file=str(written),
                    success=True,
                    curves_exported=curve_count,
                )
            )
        except Exception as exc:
            results.append(
                ExampleFormatExportResult(
                    source_file=str(file_path),
                    output_file=str(export_path),
                    success=False,
                    errors=[str(exc)],
                )
            )

    return results


def export_example_work_package_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.las",
    recursive: bool = False,
) -> List[ExampleFormatExportResult]:
    from petrophys.workflow.zoned_batch import (
        _finalize_export_curves,
        run_dynamic_workflow_with_zones,
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))

    results: List[ExampleFormatExportResult] = []
    for file_path in files:
        relative_path = file_path.relative_to(input_path)
        export_path = output_path / relative_path
        export_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            well = read_las(file_path)
            well = standardize_curves(well, rename_curves=True, convert_units=True)
            original_curves = set(well.curves)

            workflow_result = run_dynamic_workflow_with_zones(well)
            _finalize_export_curves(workflow_result.well, original_curves)

            warnings = list(workflow_result.warnings)
            warnings.extend(workflow_result.base_result.warnings)
            for zone_result in workflow_result.zone_results:
                warnings.extend(zone_result.warnings)

            errors = list(workflow_result.base_result.errors)
            for zone_result in workflow_result.zone_results:
                errors.extend(zone_result.errors)

            written = export_example_format_las(
                workflow_result.well,
                export_path,
                original_curves=original_curves,
            )
            json_file = _write_analysis_json(
                workflow_result.well,
                workflow_result,
                file_path,
                written,
                original_curves,
                warnings,
                errors,
            )

            results.append(
                ExampleFormatExportResult(
                    source_file=str(file_path),
                    output_file=str(written),
                    success=not errors,
                    json_file=str(json_file),
                    curves_exported=len(_collect_export_curves(workflow_result.well, original_curves=original_curves)),
                    warnings=warnings,
                    errors=errors,
                )
            )
        except Exception as exc:
            results.append(
                ExampleFormatExportResult(
                    source_file=str(file_path),
                    output_file=str(export_path),
                    success=False,
                    errors=[str(exc)],
                )
            )

    return results


def export_reference_json_work_package_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.las",
    recursive: bool = False,
) -> List[ExampleFormatExportResult]:
    from petrophys.workflow.zoned_batch import (
        _finalize_export_curves,
        run_dynamic_workflow_with_zones,
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))

    results: List[ExampleFormatExportResult] = []
    for file_path in files:
        relative_path = file_path.relative_to(input_path)
        export_path = output_path / relative_path
        export_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            well = read_las(file_path)
            well = standardize_curves(well, rename_curves=True, convert_units=True)
            original_curves = set(well.curves)

            workflow_result = run_dynamic_workflow_with_zones(well)
            json_well = copy.deepcopy(workflow_result.well)

            warnings = list(workflow_result.warnings)
            warnings.extend(workflow_result.base_result.warnings)
            for zone_result in workflow_result.zone_results:
                warnings.extend(zone_result.warnings)

            errors = list(workflow_result.base_result.errors)
            for zone_result in workflow_result.zone_results:
                errors.extend(zone_result.errors)

            _finalize_export_curves(workflow_result.well, original_curves)
            written = export_example_format_las(
                workflow_result.well,
                export_path,
                original_curves=original_curves,
            )
            json_file = _write_reference_analysis_json(
                json_well,
                workflow_result,
                file_path,
                written,
                original_curves,
            )

            results.append(
                ExampleFormatExportResult(
                    source_file=str(file_path),
                    output_file=str(written),
                    success=not errors,
                    json_file=str(json_file),
                    curves_exported=len(_collect_export_curves(workflow_result.well, original_curves=original_curves)),
                    warnings=warnings,
                    errors=errors,
                )
            )
        except Exception as exc:
            results.append(
                ExampleFormatExportResult(
                    source_file=str(file_path),
                    output_file=str(export_path),
                    success=False,
                    errors=[str(exc)],
                )
            )

    return results


def write_example_format_report(
    results: Iterable[ExampleFormatExportResult],
    report_path: str | Path,
) -> Path:
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with report_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_file",
                "output_file",
                "json_file",
                "success",
                "curves_exported",
                "warnings",
                "errors",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.source_file,
                    item.output_file,
                    item.json_file,
                    item.success,
                    item.curves_exported,
                    " | ".join(item.warnings),
                    " | ".join(item.errors),
                ]
            )

    return report_file
