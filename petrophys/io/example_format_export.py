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
import re
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from petrophys.analysis import HydrocarbonAnalysis
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
    "VP_VS_RATIO",
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
    "PHI_NORM",
    "FZI",
    "SWIRR_APP",
    "INVASION_RATIO",
    "MHI",
]

FAMILY_LABELS = {
    "DEPT": "Measured Depth",
    "CALI": "Caliper",
    "BS": "Bit Size",
    "DRHO": "Bulk Density Correction",
    "DTC": "Compressional Slowness",
    "DTE": "Shear Slowness",
    "VP_VS_RATIO": "Vp/Vs Ratio",
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
    "PHI_NORM": "Normalized Porosity",
    "FZI": "Flow Zone Indicator",
    "SWIRR_APP": "Apparent Irreducible Water Saturation",
    "INVASION_RATIO": "Invasion Ratio Rxo/Rt",
    "MHI": "Moveable Hydrocarbon Index",
}

ALIASED_SOURCE_CURVES = {
    "DTS",
    "PHI_Z",
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
    "LIMESTONE_INDICATOR",
    "DOLOMITE_INDICATOR",
    "CARBONATE_INDICATOR",
    "GR_CLAY_INDICATOR",
    "NPHI_CLAY_INDICATOR",
    "PEF_CLAY_INDICATOR",
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
    "VSAND",
}

DUPLICATE_RAW_CURVES = {
    "DCAL",
    "DT",
}

REFERENCE_DERIVED_LOG_EXCLUDE = {
    "SWA",
    "SWXOA",
    "SXO-ARCHIE",
    "SXO-INDONESIA",
    "SXO-NIGERIA_SCHLUMBERGER",
    "SXO-NORTH_SEA",
    "VCLAY",
    "VCLGR",
    "VCLSP",
    "VCLND",
    "VCL_GR",
    "VCL_SP",
    "VCL_ND",
}

REFERENCE_DERIVED_LOG_TARGETS = [
    "TMPC",
    "BAD_HOLE",
    "BADHOLEFLAG",
    "VCL",
    "RMF",
    "RW",
    "PHIN",
    "PHID",
    "PHIS",
    "PHIT",
    "PHIE",
    "PHI_SEC",
    "PHIE_CORECAL",
    "SW",
    "SXO",
    "PERM",
    "PERM_CC",
    "SWIRR_APP",
    "RQI",
    "PHI_NORM",
    "FZI",
    "INVASION_RATIO",
    "BVW",
    "MHI",
    "VQUAR",
    "VCALC",
    "VDOLO",
    "VANHY",
    "LITHO",
    "NET_ROCK",
    "NET_RES",
    "NET_PAY",
    "RESFLAG",
    "SEALFLAG",
    "CROSSOVER_FLAG",
    "VP_VS_RATIO",
    "PAY_FLAG_PETROPHYSICAL",
    "PRIME_GAS_TARGET",
    "PRIME_OIL_TARGET",
]

REFERENCE_CURVE_TO_STEP = {
    "TMPC": "temperature",
    "BAD_HOLE": "bad_hole",
    "BADHOLEFLAG": "bad_hole",
    "VCL": "vclay",
    "RMF": "rmf",
    "RW": "rw",
    "PHIN": "porosity",
    "PHID": "porosity",
    "PHIS": "porosity",
    "PHIT": "porosity",
    "PHIE": "porosity",
    "PHI_SEC": "porosity",
    "SW": "saturation",
    "SXO": "saturation",
    "PERM": "permeability",
    "PERM_CC": "permeability",
    "PHIE_CORECAL": "permeability",
    "SWIRR_APP": "permeability",
    "RQI": "rock_quality",
    "PHI_NORM": "rock_quality",
    "FZI": "rock_quality",
    "INVASION_RATIO": "invasion",
    "BVW": "bvw",
    "MHI": "mhi",
    "VQUAR": "lithology",
    "VCALC": "lithology",
    "VDOLO": "lithology",
    "VANHY": "lithology",
    "LITHO": "lithology",
    "NET_ROCK": "net_pay",
    "NET_RES": "net_pay",
    "NET_PAY": "net_pay",
    "RESFLAG": "net_pay",
    "SEALFLAG": "net_pay",
    "CROSSOVER_FLAG": "net_pay",
    "VP_VS_RATIO": "net_pay",
    "PAY_FLAG_PETROPHYSICAL": "net_pay",
    "PRIME_GAS_TARGET": "net_pay",
    "PRIME_OIL_TARGET": "net_pay",
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

REFERENCE_CURVE_NAME_MAPPING = {
    "canonical_curve_renames": [
        {
            "current_export_name": "SXO",
            "previous_example_name": "SWXO",
            "notes": "Final flushed-zone water saturation is exported as SXO.",
        },
        {
            "current_export_name": "VDOLO",
            "previous_example_name": "VDOL",
            "notes": "Dolomite volume is exported as VDOLO.",
        },
        {
            "current_export_name": "VCL",
            "previous_example_name": "VSH",
            "notes": "The final clay-volume curve is exported as VCL.",
        },
    ],
    "method_specific_legacy_names": [
        {
            "current_export_name": "VCL_GR",
            "previous_example_name": "VSHGR",
            "notes": "Method-specific gamma-ray clay volume.",
        },
        {
            "current_export_name": "VCL_SP",
            "previous_example_name": "VSHSP",
            "notes": "Method-specific spontaneous-potential clay volume.",
        },
        {
            "current_export_name": "VCL_ND",
            "previous_example_name": "VSHND",
            "notes": "Method-specific neutron-density clay volume.",
        },
    ],
    "explicit_replacements": [
        {
            "current_export_name": "VQUAR",
            "previous_example_name": "VSAND",
            "notes": "Quartz volume is exported as VQUAR.",
        },
        {
            "current_export_name": "VCALC",
            "previous_example_name": "VLIM",
            "notes": "Calcite volume is exported as VCALC.",
        },
    ],
    "unchanged_key_curves": [
        "SW",
        "RW",
        "RMF",
        "PHID",
        "PHIN",
        "PHIS",
        "PHIT",
        "PHIE",
        "PERM",
        "RQI",
        "FZI",
        "PHI_NORM",
        "BVW",
        "MHI",
        "INVASION_RATIO",
    ],
}


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

    phi_norm = _first_curve(well, ["PHI_NORM", "PHI_Z"])
    if phi_norm is not None:
        specs.append(
            ExportCurveSpec(
                mnemonic="PHI_NORM",
                unit="v/v",
                description="Normalized Porosity",
                family=FAMILY_LABELS["PHI_NORM"],
                data=_normalize_numeric_array(phi_norm.data),
                source_curves=[phi_norm.mnemonic],
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


def _workflow_bad_hole_analysis(
    well: "Well",
    workflow_result,
) -> Optional[CategoryAnalysisResult]:
    details = _workflow_bad_hole_details(well, workflow_result)
    if details is None:
        return None

    curve = well.get("BADHOLEFLAG") or well.get("BAD_HOLE")
    if curve is None:
        return None

    values = _normalize_numeric_array(curve.data)
    labels = np.full(well.n_samples, "NO_QC_DATA", dtype=object)
    valid = np.isfinite(values)
    labels[valid & (values > 0.5)] = "BAD_HOLE"
    labels[valid & (values <= 0.5)] = "GOOD_HOLE"

    thresholds = details["thresholds"]
    methodology = (
        "Washout is evaluated from caliper-minus-bit-size and density-correction thresholds, "
        "then combined with the large-hole geometry screen."
    )
    calculation_thresholds = (
        f"WASHOUT when (CALI - BS) > {thresholds['cali_minus_bs_washout_threshold_in']:.2f} in"
        + (
            f" or |DRHO| > {thresholds['drho_washout_threshold_g_per_cc']:.2f} g/cc"
            if details["log_availability"].get("DRHO")
            else ""
        )
        + f"; LARGE_HOLE when BS > {thresholds['large_hole_bit_size_threshold_in']:.2f} in; "
        "BAD_HOLE = WASHOUT OR LARGE_HOLE."
    )

    return CategoryAnalysisResult(
        category_key="WELLBORE_QUALITY_CONTROL",
        sections={
            "bad_hole_detection": build_section(
                well,
                labels,
                description="Bad-hole screening from caliper, bit size, and density-correction thresholds.",
                methodology=methodology,
                calculation_thresholds=calculation_thresholds,
            )
        },
        metadata={
            "workflow_bad_hole_summary": {
                "bad_hole_percentage": details["bad_hole_percentage"],
                "counts": {
                    "washout": details["counts"]["washout"],
                    "large_hole": details["counts"]["large_hole"],
                    "bad_hole": details["counts"]["bad_hole"],
                    "total_points": details["counts"]["total_points"],
                },
                "bit_size_used_in": _json_safe(details["bit_size_used"]),
            }
        },
    )


def _workflow_flag_analysis(well: "Well") -> Optional[CategoryAnalysisResult]:
    sections = {}

    reservoir = _binary_flag_section(
        well,
        ["RESFLAG", "NET_RES"],
        "reservoir_flag_screening",
        "Reservoir screening from the reservoir flag.",
        "Uses the reservoir flag derived from VCL and PHIE cutoffs.",
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
        "Seal screening from the sealflag curve.",
        "Uses sealflag generated from high clay-volume intervals.",
        "SEALFLAG when VCL >= 0.8, otherwise NON_SEALFLAG.",
        "SEALFLAG",
        "NON_SEALFLAG",
    )
    if seal is not None:
        name, section = seal
        sections[name] = section

    crossover = _binary_flag_section(
        well,
        ["CROSSOVER_FLAG"],
        "crossover_flag_screening",
        "Gas crossover screening from neutron-density separation.",
        "Uses crossover flag where PHID - PHIN > 0.01.",
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


def _canonical_delivery_curve_name(curve: str) -> str:
    upper = curve.upper()
    if upper == "TEMP":
        return "TMPC"
    if upper == "PHI_Z":
        return "PHI_NORM"
    if upper == "SWA":
        return "SW"
    if upper == "SWXOA" or upper.startswith("SXO-"):
        return "SXO"
    if upper in {"VCLAY", "VSH"}:
        return "VCL"
    return upper


def _target_curve_candidates(curve: str) -> List[str]:
    mapping = {
        "TMPC": ["TMPC", "TEMP"],
        "PHI_NORM": ["PHI_NORM", "PHI_Z"],
        "SW": ["SW", "SWA"],
        "SXO": ["SXO", "SWXOA"],
        "VCL": ["VCL", "VCLAY", "VSH"],
    }
    return mapping.get(curve, [curve])


def _all_step_results(workflow_result, step_name: Optional[str] = None) -> List[Any]:
    step_results: List[Any] = []
    all_runs = [workflow_result.base_result, *workflow_result.zone_results]
    for run in all_runs:
        for step in run.steps_executed:
            if step_name is not None and step.step != step_name:
                continue
            step_results.append(step)
    return step_results


def _workflow_output_curves(workflow_result) -> List[str]:
    curves: List[str] = []
    for step in _all_step_results(workflow_result):
        curves.extend(step.output_curves or [])
    return curves


def _derived_logs_computed(
    well: "Well",
    workflow_result,
    original_curves: set[str],
) -> List[str]:
    derived = {
        _canonical_delivery_curve_name(curve)
        for curve in _workflow_output_curves(workflow_result)
    }
    derived.update(
        _canonical_delivery_curve_name(curve)
        for curve in (set(well.curves) - set(original_curves))
    )

    # Keep the reference-style list on canonical mnemonics rather than
    # method-specific aliases that are internal compatibility artifacts.
    derived.difference_update(REFERENCE_DERIVED_LOG_EXCLUDE)
    derived.discard("TEMP")
    derived.discard("PHI_Z")
    derived.difference_update({"SWA", "SWXOA", "VCLAY", "VSH", "VCLGR", "VCLSP", "VCLND", "VCL_GR", "VCL_SP", "VCL_ND"})

    return sorted(derived)


def _step_lookup(workflow_result) -> tuple[dict[str, Any], set[str]]:
    step_results: dict[str, Any] = {}
    skipped_steps: set[str] = set()
    for run in [workflow_result.base_result, *workflow_result.zone_results]:
        skipped_steps.update(run.steps_skipped)
        for step in run.steps_executed:
            if step.step not in step_results or step.success:
                step_results[step.step] = step
    return step_results, skipped_steps


def _missing_curve_reason(
    curve: str,
    well: "Well",
    workflow_result,
    step_results: dict[str, Any],
    skipped_steps: set[str],
    computed_curves: set[str],
    original_curves: set[str],
) -> str:
    step_name = REFERENCE_CURVE_TO_STEP.get(curve)
    step = step_results.get(step_name) if step_name else None

    if curve == "SXO":
        sw_present = "SW" in computed_curves or "SW" in well.curves
        shallow_present = well.get("RSHA") is not None or well.get("RXO") is not None
        rmf_present = "RMF" in computed_curves or well.get("RMF") is not None
        if not sw_present:
            return "SW unavailable, so SXO fallback could not be applied"
        if not shallow_present and not rmf_present:
            return "SXO fallback should have used SW^(1/5), but saturation output was not written"
        if not shallow_present:
            return "SXO fallback should have used SW^(1/5) because RSHA/RXO were absent"
        if not rmf_present:
            return "RSHA/RXO available but RMF was unavailable"

    if curve == "MHI":
        if well.get("SW") is None:
            return "SW unavailable"
        if well.get("SXO") is None:
            return "SXO unavailable"

    if curve == "VP_VS_RATIO":
        has_dtc = well.get("DTC") is not None or well.get("DT") is not None
        has_dts = well.get("DTS") is not None or well.get("DTE") is not None
        if not has_dtc and not has_dts:
            return "not calculated because both DTC and DTS were unavailable"
        if not has_dtc:
            return "not calculated because DTC was unavailable"
        if not has_dts:
            return "not calculated because DTS was unavailable"
        return "DTC and DTS were present but VP_VS_RATIO was not written"

    if curve == "TMPC":
        return "not calculated because no temperature source (TEMP, BHT, or TD) was available"

    if curve in {"PHIE_CORECAL", "PERM_CC"} and step is not None:
        same_well_models = step.parameters.get("same_well_core_models")
        peer_models = step.parameters.get("peer_core_models")
        strategy = step.parameters.get("strategy") or step.parameters.get("method")
        if (same_well_models or 0) == 0 and (peer_models or 0) == 0:
            return f"no usable core permeability model; strategy={strategy}"

    if curve == "SWIRR_APP" and step is not None:
        swirr_source = step.parameters.get("swirr_source")
        if swirr_source and swirr_source != "apparent_buckles":
            return f"Swirr source was {swirr_source}"

    if step_name and step_name in skipped_steps:
        return f"{step_name} step skipped"
    if step is not None and not step.success:
        return step.message or f"{step_name} step failed"

    if curve in original_curves:
        return "present in source data"

    return "not produced by the available workflow inputs"


def _computed_curve_detail(curve: str, step_results: dict[str, Any]) -> str:
    step_name = REFERENCE_CURVE_TO_STEP.get(curve)
    step = step_results.get(step_name) if step_name else None
    if step is None:
        return ""

    if curve == "SXO":
        sxo_method = step.parameters.get("sxo_method")
        sxo_reason = step.parameters.get("sxo_reason")
        if sxo_method == "archie_rxo_rmf":
            return "computed from Archie invaded-zone equation using RSHA/RXO and RMF"
        if sxo_method == "fifth_root_quicklook":
            if sxo_reason:
                return f"computed with fifth-root quick-look fallback because {sxo_reason}"
            return "computed with fifth-root quick-look fallback SXO = SW^(1/5)"
    if curve == "RMF":
        return _method_label(step.method) if step.method else "computed"
    if curve == "RW":
        return _method_label(step.method) if step.method else "computed"
    if curve == "MHI":
        return "computed as MHI = SW / SXO"
    if curve == "TMPC":
        return "computed as depth-wise formation temperature"
    if curve == "VP_VS_RATIO":
        return "computed as VP_VS_RATIO = DTS / DTC"
    if step.method:
        return _method_label(step.method)
    return "computed"


def _build_reference_derived_log_status(
    well: "Well",
    workflow_result,
    original_curves: set[str],
) -> List[dict[str, str]]:
    computed_curves = set(_derived_logs_computed(well, workflow_result, original_curves))
    normalized_original_curves = {
        _canonical_delivery_curve_name(curve)
        for curve in original_curves
    }
    step_results, skipped_steps = _step_lookup(workflow_result)
    statuses: List[dict[str, str]] = []

    for curve in REFERENCE_DERIVED_LOG_TARGETS:
        if curve in computed_curves:
            status = "computed"
            reason = _computed_curve_detail(curve, step_results)
        elif curve in normalized_original_curves and _first_curve(well, _target_curve_candidates(curve)) is not None:
            status = "already_present"
            reason = "present in source data"
        elif _first_curve(well, _target_curve_candidates(curve)) is not None:
            status = "present"
            reason = ""
        else:
            status = "missing"
            reason = _missing_curve_reason(
                curve,
                well,
                workflow_result,
                step_results,
                skipped_steps,
                computed_curves,
                original_curves,
            )

        statuses.append(
            {
                "mnemonic": curve,
                "status": status,
                "reason": reason,
            }
        )

    return statuses


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


def _format_overall_range(values: Sequence[Any], decimals: int = 4, unit: str = "") -> str:
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
    suffix = f" {unit}".rstrip()
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
        "las_metadata": "direct LAS header value",
        "las_metadata_arps_temperature_correction": "LAS header RMF with RMFT temperature correction",
        "las_metadata_profile_temperature_correction": "LAS header RMF with RMFT depth-wise temperature correction",
        "las_metadata_assumed_reference_temperature": "LAS header RMF with assumed-reference depth-wise temperature correction",
        "arps_temperature_correction": "explicit RMF temperature correction",
        "arps_temperature_profile_correction": "explicit RMF depth-wise temperature correction",
        "rm_to_rmf_conversion_factor_0.75": "RM-to-RMF conversion (fresh mud factor 0.75)",
        "rm_to_rmf_conversion_factor_0.85": "RM-to-RMF conversion (salt mud factor 0.85)",
        "OBM_default": "OBM non-conductive default",
        "user_provided": "direct RW value",
        "user_provided_temperature_corrected": "direct RW with temperature correction",
        "user_provided_temperature_profile_corrected": "direct RW with depth-wise temperature correction",
        "user_provided_assumed_reference_temperature": "direct RW with assumed-reference depth-wise temperature correction",
        "las_metadata_temperature_corrected": "LAS header RW with RWT temperature correction",
        "las_metadata_temperature_profile_corrected": "LAS header RW with RWT depth-wise temperature correction",
        "las_metadata_assumed_reference_temperature": "LAS header RW with assumed-reference depth-wise temperature correction",
        "salinity_conversion": "salinity-to-RW conversion",
        "sp_clean_water": "SP clean-water method",
        "rxo_rt_clean_water": "Rxo/Rt clean-water method",
        "default_fallback": "default RW fallback (0.05 ohm.m)",
        "default_fallback_profile": "default RW fallback (0.05 ohm.m) with depth-wise temperature correction",
        "component_based": "component-based porosity workflow",
        "linear_gradient_from_BHT": "linear temperature gradient from BHT",
        "constant_BHT_metadata": "constant BHT from LAS header",
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
        return "LAS header"
    if field_name in zoned_defaults:
        return "LAS header"
    return "configured value"


def _config_source_phrase(workflow_result, field_name: str) -> str:
    source = _config_source(workflow_result, field_name)
    return source


def _depth_profile_range_text(step_results: Sequence[Any], min_key: str, max_key: str, decimals: int = 4, unit: str = "") -> str:
    mins = _step_parameter_values(step_results, min_key)
    maxs = _step_parameter_values(step_results, max_key)
    values: List[float] = []
    for bucket in (mins, maxs):
        for value in bucket:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                values.append(numeric)
    return _format_overall_range(values, decimals, unit)


def _saturation_equation_text(model: str) -> str:
    mapping = {
        "archie": "Archie equation: Sw = [a * Rw / (PHIE^m * Rt)]^(1/n).",
        "indonesia": "Indonesia equation: 1/sqrt(Rt) = VCL^(1 - VCL/2)/sqrt(RCL) + (PHIE^(m/2) * Sw^(n/2))/sqrt(a * Rw).",
        "simandoux": "Simandoux equation: 1/Rt = (PHIE^m / (a * Rw)) * Sw^n + (VCL / RCL) * Sw.",
        "north_sea": "North Sea equation: 1/Rt = (PHIE^m / (a * Rw)) * Sw^n + VCL^vsh_exp / Rtsh.",
        "dual_water": "Dual Water equation: phi_bw = VCL * phi_cl; Swb = phi_bw / PHIT; 1/Rwt = Swb/Rwb + (1 - Swb)/Rw; Swt = [(a * Rwt)/(PHIT^m * Rt)]^(1/n); reported Sw = (Swt - Swb)/(1 - Swb).",
        "waxman_smits": "Waxman-Smits equation: Ct = (Sw^n / F*) * (Cw + B * Qv / Sw), iterated with F* = PHIT^(-m*) and Qv = CEC * rho_ma * (1 - PHIT) / PHIT.",
        "nigeria_schlumberger": "Nigeria/Schlumberger equation: 1/sqrt(Rt) = VCL^shale_exp/sqrt(Rsh) + (PHIE^(m/2) / sqrt(a * Rw)) * Sw^(n/2).",
    }
    return mapping.get(model, model.replace("_", " "))


def _matrix_endpoint_note(step) -> str:
    endpoints = step.parameters.get("matrix_endpoints") or []
    if not endpoints:
        return ""
    parts = []
    for endpoint in endpoints:
        group = endpoint.get("lithology_group")
        rho_ma = endpoint.get("rho_ma")
        nphi_ma = endpoint.get("nphi_ma")
        dt_ma = endpoint.get("dt_ma")
        if not group:
            continue
        parts.append(
            f"{group}: rho_ma={_format_numeric(rho_ma, 3)}, "
            f"nphi_ma={_format_numeric(nphi_ma, 3)}, dt_ma={_format_numeric(dt_ma, 1)}"
        )
    return "; ".join(parts)


def _phi_clay_source_label(source: str) -> str:
    mapping = {
        "LAS PNSH parameter": "LAS PNSH parameter",
        "zoned LAS PNSH fallback": "zoned LAS PNSH fallback",
        "lithology_shale_average_nphi": "average NPHI in shale lithology samples",
        "vcl_gt_0.9_average_nphi": "average NPHI where VCL > 0.9",
        "config_or_default": "configured/default clay-porosity value",
    }
    return mapping.get(source, source)


def _bit_size_used_for_bad_hole(well: "Well", workflow_result) -> Optional[float]:
    values = _step_parameter_values(_all_step_results(workflow_result, "bad_hole"), "bit_size_used")
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric

    bs_curve = well.get("BS")
    if bs_curve is not None:
        bs_data = _normalize_numeric_array(bs_curve.data)
        finite = bs_data[np.isfinite(bs_data)]
        if finite.size:
            return float(np.nanmedian(finite))

    header_value = getattr(well.header, "bit_size", None)
    if header_value is None:
        return None
    try:
        numeric = float(header_value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _flag_intervals(depth: np.ndarray, flag: np.ndarray) -> List[str]:
    intervals: List[str] = []
    if depth.size == 0 or not np.any(flag):
        return intervals

    padded = np.concatenate([[False], flag.astype(bool), [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for start, end in zip(starts, ends):
        intervals.append(f"{depth[start]:.1f}-{depth[end - 1]:.1f}")
    return intervals


def _workflow_bad_hole_details(well: "Well", workflow_result) -> Optional[dict[str, Any]]:
    from petrophys.base.constants import DefaultParameters
    from petrophys.qaqc.bad_hole.flag_generator import generate_bad_hole_flags

    bit_size = _bit_size_used_for_bad_hole(well, workflow_result)
    if well.get("CALI") is None and well.get("DRHO") is None:
        return None

    flags = generate_bad_hole_flags(
        well,
        bit_size=bit_size,
        cali_threshold=DefaultParameters.CALI_BS_THRESHOLD,
        drho_threshold=DefaultParameters.DRHO_THRESHOLD,
        pef_max=DefaultParameters.PEF_MAX,
        pef_min=DefaultParameters.PEF_MIN,
        large_hole_threshold=DefaultParameters.LARGE_HOLE_BS,
    )

    depth = np.asarray(well.depth, dtype=float) if well.depth is not None else np.array([], dtype=float)
    washout = np.asarray(flags.get("washout", np.zeros(well.n_samples, dtype=bool)), dtype=bool)
    large_hole = np.asarray(flags.get("large_hole", np.zeros(well.n_samples, dtype=bool)), dtype=bool)
    pef_exclusion = np.asarray(flags.get("pef_exclusion", np.zeros(well.n_samples, dtype=bool)), dtype=bool)
    bad_hole = np.asarray(flags.get("bad_hole", np.zeros(well.n_samples, dtype=bool)), dtype=bool)

    total_points = int(well.n_samples)
    bad_points = int(np.count_nonzero(bad_hole))
    bad_percentage = round((bad_points / total_points) * 100.0, 4) if total_points else 0.0

    return {
        "bit_size_used": bit_size,
        "thresholds": {
            "cali_minus_bs_washout_threshold_in": float(DefaultParameters.CALI_BS_THRESHOLD),
            "drho_washout_threshold_g_per_cc": float(DefaultParameters.DRHO_THRESHOLD),
            "large_hole_bit_size_threshold_in": float(DefaultParameters.LARGE_HOLE_BS),
            "pef_exclusion_min": float(DefaultParameters.PEF_MIN),
            "pef_exclusion_max": float(DefaultParameters.PEF_MAX),
        },
        "log_availability": {
            "CALI": well.get("CALI") is not None,
            "BS": well.get("BS") is not None or bit_size is not None,
            "DRHO": well.get("DRHO") is not None,
            "PEF": well.get("PEF") is not None,
        },
        "counts": {
            "washout": int(np.count_nonzero(washout)),
            "large_hole": int(np.count_nonzero(large_hole)),
            "pef_exclusion": int(np.count_nonzero(pef_exclusion)),
            "bad_hole": bad_points,
            "total_points": total_points,
        },
        "bad_hole_percentage": bad_percentage,
        "bad_hole_intervals": _flag_intervals(depth, bad_hole),
    }


def _build_delivery_computation_notes(well: "Well", workflow_result) -> List[str]:
    notes: List[str] = []

    bad_hole_steps = _all_step_results(workflow_result, "bad_hole")
    if bad_hole_steps:
        details = _workflow_bad_hole_details(well, workflow_result)
        if details is not None:
            thresholds = details["thresholds"]
            counts = details["counts"]
            note = (
                "BAD HOLE QC: "
                f"Washout = (CALI - BS) > {thresholds['cali_minus_bs_washout_threshold_in']:.2f} in"
            )
            if details["log_availability"].get("DRHO"):
                note += (
                    f" or |DRHO| > {thresholds['drho_washout_threshold_g_per_cc']:.2f} g/cc"
                )
            note += (
                f"; large_hole = BS > {thresholds['large_hole_bit_size_threshold_in']:.2f} in. "
                "Final BAD_HOLE is the logical OR of washout and large_hole."
            )
            if details["bit_size_used"] is not None:
                note += f" Bit size used for this well: {_format_numeric(details['bit_size_used'], 2)} in."
            note += (
                f" Counts in this well: washout={counts['washout']}, large_hole={counts['large_hole']}, "
                f"final BAD_HOLE={counts['bad_hole']} of {counts['total_points']} "
                f"samples ({details['bad_hole_percentage']:.2f}%)."
            )
            if details["log_availability"].get("PEF"):
                note += (
                    f" PEF exclusion is evaluated separately as PEF < {thresholds['pef_exclusion_min']:.2f} "
                    f"or PEF > {thresholds['pef_exclusion_max']:.2f}, but it is not part of BAD_HOLE."
                )
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
                note += f" with GRmin {gr_min} and GRmax {gr_max} across intervals in this well"
            note += "."
        if any(step.parameters.get("use_sp") for step in vcl_steps):
            sp_clean = _format_numeric_summary(_step_parameter_values(vcl_steps, "sp_max"), 1, "mV")
            sp_shale = _format_numeric_summary(_step_parameter_values(vcl_steps, "sp_min"), 1, "mV")
            note += (
                " SP method: VCL_SP = clip((SPclean - SP)/(SPclean - SPshale), 0, 1)"
            )
            if sp_clean and sp_shale:
                note += f" with SPclean {sp_clean} and SPshale {sp_shale} across intervals in this well"
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
        methods = _ordered_unique(step.parameters.get("source") or step.method for step in rcl_steps)
        shale_groups = _ordered_unique(
            group
            for step in rcl_steps
            for group in (step.parameters.get("shale_lithology_groups") or [])
        )
        shale_count = _format_numeric_summary(_step_parameter_values(rcl_steps, "shale_sample_count"), 0)
        note = "RCL: "
        if methods == ["lithology_shale_median_rt"]:
            note += "For this well, RCL was calculated as the median deep resistivity in shale lithology samples."
            if shale_groups:
                note += f" Shale groups used: {_friendly_join([str(group) for group in shale_groups])}."
            if shale_count:
                note += f" Shale sample count: {shale_count}."
        elif methods == ["vcl_gt_0.9_median_rt"]:
            note += "For this well, no shale lithology mask was available, so RCL was calculated as the median deep resistivity where VCL > 0.9."
            if threshold:
                note += f" VCL threshold used: {threshold}."
            if shale_count:
                note += f" Sample count: {shale_count}."
        else:
            note += "For this well, no shale lithology samples and no VCL > 0.9 shale proxy were available, so the fallback RCL was used."
        if rcl_values:
            note += f" Final RCL used in this well: {rcl_values}."
        notes.append(note)

    rmf_steps = _all_step_results(workflow_result, "rmf")
    if rmf_steps:
        methods = _ordered_unique(step.method for step in rmf_steps if step.method)
        rmf_profile = _depth_profile_range_text(rmf_steps, "rmf_min", "rmf_max", 4, "ohm.m")
        temp_profile = _depth_profile_range_text(rmf_steps, "formation_temp_min_f", "formation_temp_max_f", 1, "F")
        if methods == ["mud_filtrate_salinity_conversion"]:
            salinity = _format_numeric_summary(_step_parameter_values(rmf_steps, "mud_filtrate_salinity_ppm"), 0, "ppm")
            rmf_measured = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_measured_temp"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "measured_temp_f"), 1, "F")
            note = (
                "RMF: For this well, RMF was derived from CLFL. "
                "CLFL was converted with Rmf75 = 0.0123 + 3647.5 / CLFL^0.955, then corrected depth-by-depth with "
                "RMF(T) = RMF_measured * (Tmeasured + 6.77) / (T + 6.77)."
            )
            if salinity:
                note += f" CLFL used: {salinity}."
            if rmf_measured and measured_temp:
                note += f" RMF at measured temperature: {rmf_measured} at {measured_temp}."
            if rmf_profile:
                note += f" Final RMF(T) range: {rmf_profile}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["las_metadata_profile_temperature_correction"]:
            rmf_measured = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_measured_temp"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "measured_temp_f"), 1, "F")
            note = (
                "RMF: For this well, RMF came directly from the LAS header and was corrected depth-by-depth with "
                "RMF(T) = RMF_measured * (Tmeasured + 6.77) / (T + 6.77)."
            )
            if rmf_measured and measured_temp:
                note += f" Input RMF: {rmf_measured} at {measured_temp}."
            if rmf_profile:
                note += f" Final RMF(T) range: {rmf_profile}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["las_metadata_assumed_reference_temperature"]:
            rmf_reference = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_reference_temp"), 4, "ohm.m")
            reference_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "reference_temp_f"), 1, "F")
            note = (
                "RMF: For this well, RMF came directly from the LAS header. "
                "Because no RMFT was provided, the value was treated as a reference-temperature input and corrected "
                "depth-by-depth with RMF(T) = RMF_reference * (Treference + 6.77) / (T + 6.77)."
            )
            if rmf_reference and reference_temp:
                note += f" Input RMF: {rmf_reference} at assumed reference temperature {reference_temp}."
            if rmf_profile:
                note += f" Final RMF(T) range: {rmf_profile}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["arps_temperature_profile_correction"]:
            source = _config_source_phrase(workflow_result, "rmf_at_meas_temp")
            rmf_surface = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf_at_surface"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "surface_temp_f"), 1, "F")
            note = (
                f"RMF: For this well, RMF came from {source}. "
                "It was corrected depth-by-depth with RMF(T) = RMF_measured * (Tmeasured + 6.77) / (T + 6.77)."
            )
            if rmf_surface and measured_temp:
                note += f" Input RMF: {rmf_surface} at {measured_temp}."
            if rmf_profile:
                note += f" Final RMF(T) range: {rmf_profile}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif all(name.startswith("rm_to_rmf_conversion_factor_") for name in methods):
            rm_value = _format_numeric_summary(_step_parameter_values(rmf_steps, "rm"), 4, "ohm.m")
            factor = _format_numeric_summary(_step_parameter_values(rmf_steps, "conversion_factor"), 2)
            measured_temp = _format_numeric_summary(_step_parameter_values(rmf_steps, "measured_temp_f"), 1, "F")
            note = (
                "RMF: For this well, RMF was derived from RM as RMF_measured = conversion_factor * RM, "
                "then corrected depth-by-depth with RMF(T) = RMF_measured * (Tmeasured + 6.77) / (T + 6.77)."
            )
            if rm_value:
                note += f" Input RM: {rm_value}."
            if factor:
                note += f" Conversion factor: {factor}."
            if measured_temp:
                note += f" RM was treated as measured at {measured_temp}."
            if rmf_profile:
                note += f" Final RMF(T) range: {rmf_profile}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["OBM_default"]:
            rmf_value = _format_numeric_summary(_step_parameter_values(rmf_steps, "rmf"), 4, "ohm.m")
            notes.append(
                "RMF: For this well, OBM was indicated, so a non-conductive constant RMF default was used."
                + (f" RMF = {rmf_value}." if rmf_value else "")
            )
        else:
            labels = _ordered_unique(_method_label(name) for name in methods)
            note = f"RMF: Multiple RMF methods were used across this well: {_friendly_join(labels)}."
            if rmf_profile:
                note += f" Final RMF(T) range: {rmf_profile}."
            notes.append(note)

    rw_steps = _all_step_results(workflow_result, "rw")
    if rw_steps:
        methods = _ordered_unique(step.method for step in rw_steps if step.method)
        rw_values = _depth_profile_range_text(rw_steps, "rw_min", "rw_max", 4, "ohm.m")
        temp_profile = _depth_profile_range_text(rw_steps, "formation_temp_min_f", "formation_temp_max_f", 1, "F")

        if methods == ["sp_clean_water"]:
            rmf_used = _depth_profile_range_text(rw_steps, "rmf_min", "rmf_max", 4, "ohm.m")
            sp_clean = _format_numeric_summary(_step_parameter_values(rw_steps, "sp_clean_mv"), 1, "mV")
            sp_shale = _format_numeric_summary(_step_parameter_values(rw_steps, "sp_shale_mv"), 1, "mV")
            ssp = _format_numeric_summary(_step_parameter_values(rw_steps, "ssp_mv"), 1, "mV")
            note = (
                "RW: For this well, RW was derived from SP clean-water calibration. "
                "SSP was taken as SPclean - SPshale, K = 61 + 0.133 * T(F), and "
                "Rw(T) = RMF(T) / 10^(-SSP / K)."
            )
            if sp_clean and sp_shale:
                note += f" SPclean = {sp_clean}; SPshale = {sp_shale}."
            if ssp:
                note += f" SSP used: {ssp}."
            if rmf_used:
                note += f" RMF(T) range used: {rmf_used}."
            if rw_values:
                note += f" Final RW(T) range: {rw_values}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["rxo_rt_clean_water"]:
            rmf_used = _depth_profile_range_text(rw_steps, "rmf_min", "rmf_max", 4, "ohm.m")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            sample_count = _format_numeric_summary(_step_parameter_values(rw_steps, "n_points"), 0)
            note = (
                "RW: For this well, RW was derived from clean-water Rxo/Rt control points. "
                "The clean-water estimate used Rw_clean = RMF * Rt / Rxo, then converted those values to salinity, "
                "and finally wrote depth-wise RW(T) from Rw75 = 0.0123 + 3647.5 / SAL^0.955 and "
                "Rw(T) = Rw75 * (75 + 6.77) / (T + 6.77)."
            )
            if rmf_used:
                note += f" RMF(T) range used: {rmf_used}."
            if sample_count:
                note += f" Clean-water sample count: {sample_count}."
            if salinity:
                note += f" Salinity inferred from clean-water points: {salinity}."
            if rw_values:
                note += f" Final RW(T) range: {rw_values}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["salinity_conversion"]:
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm"), 0, "ppm")
            note = (
                "RW: For this well, RW was derived directly from salinity using "
                "Rw75 = 0.0123 + 3647.5 / SAL^0.955 and Rw(T) = Rw75 * (75 + 6.77) / (T + 6.77)."
            )
            if salinity:
                note += f" Salinity used: {salinity}."
            if rw_values:
                note += f" Final RW(T) range: {rw_values}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["las_metadata_temperature_profile_corrected"]:
            measured_rw = _format_numeric_summary(_step_parameter_values(rw_steps, "rw_at_measured_temp"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "measured_temp_f"), 1, "F")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            note = (
                "RW: For this well, RW came from the LAS header. "
                "The header value was converted to equivalent salinity at RWT, then rewritten depth-by-depth with "
                "Rw75 = 0.0123 + 3647.5 / SAL^0.955 and Rw(T) = Rw75 * (75 + 6.77) / (T + 6.77)."
            )
            if measured_rw and measured_temp:
                note += f" Input RW: {measured_rw} at {measured_temp}."
            if salinity:
                note += f" Equivalent salinity from the LAS RW input: {salinity}."
            if rw_values:
                note += f" Final RW(T) range: {rw_values}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["las_metadata_assumed_reference_temperature"]:
            input_rw = _format_numeric_summary(_step_parameter_values(rw_steps, "rw"), 4, "ohm.m")
            reference_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "reference_temp_f"), 1, "F")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            note = (
                "RW: For this well, RW came from the LAS header. "
                "Because no RWT was provided, that input was treated as a reference-temperature value, converted to equivalent salinity, "
                "and rewritten depth-by-depth with Rw75 = 0.0123 + 3647.5 / SAL^0.955 and "
                "Rw(T) = Rw75 * (75 + 6.77) / (T + 6.77)."
            )
            if input_rw and reference_temp:
                note += f" Input RW: {input_rw} at assumed reference temperature {reference_temp}."
            if salinity:
                note += f" Equivalent salinity from the LAS RW input: {salinity}."
            if rw_values:
                note += f" Final RW(T) range: {rw_values}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["user_provided_temperature_profile_corrected"]:
            source = _config_source_phrase(workflow_result, "rw")
            measured_rw = _format_numeric_summary(_step_parameter_values(rw_steps, "rw_at_measured_temp"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "measured_temp_f"), 1, "F")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            note = (
                f"RW: For this well, RW came from {source}. "
                "That input was converted to equivalent salinity at the stated measurement temperature, then rewritten depth-by-depth with "
                "Rw75 = 0.0123 + 3647.5 / SAL^0.955 and Rw(T) = Rw75 * (75 + 6.77) / (T + 6.77)."
            )
            if measured_rw and measured_temp:
                note += f" Input RW: {measured_rw} at {measured_temp}."
            if salinity:
                note += f" Equivalent salinity from the RW input: {salinity}."
            if rw_values:
                note += f" Final RW(T) range: {rw_values}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["user_provided_assumed_reference_temperature"]:
            source = _config_source_phrase(workflow_result, "rw")
            input_rw = _format_numeric_summary(_step_parameter_values(rw_steps, "rw"), 4, "ohm.m")
            reference_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "reference_temp_f"), 1, "F")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            note = (
                f"RW: For this well, RW came from {source}. "
                "Because no RWT was provided, that input was treated as a reference-temperature value, converted to equivalent salinity, "
                "and rewritten depth-by-depth with Rw75 = 0.0123 + 3647.5 / SAL^0.955 and "
                "Rw(T) = Rw75 * (75 + 6.77) / (T + 6.77)."
            )
            if input_rw and reference_temp:
                note += f" Input RW: {input_rw} at assumed reference temperature {reference_temp}."
            if salinity:
                note += f" Equivalent salinity from the RW input: {salinity}."
            if rw_values:
                note += f" Final RW(T) range: {rw_values}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            notes.append(note)
        elif methods == ["default_fallback_profile"]:
            reference_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "reference_temp_f"), 1, "F")
            notes.append(
                "RW: For this well, no usable direct RW, salinity, RMF+SP, or RMF+Rxo/Rt source was available, "
                "so the default clean-water input RW = 0.05 ohm.m was converted to equivalent salinity and rewritten depth-by-depth."
                + (f" Assumed reference temperature: {reference_temp}." if reference_temp else "")
                + (f" Final RW(T) range: {rw_values}." if rw_values else "")
            )
        else:
            labels = _ordered_unique(_method_label(name) for name in methods)
            note = f"RW: Multiple RW methods were used across this well: {_friendly_join(labels)}."
            if rw_values:
                note += f" Final RW(T) values span {rw_values}."
            if "default_fallback_profile" in methods:
                note += " Default RW was used only where the higher-priority source for that interval was not usable."
            notes.append(note)

    porosity_steps = _all_step_results(workflow_result, "porosity")
    if porosity_steps:
        decisions = _ordered_unique(
            decision
            for step in porosity_steps
            for decision in step.decisions
        )
        phi_cl_values = _format_numeric_summary(_step_parameter_values(porosity_steps, "phi_clay"), 3)
        rho_ma = _format_numeric_summary(_step_parameter_values(porosity_steps, "rho_ma"), 3)
        rho_ma_range = _format_overall_range(
            [*_step_parameter_values(porosity_steps, "rho_ma_min"), *_step_parameter_values(porosity_steps, "rho_ma_max")],
            3,
        )
        rho_fl = _format_numeric_summary(_step_parameter_values(porosity_steps, "rho_fl"), 3)
        nphi_ma = _format_numeric_summary(_step_parameter_values(porosity_steps, "nphi_ma"), 3)
        nphi_ma_range = _format_overall_range(
            [*_step_parameter_values(porosity_steps, "nphi_ma_min"), *_step_parameter_values(porosity_steps, "nphi_ma_max")],
            3,
        )
        dt_ma = _format_numeric_summary(_step_parameter_values(porosity_steps, "dt_ma"), 1)
        dt_ma_range = _format_overall_range(
            [*_step_parameter_values(porosity_steps, "dt_ma_min"), *_step_parameter_values(porosity_steps, "dt_ma_max")],
            1,
        )
        dt_fl = _format_numeric_summary(_step_parameter_values(porosity_steps, "dt_fl"), 1)
        matrix_modes = _ordered_unique(
            str(step.parameters.get("matrix_mode"))
            for step in porosity_steps
            if step.parameters.get("matrix_mode")
        )
        endpoint_text = "; ".join(
            _ordered_unique(
                [
                    f"{endpoint.get('lithology_group')}: rho_ma={_format_numeric(endpoint.get('rho_ma'), 3)}, "
                    f"nphi_ma={_format_numeric(endpoint.get('nphi_ma'), 3)}, "
                    f"dt_ma={_format_numeric(endpoint.get('dt_ma'), 1)}"
                    for step in porosity_steps
                    for endpoint in (step.parameters.get("matrix_endpoints") or [])
                    if endpoint.get("lithology_group")
                ]
            )
        )
        phi_cl_sources = _ordered_unique(
            str(step.parameters.get("phi_clay_source"))
            for step in porosity_steps
            if step.parameters.get("phi_clay_source")
        )
        phi_cl_basis_count = _format_numeric_summary(_step_parameter_values(porosity_steps, "phi_clay_basis_count"), 0)
        phit_methods = _ordered_unique(
            str(step.parameters.get("phit_method"))
            for step in porosity_steps
            if step.parameters.get("phit_method")
        )

        details: List[str] = []
        if any("PHIN from NPHI" in decision for decision in decisions):
            detail = "PHIN = clip(NPHI - nphi_ma, 0, 0.6)"
            if "lithology_dependent" in matrix_modes:
                detail += " with lithology-dependent nphi_ma(depth)"
                if endpoint_text:
                    detail += f". Endpoints used in this well: {endpoint_text}"
                elif nphi_ma_range:
                    detail += f". nphi_ma range used: {nphi_ma_range}"
            elif nphi_ma:
                detail += f" with nphi_ma={nphi_ma}"
            details.append(detail + ".")
        if any(decision.startswith("PHID =") for decision in decisions):
            detail = "PHID = (rho_ma - RHOB) / (rho_ma - rho_fl)"
            if "lithology_dependent" in matrix_modes:
                detail = "PHID = (rho_ma(depth) - RHOB) / (rho_ma(depth) - rho_fl)"
                if rho_ma_range and rho_fl:
                    detail += f" with rho_ma range {rho_ma_range} and rho_fl={rho_fl}"
            elif rho_ma and rho_fl:
                detail += f" with rho_ma={rho_ma} and rho_fl={rho_fl}"
            details.append(detail + ".")
        if any("PHIS from Wyllie time-average equation" in decision for decision in decisions):
            if "lithology_dependent" in matrix_modes:
                detail = "PHIS from the Wyllie time-average equation using dt_ma(depth)"
                if dt_ma_range and dt_fl:
                    detail += f" with dt_ma range {dt_ma_range} and dt_fl={dt_fl}"
            else:
                detail = "PHIS from the Wyllie time-average equation"
                if dt_ma and dt_fl:
                    detail += f" with dt_ma={dt_ma} and dt_fl={dt_fl}"
                elif dt_fl:
                    detail += f" with dt_fl={dt_fl}"
            if dt_fl and "lithology_dependent" not in matrix_modes and not dt_ma:
                detail += f" with dt_fl={dt_fl}"
            details.append(detail + ".")
        if phit_methods == ["adaptive_nd"] or any("PHIT = adaptive(PHIN, PHID)" in decision for decision in decisions):
            details.append(
                "PHIT from the library's adaptive neutron-density combination: "
                "PHIT = PHID where PHID > PHIN + 0.05, otherwise PHIT = 0.6*PHID + 0.4*PHIN."
            )
        elif phit_methods == ["density_only"] or any("Density-only porosity" in decision for decision in decisions):
            details.append("PHIT was not written; density-only porosity remained as PHID.")
        elif phit_methods == ["neutron_only"] or any("Neutron-only porosity" in decision for decision in decisions):
            details.append("PHIT was not written; neutron-only porosity remained as PHIN.")
        elif phit_methods == ["sonic_only"] or any("Sonic-only porosity" in decision for decision in decisions):
            details.append("PHIT was not written; sonic-only porosity remained as PHIS.")
        elif phit_methods == ["existing_phit"] or any("PHIT reused from existing well curve" in decision for decision in decisions):
            details.append("PHIT was taken directly from the existing PHIT curve in the well.")
        if any("PHIE =" in decision for decision in decisions):
            if phi_cl_values:
                phi_cl_source = _friendly_join([_phi_clay_source_label(source) for source in phi_cl_sources])
                details.append(
                    f"PHIE = selected porosity basis - VCL * phi_clay, with phi_clay={phi_cl_values} v/v "
                    f"from {phi_cl_source if phi_cl_source else 'available shale-porosity input'}"
                    + (f" using {phi_cl_basis_count} shale-control samples." if phi_cl_basis_count else "")
                    + "."
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
        if not models and well.get("SW") is None:
            pass
        else:
            cutoff = vcl_cutoffs[0] if vcl_cutoffs else None
            reasons = _ordered_unique(_model_reason(model, cutoff) for model in models)
            if models:
                note = f"SW: For this well, SW used {_friendly_join(models)}."
            else:
                note = "SW: SW was written in this well, but no saturation-model label was recorded."
            if cutoff is not None:
                note += f" Auto-routing was applied sample-by-sample with VCL cutoff {float(cutoff):.2f}."
            if reasons:
                note += f" Depth routing in this well: {_friendly_join(reasons)}."
            equation_text = " ".join(_saturation_equation_text(model) for model in models)
            if equation_text:
                note += f" {equation_text}"
            archie_params = [triple for triple in archie_params if all(value is not None for value in triple)]
            if archie_params:
                a_value, m_value, n_value = archie_params[0]
                note += f" Archie parameters used: a={a_value}, m={m_value}, n={n_value}."
            rw_used = _depth_profile_range_text(saturation_steps, "rw_min", "rw_max", 4, "ohm.m")
            if rw_used:
                note += f" RW(T) range used in saturation: {rw_used}."
            rcl_used = _format_numeric_summary(_step_parameter_values(saturation_steps, "rcl"), 4, "ohm.m")
            if any(model in {"north_sea", "indonesia", "simandoux", "nigeria_schlumberger"} for model in models) and rcl_used:
                note += f" RCL used in shaly-sand saturation calculations: {rcl_used}."
            rtsh_used = _format_numeric_summary(_step_parameter_values(saturation_steps, "rtsh"), 4, "ohm.m")
            if any(model in {"north_sea", "indonesia", "simandoux", "nigeria_schlumberger"} for model in models) and rtsh_used:
                note += f" Shale resistivity / RTSH used: {rtsh_used}."
            vsh_exp_used = _format_numeric_summary(_step_parameter_values(saturation_steps, "vsh_exp"), 3)
            if "north_sea" in models and vsh_exp_used:
                note += f" North Sea shale exponent used: {vsh_exp_used}."
            sxo_methods = _ordered_unique(
                str(method)
                for method in _step_parameter_values(saturation_steps, "sxo_method")
                if method
            )
            sxo_reasons = _ordered_unique(
                str(reason)
                for reason in _step_parameter_values(saturation_steps, "sxo_reason")
                if reason
            )
            if well.get("SXO") is not None:
                if sxo_methods == ["archie_rxo_rmf"]:
                    note += " SXO used the invaded-zone Archie equation SXO = [a * RMF / (PHIE^m * Rxo)]^(1/n), with RSHA preferred and RXO used when RSHA was absent."
                elif sxo_methods == ["fifth_root_quicklook"]:
                    note += " SXO used the fifth-root quick-look fallback SXO = SW^(1/5)."
                    if sxo_reasons:
                        note += f" Fallback trigger: {_friendly_join(sxo_reasons)}."
                elif sxo_methods:
                    note += f" SXO methods used in this well: {_friendly_join(sxo_methods)}."
                    if sxo_reasons:
                        note += f" Context: {_friendly_join(sxo_reasons)}."
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
        swirr_scalar = _format_numeric_summary(_step_parameter_values(permeability_steps, "swirr"), 4, "v/v")
        perm_present = well.get("PERM") is not None
        perm_cc_present = well.get("PERM_CC") is not None
        failure_reasons = _ordered_unique(
            str(reason)
            for reason in (
                [step.message for step in permeability_steps if getattr(step, "message", None)]
                + [step.parameters.get("empirical_error") for step in permeability_steps if step.parameters.get("empirical_error")]
            )
            if reason
        )
        note = "PERMEABILITY: "
        if not perm_present and not perm_cc_present:
            reason = _friendly_join(failure_reasons) if failure_reasons else "required permeability inputs were unavailable"
            note += f"Not calculated because {reason}."
        elif empirical:
            method_names = _friendly_join(empirical)
            note += f"PERM was calculated with the {method_names} empirical permeability method."
            equations: List[str] = []
            if "timur" in empirical:
                equations.append("Timur equation: k = 8581 * PHIE^4.4 / Swirr^2")
            if "coates" in empirical:
                equations.append("Coates-Denoo equation: k = 100^2 * [PHIE * (1 - Swirr) / Swirr]^2")
            if "wyllie_rose" in empirical:
                equations.append("Wyllie-Rose equation: k = 250 * PHIE^3 / Swirr^2")
            if equations:
                note += " " + ". ".join(equations) + "."
        else:
            note += "PERM stores the empirical permeability curve."
        if perm_cc_present:
            note += " PERM_CC was written separately where a valid core-calibrated porosity-permeability transform was available."
        elif perm_present:
            note += " No valid core-calibrated porosity-permeability transform was available for this well, so only empirical PERM is retained."
        if strategies:
            note += f" Permeability strategy used in this well: {_friendly_join(strategies)}."
        if swirr_sources:
            note += f" Swirr source used: {_friendly_join(swirr_sources)}."
            if swirr_scalar:
                note += f" Swirr value used: {swirr_scalar}."
            if "apparent_buckles" in swirr_sources and bvw_irr:
                note += f" Apparent Buckles BVWirr used: {bvw_irr}, with SWIRR_APP = BVWirr / PHIE."
        notes.append(note)

    if any(well.get(curve) is not None for curve in ("RQI", "FZI", "PHI_Z", "PHI_NORM")):
        notes.append(
            "ROCK QUALITY: RQI = 0.0314 * sqrt(PERM / PHIE), PHI_NORM = PHIE / (1 - PHIE), and FZI = RQI / PHI_NORM."
        )

    if well.get("BVW") is not None:
        notes.append("BVW: BVW = PHIE * SW.")

    net_pay_steps = _all_step_results(workflow_result, "net_pay")
    if net_pay_steps:
        vcl_cutoff = _format_numeric_summary(_step_parameter_values(net_pay_steps, "vcl_cutoff"), 2)
        phie_cutoff = _format_numeric_summary(_step_parameter_values(net_pay_steps, "phie_cutoff"), 2)
        sw_cutoff = _format_numeric_summary(_step_parameter_values(net_pay_steps, "sw_cutoff"), 2)
        seal_cutoff = _format_numeric_summary(_step_parameter_values(net_pay_steps, "seal_vcl_cutoff"), 2)
        net_rock_count = int(np.nansum(_normalize_numeric_array(well.get("NET_ROCK").data) > 0.5)) if well.get("NET_ROCK") is not None else 0
        resflag_count = int(np.nansum(_normalize_numeric_array(well.get("RESFLAG").data) > 0.5)) if well.get("RESFLAG") is not None else 0
        payflag_curve = well.get("PAY_FLAG_PETROPHYSICAL") or well.get("NET_PAY")
        payflag_count = int(np.nansum(_normalize_numeric_array(payflag_curve.data) > 0.5)) if payflag_curve is not None else 0
        seal_count = int(np.nansum(_normalize_numeric_array(well.get("SEALFLAG").data) > 0.5)) if well.get("SEALFLAG") is not None else 0
        note = (
            "WORKFLOW FLAGS: NET_ROCK is defined where VCL is below the clean-rock cutoff, "
            "RESFLAG / NET_RES additionally require PHIE above the reservoir cut-off, and "
            "PAY_FLAG_PETROPHYSICAL / NET_PAY additionally require water saturation below the pay cut-off."
        )
        if vcl_cutoff and phie_cutoff and sw_cutoff:
            note += f" Cutoffs used in this well: VCL < {vcl_cutoff}, PHIE > {phie_cutoff}, SW < {sw_cutoff}."
        if seal_cutoff:
            note += f" SEALFLAG = 1 where VCL >= {seal_cutoff}."
        note += (
            f" Positive sample counts in this well: NET_ROCK={net_rock_count}, RESFLAG={resflag_count}, "
            f"PAY_FLAG_PETROPHYSICAL={payflag_count}, SEALFLAG={seal_count}."
        )
        if well.get("CROSSOVER_FLAG") is not None:
            note += " CROSSOVER_FLAG = 1 where PHID - PHIN > 0.01."
        if well.get("VP_VS_RATIO") is not None:
            note += " VP_VS_RATIO = DTS / DTC."
        if well.get("MHI") is not None:
            note += " MHI = SW / SXO."
        notes.append(note)

    if well.get("INVASION_RATIO") is not None:
        notes.append("INVASION: INVASION_RATIO = RXO / RT and is used as a quick invasion-screening metric.")

    return notes


def _build_analysis_summary(well: "Well", workflow_result=None) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    hydrocarbon = HydrocarbonAnalysis().analyze(well, add_curves=False)
    payload.update(hydrocarbon.to_dict())

    qc = _workflow_bad_hole_analysis(well, workflow_result)
    if qc is not None:
        payload.update(qc.to_dict())

    workflow_flags = _workflow_flag_analysis(well)
    if workflow_flags is not None:
        payload.update(workflow_flags.to_dict())

    return payload


def _reference_well_name(source_file: Path, well: "Well") -> str:
    return source_file.stem or (well.name or "UNKNOWN_WELL")


def _sanitize_path_component(value: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]+', "_", str(value or "").strip())
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text or "UNNAMED"


def _normalize_tops_well_name(value: str) -> str:
    text = str(value or "").strip().upper()
    text = text.replace("\\", "/").replace("_", "/")
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_well_selector(value: str) -> str:
    text = str(value or "").strip().upper()
    text = text.replace("\\", "_").replace("/", "_")
    text = re.sub(r"\s+", " ", text)
    return text


def _sanitize_lithology_token(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip().upper())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "NONE"


def _display_lithology_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().upper())
    return text or "NAN"


def _load_lithology_mapping_workbook(
    mapping_file: str | Path,
) -> dict[int, dict[str, str]]:
    import pandas as pd

    mapping_path = Path(mapping_file)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Lithology mapping workbook not found: {mapping_path}")

    dataframe = pd.read_excel(mapping_path)
    normalized_columns = {str(column).strip().lower(): column for column in dataframe.columns}
    required = {"attribute", "lithology attribute", "abbreviation"}
    missing = required.difference(normalized_columns)
    if missing:
        raise ValueError(
            "Lithology mapping workbook is missing required columns: "
            + ", ".join(sorted(missing))
        )

    attribute_col = normalized_columns["attribute"]
    lithology_attr_col = normalized_columns["lithology attribute"]
    abbreviation_col = normalized_columns["abbreviation"]

    mapping: dict[int, dict[str, str]] = {}
    for _, row in dataframe.iterrows():
        try:
            code = int(round(float(row[abbreviation_col])))
        except (TypeError, ValueError):
            continue
        mapping[code] = {
            "attribute": str(row[attribute_col]).strip(),
            "lithology_attribute": str(row[lithology_attr_col]).strip(),
        }
    return mapping


def _strip_top_suffix(name: str) -> str:
    text = str(name or "").strip()
    if text.lower().endswith(" top"):
        return text[:-4].rstrip()
    return text


def _load_excel_formation_segments(
    tops_file: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    tops_path = Path(tops_file)
    if not tops_path.exists():
        raise FileNotFoundError(f"Formation tops workbook not found: {tops_path}")

    dataframe = pd.read_excel(tops_path)
    normalized_columns = {str(column).strip().lower(): column for column in dataframe.columns}
    required = {"well", "formation", "md_top"}
    missing = required.difference(normalized_columns)
    if missing:
        raise ValueError(
            "Formation tops workbook is missing required columns: "
            + ", ".join(sorted(missing))
        )

    well_col = normalized_columns["well"]
    formation_col = normalized_columns["formation"]
    top_col = normalized_columns["md_top"]

    def _is_formation_unit(name: str) -> bool:
        return " FM." in str(name).upper()

    segments_by_well: dict[str, list[dict[str, Any]]] = {}
    valid_rows = dataframe[[well_col, formation_col, top_col]].dropna()
    for well_name, group in valid_rows.groupby(well_col):
        group = group.copy()
        group["_formation_name"] = group[formation_col].map(_strip_top_suffix)
        group["_formation_display_name"] = group[formation_col].map(lambda value: str(value).strip().upper())
        group["_md_top"] = group[top_col].astype(float)
        group = group.sort_values(["_md_top", "_formation_display_name"], kind="stable")

        selected_rows = []
        for _, top_group in group.groupby("_md_top", sort=True):
            formation_rows = top_group[top_group["_formation_display_name"].map(_is_formation_unit)]
            if len(formation_rows) == 1:
                selected_rows.append(formation_rows.iloc[[0]])
            elif len(formation_rows) > 1:
                continue
            else:
                selected_rows.append(top_group)

        if selected_rows:
            group = pd.concat(selected_rows, ignore_index=True)
            group = group.sort_values(["_md_top", "_formation_display_name"], kind="stable")
        else:
            group = group.iloc[0:0]

        unique_tops = sorted(float(value) for value in group["_md_top"].dropna().unique())
        next_top_by_top = {
            unique_tops[index]: (unique_tops[index + 1] if index + 1 < len(unique_tops) else None)
            for index in range(len(unique_tops))
        }

        segments: list[dict[str, Any]] = []
        seen: set[tuple[str, float]] = set()
        for row in group[["_formation_name", "_formation_display_name", "_md_top"]].to_dict("records"):
            formation_name = str(row["_formation_name"]).strip()
            formation_display_name = str(row["_formation_display_name"]).strip()
            top_depth = float(row["_md_top"])
            dedupe_key = (formation_name, top_depth)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            segments.append(
                {
                    "name": formation_name,
                    "display_name": formation_display_name,
                    "top_depth": top_depth,
                    "bottom_depth": next_top_by_top.get(top_depth),
                }
            )

        segments_by_well[_normalize_tops_well_name(str(well_name))] = segments

    return segments_by_well


def _formation_segments_for_well(
    source_file: Path,
    well: "Well",
    tops_by_well: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidates = [
        _normalize_tops_well_name(source_file.stem),
        _normalize_tops_well_name(well.name),
        _normalize_tops_well_name(getattr(well.header, "well_name", "")),
    ]
    for candidate in candidates:
        if candidate and candidate in tops_by_well:
            return tops_by_well[candidate]

    prefix_candidates: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        base_candidate = candidate.split(" ", 1)[0].strip()
        if not base_candidate:
            continue
        matches = [name for name in tops_by_well if name.startswith(base_candidate)]
        if len(matches) == 1:
            prefix_candidates.append(matches[0])

    unique_prefix_candidates = sorted(set(prefix_candidates))
    if len(unique_prefix_candidates) == 1:
        return tops_by_well[unique_prefix_candidates[0]]
    return []


def _filter_input_files_by_well_names(
    files: Sequence[Path],
    well_names: Optional[Sequence[str]],
) -> list[Path]:
    if not well_names:
        return list(files)

    selectors = {_normalize_well_selector(name) for name in well_names if str(name).strip()}
    if not selectors:
        return list(files)

    filtered: list[Path] = []
    for file_path in files:
        if _normalize_well_selector(file_path.stem) in selectors:
            filtered.append(file_path)
    return filtered


def _well_mask_slice(well: "Well", mask: np.ndarray) -> "Well":
    from petrophys.base.well import Well

    boolean_mask = np.asarray(mask, dtype=bool)
    if well.depth is None:
        raise ValueError("Well has no depth data")
    if boolean_mask.shape[0] != len(well.depth):
        raise ValueError("Mask length does not match well depth length")

    sliced = Well(name=well.name)
    sliced._depth = np.asarray(well.depth)[boolean_mask]
    sliced._depth_unit = well.depth_unit
    sliced.header = copy.deepcopy(well.header)
    sliced._parameters = copy.deepcopy(well._parameters)
    sliced._results = copy.deepcopy(well._results)

    for mnemonic, curve in well._curves.items():
        sliced._curves[mnemonic] = curve[boolean_mask]

    return sliced


def _build_lithology_group_segments(
    well: "Well",
    minimum_group_thickness: float = 2.0,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> list[dict[str, Any]]:
    if well.depth is None or well.n_samples == 0:
        return []

    depth = np.asarray(well.depth, dtype=float)
    lithology_curve = well.get("LITHOLOGY_GEOLINK")
    if lithology_curve is None:
        labels = np.asarray(["NAN"] * well.n_samples, dtype=object)
    else:
        labels_list = []
        for value in lithology_curve.data:
            if _is_null_like_lithology(value) or _is_sentinel_lithology(value):
                labels_list.append("NAN")
            else:
                labels_list.append(
                    _normalize_lithology_label(value, lithology_mapping=lithology_mapping) or "NAN"
                )
        labels = np.asarray(labels_list, dtype=object)

    group_changes = np.ones(labels.shape[0], dtype=bool)
    if labels.shape[0] > 1:
        group_changes[1:] = labels[1:] != labels[:-1]
    group_ids = np.cumsum(group_changes)

    counters: dict[str, int] = {}
    segments: list[dict[str, Any]] = []
    for group_id in np.unique(group_ids):
        indices = np.flatnonzero(group_ids == group_id)
        if indices.size == 0:
            continue
        lithology_name = str(labels[indices[0]])
        counters[lithology_name] = counters.get(lithology_name, 0) + 1
        group_name = f"{lithology_name}_GROUP_{counters[lithology_name]}"

        top_depth = float(depth[indices[0]])
        bottom_depth = float(depth[indices[-1]])
        thickness = float(bottom_depth - top_depth)
        median_dz = well.step
        if (
            median_dz is not None
            and np.isfinite(median_dz)
            and indices.size > 1
            and thickness <= 0.0
        ):
            thickness = float((indices.size - 1) * abs(median_dz))
        if not np.isfinite(thickness) or thickness <= minimum_group_thickness:
            continue

        mask = np.zeros(labels.shape[0], dtype=bool)
        mask[indices] = True
        segments.append(
            {
                "group_name": group_name,
                "lithology": lithology_name,
                "top_depth": round(top_depth, 4),
                "bottom_depth": round(bottom_depth, 4),
                "thickness": round(thickness, 4),
                "mask": mask,
            }
        )

    return segments


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


def _normalize_lithology_label(
    value: Any,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> Optional[str]:
    if _is_null_like_lithology(value) or _is_sentinel_lithology(value):
        return None

    text = str(value).strip()
    try:
        numeric = int(round(float(text)))
    except (TypeError, ValueError):
        return _display_lithology_text(text)

    if lithology_mapping and numeric in lithology_mapping:
        mapped = (
            lithology_mapping[numeric].get("lithology_attribute")
            or lithology_mapping[numeric].get("attribute")
            or text
        )
        return _display_lithology_text(mapped)

    try:
        from petrophys.analysis.lithology_analysis import (
            GROUND_TRUTH_GROUPS,
            COMPUTED_LITHO_GROUPS,
            GROUND_TRUTH_NAMES,
        )
    except Exception:
        return _display_lithology_text(text)

    if numeric in GROUND_TRUTH_NAMES:
        return _display_lithology_text(str(GROUND_TRUTH_NAMES[numeric]))

    for group_name, codes in GROUND_TRUTH_GROUPS.items():
        if numeric in codes:
            return _display_lithology_text(group_name.replace("_", " "))
    if numeric in COMPUTED_LITHO_GROUPS:
        return _display_lithology_text(str(COMPUTED_LITHO_GROUPS[numeric]).replace("_", " "))
    return _display_lithology_text(text)


def _valid_lithology_labels(
    well: "Well",
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> List[str]:
    curve = well.get("LITHOLOGY_GEOLINK")
    if curve is None:
        return []
    labels: List[str] = []
    for value in np.asarray(curve.data, dtype=object):
        label = _normalize_lithology_label(value, lithology_mapping=lithology_mapping)
        if label is None:
            continue
        labels.append(label)
    return labels


def _dominant_lithology(
    well: "Well",
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> str:
    labels = _valid_lithology_labels(well, lithology_mapping=lithology_mapping)
    if not labels:
        return "UNKNOWN"
    unique, counts = np.unique(np.asarray(labels, dtype=object), return_counts=True)
    return str(unique[int(np.argmax(counts))])


def _build_reference_lithology_info(
    well: "Well",
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> dict[str, Any]:
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
        label = _normalize_lithology_label(value, lithology_mapping=lithology_mapping)
        if label is None:
            continue
        valid_labels.append(label)

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


def _build_lithology_distribution(
    well: "Well",
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> dict[str, Any]:
    labels = _valid_lithology_labels(well, lithology_mapping=lithology_mapping)
    if not labels:
        return {}
    values = np.asarray(labels, dtype=object)
    unique, counts = np.unique(values, return_counts=True)
    total = int(np.sum(counts))
    distribution: dict[str, Any] = {}
    for name, count in zip(unique, counts):
        distribution[str(name)] = {
            "count": int(count),
            "proportion": round(float(count) / total, 6) if total else 0.0,
        }
    return distribution


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


def _build_reference_bad_hole_qc(well: "Well", workflow_result) -> dict[str, Any]:
    details = _workflow_bad_hole_details(well, workflow_result)
    if details is None:
        return {
            "total_points": int(well.n_samples),
            "bad_hole_points": 0,
            "bad_hole_percentage": 0.0,
            "bad_hole_sequences": [],
            "log_availability": {
                "CALI": False,
                "BS": False,
                "DRHO": False,
                "PEF": False,
            },
            "qc_method": "washout_or_large_hole",
            "summary_text": "Bad-hole screening could not be evaluated because CALI and DRHO were unavailable.",
            "parameters_used": {},
        }

    thresholds = details["thresholds"]
    counts = details["counts"]
    return {
        "total_points": counts["total_points"],
        "bad_hole_points": counts["bad_hole"],
        "bad_hole_percentage": details["bad_hole_percentage"],
        "bad_hole_sequences": details["bad_hole_intervals"],
        "log_availability": details["log_availability"],
        "qc_method": "washout_or_large_hole",
        "summary_text": (
            "Bad-hole screening uses washout OR large-hole logic: "
            f"washout={(counts['washout'])}, large_hole={counts['large_hole']}, "
            f"final BAD_HOLE={counts['bad_hole']} of "
            f"{counts['total_points']} samples ({details['bad_hole_percentage']:.2f}%)."
        ),
        "parameters_used": {
            "bit_size_used_in": _json_safe(details["bit_size_used"]),
            "washout_formula": (
                f"(CALI - BS) > {thresholds['cali_minus_bs_washout_threshold_in']:.2f} in"
                + (
                    f" OR |DRHO| > {thresholds['drho_washout_threshold_g_per_cc']:.2f} g/cc"
                    if details["log_availability"].get("DRHO")
                    else ""
                )
            ),
            "large_hole_formula": f"BS > {thresholds['large_hole_bit_size_threshold_in']:.2f} in",
            "pef_exclusion_formula": (
                f"PEF < {thresholds['pef_exclusion_min']:.2f} or "
                f"PEF > {thresholds['pef_exclusion_max']:.2f}"
            ),
            "combined_logic": "BAD_HOLE = washout OR large_hole",
        },
    }


def _build_reference_json_payload(
    well: "Well",
    workflow_result,
    source_file: Path,
    original_curves: set[str],
    formation_name: str = "WHOLE_WELL",
    formation_depth_info: Optional[dict[str, Any]] = None,
    lithology_group_name: Optional[str] = None,
    lithology_name: Optional[str] = None,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> dict[str, Any]:
    well_name = _reference_well_name(source_file, well)
    depth_info = _depth_summary(well)
    formation_info = formation_depth_info or depth_info
    lithology_info = _build_reference_lithology_info(well, lithology_mapping=lithology_mapping)
    dominant_lithology = _dominant_lithology(well, lithology_mapping=lithology_mapping)
    derived_logs = _derived_logs_computed(well, workflow_result, original_curves)
    derived_log_status = _build_reference_derived_log_status(well, workflow_result, original_curves)
    computation_notes = _build_delivery_computation_notes(well, workflow_result)

    hydrocarbon_summary = HydrocarbonAnalysis().analyze(well, add_curves=False).to_dict()
    full_analysis_summary = _build_analysis_summary(well, workflow_result)

    return {
        "formation_info": {
            "name": formation_name,
            "min_depth": formation_info["min_depth"],
            "max_depth": formation_info["max_depth"],
            "thickness": formation_info["thickness"],
        },
        "analysis_results": {
            "well_name": well_name,
            "formation_name": formation_name,
            "lithology group": lithology_group_name or formation_name,
            "lithology": lithology_name or dominant_lithology,
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
            "lithology_distribution": _build_lithology_distribution(well, lithology_mapping=lithology_mapping),
            "statistics": _build_reference_statistics(well),
            "analysis_summary": full_analysis_summary or hydrocarbon_summary,
            "data_completeness": {
                "missing_primary_logs": [name for name in PRIMARY_LOGS_FOR_COMPLETENESS if name not in original_curves],
                "derived_log_computation_notes": computation_notes,
                "derived_log_status": derived_log_status,
            },
            "lithology_info": lithology_info,
            "input_logs_present": sorted(original_curves),
            "BADHOLE_QC_CHECK": _build_reference_bad_hole_qc(well, workflow_result),
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
    formation_name: str = "WHOLE_WELL",
    formation_depth_info: Optional[dict[str, Any]] = None,
    lithology_group_name: Optional[str] = None,
    lithology_name: Optional[str] = None,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> Path:
    json_path = las_output.with_suffix(".json")
    payload = _build_reference_json_payload(
        well,
        workflow_result,
        source_file,
        original_curves,
        formation_name=formation_name,
        formation_depth_info=formation_depth_info,
        lithology_group_name=lithology_group_name,
        lithology_name=lithology_name,
        lithology_mapping=lithology_mapping,
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)
    return json_path


def _write_reference_analysis_json_to_path(
    json_path: Path,
    well: "Well",
    workflow_result,
    source_file: Path,
    original_curves: set[str],
    formation_name: str = "WHOLE_WELL",
    formation_depth_info: Optional[dict[str, Any]] = None,
    lithology_group_name: Optional[str] = None,
    lithology_name: Optional[str] = None,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> Path:
    payload = _build_reference_json_payload(
        well,
        workflow_result,
        source_file,
        original_curves,
        formation_name=formation_name,
        formation_depth_info=formation_depth_info,
        lithology_group_name=lithology_group_name,
        lithology_name=lithology_name,
        lithology_mapping=lithology_mapping,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
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
        "analysis_summary": _build_analysis_summary(well, workflow_result),
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


def export_reference_json_work_package_by_formation(
    input_dir: str | Path,
    output_dir: str | Path,
    tops_file: str | Path,
    pattern: str = "*.las",
    recursive: bool = False,
) -> List[ExampleFormatExportResult]:
    from petrophys.workflow.zoned_batch import (
        _finalize_export_curves,
        run_dynamic_workflow_with_zones,
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))
    tops_by_well = _load_excel_formation_segments(tops_file)

    results: List[ExampleFormatExportResult] = []
    for file_path in files:
        try:
            full_well = read_las(file_path)
            full_well = standardize_curves(full_well, rename_curves=True, convert_units=True)
            formation_segments = _formation_segments_for_well(file_path, full_well, tops_by_well)

            well_folder = output_path / _sanitize_path_component(file_path.stem)
            well_folder.mkdir(parents=True, exist_ok=True)
            whole_well_original_curves = set(full_well.curves)
            whole_well_result = run_dynamic_workflow_with_zones(copy.deepcopy(full_well))
            _finalize_export_curves(whole_well_result.well, whole_well_original_curves)
            whole_well_las_path = well_folder / f"{_sanitize_path_component(file_path.stem)}.las"
            export_example_format_las(
                whole_well_result.well,
                whole_well_las_path,
                original_curves=whole_well_original_curves,
            )
            fallback_warnings: list[str] = []
            formation_name_counts: dict[str, int] = {}

            well_top = float(full_well.top) if full_well.top is not None else None
            well_bottom = float(full_well.bottom) if full_well.bottom is not None else None
            if well_top is None or well_bottom is None:
                results.append(
                    ExampleFormatExportResult(
                        source_file=str(file_path),
                        output_file="",
                        success=False,
                        errors=["Well has no valid depth range for formation slicing."],
                    )
                )
                continue

            if not formation_segments:
                formation_segments = [
                    {
                        "name": "WHOLE_WELL",
                        "display_name": "WHOLE_WELL",
                        "top_depth": well_top,
                        "bottom_depth": None,
                    }
                ]
                fallback_warnings.append(
                    "No formation tops matched this well in the Excel workbook. Falling back to WHOLE_WELL segmentation."
                )

            for segment in formation_segments:
                display_name = str(segment.get("display_name") or segment["name"])
                formation_name_counts[display_name] = formation_name_counts.get(display_name, 0) + 1

            for segment in formation_segments:
                raw_top = float(segment["top_depth"])
                raw_bottom = (
                    float(segment["bottom_depth"])
                    if segment["bottom_depth"] is not None
                    else well_bottom
                )
                if raw_bottom <= well_top or raw_top >= well_bottom:
                    continue

                slice_top = max(raw_top, well_top)
                slice_bottom = min(raw_bottom, well_bottom)
                if segment["bottom_depth"] is not None:
                    slice_bottom = float(np.nextafter(slice_bottom, -np.inf))
                if slice_bottom <= slice_top:
                    continue

                sliced_well = full_well.depth_slice(slice_top, slice_bottom)
                if sliced_well.n_samples == 0:
                    continue

                original_curves = set(sliced_well.curves)
                workflow_result = run_dynamic_workflow_with_zones(sliced_well)
                json_well = copy.deepcopy(workflow_result.well)

                warnings = list(fallback_warnings)
                warnings.extend(workflow_result.warnings)
                warnings.extend(workflow_result.base_result.warnings)
                for zone_result in workflow_result.zone_results:
                    warnings.extend(zone_result.warnings)

                errors = list(workflow_result.base_result.errors)
                for zone_result in workflow_result.zone_results:
                    errors.extend(zone_result.errors)

                _finalize_export_curves(workflow_result.well, original_curves)

                formation_name = str(segment["name"])
                formation_folder = well_folder / _sanitize_path_component(formation_name)
                formation_folder.mkdir(parents=True, exist_ok=True)

                depth_label = f"{slice_top:.1f}_{slice_bottom:.1f}"
                file_label = (
                    f"{_sanitize_path_component(file_path.stem)}"
                    f"__{_sanitize_path_component(formation_name)}"
                    f"__{_sanitize_path_component(depth_label)}"
                )
                export_path = formation_folder / f"{file_label}.las"

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
                    formation_name=formation_name,
                )

                results.append(
                    ExampleFormatExportResult(
                        source_file=str(file_path),
                        output_file=str(written),
                        success=not errors,
                        json_file=str(json_file),
                        curves_exported=len(
                            _collect_export_curves(
                                workflow_result.well,
                                original_curves=original_curves,
                            )
                        ),
                        warnings=warnings,
                        errors=errors,
                    )
                )
        except Exception as exc:
            results.append(
                ExampleFormatExportResult(
                    source_file=str(file_path),
                    output_file="",
                    success=False,
                    errors=[str(exc)],
                )
            )

    return results


def export_reference_json_work_package_by_formation_lithology(
    input_dir: str | Path,
    output_dir: str | Path,
    tops_file: str | Path,
    pattern: str = "*.las",
    recursive: bool = False,
    minimum_group_thickness: float = 2.0,
    well_names: Optional[Sequence[str]] = None,
    lithology_mapping_file: Optional[str | Path] = None,
) -> List[ExampleFormatExportResult]:
    from petrophys.workflow.zoned_batch import (
        _finalize_export_curves,
        run_dynamic_workflow_with_zones,
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))
    files = _filter_input_files_by_well_names(files, well_names)
    tops_by_well = _load_excel_formation_segments(tops_file)
    lithology_mapping = (
        _load_lithology_mapping_workbook(lithology_mapping_file)
        if lithology_mapping_file
        else None
    )

    results: List[ExampleFormatExportResult] = []
    for file_path in files:
        try:
            full_well = read_las(file_path)
            full_well = standardize_curves(full_well, rename_curves=True, convert_units=True)
            formation_segments = _formation_segments_for_well(file_path, full_well, tops_by_well)

            well_folder = output_path / _sanitize_path_component(file_path.stem)
            well_folder.mkdir(parents=True, exist_ok=True)
            whole_well_original_curves = set(full_well.curves)
            whole_well_result = run_dynamic_workflow_with_zones(copy.deepcopy(full_well))
            _finalize_export_curves(whole_well_result.well, whole_well_original_curves)
            whole_well_las_path = well_folder / f"{_sanitize_path_component(file_path.stem)}.las"
            export_example_format_las(
                whole_well_result.well,
                whole_well_las_path,
                original_curves=whole_well_original_curves,
            )
            fallback_warnings: list[str] = []
            formation_name_counts: dict[str, int] = {}

            well_top = float(full_well.top) if full_well.top is not None else None
            well_bottom = float(full_well.bottom) if full_well.bottom is not None else None
            if well_top is None or well_bottom is None:
                results.append(
                    ExampleFormatExportResult(
                        source_file=str(file_path),
                        output_file="",
                        success=False,
                        errors=["Well has no valid depth range for formation slicing."],
                    )
                )
                continue

            if not formation_segments:
                formation_segments = [
                    {
                        "name": "WHOLE_WELL",
                        "display_name": "WHOLE_WELL",
                        "top_depth": well_top,
                        "bottom_depth": None,
                    }
                ]
                fallback_warnings.append(
                    "No formation tops matched this well in the Excel workbook. Falling back to WHOLE_WELL segmentation."
                )

            for segment in formation_segments:
                display_name = str(segment.get("display_name") or segment["name"])
                formation_name_counts[display_name] = formation_name_counts.get(display_name, 0) + 1

            for segment in formation_segments:
                raw_top = float(segment["top_depth"])
                raw_bottom = (
                    float(segment["bottom_depth"])
                    if segment["bottom_depth"] is not None
                    else well_bottom
                )
                if raw_bottom <= well_top or raw_top >= well_bottom:
                    continue

                slice_top = max(raw_top, well_top)
                slice_bottom = min(raw_bottom, well_bottom)
                if segment["bottom_depth"] is not None:
                    slice_bottom = float(np.nextafter(slice_bottom, -np.inf))
                if slice_bottom <= slice_top:
                    continue

                formation_input_well = full_well.depth_slice(slice_top, slice_bottom)
                if formation_input_well.n_samples == 0:
                    continue

                formation_original_curves = set(formation_input_well.curves)
                workflow_result = run_dynamic_workflow_with_zones(formation_input_well)
                warnings = list(fallback_warnings)
                warnings.extend(workflow_result.warnings)
                warnings.extend(workflow_result.base_result.warnings)
                for zone_result in workflow_result.zone_results:
                    warnings.extend(zone_result.warnings)

                errors = list(workflow_result.base_result.errors)
                for zone_result in workflow_result.zone_results:
                    errors.extend(zone_result.errors)

                _finalize_export_curves(workflow_result.well, formation_original_curves)
                formation_depth_info = _depth_summary(formation_input_well)
                lithology_segments = _build_lithology_group_segments(
                    workflow_result.well,
                    minimum_group_thickness=minimum_group_thickness,
                    lithology_mapping=lithology_mapping,
                )

                formation_display_name = str(segment.get("display_name") or segment["name"])
                formation_folder = well_folder / _sanitize_path_component(formation_display_name)
                formation_folder.mkdir(parents=True, exist_ok=True)

                for lithology_segment in lithology_segments:
                    group_well = _well_mask_slice(workflow_result.well, lithology_segment["mask"])
                    if group_well.n_samples == 0:
                        continue

                    raw_group_name = str(lithology_segment["group_name"])
                    raw_lithology_name = str(lithology_segment["lithology"])
                    json_group_name = raw_group_name
                    json_lithology_name = raw_lithology_name
                    file_group_name = raw_group_name
                    if raw_lithology_name == "NAN":
                        file_group_name = raw_group_name
                        json_group_name = "No lithology information available"
                        json_lithology_name = "No lithology information available"

                    file_stem = f"{formation_display_name}_{file_group_name}"
                    if formation_name_counts.get(formation_display_name, 0) > 1:
                        file_stem = (
                            f"{file_stem}"
                            f"__{formation_depth_info['min_depth']:.1f}_{formation_depth_info['max_depth']:.1f}"
                        )
                    json_path = formation_folder / f"{_sanitize_path_component(file_stem)}.json"

                    json_file = _write_reference_analysis_json_to_path(
                        json_path,
                        group_well,
                        workflow_result,
                        file_path,
                        formation_original_curves,
                        formation_name=formation_display_name,
                        formation_depth_info=formation_depth_info,
                        lithology_group_name=json_group_name,
                        lithology_name=json_lithology_name,
                        lithology_mapping=lithology_mapping,
                    )

                    results.append(
                        ExampleFormatExportResult(
                            source_file=str(file_path),
                            output_file=str(whole_well_las_path),
                            success=not errors,
                            json_file=str(json_file),
                            curves_exported=0,
                            warnings=warnings,
                            errors=errors,
                        )
                    )
        except Exception as exc:
            results.append(
                ExampleFormatExportResult(
                    source_file=str(file_path),
                    output_file="",
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


def write_reference_curve_name_mapping(
    output_dir: str | Path,
    file_name: str = "curve_name_mapping.json",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mapping_path = output_path / file_name
    payload = {
        "description": (
            "Mapping between the current LAS/JSON export mnemonics and the older/example naming "
            "conventions used in prior reference files."
        ),
        "mapping": REFERENCE_CURVE_NAME_MAPPING,
    }
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)
    return mapping_path
