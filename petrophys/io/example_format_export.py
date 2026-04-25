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

from petrophys.analysis import (
    HydrocarbonAnalysis,
    LithologyAnalysis,
    NumericalStatisticalAnalysis,
    PermeabilityAnalysis,
    PorosityAnalysis,
    SaturationAnalysis,
)
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

CATEGORY_CONTEXT_SPECS = {
    "hydrocarbon": {
        "analyzer": HydrocarbonAnalysis,
        "targets": {
            "TMPC",
            "RMF",
            "RW",
            "RWA",
            "SAL",
            "SALAPP",
            "SALMUDAPP",
            "PHID",
            "PHIN",
            "PHIS",
            "PHIT",
            "PHIE",
            "SW",
            "SXO",
            "PERM",
            "PERM_CC",
            "VCL",
            "BVW",
            "MHI",
            "PAY_FLAG_PETROPHYSICAL",
            "CROSSOVER_FLAG",
            "VP_VS_RATIO",
            "PRIME_GAS_TARGET",
            "PRIME_OIL_TARGET",
            "RESFLAG",
            "SEALFLAG",
        },
    },
    "saturation": {
        "analyzer": SaturationAnalysis,
        "targets": {
            "TMPC",
            "RMF",
            "RW",
            "RWA",
            "SAL",
            "SALAPP",
            "SALMUDAPP",
            "PHID",
            "PHIN",
            "PHIS",
            "PHIT",
            "PHIE",
            "SW",
            "SXO",
            "BVW",
            "MHI",
            "ND_SEPARATION",
            "PHIT_GAS_CORR",
        },
    },
    "permeability": {
        "analyzer": PermeabilityAnalysis,
        "targets": {
            "PERM",
            "PERM_CC",
            "RQI",
            "FZI",
            "KH",
            "PERM_QUALITY_INDEX",
            "RXO_RDEP_RATIO",
            "PHIE",
            "PHIT",
            "VCL",
            "MHI",
        },
    },
    "porosity": {
        "analyzer": PorosityAnalysis,
        "targets": {
            "PHID",
            "PHIN",
            "PHIS",
            "PHIT",
            "PHIE",
            "PHIE_CORECAL",
            "PHI_SEC",
            "DS_SEPARATION",
            "PORO_QUALITY_INDEX",
            "RQI",
            "FZI",
            "VCL",
        },
    },
    "numerical": {
        "analyzer": NumericalStatisticalAnalysis,
        "targets": {
            "TMPC",
            "RW",
            "RMF",
            "BAD_HOLE",
            "BADHOLEFLAG",
            "DATA_COMPLETENESS",
            "GR_GRAD",
            "RHOB_GRAD",
            "RDEP_GRAD",
            "NPHI_GRAD",
            "DTC_GRAD",
            "CONSISTENCY_RATIO",
            "ACOUSTIC_IMPEDANCE",
            "POISSON_RATIO",
            "YOUNG_MOD",
            "BRITTLENESS_INDEX",
            "ELECTROFACIES",
            "S_V",
            "PP_HYDROSTATIC",
            "SH_MIN",
            "NET_PAY",
        },
    },
    "lithology": {
        "analyzer": LithologyAnalysis,
        "targets": {
            "LITHO",
            "VCL",
            "VQUAR",
            "VCALC",
            "VDOLO",
            "VANHY",
            "GT_LITHOLOGY_GROUP_CODE",
            "LIMESTONE_INDICATOR",
            "DOLOMITE_INDICATOR",
            "CARBONATE_INDICATOR",
            "GR_CLAY_INDICATOR",
            "NPHI_CLAY_INDICATOR",
            "PEF_CLAY_INDICATOR",
            "COMPOSITE_CLAY",
            "CONSISTENCY_SCORE",
        },
    },
}

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


def _display_curve_name(curve: Any) -> str:
    text = str(curve or "")
    if text.upper() == "LITHOLOGY_GEOLINK":
        return "lithology"
    return text


def _display_curve_list(curves: Sequence[Any]) -> List[str]:
    return [_display_curve_name(curve) for curve in curves]


def _display_trace_text(text: Any) -> str:
    value = str(text or "")
    replacements = {
        "LITHOLOGY_GEOLINK": "lithology",
        "ground_truth_lithology_geolink": "lithology",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _display_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _display_trace_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_display_trace_value(item) for item in value]
    if isinstance(value, tuple):
        return [_display_trace_value(item) for item in value]
    if isinstance(value, str):
        if value.upper() == "LITHOLOGY_GEOLINK":
            return "lithology"
        return _display_trace_text(value)
    return value


def _raw_input_dir_for_processed(input_path: Path) -> Optional[Path]:
    """Infer a sibling raw LAS directory for processed inputs when possible."""
    name = input_path.name
    candidates: list[Path] = []
    if name.endswith("_processed"):
        candidates.append(input_path.with_name(name[: -len("_processed")]))
    if name.lower() == "eval_processed":
        candidates.append(input_path.with_name("eval"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _raw_curve_lookup(
    raw_input_dir: Optional[str | Path],
    pattern: str,
    recursive: bool,
) -> dict[str, set[str]]:
    """Return standardized raw LAS curve names keyed by LAS stem."""
    if raw_input_dir is None:
        return {}

    raw_path = Path(raw_input_dir)
    if not raw_path.exists() or not raw_path.is_dir():
        return {}

    files = sorted(raw_path.rglob(pattern) if recursive else raw_path.glob(pattern))
    lookup: dict[str, set[str]] = {}
    for file_path in files:
        try:
            raw_well = read_las(file_path)
            raw_well = standardize_curves(raw_well, rename_curves=True, convert_units=True)
            lookup[file_path.stem.lower()] = set(raw_well.curves)
        except Exception:
            continue
    return lookup


def _raw_original_curves_for_file(
    file_path: Path,
    processed_original_curves: set[str],
    raw_curves_by_stem: dict[str, set[str]],
) -> set[str]:
    """Use raw LAS curves for context provenance, falling back to processed input."""
    return set(raw_curves_by_stem.get(file_path.stem.lower()) or processed_original_curves)


def _input_log_display_list(curves: set[str]) -> List[str]:
    return sorted(_display_curve_name(name) for name in curves)


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


def _category_analysis_result(category: str, well: "Well") -> CategoryAnalysisResult:
    spec = CATEGORY_CONTEXT_SPECS[category]
    analyzer = spec["analyzer"]()
    return analyzer.analyze(well, add_curves=False)


def _category_derived_logs_computed(
    category: str,
    well: "Well",
    workflow_result,
    original_curves: set[str],
    category_result: CategoryAnalysisResult,
) -> List[str]:
    base_derived = set(_derived_logs_computed(well, workflow_result, original_curves))
    category_targets = {
        _canonical_delivery_curve_name(name)
        for name in CATEGORY_CONTEXT_SPECS[category]["targets"]
    }

    derived_curves = {
        _canonical_delivery_curve_name(name)
        for name, curve in category_result.derived_curves.items()
        if curve is not None and np.any(np.isfinite(_normalize_numeric_array(curve.data)))
    }

    return sorted((base_derived & category_targets) | (derived_curves & category_targets))


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
            return "SXO fallback should have used SW^(1/5) because RSHA or RXO was absent"
        if not rmf_present:
            return "RSHA or RXO was available but RMF was unavailable"

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
        return "not calculated because TEMP was unavailable and a fallback temperature profile could not be written"

    if curve == "RMF":
        return "no RMF, RM, CLFL salinity, or RMF salinity input was available, and mud type was not OBM"

    if curve in {"PHIE_CORECAL", "PERM_CC"} and step is not None:
        same_well_models = step.parameters.get("same_well_core_models")
        peer_models = step.parameters.get("peer_core_models")
        if (same_well_models or 0) == 0 and (peer_models or 0) == 0:
            if curve == "PHIE_CORECAL":
                return "no usable core porosity calibration was available; requires paired log porosity and core porosity"
            return "no usable core permeability calibration was available; requires paired porosity and core permeability"

    if curve == "SWIRR_APP" and step is not None:
        swirr_source = step.parameters.get("swirr_source")
        if swirr_source:
            return f"Swirr source was {swirr_source}"

    if curve == "INVASION_RATIO":
        has_shallow = well.get("RSHA") is not None or well.get("RESS") is not None or well.get("RXO") is not None
        has_deep = well.get("RDEP") is not None or well.get("RT") is not None
        if not has_shallow and not has_deep:
            return "shallow resistivity and deep resistivity curves were unavailable; requires RSHA, RESS, or RXO plus RDEP or RT"
        if not has_shallow:
            return "shallow resistivity curve was unavailable; requires RSHA, RESS, or RXO"
        if not has_deep:
            return "deep resistivity curve was unavailable; requires RDEP or RT"

    if curve == "PAY_FLAG_PETROPHYSICAL":
        if well.get("NET_PAY") is not None:
            return "not exported as a separate curve; required inputs were available and NET_PAY is the pay flag"
        missing_inputs = [
            name
            for name, present in (
                ("PHIE", well.get("PHIE") is not None),
                ("SW", well.get("SW") is not None),
            )
            if not present
        ]
        if missing_inputs:
            return "required input curves unavailable: " + ", ".join(missing_inputs)

    if curve == "PRIME_GAS_TARGET":
        required = {
            "CROSSOVER_FLAG": well.get("CROSSOVER_FLAG") is not None,
            "VP_VS_RATIO": well.get("VP_VS_RATIO") is not None,
            "PHIE": well.get("PHIE") is not None,
            "PERM or PERM_CC": well.get("PERM") is not None or well.get("PERM_CC") is not None,
        }
        missing_inputs = [name for name, present in required.items() if not present]
        if missing_inputs:
            return "required input curves unavailable: " + ", ".join(missing_inputs)
        return "not exported as a separate curve in this context; required inputs were available"

    if curve == "PRIME_OIL_TARGET":
        required = {
            "NET_PAY or PAY_FLAG_PETROPHYSICAL": well.get("NET_PAY") is not None or well.get("PAY_FLAG_PETROPHYSICAL") is not None,
            "PHIE": well.get("PHIE") is not None,
            "PERM or PERM_CC": well.get("PERM") is not None or well.get("PERM_CC") is not None,
        }
        missing_inputs = [name for name, present in required.items() if not present]
        if missing_inputs:
            return "required input curves unavailable: " + ", ".join(missing_inputs)
        return "not exported as a separate curve in this context; required inputs were available"

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
            return "computed from Archie invaded-zone equation using RSHA or RXO plus RMF"
        if sxo_method == "fifth_root_quicklook":
            if sxo_reason:
                return f"computed with fifth-root quick-look fallback because {sxo_reason}"
            return "computed with fifth-root quick-look fallback SXO = SW^(1/5)"
    if curve == "RMF":
        return _method_label(step.method) if step.method else "computed"
    if curve == "RW":
        return _method_label(step.method) if step.method else "computed"
    if curve == "INVASION_RATIO":
        return "computed as shallow/deep resistivity ratio; invalid denominators and extreme ratios are written as NaN"
    if curve in {"PERM", "PERM_CC"}:
        cap = step.parameters.get("perm_max_clip_md")
        if cap is not None:
            try:
                cap_text = f"{float(cap):g}"
            except (TypeError, ValueError):
                cap_text = str(cap)
            return f"{_method_label(step.method) if step.method else 'computed'}; finite values clipped to <= {cap_text} mD"
        return _method_label(step.method) if step.method else "computed"
    if curve == "SWIRR_APP":
        method = step.parameters.get("swirr_app_method") or step.parameters.get("swirr_source")
        if method:
            return f"computed from permeability Swirr route: {method}"
        return "computed from permeability Swirr route"
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


def _curve_value_summary(well: "Well", curve: str) -> dict[str, Any]:
    curve_obj = _first_curve(well, _target_curve_candidates(curve))
    if curve_obj is None:
        return {"present": False}

    values = _normalize_numeric_array(curve_obj.data)
    finite = values[np.isfinite(values)]
    sample_count = int(values.size)
    finite_count = int(finite.size)
    nan_mask = ~np.isfinite(values)
    summary: dict[str, Any] = {
        "present": True,
        "curve_name_in_well": curve_obj.mnemonic,
        "unit": curve_obj.unit or "",
        "description": _curve_description(curve, curve_obj),
        "sample_count": sample_count,
        "finite_count": finite_count,
        "nan_count": int(np.count_nonzero(nan_mask)),
        "finite_percent": float(finite_count / max(sample_count, 1) * 100.0),
    }
    if finite_count:
        summary.update(
            {
                "min": float(np.nanmin(finite)),
                "p05": float(np.nanpercentile(finite, 5)),
                "p50": float(np.nanpercentile(finite, 50)),
                "p95": float(np.nanpercentile(finite, 95)),
                "max": float(np.nanmax(finite)),
            }
        )

    depth = getattr(well, "depth", None)
    if depth is not None and len(depth) == sample_count and np.any(nan_mask):
        intervals = _flag_intervals(np.asarray(depth, dtype=float), nan_mask)
        summary["nan_interval_count"] = len(intervals)
        summary["nan_intervals_first_25"] = intervals[:25]
    return summary


def _invasion_ratio_rejection_details(well: "Well") -> Optional[dict[str, Any]]:
    ratio_curve = well.get("INVASION_RATIO")
    rxo_curve = well.get("RSHA") or well.get("RXO") or well.get("RESS")
    rt_curve = well.get("RDEP") or well.get("RT")
    if ratio_curve is None or rxo_curve is None or rt_curve is None:
        return None

    rxo, rt = np.broadcast_arrays(
        _normalize_numeric_array(rxo_curve.data),
        _normalize_numeric_array(rt_curve.data),
    )
    ratio = _normalize_numeric_array(ratio_curve.data)
    rt_min = 0.2
    max_ratio = 100.0
    core_valid = (
        np.isfinite(rxo)
        & np.isfinite(rt)
        & (rxo > 0.0)
        & (rt >= rt_min)
    )
    raw_ratio = np.full(rxo.shape, np.nan, dtype=float)
    raw_ratio[core_valid] = rxo[core_valid] / rt[core_valid]
    finite_ratio = np.isfinite(ratio)
    nan_mask = ~finite_ratio
    details: dict[str, Any] = {
        "rxo_curve": rxo_curve.mnemonic,
        "rt_curve": rt_curve.mnemonic,
        "total_samples": int(ratio.size),
        "valid_samples": int(np.count_nonzero(finite_ratio)),
        "nan_samples": int(np.count_nonzero(nan_mask)),
        "rt_min_used": rt_min,
        "max_ratio_used": max_ratio,
        "nonfinite_rxo_samples": int(np.count_nonzero(~np.isfinite(rxo))),
        "nonfinite_rt_samples": int(np.count_nonzero(~np.isfinite(rt))),
        "nonpositive_rxo_samples": int(np.count_nonzero(np.isfinite(rxo) & (rxo <= 0.0))),
        "rt_below_min_samples": int(np.count_nonzero(np.isfinite(rt) & (rt < rt_min))),
        "ratio_above_max_samples": int(np.count_nonzero(core_valid & (raw_ratio > max_ratio))),
    }
    depth = getattr(well, "depth", None)
    if depth is not None and len(depth) == ratio.size and np.any(nan_mask):
        intervals = _flag_intervals(np.asarray(depth, dtype=float), nan_mask)
        details["nan_interval_count"] = len(intervals)
        details["nan_intervals"] = intervals[:25]
    return details


def _compose_curve_computation_note(
    curve: str,
    status: str,
    reason: str,
    step: Any,
    value_summary: dict[str, Any],
) -> str:
    label = FAMILY_LABELS.get(curve, curve)
    parts = [f"{curve} ({label}): status={status}"]
    if reason:
        parts.append(reason)
    if step is not None:
        step_name = str(getattr(step, "step", "") or "")
        if step_name == "porosity":
            route = _porosity_curve_route_reason(curve, step)
            if route:
                parts.append(route)
        if step.method:
            parts.append(f"workflow method={step.method}")
        if step.input_curves:
            parts.append(f"input curves={', '.join(_display_curve_list(step.input_curves))}")
        if step.decisions and curve != "INVASION_RATIO" and step_name != "porosity":
            parts.append("workflow decisions: " + " ".join(_display_trace_text(item) for item in step.decisions))
    if value_summary.get("present"):
        parts.append(
            f"segment finite samples={value_summary.get('finite_count')} of {value_summary.get('sample_count')}"
        )
        if value_summary.get("nan_count"):
            parts.append(f"NaN samples={value_summary.get('nan_count')}")
    return ". ".join(part.rstrip(".") for part in parts if part) + "."


def _build_reference_curve_computation_details(
    well: "Well",
    workflow_result,
    original_curves: set[str],
) -> List[dict[str, Any]]:
    statuses = _build_reference_derived_log_status(well, workflow_result, original_curves)
    step_results, _ = _step_lookup(workflow_result)
    details: List[dict[str, Any]] = []

    for status_item in statuses:
        curve = status_item["mnemonic"]
        step_name = REFERENCE_CURVE_TO_STEP.get(curve)
        step = step_results.get(step_name) if step_name else None
        value_summary = _curve_value_summary(well, curve)
        item: dict[str, Any] = {
            "mnemonic": curve,
            "family": FAMILY_LABELS.get(curve, curve),
            "status": status_item["status"],
            "reason": status_item["reason"],
            "value_summary": value_summary,
            "note": _compose_curve_computation_note(
                curve,
                status_item["status"],
                status_item["reason"],
                step,
                value_summary,
            ),
        }
        if step is not None:
            item["source_step"] = step.step
            item["method"] = step.method
            item["input_curves"] = _display_curve_list(step.input_curves or [])
            item["output_curves"] = _display_curve_list(step.output_curves or [])
            item["workflow_decisions"] = [_display_trace_text(decision) for decision in (step.decisions or [])]
            item["workflow_parameters"] = _display_trace_value(_json_safe(dict(step.parameters or {})))
            if not step.success:
                item["workflow_error"] = step.message
        if curve == "INVASION_RATIO":
            rejection_details = _invasion_ratio_rejection_details(well)
            item["invasion_rejection_details"] = rejection_details
            if rejection_details is not None:
                item["note"] += (
                    f" Segment invasion-ratio validity: {rejection_details['valid_samples']} valid "
                    f"of {rejection_details['total_samples']} samples; "
                    f"{rejection_details['nan_samples']} samples written as NaN/rejected."
                )
                reasons = []
                if rejection_details["rt_below_min_samples"]:
                    reasons.append(f"deep resistivity < 0.2 ohm.m={rejection_details['rt_below_min_samples']}")
                if rejection_details["ratio_above_max_samples"]:
                    reasons.append(f"raw ratio > 100={rejection_details['ratio_above_max_samples']}")
                if rejection_details["nonfinite_rxo_samples"]:
                    reasons.append(f"non-finite shallow resistivity={rejection_details['nonfinite_rxo_samples']}")
                if rejection_details["nonfinite_rt_samples"]:
                    reasons.append(f"non-finite deep resistivity={rejection_details['nonfinite_rt_samples']}")
                if rejection_details["nonpositive_rxo_samples"]:
                    reasons.append(f"shallow resistivity <= 0={rejection_details['nonpositive_rxo_samples']}")
                if reasons:
                    item["note"] += " Rejection reasons: " + "; ".join(reasons) + "."
                if rejection_details.get("nan_intervals"):
                    item["note"] += " NaN/rejected intervals: " + ", ".join(rejection_details["nan_intervals"][:10]) + "."
        details.append(item)

    return details


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
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not set" if value is None else str(value)
    if not np.isfinite(numeric):
        return "nan"
    text = f"{numeric:.{decimals}f}"
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
        "rwa_clean_wet_p5": "Rwa clean-wet P5 fallback",
        "default_fallback": "default RW fallback (0.05 ohm.m)",
        "default_fallback_profile": "default RW fallback (0.05 ohm.m) with depth-wise temperature correction",
        "component_based": "component-based porosity workflow",
        "linear_gradient_from_BHT": "linear temperature gradient from BHT",
        "fallback_geothermal_gradient": "fallback 30 C/km geothermal gradient",
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
        "archie": f"Archie route: samples with VCL < {cutoff_text}",
        "indonesia": f"Indonesia route: shaly clastic samples with VCL >= {cutoff_text}; selected because the required RCL input was available",
        "simandoux": f"Simandoux route: shaly clastic samples with VCL >= {cutoff_text}; selected because saline shaly-sand routing inputs were available",
        "north_sea": "North Sea route: shaly-sand samples with shale-exponent inputs available",
        "dual_water": f"Dual Water route: shaly carbonate samples with VCL >= {cutoff_text}",
        "waxman_smits": "Waxman-Smits route: shaly samples with shale-conductivity inputs available",
        "nigeria_schlumberger": "Nigeria Schlumberger route: selected shaly-sand samples",
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


def _rw_route_reason_text(rw_steps: Sequence[Any], selected_method: str) -> str:
    decisions = _ordered_unique(
        str(decision)
        for step in rw_steps
        for decision in (step.decisions or [])
        if decision
    )
    selected = str(selected_method or "")
    reasons: List[str] = []

    direct_methods = {
        "user_provided_temperature_profile_corrected",
        "user_provided_assumed_reference_temperature",
        "las_metadata_temperature_profile_corrected",
        "las_metadata_assumed_reference_temperature",
    }
    if selected in direct_methods:
        return ""

    if any("Rw direct source unavailable" in decision for decision in decisions):
        reasons.append("direct RW route skipped because no usable config/LAS RW value was found")
    elif selected != "salinity_conversion":
        reasons.append("direct RW route did not supply the selected RW curve")

    if selected != "salinity_conversion":
        reasons.append("salinity route skipped because no usable salinity_ppm input was supplied")

    sp_decision = next(
        (decision for decision in decisions if decision.startswith("SP method skipped because")),
        "",
    )
    sp_failure = next(
        (decision for decision in decisions if decision.startswith("Rw SP heuristic failed:")),
        "",
    )
    if selected != "sp_clean_water":
        if sp_decision:
            reasons.append("SP route skipped because " + sp_decision.split("because", 1)[1].strip())
        elif sp_failure:
            reasons.append("SP route failed because " + sp_failure.split(":", 1)[1].strip())
        elif selected in {"rxo_rt_clean_water", "rwa_clean_wet_p5", "default_fallback_profile"}:
            reasons.append("SP route did not supply the selected RW curve")

    rxo_failure = next(
        (decision for decision in decisions if decision.startswith("Rw log heuristic failed:")),
        "",
    )
    if selected != "rxo_rt_clean_water":
        if rxo_failure:
            reasons.append("Rxo/Rt route failed because " + rxo_failure.split(":", 1)[1].strip())
        elif selected in {"rwa_clean_wet_p5", "default_fallback_profile"}:
            reasons.append("Rxo/Rt route did not supply the selected RW curve")

    if selected == "default_fallback_profile":
        rwa_skip = next(
            (decision for decision in decisions if decision.startswith("Rwa clean-wet fallback skipped")),
            "",
        )
        if rwa_skip:
            reasons.append("RWA clean-wet route skipped because " + rwa_skip.split("because", 1)[1].strip())

    return "; ".join(_ordered_unique(reasons))


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


def _curve_value_note(summary: dict[str, Any]) -> str:
    if not summary.get("present"):
        return "no curve values in this exported segment"
    finite = summary.get("finite_count")
    total = summary.get("sample_count")
    text = f"finite samples={finite}/{total}"
    if finite:
        text += (
            f", min={_format_numeric(summary.get('min'), 4)}, "
            f"p50={_format_numeric(summary.get('p50'), 4)}, "
            f"max={_format_numeric(summary.get('max'), 4)}"
        )
    nan_count = summary.get("nan_count")
    if nan_count:
        text += f", NaN samples={nan_count}"
        interval_count = summary.get("nan_interval_count")
        if interval_count:
            text += f", NaN interval count={interval_count}"
    return text


def _porosity_endpoint_text(step: Any, key: str, decimals: int, label: Optional[str] = None) -> str:
    params = getattr(step, "parameters", {}) or {}
    name = label or key
    endpoints = params.get("matrix_endpoints") or []
    parts = []
    for endpoint in endpoints:
        group = endpoint.get("lithology_group")
        value = endpoint.get(key)
        if not group or value is None:
            continue
        parts.append(f"{group}: {name}={_format_numeric(value, decimals)}")
    if parts:
        return "; ".join(_ordered_unique(parts))

    value = params.get(key)
    if value is not None:
        return f"{name}={_format_numeric(value, decimals)}"

    range_text = _format_overall_range(
        [params.get(f"{key}_min"), params.get(f"{key}_max")],
        decimals,
    )
    if range_text:
        return f"{name} range {range_text}"
    return ""


def _porosity_phi_clay_text(params: dict[str, Any]) -> str:
    phi_clay = params.get("phi_clay")
    if phi_clay is None:
        return ""
    source = str(params.get("phi_clay_source") or "")
    source_text = _phi_clay_source_label(source) if source else "configured/default clay-porosity value"
    text = f"phi_clay={_format_numeric(phi_clay, 3)} v/v from {source_text}"
    count = params.get("phi_clay_basis_count")
    if count is not None:
        text += f" using {count} control samples"
    return text


def _porosity_curve_route_reason(curve: str, step: Any) -> str:
    params = getattr(step, "parameters", {}) or {}
    inputs = [str(item).upper() for item in (getattr(step, "input_curves", []) or [])]
    outputs = [str(item).upper() for item in (getattr(step, "output_curves", []) or [])]
    matrix_mode = params.get("matrix_mode")
    depth_suffix = "(depth)" if matrix_mode == "lithology_dependent" else ""

    if curve == "PHIN":
        endpoint_text = _porosity_endpoint_text(step, "nphi_ma", 3)
        if "PHIN" in outputs:
            text = (
                f"PHIN calculated from NPHI as clip(NPHI - nphi_ma{depth_suffix}, 0, 0.6) "
                "because NPHI was available"
            )
        elif "PHIN" in inputs:
            text = "PHIN reused from the existing PHIN curve because it was already present in the well"
        else:
            text = "PHIN was not calculated because NPHI and existing PHIN were unavailable"
        if endpoint_text:
            text += f"; nphi_ma used: {endpoint_text}"
        return text

    if curve == "PHID":
        rho_text = _porosity_endpoint_text(step, "rho_ma", 3)
        rho_fl = params.get("rho_fl")
        text = (
            f"PHID calculated from RHOB as clip((rho_ma{depth_suffix} - RHOB) / "
            f"(rho_ma{depth_suffix} - rho_fl), 0, 0.5) because RHOB was available"
        )
        details = []
        if rho_text:
            details.append(f"rho_ma used: {rho_text}")
        if rho_fl is not None:
            details.append(f"rho_fl={_format_numeric(rho_fl, 3)}")
        if details:
            text += "; " + "; ".join(details)
        return text

    if curve == "PHIS":
        dt_text = _porosity_endpoint_text(step, "dt_ma", 1)
        dt_fl = params.get("dt_fl")
        text = (
            f"PHIS calculated from DTC by Wyllie as clip((DTC - dt_ma{depth_suffix}) / "
            f"(dt_fl - dt_ma{depth_suffix}), 0, 0.5) because DTC was available"
        )
        details = []
        if dt_text:
            details.append(f"dt_ma used: {dt_text} us/ft")
        if dt_fl is not None:
            details.append(f"dt_fl={_format_numeric(dt_fl, 1)} us/ft")
        if details:
            text += "; " + "; ".join(details)
        return text

    if curve == "PHIT":
        method = params.get("phit_method")
        if method == "adaptive_nd":
            return (
                "PHIT calculated from PHIN and PHID by the adaptive neutron-density rule because both curves were available: "
                "PHIT = PHID where PHID > PHIN + 0.05; otherwise PHIT = 0.6*PHID + 0.4*PHIN; clipped to 0-0.5"
            )
        if method == "existing_phit":
            return "PHIT reused from the existing PHIT curve because no new PHIN/PHID/PHIS route replaced it"
        if method == "density_only":
            return "PHIT was not written by the porosity step; density-only porosity used PHID as the effective-porosity basis"
        if method == "neutron_only":
            return "PHIT was not written by the porosity step; neutron-only porosity used PHIN as the effective-porosity basis"
        if method == "sonic_only":
            return "PHIT was not written by the porosity step; sonic-only porosity used PHIS as the effective-porosity basis"
        return "PHIT route was evaluated by the porosity step"

    if curve == "PHIE":
        basis = params.get("phie_basis") or "selected porosity"
        clay_text = _porosity_phi_clay_text(params)
        vcl_inputs = [name for name in inputs if name in {"VCL", "VCLAY", "VSH"}]
        if vcl_inputs:
            text = (
                f"PHIE calculated as clip({basis} - {vcl_inputs[0]} * phi_clay, 0, {basis}) "
                f"because {basis} and {vcl_inputs[0]} were available"
            )
            if clay_text:
                text += f"; {clay_text}"
            return text
        return f"PHIE copied from {basis} because no clay-volume curve was available"

    if curve == "PHI_SEC":
        return "PHI_SEC calculated as max(PHIT - PHIS, 0) because PHIT and PHIS were available"

    return ""


def _curve_route_reason(curve: str, step: Any, well: "Well") -> str:
    if step is None:
        return ""

    step_name = str(getattr(step, "step", "") or "")
    method = str(getattr(step, "method", "") or "")
    params = getattr(step, "parameters", {}) or {}
    inputs = list(getattr(step, "input_curves", []) or [])

    if step_name == "temperature":
        if method == "linear_gradient_from_BHT":
            return (
                f"BHT route selected because a BHT anchor was available; surface_temp={params.get('surface_temp_f')} F, "
                f"BHT={params.get('bht_f')} F, reference_depth={params.get('reference_depth_ft')} ft, "
                f"gradient={_format_numeric(params.get('gradient_f_per_100ft'), 4)} F/100ft"
            )
        if method == "fallback_geothermal_gradient":
            return (
                f"fallback geothermal route selected because no BHT anchor was available; surface_temp={params.get('surface_temp_f')} F, "
                f"gradient={_format_numeric(params.get('gradient_f_per_100ft'), 4)} F/100ft"
            )

    if step_name == "bad_hole":
        from petrophys.base.constants import DefaultParameters

        return (
            f"caliper/DRHO QC selected because input curves present were {', '.join(_display_curve_list(inputs)) or 'none'}; "
            f"washout=(CALI-BS)>{DefaultParameters.CALI_BS_THRESHOLD:.2f} in, "
            f"DRHO threshold={DefaultParameters.DRHO_THRESHOLD:.2f} g/cc, "
            f"large-hole BS>{DefaultParameters.LARGE_HOLE_BS:.2f} in, "
            f"bit_size_used={_format_numeric(params.get('bit_size_used'), 2)} in"
        )

    if step_name == "vclay":
        used = ["GR"]
        skipped = []
        if params.get("use_sp"):
            used.append("SP")
        else:
            sp_reason = "SP curve was unavailable" if well.get("SP") is None else "mud_has_clay_or_kcl=True"
            skipped.append(f"SP not used because {sp_reason}")
        if params.get("use_nd"):
            used.append("NPHI/RHOB neutron-density")
        else:
            skipped.append("neutron-density not used because NPHI or RHOB was unavailable")
        text = (
            f"VCL route selected because {', '.join(used)} inputs were usable; "
            f"GRmin={_format_numeric(params.get('gr_min'), 1)} API, "
            f"GRmax={_format_numeric(params.get('gr_max'), 1)} API"
        )
        if skipped:
            text += "; " + "; ".join(skipped)
        return text

    if step_name == "rcl":
        source = params.get("source") or method
        if source == "lithology_shale_median_rt":
            return (
                f"RCL route selected because shale lithology samples were present; "
                f"shale_sample_count={params.get('shale_sample_count')}, "
                f"RCL={_format_numeric(params.get('rcl_computed'), 4)} ohm.m"
            )
        if source == "vcl_gt_0.9_median_rt":
            return (
                f"RCL route selected because shale lithology samples were absent and VCL>0.9 samples were present; "
                f"sample_count={params.get('shale_sample_count')}, "
                f"RCL={_format_numeric(params.get('rcl_computed'), 4)} ohm.m"
            )
        return (
            f"RCL fallback selected because shale lithology and VCL>0.9 routes did not provide a usable value; "
            f"RCL={_format_numeric(params.get('rcl_computed'), 4)} ohm.m"
        )

    if step_name == "rmf":
        if method == "OBM_default":
            return f"OBM route selected because mud_type=OBM; RMF constant={_format_numeric(params.get('rmf'), 4)} ohm.m"
        if method == "mud_filtrate_salinity_conversion":
            return (
                "CLFL route selected because mud-filtrate salinity was available; "
                f"CLFL={_format_numeric(params.get('mud_filtrate_salinity_ppm'), 0)} ppm, "
                f"RMF_measured={_format_numeric(params.get('rmf_at_measured_temp'), 4)} ohm.m"
            )
        if method.startswith("las_metadata"):
            return "LAS/header RMF route selected because RMF metadata was available and higher-priority explicit config RMF was not used"
        if method.startswith("rm_to_rmf"):
            return (
                f"RM-to-RMF route selected because RM was available; conversion_factor={params.get('conversion_factor')}, "
                f"mud_water={params.get('mud_water')}"
            )
        if method == "arps_temperature_profile_correction":
            return "explicit RMF route selected because rmf_at_meas_temp was provided"

    if step_name == "rw":
        return _rw_route_reason_text([step], method)

    if step_name == "porosity":
        return _porosity_curve_route_reason(curve, step)

    if step_name == "saturation":
        counts = params.get("sw_model_sample_counts") or {}
        counts_text = ", ".join(f"{name}={count}" for name, count in sorted(counts.items())) if isinstance(counts, dict) else ""
        return (
            f"saturation route selected because deep resistivity (RDEP or RT), PHIE, RW, and VCL/lithology inputs were available; "
            f"model={params.get('sw_model')}, requested={params.get('sw_model_requested')}, "
            f"VCL_cutoff={params.get('vcl_cutoff_for_routing')}, sample_counts={counts_text}"
        )

    if step_name == "permeability":
        swirr_metadata = params.get("swirr_metadata") or {}
        text = (
            f"permeability route selected because {', '.join(_display_curve_list(inputs)) or 'required inputs'} were usable; "
            f"strategy={params.get('strategy')}, empirical_method={params.get('empirical_method')}, "
            f"Swirr_source={params.get('swirr_source')}, max_clip={params.get('perm_max_clip_md')} mD"
        )
        if curve == "SWIRR_APP" and isinstance(swirr_metadata, dict):
            text += (
                f"; SWIRR_APP audit: percentile={swirr_metadata.get('percentile')}, "
                f"rt_cutoff={swirr_metadata.get('rt_cutoff')}, min_samples={swirr_metadata.get('min_samples')}, "
                f"hc_mask_samples={swirr_metadata.get('hc_mask_samples')}"
            )
        return text

    if step_name == "rock_quality":
        return f"rock-quality curves calculated because PERM and PHIE were available; inputs={', '.join(_display_curve_list(inputs)) or 'PERM, PHIE'}"

    if step_name == "invasion":
        return (
            f"invasion ratio calculated because shallow and deep resistivity inputs were available; "
            f"valid_samples={params.get('valid_samples')}, nan_samples={params.get('nan_samples')}, "
            f"rt_min={params.get('rt_min_used')}, max_ratio={params.get('max_ratio_used')}"
        )

    if step_name == "bvw":
        return "BVW calculated because PHIE and SW were available"

    if step_name == "mhi":
        return "MHI calculated because SW and SXO were available"

    if step_name == "net_pay":
        return (
            f"flag curve calculated because VCL, PHIE, and SW were available; "
            f"VCL_cutoff={params.get('vcl_cutoff')}, PHIE_cutoff={params.get('phie_cutoff')}, "
            f"SW_cutoff={params.get('sw_cutoff')}, seal_VCL_cutoff={params.get('seal_vcl_cutoff')}"
        )

    if step_name == "lithology":
        return f"lithology/mineral route selected because usable inputs were {', '.join(_display_curve_list(inputs)) or 'none'}"

    return ""


def _include_workflow_decision_in_notes(decision: str) -> bool:
    text = str(decision or "").strip().lower()
    if not text:
        return False
    output_stat_prefixes = (
        "perm range:",
        "sw range:",
        "temp range:",
        "final vcl = min",
    )
    return not text.startswith(output_stat_prefixes)


def _build_curve_routing_notes(
    well: "Well",
    workflow_result,
    original_curves: Optional[set[str]],
) -> List[str]:
    if original_curves is None:
        return []

    details = _build_reference_curve_computation_details(
        well,
        workflow_result,
        original_curves,
    )
    step_results, _ = _step_lookup(workflow_result)
    notes: List[str] = []
    for item in details:
        curve = str(item.get("mnemonic") or "")
        status = str(item.get("status") or "unknown")
        method = item.get("method")
        source_step = item.get("source_step")
        step = step_results.get(source_step) if source_step else None
        reason = str(item.get("reason") or "").strip()
        route_reason = _curve_route_reason(curve, step, well)
        label = FAMILY_LABELS.get(curve, curve)
        clauses: List[str] = []
        if route_reason:
            clauses.append(route_reason)
        elif reason:
            clauses.append(reason)
        else:
            clauses.append(f"routing status={status}")
        if status in {"missing", "already_present"} and reason and route_reason:
            clauses.append(f"status={status}; reason={reason}")
        elif status not in {"computed", "present"} and not clauses[-1].startswith("status="):
            clauses.append(f"status={status}")
        if source_step and status == "computed":
            clauses.append(f"workflow step={source_step}")
        if method and status == "computed" and source_step != "porosity":
            clauses.append(f"method={method}")
        input_curves = item.get("input_curves") or []
        if input_curves and source_step != "porosity":
            clauses.append(f"input curves={', '.join(_display_curve_list(input_curves))}")
        note = f"{curve} ({label}): " + ". ".join(clause.rstrip(".") for clause in clauses if clause) + "."
        decisions = []
        if source_step != "porosity":
            decisions = [
                str(value)
                for value in item.get("workflow_decisions") or []
                if _include_workflow_decision_in_notes(str(value))
            ]
        if decisions:
            note += " Workflow decisions: " + "; ".join(decisions) + "."
        if curve == "SWIRR_APP":
            metadata = ((item.get("workflow_parameters") or {}).get("swirr_metadata") or {})
            if metadata:
                note += (
                    " Swirr route parameters: "
                    f"percentile={metadata.get('percentile')}, "
                    f"rt_cutoff={metadata.get('rt_cutoff')} ohm.m, "
                    f"min_samples={metadata.get('min_samples')}, "
                    f"hc_mask_samples={metadata.get('hc_mask_samples')}, "
                    f"fallback={metadata.get('fallback_swirr')}."
                )
        notes.append(note)
    return notes


def _build_lithology_method_notes(
    well: "Well",
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> List[str]:
    from petrophys.base.constants import ArchieParameterDefaults

    info = _build_reference_lithology_info(well, lithology_mapping=lithology_mapping)
    notes = [
        (
            "LITHOLOGY DATA STATUS: "
            f"{info['valid_lithology_percentage']:.1f}% of rows have valid lithology labels "
            f"({info['valid_lithology_count']}/{info['total_rows']}); "
            f"NaN rows={info['nan_count']}; sentinel rows={info['sentinel_count']}; "
            f"unique lithologies={', '.join(info['unique_lithologies']) if info['unique_lithologies'] else 'none'}."
        ),
    ]
    if info["nan_count"] or info["sentinel_count"] or not info["has_valid_lithology"]:
        notes.append(
            "LITHOLOGY FALLBACK USED: invalid or missing lithology rows were routed to the generic clean-rock/sandstone-style defaults recorded in constants."
        )
    else:
        notes.append(
            "LITHOLOGY FALLBACK NOT USED: every row in this exported segment has a valid lithology label."
        )

    archie_parts: List[str] = []
    for lithology in info["unique_lithologies"][:12]:
        coeffs = ArchieParameterDefaults.get(lithology)
        archie_parts.append(
            f"{lithology} -> {coeffs.lithology} (a={coeffs.a}, m={coeffs.m}, n={coeffs.n}; closest={coeffs.closest_lithology})"
        )
    if archie_parts:
        notes.append(
            "ARCHIE COEFFICIENTS FOR SEGMENT LITHOLOGIES: "
            + "; ".join(archie_parts)
            + "."
        )
    return notes


def _first_step_result(workflow_result, step_name: str):
    steps = _all_step_results(workflow_result, step_name)
    return steps[0] if steps else None


def _curve_range_text(well: "Well", names: Sequence[str], decimals: int = 4, unit: str = "") -> str:
    curve = _first_curve(well, names)
    if curve is None:
        return ""
    values = _normalize_numeric_array(curve.data)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    text = f"{_format_numeric(float(np.nanmin(finite)), decimals)}-{_format_numeric(float(np.nanmax(finite)), decimals)}"
    return text + (f" {unit}" if unit else "")


def _compact_status_lists(
    well: "Well",
    workflow_result,
    original_curves: Optional[set[str]],
) -> tuple[list[str], list[str]]:
    if original_curves is None:
        return [], []
    statuses = _build_reference_derived_log_status(well, workflow_result, original_curves)
    computed = [item["mnemonic"] for item in statuses if item.get("status") == "computed"]
    missing = [item["mnemonic"] for item in statuses if item.get("status") == "missing"]
    return computed, missing


def _compact_lithology_notes(
    well: "Well",
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> list[str]:
    from petrophys.base.constants import ArchieParameterDefaults

    info = _build_reference_lithology_info(well, lithology_mapping=lithology_mapping)
    unique = ", ".join(info["unique_lithologies"]) if info["unique_lithologies"] else "none"
    notes = [
        (
            "Lithology: "
            f"valid={info['valid_lithology_count']}/{info['total_rows']} "
            f"({info['valid_lithology_percentage']:.0f}%), NaN={info['nan_count']}, "
            f"sentinel={info['sentinel_count']}, unique={unique}."
        ),
        "Lithology fallback: "
        + (
            "generic clean-rock defaults used for invalid/missing rows."
            if info["nan_count"] or info["sentinel_count"] or not info["has_valid_lithology"]
            else "not used."
        ),
    ]

    archie_parts = []
    for lithology in info["unique_lithologies"][:8]:
        coeffs = ArchieParameterDefaults.get(lithology)
        archie_parts.append(f"{lithology}: a={coeffs.a}, m={coeffs.m}, n={coeffs.n} ({coeffs.closest_lithology})")
    if archie_parts:
        notes.append("Archie coefficients: " + "; ".join(archie_parts) + ".")
    return notes


def _segment_lithology_groups_for_saturation(well: "Well", fallback: Any = "clastic") -> np.ndarray:
    """Return the same per-depth lithology groups used by saturation routing."""
    try:
        from petrophys.workflow.dynamic_workflow import (
            COMPUTED_LITHO_CODE_TO_GROUP,
            GROUND_TRUTH_CODE_TO_GROUP,
        )
    except Exception:
        GROUND_TRUTH_CODE_TO_GROUP = {}
        COMPUTED_LITHO_CODE_TO_GROUP = {}

    fallback_text = str(fallback or "clastic")
    if fallback_text == "per-depth array":
        fallback_text = "clastic"

    def _groups_from_curve(curve_name: str, mapping: dict[int, str]) -> Optional[np.ndarray]:
        curve = well.get(curve_name)
        if curve is None:
            return None
        try:
            data = np.asarray(curve.data, dtype=float)
        except (TypeError, ValueError):
            return None
        groups = np.full(data.shape, fallback_text, dtype=object)
        finite = np.isfinite(data)
        if np.any(finite):
            for idx, code in zip(np.where(finite)[0], np.rint(data[finite]).astype(int)):
                groups[idx] = mapping.get(int(code), fallback_text)
        return groups

    geolink_groups = _groups_from_curve("LITHOLOGY_GEOLINK", GROUND_TRUTH_CODE_TO_GROUP)
    if geolink_groups is not None:
        return geolink_groups
    computed_groups = _groups_from_curve("LITHO", COMPUTED_LITHO_CODE_TO_GROUP)
    if computed_groups is not None:
        return computed_groups
    return np.full(int(well.n_samples), fallback_text, dtype=object)


def _segment_saturation_model_counts(well: "Well", params: dict[str, Any]) -> dict[str, int]:
    """Recreate the saturation model routing for the exported interval only."""
    n_samples = int(well.n_samples)
    requested = str(params.get("sw_model_requested") or "auto").strip().lower()
    sw_curve = well.get("SW")
    if sw_curve is not None:
        finite_sw = np.isfinite(_normalize_numeric_array(sw_curve.data))
        routed_samples = int(np.count_nonzero(finite_sw))
    else:
        routed_samples = n_samples

    if requested and requested != "auto":
        return {requested: routed_samples}

    vcl_curve = well.get("VCL") or well.get("VCLAY") or well.get("VSH")
    if vcl_curve is not None:
        vcl = _normalize_numeric_array(vcl_curve.data)
    else:
        vcl = np.zeros(n_samples, dtype=float)
    vcl = np.resize(vcl, n_samples)

    lithology = _segment_lithology_groups_for_saturation(
        well, params.get("lithology_type") or "clastic"
    )
    lithology = np.resize(lithology, n_samples)

    cutoff = float(params.get("vcl_cutoff_for_routing") or 0.3)
    salinity = params.get("salinity_ppm")
    cec_present = bool(params.get("cec_present") or params.get("cec") is not None)
    b_coeff_present = bool(params.get("b_coeff_present") or params.get("b_coeff") is not None)
    counts: dict[str, int] = {}
    for idx in range(n_samples):
        if sw_curve is not None and idx < finite_sw.size and not bool(finite_sw[idx]):
            continue
        vcl_value = float(vcl[idx]) if np.isfinite(vcl[idx]) else 0.0
        if vcl_value < cutoff:
            model = "archie"
        else:
            group = str(lithology[idx] or "").strip().lower()
            if group in {"carbonate", "limestone", "dolomite", "calcite"}:
                model = "dual_water"
            elif cec_present and b_coeff_present:
                model = "waxman_smits"
            elif salinity is not None and float(salinity) >= 20000.0:
                model = "simandoux"
            else:
                model = "indonesia"
        counts[model] = counts.get(model, 0) + 1
    return counts


def _model_display_name(model: str) -> str:
    return {
        "archie": "Archie",
        "indonesia": "Indonesia",
        "simandoux": "Simandoux",
        "dual_water": "Dual Water",
        "waxman_smits": "Waxman-Smits",
        "north_sea": "North Sea",
        "nigeria_schlumberger": "Nigeria-Schlumberger",
    }.get(str(model), str(model).replace("_", " ").title())


def _sw_equation_text(models: Iterable[str], params: dict[str, Any]) -> str:
    parts: list[str] = []
    model_set = {str(model) for model in models}
    if "archie" in model_set:
        parts.append("Archie Sw=(a*RW/(PHIE^m*Rt))^(1/n)")
    if "indonesia" in model_set:
        parts.append(
            "Indonesia 1/sqrt(Rt)=VCL^(1-VCL/2)/sqrt(RCL)+PHIE^(m/2)*Sw^(n/2)/sqrt(a*RW)"
        )
    if "simandoux" in model_set:
        parts.append("Simandoux 1/Rt=(PHIE^m/(a*RW))*Sw^n+(VCL/RCL)*Sw")
    if "dual_water" in model_set:
        rwb = params.get("rwb")
        phi_cl = params.get("phi_cl")
        rwb_text = f", Rwb={_format_numeric(rwb, 4)} ohm.m" if rwb is not None else ", Rwb=0.3*RW"
        phi_cl_text = f", phi_cl={_format_numeric(phi_cl, 3)}" if phi_cl is not None else ""
        parts.append(
            "Dual Water phi_bw=VCL*phi_cl, Swb=phi_bw/PHIT, "
            f"1/Rwt=Swb/Rwb+(1-Swb)/RW, Sw=[a*Rwt/(PHIT^m*Rt)]^(1/n){rwb_text}{phi_cl_text}"
        )
    if "waxman_smits" in model_set:
        parts.append("Waxman-Smits Ct=(Sw^n/F*)*(Cw+B*Qv/Sw), solved iteratively")
    if "north_sea" in model_set:
        parts.append("North Sea 1/Rt=(PHIE^m/(a*RW))*Sw^n+VCL^vsh_exp/RTSH")
    if "nigeria_schlumberger" in model_set:
        parts.append("Nigeria-Schlumberger 1/sqrt(Rt)=VCL^shale_exp/sqrt(RSH)+PHIE^(m/2)*Sw^(n/2)/sqrt(a*RW)")
    return "; ".join(parts)


def _sw_route_reason_text(model: str, params: dict[str, Any]) -> str:
    cutoff = params.get("vcl_cutoff_for_routing")
    salinity = params.get("salinity_ppm")
    cec_present = bool(params.get("cec_present") or params.get("cec") is not None)
    b_coeff_present = bool(params.get("b_coeff_present") or params.get("b_coeff") is not None)
    if model == "archie":
        return f"VCL<{cutoff}"
    if model == "dual_water":
        return f"VCL>={cutoff} and carbonate lithology"
    if model == "waxman_smits":
        return f"VCL>={cutoff} with CEC and B"
    if model == "simandoux":
        return f"VCL>={cutoff} and salinity={_format_numeric(salinity, 0)} ppm"
    if model == "indonesia":
        if not cec_present or not b_coeff_present:
            sal_text = "no salinity>=20000 ppm" if salinity is None or float(salinity) < 20000.0 else f"salinity={_format_numeric(salinity, 0)} ppm"
            return f"VCL>={cutoff}, non-carbonate, no CEC/B, {sal_text}"
        return f"VCL>={cutoff}, non-carbonate"
    return "configured saturation model"


def _permeability_equation_text(method: str, params: dict[str, Any]) -> str:
    method_norm = str(method or "").strip().lower()
    if method_norm == "timur":
        return "Timur k=8581*PHIE^4.4/SWIRR^2"
    if method_norm in {"coates", "coates_denoo"}:
        return "Coates-Denoo k=100^2*(PHIE*(1-SWIRR)/SWIRR)^2"
    if method_norm == "wyllie_rose":
        return "Wyllie-Rose k=250*PHIE^3/SWIRR^2"
    if method_norm == "lith_vcl":
        metadata = params.get("empirical_method_metadata") or {}
        groups = metadata.get("lithology_parameter_groups") or []
        group_text = f"; parameter groups={', '.join(map(str, groups))}" if groups else ""
        return "lith_vcl k=10^min(4.5, slope*PHIT*(1-VCL*1.2)+intercept); valid VCL>0 and PHIT>0.001" + group_text
    return f"{method} empirical equation"


def _rw_temperature_formula_text() -> str:
    return "Rw75=0.0123+3647.5/salinity^0.955; RW(T)=Rw75*(75+6.77)/(TEMP+6.77)"


def _vcl_gr_equation(method: str) -> str:
    method_norm = str(method or "linear").lower()
    if method_norm == "linear":
        return "VCL_GR=IGR"
    if method_norm == "larionov_tertiary":
        return "VCL_GR=0.083*(2^(3.7*IGR)-1)"
    if method_norm == "larionov_older":
        return "VCL_GR=0.33*(2^(2*IGR)-1)"
    if method_norm == "steiber":
        return "VCL_GR=IGR/(3-2*IGR)"
    if method_norm == "clavier":
        return "VCL_GR=1.7-sqrt(3.38-(IGR+0.7)^2)"
    return f"VCL_GR={method_norm}(IGR)"


def _vcl_curve_note(step: Any) -> str:
    params = getattr(step, "parameters", {}) or {}
    gr_min_raw = params.get("gr_min")
    gr_max_raw = params.get("gr_max")
    try:
        gr_clean = min(float(gr_min_raw), 20.0)
    except (TypeError, ValueError):
        gr_clean = gr_min_raw
    try:
        gr_shale = max(float(gr_max_raw), 100.0)
    except (TypeError, ValueError):
        gr_shale = gr_max_raw

    formulas = [
        f"IGR=clip((GR-{_format_numeric(gr_clean, 1)})/({_format_numeric(gr_shale, 1)}-{_format_numeric(gr_clean, 1)}),0,1)",
        _vcl_gr_equation(params.get("gr_method")),
    ]
    used = ["GR"]
    if params.get("use_sp"):
        used.append("SP")
        formulas.append(
            "VCL_SP=clip((SPclean-SP)/(SPclean-SPshale),0,1); "
            f"SPclean={_format_numeric(params.get('sp_max'), 1)} mV, "
            f"SPshale={_format_numeric(params.get('sp_min'), 1)} mV"
        )
    if params.get("use_nd"):
        used.append("ND")
        formulas.append(
            "VCL_ND=clip(((RHOB2-RHOB1)*(NPHI-NPHI1)-(RHOB-RHOB1)*(NPHI2-NPHI1))/"
            "((RHOB2-RHOB1)*(NPHIclay-NPHI1)-(RHOBclay-RHOB1)*(NPHI2-NPHI1)),0,1); "
            f"clean1=({_format_numeric(params.get('nd_clean1_density'), 2)},"
            f"{_format_numeric(params.get('nd_clean1_neutron'), 2)}), "
            f"clean2=({_format_numeric(params.get('nd_clean2_density'), 2)},"
            f"{_format_numeric(params.get('nd_clean2_neutron'), 2)}), "
            f"clay=({_format_numeric(params.get('nd_clay_density'), 2)},"
            f"{_format_numeric(params.get('nd_clay_neutron'), 2)})"
        )
    return f"VCL: final=min({', '.join(used)}); " + "; ".join(formulas) + "."


def _rmf_curve_note(step: Any) -> str:
    params = getattr(step, "parameters", {}) or {}
    rmf_range = _depth_profile_range_text([step], "rmf_min", "rmf_max", 4, "ohm.m")
    method = getattr(step, "method", "")
    arps = "Arps R(T)=Rmeas*(Tmeas+6.77)/(TEMP+6.77)"
    if method == "OBM_default":
        return f"RMF: OBM default; RMF=200 ohm.m constant; RMF(T)={rmf_range}."
    if method in {"arps_temperature_profile_correction", "las_metadata_profile_temperature_correction"}:
        measured = params.get("rmf_at_surface", params.get("rmf_at_measured_temp"))
        return (
            f"RMF: {arps}; Rmeas={_format_numeric(measured, 4)} ohm.m, "
            f"Tmeas={_format_numeric(params.get('measured_temp_f'), 1)} F; RMF(T)={rmf_range}."
        )
    if method == "las_metadata_assumed_reference_temperature":
        return (
            f"RMF: {arps}; Rmeas={_format_numeric(params.get('rmf_at_reference_temp'), 4)} ohm.m, "
            f"Tmeas={_format_numeric(params.get('reference_temp_f'), 1)} F; RMF(T)={rmf_range}."
        )
    if method == "mud_filtrate_salinity_conversion":
        return (
            f"RMF: CLFL salinity conversion; Rmf75=0.0123+3647.5/CLFL^0.955, then {arps}; "
            f"CLFL={_format_numeric(params.get('mud_filtrate_salinity_ppm'), 0)} ppm, "
            f"Rmeas={_format_numeric(params.get('rmf_at_measured_temp'), 4)} ohm.m at "
            f"{_format_numeric(params.get('measured_temp_f'), 1)} F; RMF(T)={rmf_range}."
        )
    if str(method).startswith("rm_to_rmf_conversion_factor"):
        return (
            f"RMF: RM first corrected by {arps}, then RMF=RM(T)*{_format_numeric(params.get('conversion_factor'), 2)}; "
            f"RM={_format_numeric(params.get('rm_at_measured_temp'), 4)} ohm.m at "
            f"{_format_numeric(params.get('rm_measured_temp_f'), 1)} F; RMF(T)={rmf_range}."
        )
    return f"RMF: {method}; {arps}; RMF(T)={rmf_range or 'not recorded'}."


def _rw_curve_note(step: Any, well: "Well") -> str:
    params = getattr(step, "parameters", {}) or {}
    rw_range = _depth_profile_range_text([step], "rw_min", "rw_max", 4, "ohm.m")
    method = getattr(step, "method", "")
    formula = _rw_temperature_formula_text()

    if method == "rwa_clean_wet_p5":
        return (
            "RW: RWA clean-wet P5; "
            f"RWA=RDEP*PHIE^{_format_numeric(params.get('archie_m'), 3)}/a, "
            f"a={_format_numeric(params.get('archie_a'), 3)}, "
            f"mask=VCL<{params.get('vcl_cutoff')} and PHIE>{params.get('phie_cutoff')}; "
            f"P5={_format_numeric(params.get('rw_from_rwa'), 4)} ohm.m, "
            f"samples={params.get('rwa_clean_wet_samples')}, "
            f"salinity={_format_numeric(params.get('salinity_ppm_estimated'), 0)} ppm; {formula}; RW(T)={rw_range}."
        )
    if method == "sp_clean_water":
        rmf_range = _depth_profile_range_text([step], "rmf_min", "rmf_max", 4, "ohm.m")
        return (
            "RW: SP clean-water; SSP=SPclean-SPshale, K=61+0.133*TEMP(F), "
            "RW=RMF(T)/10^(-SSP/K), clipped 0.01-10 ohm.m; "
            f"SPclean={_format_numeric(params.get('sp_clean_mv'), 2)} mV, "
            f"SPshale={_format_numeric(params.get('sp_shale_mv'), 2)} mV, "
            f"SSP={_format_numeric(params.get('ssp_mv'), 2)} mV, RMF(T)={rmf_range}; RW(T)={rw_range}."
        )
    if method == "rxo_rt_clean_water":
        return (
            "RW: clean-water Rxo/Rt; RW_clean=RMF(T)*RT/RXO, converted to salinity and corrected by temperature; "
            f"samples={params.get('n_points')}, salinity={_format_numeric(params.get('salinity_ppm_estimated'), 0)} ppm, RW(T)={rw_range}."
        )
    if method == "salinity_conversion":
        return (
            f"RW: salinity conversion; salinity={_format_numeric(params.get('salinity_ppm'), 0)} ppm; "
            f"{formula}; RW(T)={rw_range}."
        )
    if method in {"user_provided_temperature_profile_corrected", "las_metadata_temperature_profile_corrected"}:
        source = "user input" if method.startswith("user") else "LAS metadata"
        return (
            f"RW: {source}; measured RW={_format_numeric(params.get('rw_at_measured_temp'), 4)} ohm.m "
            f"at { _format_numeric(params.get('measured_temp_f'), 1)} F, converted to salinity "
            f"{_format_numeric(params.get('salinity_ppm_estimated'), 0)} ppm; {formula}; RW(T)={rw_range}."
        )
    if method in {"user_provided_assumed_reference_temperature", "las_metadata_assumed_reference_temperature"}:
        source = "user input" if method.startswith("user") else "LAS metadata"
        return (
            f"RW: {source}; RW={_format_numeric(params.get('rw'), 4)} ohm.m at reference "
            f"{_format_numeric(params.get('reference_temp_f'), 1)} F, converted to salinity "
            f"{_format_numeric(params.get('salinity_ppm_estimated'), 0)} ppm; {formula}; RW(T)={rw_range}."
        )
    if method == "default_fallback_profile":
        return (
            f"RW: default fallback; RW={_format_numeric(params.get('rw_default'), 4)} ohm.m at "
            f"{_format_numeric(params.get('reference_temp_f'), 1)} F, converted to salinity and temperature-corrected; "
            f"{formula}; RW(T)={rw_range}."
        )
    return f"RW: {method}; {formula}; RW(T)={rw_range or 'not recorded'}."


def _compact_curve_note(curve: str, status: str, reason: str, step: Any, well: "Well") -> str:
    params = getattr(step, "parameters", {}) or {}

    if curve in {"BAD_HOLE", "BADHOLEFLAG"} and step is not None:
        from petrophys.base.constants import DefaultParameters

        bad_curve = well.get("BAD_HOLE") or well.get("BADHOLEFLAG")
        bad_count = int(np.nansum(_normalize_numeric_array(bad_curve.data) > 0.5)) if bad_curve is not None else 0
        bad_pct = bad_count / max(int(well.n_samples), 1) * 100.0
        bit_size = params.get("bit_size_used")
        return (
            f"{curve}: washout=(CALI-BS)>{DefaultParameters.CALI_BS_THRESHOLD:.2f} in or "
            f"|DRHO|>{DefaultParameters.DRHO_THRESHOLD:.2f} g/cc; "
            f"large_hole=BS>{DefaultParameters.LARGE_HOLE_BS:.2f} in; rule=washout OR large_hole"
            + (f"; BS={_format_numeric(bit_size, 2)} in" if bit_size is not None else "")
            + f"; flagged={bad_count}/{well.n_samples} ({bad_pct:.0f}%)."
        )

    if curve == "TMPC" and step is not None:
        method = getattr(step, "method", "")
        temp_range = _curve_range_text(well, ["TEMP", "TMPC"], 1, "F")
        if method == "linear_gradient_from_BHT":
            return (
                "TMPC: TEMP=surface+gradient*(DEPT-minDEPT)/100; BHT-derived gradient; "
                f"surface={_format_numeric(params.get('surface_temp_f'), 1)} F, "
                f"BHT={_format_numeric(params.get('bht_f'), 1)} F, "
                f"gradient={_format_numeric(params.get('gradient_f_per_100ft'), 4)} F/100ft"
                + (f"; TEMP={temp_range}." if temp_range else ".")
            )
        return (
            "TMPC: TEMP=surface+gradient*(DEPT-minDEPT)/100; fallback geothermal gradient; "
            f"surface={_format_numeric(params.get('surface_temp_f'), 1)} F, "
            f"gradient={_format_numeric(params.get('gradient_f_per_100ft'), 4)} F/100ft"
            + (f"; TEMP={temp_range}." if temp_range else ".")
        )

    if curve == "VCL" and step is not None:
        return _vcl_curve_note(step)

    if curve == "RMF":
        if step is None or status == "missing":
            return f"RMF: skipped because {reason}."
        return _rmf_curve_note(step)

    if curve == "RW":
        if step is None or status == "missing":
            return f"RW: skipped because {reason}."
        return _rw_curve_note(step, well)

    if curve == "PHIN" and step is not None:
        endpoint = _porosity_endpoint_text(step, "nphi_ma", 3)
        return "PHIN: clip(NPHI-nphi_ma(depth),0,0.6)" + (f"; {endpoint}." if endpoint else ".")
    if curve == "PHID" and step is not None:
        endpoint = _porosity_endpoint_text(step, "rho_ma", 3)
        rho_fl = params.get("rho_fl")
        return (
            "PHID: clip((rho_ma(depth)-RHOB)/(rho_ma(depth)-rho_fl),0,0.5)"
            + (f"; {endpoint}" if endpoint else "")
            + (f"; rho_fl={_format_numeric(rho_fl, 3)}." if rho_fl is not None else ".")
        )
    if curve == "PHIS" and step is not None:
        endpoint = _porosity_endpoint_text(step, "dt_ma", 1)
        return (
            "PHIS: Wyllie clip((DTC-dt_ma(depth))/(dt_fl-dt_ma(depth)),0,0.5)"
            + (f"; {endpoint}" if endpoint else "")
            + f"; dt_fl={_format_numeric(params.get('dt_fl'), 1)} us/ft."
        )
    if curve == "PHIT" and step is not None:
        return "PHIT: PHID where PHID>PHIN+0.05; otherwise 0.6*PHID+0.4*PHIN; clipped 0-0.5."
    if curve == "PHIE" and step is not None:
        return "PHIE: clip(PHIT-VCL*phi_clay,0,PHIT); " + _porosity_phi_clay_text(params) + "."
    if curve == "PHI_SEC":
        return "PHI_SEC: max(PHIT-PHIS,0)."

    if curve == "SW" and step is not None:
        counts = _segment_saturation_model_counts(well, params)
        counts_text = ", ".join(
            f"{_model_display_name(name)}={count}/{well.n_samples} ({_sw_route_reason_text(name, params)})"
            for name, count in sorted(counts.items())
        ) if counts else "not recorded"
        rw_range = _depth_profile_range_text([step], "rw_min", "rw_max", 4, "ohm.m")
        requested = str(params.get("sw_model_requested") or "auto")
        route_text = "requested model" if requested != "auto" else "depth-by-depth routing"
        equations = _sw_equation_text(counts.keys(), params)
        rcl = params.get("rcl")
        rcl_text = f"; RCL={_format_numeric(rcl, 4)} ohm.m" if rcl is not None and any(name != "archie" for name in counts) else ""
        return (
            f"SW: {route_text}; {counts_text}; "
            f"equation(s): {equations}; a={params.get('a')}, m={params.get('m')}, n={params.get('n')}"
            + (f"; RW(T)={rw_range}" if rw_range else "")
            + rcl_text
            + "."
        )
    if curve == "SXO" and step is not None:
        method = params.get("sxo_method")
        if method == "archie_rxo_rmf":
            rmf_range = _depth_profile_range_text([step], "rmf_min", "rmf_max", 4, "ohm.m")
            return (
                "SXO: invaded-zone Archie; SXO=(a*RMF/(PHIE^m*Rxo))^(1/n), clipped 0-1; "
                f"a={params.get('a')}, m={params.get('m')}, n={params.get('n')}"
                + (f"; RMF(T)={rmf_range}." if rmf_range else ".")
            )
        if method == "fifth_root_quicklook":
            return "SXO: fallback SXO=SW^(1/5)."
        return f"SXO: method={method or step.method}."

    if curve == "PERM" and step is not None:
        empirical = params.get("empirical_method") or step.method
        equation = _permeability_equation_text(empirical, params)
        swirr_text = ""
        if empirical != "lith_vcl" and params.get("swirr") is not None:
            source = params.get("swirr_source")
            if source == "hc_mask_p5":
                swirr_text = f"; SWIRR=SWIRR_APP={_format_numeric(params.get('swirr'), 4)}"
            elif source == "default_0.20":
                swirr_text = f"; SWIRR=fallback {_format_numeric(params.get('swirr'), 4)}"
            else:
                swirr_text = f"; SWIRR={_format_numeric(params.get('swirr'), 4)}"
        cap = params.get("perm_max_clip_md")
        cap_text = f"; clipped <= {cap:g} mD" if isinstance(cap, (int, float)) else ""
        return f"PERM: {equation}{swirr_text}{cap_text}."
    if curve == "PERM_CC":
        if status == "computed" and step is not None:
            return "PERM_CC: core-calibrated transform; PHIE_CORECAL=A*PHIE+B, PERM_CC=10^(A2*PHIE_CORECAL+B2), clipped 0.001-10000 mD."
        return f"PERM_CC: skipped because {reason}."
    if curve == "PHIE_CORECAL":
        if status == "computed" and step is not None:
            return "PHIE_CORECAL: core porosity correction; PHIE_CORECAL=A*PHIE+B, clipped 0-0.5."
        return f"PHIE_CORECAL: skipped because {reason}."
    if curve == "SWIRR_APP" and step is not None:
        metadata = params.get("swirr_metadata") or {}
        return (
            "SWIRR_APP: P5(SW) over RESFLAG or NET_RES and deep resistivity (RDEP or RT)>5 ohm.m; "
            f"percentile={metadata.get('percentile', 5)}, min_samples={metadata.get('min_samples', 10)}, "
            f"hc_samples={metadata.get('hc_mask_samples')}."
        )

    simple = {
        "RQI": "RQI: 0.0314*sqrt(PERM/PHIE).",
        "PHI_NORM": "PHI_NORM: PHIE/(1-PHIE).",
        "FZI": "FZI: RQI/PHI_NORM.",
        "BVW": "BVW: PHIE*SW.",
        "MHI": "MHI: SW/SXO.",
        "CROSSOVER_FLAG": "CROSSOVER_FLAG: PHID-PHIN>0.01.",
        "VP_VS_RATIO": "VP_VS_RATIO: DTS divided by DTC where DTC=20-300 us/ft, DTS=20-600 us/ft, and ratio=0.5-5.0.",
    }
    if curve in simple and status != "missing":
        return simple[curve]

    if curve == "INVASION_RATIO":
        return (
            f"INVASION_RATIO: skipped because {reason}."
            if status == "missing"
            else "INVASION_RATIO: shallow resistivity / deep resistivity; NaN where deep resistivity<0.2 ohm.m or ratio>100."
        )
    if curve in {"VQUAR", "VCALC", "VDOLO", "VANHY", "LITHO"}:
        return (
            f"{curve}: multi-mineral output; sum(Vi)=1-VCL-PHIE, "
            "RHOB=sum(Vi*rho_i)+fluids, NPHI/U/DTC mixing equations where inputs are available."
        )
    if curve == "NET_ROCK" and step is not None:
        return f"NET_ROCK: 1 where VCL<{params.get('vcl_cutoff')}; else 0."
    if curve in {"NET_RES", "RESFLAG"} and step is not None:
        cutoff = params.get("phie_cutoff")
        cutoff_text = cutoff if cutoff is not None else "lithology cutoff: 0.08 clastic, 0.045 carbonate"
        return f"{curve}: 1 where NET_ROCK=1 and PHIE>{cutoff_text}; else 0."
    if curve == "NET_PAY" and step is not None:
        return f"NET_PAY: 1 where NET_RES=1 and SW<{params.get('sw_cutoff')}; else 0."
    if curve == "SEALFLAG" and step is not None:
        return f"SEALFLAG: 1 where VCL>={params.get('seal_vcl_cutoff', 0.8)}; else 0."
    if curve == "PAY_FLAG_PETROPHYSICAL" and status != "missing":
        return "PAY_FLAG_PETROPHYSICAL: resolved pay flag; 1 where existing PAY_FLAG_PETROPHYSICAL/NET_PAY>0.5."
    if curve == "PRIME_GAS_TARGET" and status != "missing":
        return "PRIME_GAS_TARGET: 1 where CROSSOVER_FLAG=1, VP_VS_RATIO<1.75, and reservoir quality is GOOD/EXCELLENT."
    if curve == "PRIME_OIL_TARGET" and status != "missing":
        return "PRIME_OIL_TARGET: 1 where pay flag=1 and reservoir quality is EXCELLENT (PERM>10 mD and PHIE>0.15)."
    if status == "missing":
        return f"{curve}: skipped because {reason}."
    if status == "already_present":
        return f"{curve}: raw input curve; not recalculated."
    return f"{curve}: computed by {getattr(step, 'step', 'workflow') if step is not None else 'workflow'}."


def _build_compact_curve_routing_notes(
    well: "Well",
    workflow_result,
    original_curves: Optional[set[str]],
) -> List[str]:
    if original_curves is None:
        return []
    statuses = _build_reference_derived_log_status(well, workflow_result, original_curves)
    step_results, _ = _step_lookup(workflow_result)
    notes: List[str] = []
    for item in statuses:
        curve = item["mnemonic"]
        if curve in {"VQUAR", "VCALC", "VDOLO", "VANHY", "LITHO"}:
            continue
        step_name = REFERENCE_CURVE_TO_STEP.get(curve)
        step = step_results.get(step_name) if step_name else None
        note = _compact_curve_note(curve, item["status"], item.get("reason", ""), step, well)
        if note:
            notes.append(note)
    return notes


def _build_concise_delivery_computation_notes(
    well: "Well",
    workflow_result,
    original_curves: Optional[set[str]] = None,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> List[str]:
    notes: List[str] = [
        "Analysis started: dynamic petrophysical workflow with per-well routing, temperature correction, and NaN handling."
    ]
    computed, missing = _compact_status_lists(well, workflow_result, original_curves)
    if computed:
        notes.append(f"Computed {len(computed)} derived logs: {', '.join(computed)}.")
    if missing:
        notes.append(f"Missing derived logs ({len(missing)}): {', '.join(missing)}.")

    notes.extend(_compact_lithology_notes(well, lithology_mapping=lithology_mapping))

    for curve, step_name in (
        ("TMPC", "temperature"),
        ("BAD_HOLE", "bad_hole"),
        ("VCL", "vclay"),
        ("RW", "rw"),
    ):
        step = _first_step_result(workflow_result, step_name)
        if step is not None:
            notes.append(_compact_curve_note(curve, "computed", "", step, well))

    rcl_step = _first_step_result(workflow_result, "rcl")
    if rcl_step is not None:
        source = rcl_step.parameters.get("source") or rcl_step.method
        if source == "lithology_shale_median_rt":
            method_text = "median(deep resistivity where lithology is shale)"
        elif source == "vcl_gt_0.9_median_rt":
            method_text = "median(deep resistivity where VCL>0.9)"
        else:
            method_text = "default RCL=5.0 ohm.m"
        notes.append(
            f"RCL: {method_text}; "
            f"samples={rcl_step.parameters.get('shale_sample_count')}; "
            f"RCL={_format_numeric(rcl_step.parameters.get('rcl_computed'), 4)} ohm.m."
        )

    por_step = _first_step_result(workflow_result, "porosity")
    if por_step is not None:
        phi_clay = _porosity_phi_clay_text(por_step.parameters)
        notes.append(
            "Porosity: PHIN=clip(NPHI-nphi_ma,0,0.6); PHID=clip((rho_ma-RHOB)/(rho_ma-rho_fl),0,0.5); "
            "PHIS=clip((DTC-dt_ma)/(dt_fl-dt_ma),0,0.5); PHIT=PHID where PHID>PHIN+0.05 else 0.6*PHID+0.4*PHIN; "
            "PHIE=clip(PHIT-VCL*phi_clay,0,PHIT); PHI_SEC=max(PHIT-PHIS,0)"
            + (f"; {phi_clay}." if phi_clay else ".")
        )

    sat_step = _first_step_result(workflow_result, "saturation")
    if sat_step is not None:
        notes.append(_compact_curve_note("SW", "computed", "", sat_step, well))
        if well.get("SXO") is not None:
            notes.append(_compact_curve_note("SXO", "computed", "", sat_step, well))

    if any(well.get(name) is not None for name in ("VQUAR", "VCALC", "VDOLO", "VANHY", "LITHO")):
        notes.append(
            "Mineral volumes (VQUAR, VCALC, VDOLO, VANHY, LITHO): solve sum(Vi)=1-VCL-PHIE plus RHOB, NPHI, U=PEF*RHOB, and DTC mixing equations. LITHO is the computed mineral-class curve, separate from input lithology labels."
        )

    perm_step = _first_step_result(workflow_result, "permeability")
    if perm_step is not None:
        notes.append(_compact_curve_note("PERM", "computed", "", perm_step, well))
        if well.get("SWIRR_APP") is not None:
            notes.append(_compact_curve_note("SWIRR_APP", "computed", "", perm_step, well))

    if any(well.get(name) is not None for name in ("RQI", "PHI_NORM", "FZI")):
        notes.append("Rock quality: RQI=0.0314*sqrt(PERM/PHIE); PHI_NORM=PHIE/(1-PHIE); FZI=RQI/PHI_NORM.")
    if well.get("BVW") is not None:
        notes.append("BVW: PHIE*SW.")
    if well.get("MHI") is not None:
        notes.append("MHI: SW/SXO.")

    net_step = _first_step_result(workflow_result, "net_pay")
    if net_step is not None:
        flag_parts = [f"NET_ROCK=VCL<{net_step.parameters.get('vcl_cutoff')}"]
        phie_cutoff = net_step.parameters.get("phie_cutoff")
        if phie_cutoff is not None:
            flag_parts.append(f"NET_RES and RESFLAG add PHIE>{phie_cutoff}")
        else:
            flag_parts.append("NET_RES and RESFLAG use reservoir screening after NET_ROCK")
        flag_parts.extend(
            [
                f"NET_PAY adds SW<{net_step.parameters.get('sw_cutoff')}",
                f"SEALFLAG=VCL>={net_step.parameters.get('seal_vcl_cutoff', 0.8)}",
                "CROSSOVER_FLAG=PHID-PHIN>0.01",
                "VP_VS_RATIO=DTS divided by DTC with plausibility screens",
            ]
        )
        notes.append("Flags: " + "; ".join(flag_parts) + ".")

    notes.extend(_build_compact_curve_routing_notes(well, workflow_result, original_curves))
    notes.append("Analysis completed: compact routing notes recorded.")
    return list(dict.fromkeys(note for note in notes if note))


def _build_delivery_computation_notes(
    well: "Well",
    workflow_result,
    original_curves: Optional[set[str]] = None,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> List[str]:
    return _build_concise_delivery_computation_notes(
        well,
        workflow_result,
        original_curves=original_curves,
        lithology_mapping=lithology_mapping,
    )

    notes: List[str] = [
        "ANALYSIS STARTED: Dynamic petrophysical calculations with per-well routing, depth-wise temperature correction, and NaN handling.",
    ]

    statuses = (
        _build_reference_derived_log_status(well, workflow_result, original_curves)
        if original_curves is not None
        else []
    )
    computed = [item["mnemonic"] for item in statuses if item.get("status") == "computed"]
    missing = [item["mnemonic"] for item in statuses if item.get("status") == "missing"]
    if computed:
        notes.append(f"COMPUTING {len(computed)} derived logs: {', '.join(computed)}.")
    if missing:
        notes.append(
            f"MISSING DERIVED LOGS AFTER ROUTING ({len(missing)}): {', '.join(missing)}. "
            "Reasons are listed in the per-curve routing notes and curve_computation_details."
        )

    notes.append(
        "CALCULATION ENGINE METHODOLOGY: the notes below report the workflow routes, input curves, parameters, counts, and output statistics actually used for this processed well and exported segment."
    )
    notes.extend(_build_lithology_method_notes(well, lithology_mapping=lithology_mapping))

    temperature_steps = _all_step_results(workflow_result, "temperature")
    if temperature_steps:
        methods = _ordered_unique(step.method for step in temperature_steps if step.method)
        temp_profile = _depth_profile_range_text(temperature_steps, "gradient_f_per_100ft", "gradient_f_per_100ft", 4, "F/100ft")
        surface_temp = _format_numeric_summary(_step_parameter_values(temperature_steps, "surface_temp_f"), 1, "F")
        bht = _format_numeric_summary(_step_parameter_values(temperature_steps, "bht_f"), 1, "F")
        ref_depth = _format_numeric_summary(_step_parameter_values(temperature_steps, "reference_depth_ft"), 1, "ft")
        temp_curve = well.get("TEMP")
        temp_range = ""
        if temp_curve is not None:
            temp_values = _normalize_numeric_array(temp_curve.data)
            finite_temp = temp_values[np.isfinite(temp_values)]
            if finite_temp.size:
                temp_range = f"{_format_numeric(float(np.nanmin(finite_temp)), 1)} to {_format_numeric(float(np.nanmax(finite_temp)), 1)} F"
        note = "TEMPERATURE: "
        if methods == ["linear_gradient_from_BHT"]:
            note += "Temperature was calculated as a linear profile from surface temperature to BHT."
            if surface_temp:
                note += f" Surface temperature used: {surface_temp}."
            if bht:
                note += f" BHT anchor: {bht}."
            if ref_depth:
                note += f" Reference depth: {ref_depth}."
        elif methods == ["fallback_geothermal_gradient"]:
            note += "No BHT anchor was available, so the fallback geothermal gradient was used from the surface temperature."
            if surface_temp:
                note += f" Surface temperature used: {surface_temp}."
        else:
            note += f"Temperature methods used in this well: {_friendly_join([_method_label(method) for method in methods])}."
        if temp_profile:
            note += f" Gradient used: {temp_profile}."
        if temp_range:
            note += f" Final TEMP range in this exported segment: {temp_range}."
        notes.append(note)

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
        use_sp = any(bool(step.parameters.get("use_sp")) for step in vcl_steps)
        use_nd = any(bool(step.parameters.get("use_nd")) for step in vcl_steps)
        methods_used = ["GR linear method"]
        if use_sp:
            methods_used.append("SP method")
        if use_nd:
            methods_used.append("neutron-density crossplot")
        note = f"VCL: Final VCL used {_friendly_join(methods_used)}. Final VCL = row-wise minimum of those method curves."
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
        if use_sp:
            sp_clean = _format_numeric_summary(_step_parameter_values(vcl_steps, "sp_max"), 1, "mV")
            sp_shale = _format_numeric_summary(_step_parameter_values(vcl_steps, "sp_min"), 1, "mV")
            note += (
                " SP method: VCL_SP = clip((SPclean - SP)/(SPclean - SPshale), 0, 1)"
            )
            if sp_clean and sp_shale:
                note += f" with SPclean {sp_clean} and SPshale {sp_shale} across intervals in this well"
            note += "."
        else:
            sp_reason = (
                "mud_has_clay_or_kcl=True"
                if any("mud has clay/KCl" in decision for step in vcl_steps for decision in step.decisions)
                else "SP curve was unavailable"
            )
            note += f" SP method was not used because {sp_reason}."
        if use_nd:
            note += (
                " Neutron-density method: 3-point crossplot using clean points "
                "(RHOB=2.71 g/cc, NPHI=0.00) and (RHOB=2.03 g/cc, NPHI=0.40), "
                "and clay point (RHOB=2.62 g/cc, NPHI=0.38)."
            )
        else:
            note += " Neutron-density method was not used because NPHI or RHOB was unavailable."
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
        selected_rw_method = str(methods[0]) if methods else ""
        rw_route_reasons = _rw_route_reason_text(rw_steps, selected_rw_method)
        rw_values = _depth_profile_range_text(rw_steps, "rw_min", "rw_max", 4, "ohm.m")
        temp_profile = _depth_profile_range_text(rw_steps, "formation_temp_min_f", "formation_temp_max_f", 1, "F")

        if methods == ["sp_clean_water"]:
            rmf_used = _depth_profile_range_text(rw_steps, "rmf_min", "rmf_max", 4, "ohm.m")
            sp_clean = _format_numeric_summary(_step_parameter_values(rw_steps, "sp_clean_mv"), 1, "mV")
            sp_shale = _format_numeric_summary(_step_parameter_values(rw_steps, "sp_shale_mv"), 1, "mV")
            ssp = _format_numeric_summary(_step_parameter_values(rw_steps, "ssp_mv"), 1, "mV")
            note = (
                "RW: RW was derived from SP clean-water calibration because no direct RW or salinity input was used, and both RMF and SP were usable. "
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
            if rw_route_reasons:
                note += f" Route reasons: {rw_route_reasons}."
            notes.append(note)
        elif methods == ["rxo_rt_clean_water"]:
            rmf_used = _depth_profile_range_text(rw_steps, "rmf_min", "rmf_max", 4, "ohm.m")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            sample_count = _format_numeric_summary(_step_parameter_values(rw_steps, "n_points"), 0)
            note = (
                "RW: RW was derived from clean-water Rxo/Rt control points because no direct RW or salinity input was used and the SP route did not produce the selected RW. "
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
            if rw_route_reasons:
                note += f" Route reasons: {rw_route_reasons}."
            notes.append(note)
        elif methods == ["salinity_conversion"]:
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm"), 0, "ppm")
            note = (
                "RW: RW was derived directly from salinity because salinity_ppm was available in the workflow configuration/LAS metadata. Formula used: "
                "Rw75 = 0.0123 + 3647.5 / SAL^0.955 and Rw(T) = Rw75 * (75 + 6.77) / (T + 6.77)."
            )
            if salinity:
                note += f" Salinity used: {salinity}."
            if rw_values:
                note += f" Final RW(T) range: {rw_values}"
                if temp_profile:
                    note += f" over {temp_profile}"
                note += "."
            if rw_route_reasons:
                note += f" Route reasons: {rw_route_reasons}."
            notes.append(note)
        elif methods == ["las_metadata_temperature_profile_corrected"]:
            measured_rw = _format_numeric_summary(_step_parameter_values(rw_steps, "rw_at_measured_temp"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "measured_temp_f"), 1, "F")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            note = (
                "RW: RW came from the LAS header because RW and RWT metadata were available. "
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
            if rw_route_reasons:
                note += f" Route reasons: {rw_route_reasons}."
            notes.append(note)
        elif methods == ["las_metadata_assumed_reference_temperature"]:
            input_rw = _format_numeric_summary(_step_parameter_values(rw_steps, "rw"), 4, "ohm.m")
            reference_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "reference_temp_f"), 1, "F")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            note = (
                "RW: RW came from the LAS header because RW metadata was available. "
                "No RWT metadata was provided, so the RW input was treated as a reference-temperature value, converted to equivalent salinity, "
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
            if rw_route_reasons:
                note += f" Route reasons: {rw_route_reasons}."
            notes.append(note)
        elif methods == ["user_provided_temperature_profile_corrected"]:
            source = _config_source_phrase(workflow_result, "rw")
            measured_rw = _format_numeric_summary(_step_parameter_values(rw_steps, "rw_at_measured_temp"), 4, "ohm.m")
            measured_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "measured_temp_f"), 1, "F")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            note = (
                f"RW: RW came from {source} because an explicit RW value and measurement temperature were available. "
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
            if rw_route_reasons:
                note += f" Route reasons: {rw_route_reasons}."
            notes.append(note)
        elif methods == ["user_provided_assumed_reference_temperature"]:
            source = _config_source_phrase(workflow_result, "rw")
            input_rw = _format_numeric_summary(_step_parameter_values(rw_steps, "rw"), 4, "ohm.m")
            reference_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "reference_temp_f"), 1, "F")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            note = (
                f"RW: RW came from {source} because an explicit RW value was available. "
                "No RWT metadata was provided, so the RW input was treated as a reference-temperature value, converted to equivalent salinity, "
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
            if rw_route_reasons:
                note += f" Route reasons: {rw_route_reasons}."
            notes.append(note)
        elif methods == ["rwa_clean_wet_p5"]:
            n_samples = _format_numeric_summary(_step_parameter_values(rw_steps, "rwa_clean_wet_samples"), 0, "samples")
            rw_from_rwa = _format_numeric_summary(_step_parameter_values(rw_steps, "rw_from_rwa"), 4, "ohm.m")
            reference_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "rwa_reference_temp_f"), 1, "F")
            archie_a = _format_numeric_summary(_step_parameter_values(rw_steps, "archie_a"), 3)
            archie_m = _format_numeric_summary(_step_parameter_values(rw_steps, "archie_m"), 3)
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            notes.append(
                "RW: RW was derived from apparent Rw because direct RW metadata/configuration, salinity input, RMF+SP, and RMF+Rxo/Rt routes did not produce usable RW. "
                "Formula used: RWA = RDEP * PHIE^m / a. Mask used: VCL < 0.30 and PHIE > 0.05. Percentile used: P5. "
                "The selected RWA value was converted to equivalent salinity and rewritten depth-by-depth as RW(T)."
                + (f" Archie coefficients: a={archie_a}, m={archie_m}." if archie_a and archie_m else "")
                + (f" Samples used: {n_samples}." if n_samples else "")
                + (f" P5 RWA reference value: {rw_from_rwa}." if rw_from_rwa else "")
                + (f" Reference temperature: {reference_temp}." if reference_temp else "")
                + (f" Equivalent salinity: {salinity}." if salinity else "")
                + (f" Final RW(T) range: {rw_values}." if rw_values else "")
                + (f" Temperature profile: {temp_profile}." if temp_profile else "")
                + (f" Route reasons: {rw_route_reasons}." if rw_route_reasons else "")
            )
        elif methods == ["default_fallback_profile"]:
            reference_temp = _format_numeric_summary(_step_parameter_values(rw_steps, "reference_temp_f"), 1, "F")
            notes.append(
                "RW: RW used the default 0.05 ohm.m input because direct RW metadata/configuration, salinity input, RMF+SP, RMF+Rxo/Rt, and RWA clean-wet routes all failed. "
                "The default RW was converted to equivalent salinity and rewritten depth-by-depth."
                + (f" Assumed reference temperature: {reference_temp}." if reference_temp else "")
                + (f" Final RW(T) range: {rw_values}." if rw_values else "")
                + (f" Route reasons: {rw_route_reasons}." if rw_route_reasons else "")
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
        if any(("PHIN from NPHI" in decision) or decision.startswith("PHIN =") for decision in decisions):
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
                "PHIT = PHID for samples with PHID > PHIN + 0.05; PHIT = 0.6*PHID + 0.4*PHIN for the remaining samples."
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
            model_counts: dict[str, int] = {}
            for step in saturation_steps:
                counts = step.parameters.get("sw_model_sample_counts") or {}
                if isinstance(counts, dict):
                    for model_name, count in counts.items():
                        try:
                            model_counts[str(model_name)] = model_counts.get(str(model_name), 0) + int(count)
                        except (TypeError, ValueError):
                            continue
            if model_counts:
                note += (
                    " Saturation model sample counts in the processed well: "
                    + ", ".join(f"{model}={count}" for model, count in sorted(model_counts.items()))
                    + "."
                )
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
            if "lith_vcl" in empirical:
                equations.append("lith_vcl equation: k = 10 ** min(4.5, slope * PHIT * (1 - VCL * 1.2) + intercept)")
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
        swirr_sources_without_not_required = [source for source in swirr_sources if source != "not_required"]
        if swirr_sources_without_not_required:
            note += f" Swirr source used: {_friendly_join(swirr_sources_without_not_required)}."
            if swirr_scalar:
                note += f" Swirr value used: {swirr_scalar}."
            if "hc_mask_p5" in swirr_sources_without_not_required:
                swirr_metadata: dict[str, Any] = {}
                for step in permeability_steps:
                    metadata = step.parameters.get("swirr_metadata") or {}
                    if isinstance(metadata, dict):
                        swirr_metadata.update(metadata)
                percentile = swirr_metadata.get("percentile", 5.0)
                rt_cutoff = swirr_metadata.get("rt_cutoff", 5.0)
                min_samples = swirr_metadata.get("min_samples", 10)
                hc_mask_samples = swirr_metadata.get("hc_mask_samples")
                note += (
                    " SWIRR_APP was calculated from log-derived P5(SW) because no explicit SWIRR/SWI/SWIR curve was supplied and the hydrocarbon mask had enough samples."
                    " Mask used: RESFLAG or NET_RES true and deep resistivity (RDEP or RT) > 5 ohm.m."
                    f" Parameters used: percentile={percentile}, deep-resistivity cutoff={rt_cutoff} ohm.m, "
                    f"minimum samples={min_samples}"
                    + (f", hydrocarbon-mask samples={hc_mask_samples}." if hc_mask_samples is not None else ".")
                )
            if "default_0.20" in swirr_sources_without_not_required:
                swirr_metadata: dict[str, Any] = {}
                for step in permeability_steps:
                    metadata = step.parameters.get("swirr_metadata") or {}
                    if isinstance(metadata, dict):
                        swirr_metadata.update(metadata)
                reason = swirr_metadata.get("reason")
                n_samples = swirr_metadata.get("hc_mask_samples")
                note += " Swirr used the 0.20 fallback."
                if reason:
                    note += f" Reason: {reason}."
                elif n_samples is not None:
                    note += f" Reason: hydrocarbon mask had {n_samples} samples, below the minimum sample requirement."
                else:
                    note += " Reason: hydrocarbon mask was unavailable or too small."
        elif "not_required" in swirr_sources:
            note += " Swirr was not used by the selected empirical permeability method."
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

    invasion_steps = _all_step_results(workflow_result, "invasion")
    if well.get("INVASION_RATIO") is not None:
        ratio_details = _invasion_ratio_rejection_details(well)
        note = (
            "INVASION: INVASION_RATIO = shallow resistivity / deep resistivity "
            "(RSHA preferred, then RXO/RESS; RDEP preferred, then RT). "
            "Values are written as NaN where inputs are non-finite, shallow resistivity is non-positive, "
            "deep resistivity is below 0.2 ohm.m, or the raw ratio exceeds 100."
        )
        if ratio_details:
            note += (
                f" In this exported segment, valid samples={ratio_details['valid_samples']} of "
                f"{ratio_details['total_samples']}; NaN/rejected samples={ratio_details['nan_samples']}."
            )
            reasons = []
            if ratio_details["nonfinite_rxo_samples"]:
                reasons.append(f"non-finite shallow resistivity={ratio_details['nonfinite_rxo_samples']}")
            if ratio_details["nonfinite_rt_samples"]:
                reasons.append(f"non-finite deep resistivity={ratio_details['nonfinite_rt_samples']}")
            if ratio_details["nonpositive_rxo_samples"]:
                reasons.append(f"shallow resistivity <= 0={ratio_details['nonpositive_rxo_samples']}")
            if ratio_details["rt_below_min_samples"]:
                reasons.append(f"deep resistivity < 0.2 ohm.m={ratio_details['rt_below_min_samples']}")
            if ratio_details["ratio_above_max_samples"]:
                reasons.append(f"raw ratio > 100={ratio_details['ratio_above_max_samples']}")
            if reasons:
                note += " Rejection reasons: " + "; ".join(reasons) + "."
            if ratio_details.get("nan_intervals"):
                note += (
                    " First NaN/rejected depth intervals: "
                    + ", ".join(ratio_details["nan_intervals"][:10])
                    + "."
                )
        elif invasion_steps:
            rejected = _format_numeric_summary(_step_parameter_values(invasion_steps, "nan_samples"), 0)
            total = _format_numeric_summary(_step_parameter_values(invasion_steps, "total_samples"), 0)
            if rejected and total:
                note += f" Workflow run rejected {rejected} of {total} samples."
        notes.append(note)

    notes.extend(_build_curve_routing_notes(well, workflow_result, original_curves))
    notes.append(
        "ANALYSIS COMPLETED: derived curves, missing-curve reasons, routing decisions, parameter values, and segment output statistics were recorded in this JSON."
    )
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


def _load_reference_group_segments(
    reference_structure_dir: Optional[str | Path],
    well_stem: str,
) -> list[dict[str, Any]]:
    if reference_structure_dir is None:
        return []

    root = Path(reference_structure_dir)
    if not root.exists():
        return []

    normalized_well = _normalize_well_selector(well_stem)
    candidates = [
        root / f"{well_stem}_corrected",
        root / well_stem,
    ]
    if _normalize_well_selector(root.name.replace("_corrected", "")) == normalized_well:
        candidates.insert(0, root)

    folder = next((candidate for candidate in candidates if candidate.exists() and candidate.is_dir()), None)
    if folder is None:
        return []

    segments: list[dict[str, Any]] = []
    for json_path in sorted(folder.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        formation_info = payload.get("formation_info") or {}
        analysis_results = payload.get("analysis_results") or {}
        group_info = analysis_results.get("lithology_group_info") or {}
        start_depth = group_info.get("start_depth", formation_info.get("min_depth"))
        end_depth = group_info.get("end_depth", formation_info.get("max_depth"))
        try:
            start_depth = float(start_depth)
            end_depth = float(end_depth)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(start_depth) or not np.isfinite(end_depth) or end_depth < start_depth:
            continue

        formation_name = str(formation_info.get("name") or analysis_results.get("formation_name") or "WHOLE_WELL")
        group_name = str(analysis_results.get("lithology group") or json_path.stem)
        lithology_name = str(analysis_results.get("lithology") or group_name)
        segments.append(
            {
                "file_stem": json_path.stem,
                "formation_name": formation_name,
                "formation_depth_info": {
                    "min_depth": formation_info.get("min_depth"),
                    "max_depth": formation_info.get("max_depth"),
                    "thickness": formation_info.get("thickness"),
                },
                "group_name": group_name,
                "lithology": lithology_name,
                "start_depth": start_depth,
                "end_depth": end_depth,
                "source_file": str(json_path),
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
    analysis_summary_override: Optional[dict[str, Any]] = None,
    derived_logs_override: Optional[Sequence[str]] = None,
    include_derived_log_status: bool = True,
) -> dict[str, Any]:
    well_name = _reference_well_name(source_file, well)
    depth_info = _depth_summary(well)
    formation_info = formation_depth_info or depth_info
    lithology_info = _build_reference_lithology_info(well, lithology_mapping=lithology_mapping)
    dominant_lithology = _dominant_lithology(well, lithology_mapping=lithology_mapping)
    derived_logs = list(derived_logs_override) if derived_logs_override is not None else _derived_logs_computed(well, workflow_result, original_curves)
    computation_notes = _build_delivery_computation_notes(
        well,
        workflow_result,
        original_curves=original_curves,
        lithology_mapping=lithology_mapping,
    )

    hydrocarbon_summary = HydrocarbonAnalysis().analyze(well, add_curves=False).to_dict()
    full_analysis_summary = _build_analysis_summary(well, workflow_result)
    selected_analysis_summary = analysis_summary_override or full_analysis_summary or hydrocarbon_summary
    data_completeness = {
        "missing_primary_logs": [_display_curve_name(name) for name in PRIMARY_LOGS_FOR_COMPLETENESS if name not in original_curves],
        "derived_log_computation_notes": computation_notes,
    }
    if include_derived_log_status:
        data_completeness["derived_log_status"] = _build_reference_derived_log_status(
            well,
            workflow_result,
            original_curves,
        )

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
            "analysis_summary": selected_analysis_summary,
            "data_completeness": data_completeness,
            "lithology_info": lithology_info,
            "input_logs_present": _input_log_display_list(original_curves),
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
    analysis_summary_override: Optional[dict[str, Any]] = None,
    derived_logs_override: Optional[Sequence[str]] = None,
    include_derived_log_status: bool = True,
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
        analysis_summary_override=analysis_summary_override,
        derived_logs_override=derived_logs_override,
        include_derived_log_status=include_derived_log_status,
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
        "derived_log_computation_notes": _build_delivery_computation_notes(
            well,
            workflow_result,
            original_curves=original_curves,
        ),
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
                warnings.append(fallback_warnings[-1])

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


def export_reference_json_work_package_by_category_formation_lithology(
    input_dir: str | Path,
    output_dir: str | Path,
    tops_file: str | Path,
    pattern: str = "*.las",
    recursive: bool = False,
    minimum_group_thickness: float = 2.0,
    well_names: Optional[Sequence[str]] = None,
    lithology_mapping_file: Optional[str | Path] = None,
    reference_structure_dir: Optional[str | Path] = None,
    raw_input_dir: Optional[str | Path] = None,
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
    raw_context_dir = Path(raw_input_dir) if raw_input_dir is not None else _raw_input_dir_for_processed(input_path)
    raw_curves_by_stem = _raw_curve_lookup(raw_context_dir, pattern, recursive)
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
            fallback_warnings: list[str] = []
            processed_original_curves = set(full_well.curves)
            raw_original_curves = _raw_original_curves_for_file(
                file_path,
                processed_original_curves,
                raw_curves_by_stem,
            )
            workflow_result = run_dynamic_workflow_with_zones(full_well)
            warnings = list(workflow_result.warnings)
            warnings.extend(workflow_result.base_result.warnings)
            for zone_result in workflow_result.zone_results:
                warnings.extend(zone_result.warnings)

            errors = list(workflow_result.base_result.errors)
            for zone_result in workflow_result.zone_results:
                errors.extend(zone_result.errors)
            if raw_context_dir is not None and file_path.stem.lower() not in raw_curves_by_stem:
                warnings.append(
                    f"Raw LAS not found in {raw_context_dir}; input_logs_present fell back to processed LAS curves."
                )
            _finalize_export_curves(workflow_result.well, processed_original_curves)

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

            reference_group_segments = _load_reference_group_segments(
                reference_structure_dir,
                file_path.stem,
            )
            if reference_group_segments:
                source_folder = Path(reference_group_segments[0]["source_file"]).parent
                reference_warnings = list(warnings)
                reference_warnings.append(
                    "Using reference formation/lithology group structure from "
                    f"{source_folder}; the tops workbook was not used to rename or split this well."
                )
                formation_original_curves = raw_original_curves

                for reference_segment in reference_group_segments:
                    group_well = workflow_result.well.depth_slice(
                        float(reference_segment["start_depth"]),
                        float(reference_segment["end_depth"]),
                    )
                    if group_well.n_samples == 0:
                        continue

                    formation_display_name = str(reference_segment["formation_name"])
                    file_stem = str(reference_segment["file_stem"])
                    json_group_name = str(reference_segment["group_name"])
                    json_lithology_name = str(reference_segment["lithology"])
                    formation_depth_info = dict(
                        reference_segment.get("formation_depth_info") or {}
                    )
                    if not all(
                        formation_depth_info.get(key) is not None
                        for key in ("min_depth", "max_depth", "thickness")
                    ):
                        formation_depth_info = _depth_summary(group_well)

                    for category_name in CATEGORY_CONTEXT_SPECS:
                        category_result = _category_analysis_result(category_name, group_well)
                        derived_logs = _category_derived_logs_computed(
                            category_name,
                            group_well,
                            workflow_result,
                            formation_original_curves,
                            category_result,
                        )
                        json_path = (
                            output_path
                            / category_name
                            / _sanitize_path_component(file_path.stem)
                            / _sanitize_path_component(formation_display_name)
                            / f"{_sanitize_path_component(file_stem)}_{category_name}.json"
                        )

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
                            analysis_summary_override=category_result.to_dict(),
                            derived_logs_override=derived_logs,
                            include_derived_log_status=True,
                        )

                        results.append(
                            ExampleFormatExportResult(
                                source_file=str(file_path),
                                output_file=str(json_file),
                                success=not errors,
                                json_file=str(json_file),
                                curves_exported=0,
                                warnings=reference_warnings,
                                errors=errors,
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

            formation_name_counts: dict[str, int] = {}
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

                formation_well = workflow_result.well.depth_slice(slice_top, slice_bottom)
                if formation_well.n_samples == 0:
                    continue

                formation_original_curves = raw_original_curves
                formation_depth_info = _depth_summary(formation_well)
                lithology_segments = _build_lithology_group_segments(
                    formation_well,
                    minimum_group_thickness=minimum_group_thickness,
                    lithology_mapping=lithology_mapping,
                )

                formation_display_name = str(segment.get("display_name") or segment["name"])
                for lithology_segment in lithology_segments:
                    group_well = _well_mask_slice(formation_well, lithology_segment["mask"])
                    if group_well.n_samples == 0:
                        continue

                    raw_group_name = str(lithology_segment["group_name"])
                    raw_lithology_name = str(lithology_segment["lithology"])
                    json_group_name = raw_group_name
                    json_lithology_name = raw_lithology_name
                    file_group_name = raw_group_name
                    if raw_lithology_name == "NAN":
                        json_group_name = "No lithology information available"
                        json_lithology_name = "No lithology information available"

                    file_stem = f"{formation_display_name}_{file_group_name}"
                    if formation_name_counts.get(formation_display_name, 0) > 1:
                        file_stem = (
                            f"{file_stem}"
                            f"__{formation_depth_info['min_depth']:.1f}_{formation_depth_info['max_depth']:.1f}"
                        )

                    for category_name in CATEGORY_CONTEXT_SPECS:
                        category_result = _category_analysis_result(category_name, group_well)
                        derived_logs = _category_derived_logs_computed(
                            category_name,
                            group_well,
                            workflow_result,
                            formation_original_curves,
                            category_result,
                        )
                        json_path = (
                            output_path
                            / category_name
                            / _sanitize_path_component(file_path.stem)
                            / _sanitize_path_component(formation_display_name)
                            / f"{_sanitize_path_component(file_stem)}_{category_name}.json"
                        )

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
                            analysis_summary_override=category_result.to_dict(),
                            derived_logs_override=derived_logs,
                            include_derived_log_status=True,
                        )

                        results.append(
                            ExampleFormatExportResult(
                                source_file=str(file_path),
                                output_file=str(json_file),
                                success=not errors,
                                json_file=str(json_file),
                                curves_exported=0,
                                warnings=[*warnings, *fallback_warnings],
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

    try:
        write_reference_curve_name_mapping(output_path)
    except OSError:
        # Parallel category-generation batches may try to write the same static
        # mapping file at the same time. The JSON packages are already written;
        # the caller writes the mapping again after all batches finish.
        pass
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
