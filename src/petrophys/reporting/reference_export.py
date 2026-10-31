"""
Export helpers for writing delivery LAS files and paired JSON summaries.
"""

from __future__ import annotations


from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import functools
import hashlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import re
from typing import TYPE_CHECKING, Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from petrophys.evaluation.diagnostics import (
    HydrocarbonAnalysis,
    LithologyAnalysis,
    NumericalStatisticalAnalysis,
    PermeabilityAnalysis,
    PorosityAnalysis,
    SaturationAnalysis,
)
from petrophys.interpretation.lithology.matrix_inference import (
    _recorded_code_curve,
    interpreted_lithology_code_curve,
)
from petrophys.io.las_reader import read_las
from petrophys.preprocessing.quality.value_ranges import (
    VALUE_RANGE_QC_PARAMETER,
    ValueRangeRule,
    apply_value_range_qc,
    clean_curve_values,
    value_range_qc_note,
)
from petrophys.preprocessing.standardization import standardize_curves
from petrophys._internal.analysis_helpers import CategoryAnalysisResult, build_section, get_curve, get_curve_data_depthwise
from petrophys._internal.benchmark import reproducing_benchmark
if TYPE_CHECKING:
    from petrophys.model.well import Well
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig


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
    "TEMP",
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
    "PHIE_CC",
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
    "TEMP": "Temperature",
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
    "PHIE_CC": "Core-Calibrated Porosity",
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
    "LITHOLOGY_GROUP": "Lithology Group",
    "LITHOLOGY_NAME": "Lithology Name",
    "GT_LITHOLOGY_GROUP_CODE": "Lithology Group Code",
    "LIMESTONE_INDICATOR": "Limestone Indicator",
    "DOLOMITE_INDICATOR": "Dolomite Indicator",
    "CARBONATE_INDICATOR": "Carbonate Indicator",
    "GR_CLAY_INDICATOR": "GR Clay Indicator",
    "NPHI_CLAY_INDICATOR": "NPHI Clay Indicator",
    "PEF_CLAY_INDICATOR": "PEF Clay Indicator",
    "COMPOSITE_CLAY": "Composite Clay Indicator",
    "COMPOSITE_CLAY_INDICATOR": "Composite Clay Indicator",
    "CONSISTENCY_SCORE": "Lithology Consistency Score",
    "ND_SEPARATION": "Neutron-Density Separation",
    "PHIT_GAS_CORR": "Gas-Corrected Total Porosity Proxy",
    "DS_SEPARATION": "Density-Sonic Separation",
    "PORO_QUALITY_INDEX": "Porosity Quality Index",
    "RXO_RDEP_RATIO": "Flushed-to-Deep Resistivity Ratio",
    "SEPARATION_RATIO": "Invasion Separation Ratio",
    "KH": "Flow Capacity",
    "PERM_QUALITY_INDEX": "Permeability Quality Index",
    "DATA_COMPLETENESS": "Data Completeness",
    "GR_GRAD": "Gamma Ray Gradient",
    "RHOB_GRAD": "Bulk Density Gradient",
    "RDEP_GRAD": "Deep Resistivity Gradient",
    "NPHI_GRAD": "Neutron Porosity Gradient",
    "DTC_GRAD": "Compressional Slowness Gradient",
    "CONSISTENCY_RATIO": "Log Consistency Ratio",
    "ACOUSTIC_IMPEDANCE": "Acoustic Impedance",
    "POISSON_RATIO": "Poisson Ratio",
    "YOUNG_MOD": "Young's Modulus",
    "BRITTLENESS_INDEX": "Brittleness Index",
    "ELECTROFACIES": "Electrofacies",
    "S_V": "Overburden Stress",
    "PP_HYDROSTATIC": "Hydrostatic Pore Pressure",
    "SH_MIN": "Minimum Horizontal Stress",
    "PRIME_GAS_TARGET": "Prime Gas Target Flag",
    "PRIME_OIL_TARGET": "Prime Oil Target Flag",
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
    "COMPOSITE_CLAY_INDICATOR",
    "CONSISTENCY_SCORE",
    "CROSSOVER_FLAG",
    "GT_LITHOLOGY_GROUP_CODE",
    "LITHOLOGY_GROUP",
    "LITHOLOGY_NAME",
    "LIMESTONE_INDICATOR",
    "DOLOMITE_INDICATOR",
    "CARBONATE_INDICATOR",
    "GR_CLAY_INDICATOR",
    "NPHI_CLAY_INDICATOR",
    "PEF_CLAY_INDICATOR",
    "ND_SEPARATION",
    "PHIT_GAS_CORR",
    "DS_SEPARATION",
    "PORO_QUALITY_INDEX",
    "RXO_RDEP_RATIO",
    "SEPARATION_RATIO",
    "KH",
    "PERM_QUALITY_INDEX",
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
    "NET_RES",
    "NET_ROCK",
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

CONTEXT_NOTE_OMIT_CURVES = {"SAL", "SALAPP", "SALMUDAPP"}

DUPLICATE_RAW_CURVES = {
    "DCAL",
    "DT",
}

MISSING_LITHOLOGY_LABEL = "NO_LITHOLOGY_DATA"

REFERENCE_DERIVED_LOG_EXCLUDE = {
    "BADHOLEFLAG",
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
    "TEMP",
    "BAD_HOLE",
    "VCL",
    "RMF",
    "RW",
    "PHIN",
    "PHID",
    "PHIS",
    "PHIT",
    "PHIE",
    "PHI_SEC",
    "PHIE_CC",
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
]

CORE_ONLY_CONTEXT_CURVES = {
    "CPERM",
    "CPOR",
    "CSW",
    "GDEN",
    "KH_CORE",
    "KV_CORE",
    "PERM_CORE",
    "PHIC",
    "SOC",
    "SWC",
}

REFERENCE_CURVE_TO_STEP = {
    "TEMP": "temperature",
    "BAD_HOLE": "bad_hole",
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
    "PERM_CC": "core_correction",
    "PHIE_CC": "core_correction",
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
    "TEMP",
    "RMF",
    "RW",
    "LITHOLOGY_GEOLINK",
]

# Ordered, not sets: these feed ``_derived_status_target_list``, so iteration
# order reaches ``derived_log_computation_notes`` verbatim. Set literals made
# that order depend on PYTHONHASHSEED, which changed every note block between
# runs. Curves already in REFERENCE_DERIVED_LOG_TARGETS keep their position
# from that list; the trailing entries here are the ones whose order is only
# defined by this tuple, so keep them where they are.
# Value-range rules applied to the context statistics (lower RMF bound 0.01;
# no SP, SO or NMR rules). The Danish and COSTA packages were written with the
# later rules of _DATASET_CONTEXT_VALUE_RANGE_RULES.
_CONTEXT_VALUE_RANGE_RULES = {
    "RMF": ValueRangeRule(0.01, 10.0, "nan", ('RMF',)),
    "TEMP": ValueRangeRule(-5.0, 400.0, "nan", ('TEMP', 'TMPC')),
    "RXO": ValueRangeRule(0.01, 2000.0, "nan", ('RXO', 'RSHA', 'RESS', 'MSFL', 'SFL')),
    "RDEP": ValueRangeRule(0.01, 2000.0, "nan", ('RDEP', 'RT', 'RMED', 'RESM', 'ILD')),
    "GR": ValueRangeRule(0.0, 400.0, "nan", ('GR', 'GRC')),
    "DTC": ValueRangeRule(20.0, 300.0, "nan", ('DTC', 'DT')),
    "DTS": ValueRangeRule(20.0, 600.0, "nan", ('DTS', 'DTE')),
    "RHOB": ValueRangeRule(1.0, 3.5, "nan", ('RHOB',)),
    "NPHI": ValueRangeRule(-0.15, 0.6, "nan", ('NPHI',)),
    "PEF": ValueRangeRule(0.0, 15.0, "nan", ('PEF',)),
    "CALI": ValueRangeRule(4.0, 40.0, "nan", ('CALI', 'CAL', 'DCAL')),
    "BS": ValueRangeRule(4.0, 40.0, "nan", ('BS',)),
    "PHIS": ValueRangeRule(-0.05, 0.6, "nan", ('PHIS',)),
    "PERM": ValueRangeRule(0.0001, 20000.0, "nan", ('PERM',)),
    "PERM_CC": ValueRangeRule(0.0001, 20000.0, "nan", ('PERM_CC',)),
    "PHID": ValueRangeRule(-0.05, 0.6, "nan", ('PHID',)),
    "PHIN": ValueRangeRule(-0.15, 0.6, "nan", ('PHIN',)),
    "PHIND": ValueRangeRule(-0.05, 0.6, "nan", ('PHIND',)),
    "PHIRH": ValueRangeRule(-0.05, 0.6, "nan", ('PHIRH',)),
    "PHIT": ValueRangeRule(0.0, 0.45, "nan", ('PHIT',)),
    "PHIE": ValueRangeRule(0.0, 0.45, "nan", ('PHIE',)),
    "PHIE_CC": ValueRangeRule(0.0, 0.45, "nan", ('PHIE_CC',)),
    "PHI_SEC": ValueRangeRule(-0.05, 0.6, "nan", ('PHI_SEC',)),
    "RESFLAG": ValueRangeRule(0.0, 1.0, "nan", ('RESFLAG', 'NET_ROCK', 'NET_RES', 'NET_PAY', 'SEALFLAG', 'CROSSOVER_FLAG')),
    "RW": ValueRangeRule(0.005, 20.0, "nan", ('RW',)),
    "RWA": ValueRangeRule(0.005, 20.0, "nan", ('RWA',)),
    "SW": ValueRangeRule(0.0, 1.0, "clip", ('SW', 'SWA', 'SWT-LOG')),
    "SXO": ValueRangeRule(0.0, 1.0, "clip", ('SXO', 'SWXO', 'SWXOA', 'SXO-LOG')),
    "SWIRR_APP": ValueRangeRule(0.0, 1.0, "nan", ('SWIRR_APP', 'SWIRR', 'SWI', 'SWIR')),
    "BVW": ValueRangeRule(0.0, 1.0, "nan", ('BVW',)),
    "MHI": ValueRangeRule(0.0, 1.0, "nan", ('MHI',)),
    "VDOL": ValueRangeRule(0.0, 1.0, "nan", ('VDOL', 'VDOLO')),
    "VLIM": ValueRangeRule(0.0, 1.0, "nan", ('VLIM', 'VCALC')),
    "VSAND": ValueRangeRule(0.0, 1.0, "nan", ('VSAND', 'VQUAR')),
    "VANHY": ValueRangeRule(0.0, 1.0, "nan", ('VANHY',)),
    "VSH": ValueRangeRule(0.0, 1.0, "nan", ('VSH', 'VCL', 'VCLAY', 'VSHGR', 'VSHND', 'VCL_GR', 'VCL_SP', 'VCL_ND')),
    "VP_VS_RATIO": ValueRangeRule(0.5, 5.0, "nan", ('VP_VS_RATIO',)),
    "POISSON_RATIO": ValueRangeRule(0.0, 0.49, "nan", ('POISSON_RATIO',)),
    "BRITTLENESS_INDEX": ValueRangeRule(0.0, 1.0, "nan", ('BRITTLENESS_INDEX',)),
    "ACOUSTIC_IMPEDANCE": ValueRangeRule(1000000.0, 30000000.0, "nan", ('ACOUSTIC_IMPEDANCE',)),
    "YOUNG_MOD": ValueRangeRule(0.0, 150.0, "nan", ('YOUNG_MOD',)),
    "S_V": ValueRangeRule(0.0, 250.0, "nan", ('S_V',)),
    "PP_HYDROSTATIC": ValueRangeRule(0.0, 150.0, "nan", ('PP_HYDROSTATIC',)),
    "SH_MIN": ValueRangeRule(0.0, 250.0, "nan", ('SH_MIN',)),
    "DATA_COMPLETENESS": ValueRangeRule(0.0, 1.0, "nan", ('DATA_COMPLETENESS',)),
    "CONSISTENCY_RATIO": ValueRangeRule(0.0, 1.0, "nan", ('CONSISTENCY_RATIO',)),
}


# Target order reaches the benchmark's computed/missing summaries and compact
# notes. Keep the published order explicit so JSON is reproducible across
# Python hash seeds.
CATEGORY_CONTEXT_SPECS = {
    "hydrocarbon": {
        "analyzer": HydrocarbonAnalysis,
        "targets": (
            "TEMP",
            "RMF",
            "RW",
            "RWA",
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
            "CROSSOVER_FLAG",
            "VP_VS_RATIO",
            "RESFLAG",
            "SEALFLAG",
            "PRIME_GAS_TARGET",
            "PRIME_OIL_TARGET",
        ),
    },
    "saturation": {
        "analyzer": SaturationAnalysis,
        "targets": (
            "TEMP",
            "RMF",
            "RW",
            "PHID",
            "PHIN",
            "PHIS",
            "PHIT",
            "PHIE",
            "SW",
            "SXO",
            "BVW",
            "MHI",
            "PHIT_GAS_CORR",
            "ND_SEPARATION",
            "RWA",
        ),
    },
    "permeability": {
        "analyzer": PermeabilityAnalysis,
        "targets": (
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
        ),
    },
    "porosity": {
        "analyzer": PorosityAnalysis,
        "targets": (
            "PHID",
            "PHIN",
            "PHIS",
            "PHIT",
            "PHIE",
            "PHIE_CC",
            "PHI_SEC",
            "PORO_QUALITY_INDEX",
            "DS_SEPARATION",
            "RQI",
            "FZI",
            "VCL",
        ),
    },
    "numerical": {
        "analyzer": NumericalStatisticalAnalysis,
        "targets": (
            "TEMP",
            "RW",
            "RMF",
            "BAD_HOLE",
            "SH_MIN",
            "ELECTROFACIES",
            "ACOUSTIC_IMPEDANCE",
            "DATA_COMPLETENESS",
            "NPHI_GRAD",
            "YOUNG_MOD",
            "BRITTLENESS_INDEX",
            "PP_HYDROSTATIC",
            "RHOB_GRAD",
            "S_V",
            "RDEP_GRAD",
            "CONSISTENCY_RATIO",
            "GR_GRAD",
            "POISSON_RATIO",
            "DTC_GRAD",
            "NET_PAY",
        ),
    },
    "lithology": {
        "analyzer": LithologyAnalysis,
        "targets": (
            "LITHO",
            "VCL",
            "VQUAR",
            "VCALC",
            "VDOLO",
            "VANHY",
            "COMPOSITE_CLAY_INDICATOR",
            "LITHOLOGY_GROUP",
            "DOLOMITE_INDICATOR",
            "GT_LITHOLOGY_GROUP_CODE",
            "LIMESTONE_INDICATOR",
            "LITHOLOGY_NAME",
            "CARBONATE_INDICATOR",
            "GR_CLAY_INDICATOR",
            "NPHI_CLAY_INDICATOR",
            "PEF_CLAY_INDICATOR",
            "COMPOSITE_CLAY",
            "CONSISTENCY_SCORE",
        ),
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
        {
            "current_export_name": "GT_LITHOLOGY_GROUP_CODE",
            "previous_example_name": "LITHOLOGY_GROUP",
            "notes": "Lithology group is exported as a numeric grouped-code curve.",
        },
        {
            "current_export_name": "COMPOSITE_CLAY",
            "previous_example_name": "COMPOSITE_CLAY_INDICATOR",
            "notes": "Composite clay indicator is exported as COMPOSITE_CLAY.",
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
    """Outcome of exporting one source file in a batch export.

    The batch exporters in :mod:`petrophys.reporting` return one per file
    written (one per formation or lithology group for the split exporters);
    :func:`write_example_format_report` tabulates them.

    Attributes
    ----------
    source_file : str
        The LAS file that was read.
    output_file : str
        The LAS file written, or the intended path (or ``""``) when the
        export failed.
    success : bool
        True when the export finished without workflow errors.
    json_file : str
        The JSON analysis file written next to the LAS file, if any.
    curves_exported : int
        Number of curves written to the LAS file, excluding depth (0 when the
        result is a JSON context only).
    warnings, errors : list of str
        Workflow warnings and errors collected for this file.
    """
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
    """Convert a temperature array to Celsius; unlabelled data is assumed Celsius."""
    from petrophys.model.units import temperature_unit

    text = (unit or "").strip().lower()
    numeric = _normalize_numeric_array(data)
    if not reproducing_benchmark():
        # Whole-word unit parsing: "Celsius (ref)" is Celsius, "K" is Kelvin.
        scale = temperature_unit(unit)
        if scale == "F":
            return (numeric - 32.0) * 5.0 / 9.0
        if scale == "K":
            return numeric - 273.15
        return numeric
    if "f" in text and "c" not in text:
        return (numeric - 32.0) * 5.0 / 9.0
    return numeric


def _convert_temperature_to_fahrenheit(data: np.ndarray, unit: str) -> np.ndarray:
    """Convert a temperature array to Fahrenheit; unlabelled data is assumed Fahrenheit."""
    from petrophys.model.units import temperature_unit

    text = (unit or "").strip().lower()
    numeric = _normalize_numeric_array(data)
    if not reproducing_benchmark():
        # Whole-word unit parsing: "Celsius (ref)" is Celsius, "K" is Kelvin.
        scale = temperature_unit(unit)
        if scale == "C":
            return (numeric * 9.0 / 5.0) + 32.0
        if scale == "K":
            return numeric * 9.0 / 5.0 - 459.67
        return numeric
    if "c" in text and "f" not in text:
        return (numeric * 9.0 / 5.0) + 32.0
    return numeric


def _fraction_data(curve) -> np.ndarray:
    """A volume or saturation curve in v/v; percent-labelled data is divided by 100.

    The benchmark context profiles keep the packages' original calculation.
    """
    numeric = _normalize_numeric_array(curve.data)
    unit = str(curve.unit or "").strip().lower().replace(" ", "")
    if not reproducing_benchmark() and unit in {"%", "pu", "p.u.", "percent"}:
        return numeric / 100.0
    return numeric


def _first_curve(well: "Well", candidates: Sequence[str], require_finite: bool = True):
    return get_curve(well, candidates, require_finite=require_finite)


def _first_sxo_curve(well: "Well"):
    direct = _first_curve(well, ["SXO", "SWXOA"])
    if direct is not None:
        return direct
    for mnemonic in well.curves:
        if mnemonic.upper().startswith("SXO-"):
            curve = _first_curve(well, [mnemonic])
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
                data=_fraction_data(sw),
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
                data=_fraction_data(sxo),
                source_curves=[sxo.mnemonic],
            )
        )

    temp = _first_curve(well, ["TEMP", "TMPC"])
    if temp is not None:
        specs.append(
            ExportCurveSpec(
                mnemonic="TEMP",
                unit="degF",
                description="Temperature",
                family=FAMILY_LABELS["TEMP"],
                data=_convert_temperature_to_fahrenheit(temp.data, temp.unit),
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
                data=_fraction_data(vcl),
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
                data=_fraction_data(phi_norm),
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
        if not reproducing_benchmark():
            # Six decimals write values below 5e-7 as 0 (small PERM) and keep
            # few digits of small values; write those to 6 significant digits.
            if numeric == 0.0:
                return "0"
            if abs(numeric) < 0.1:
                return f"{numeric:.6g}"
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
    """Write a well to LAS 2.0 in the example delivery format.

    The file has the delivery header blocks (well information, a parameter
    block with a ``TLFamily_<curve>`` log-family entry per curve, curve
    information) and one row per depth sample, with nulls written as
    ``-9999``. Curves are written in the standard delivery order; internal
    flags and duplicate raw curves are left out.

    Parameters
    ----------
    well : Well
        The well to write, usually an interpreted one.
    filepath : str or Path
        Output LAS path. Missing parent folders are created.
    set_name : str, optional
        Value of the ``SET`` parameter; defaults to
        ``"<well name> Petrophys Export"``.
    original_curves : set of str, optional
        Names of the curves read from the input file. Curves outside the
        standard delivery list are written only if they are in this set, so
        working curves added during interpretation are left out. Defaults to
        every curve of ``well``.

    Returns
    -------
    Path
        The file written.

    Examples
    --------
    >>> import petrophys as pp
    >>> from petrophys.reporting import export_example_format_las
    >>> result = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07, rw_meas_temp_f=230)
    >>> export_example_format_las(result.well, "delivery/30_6-5.las")
    PosixPath('delivery/30_6-5.las')
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    curve_specs = _collect_export_curves(well, original_curves=original_curves)
    date_value = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    set_value = set_name or f"{well.name} Petrophys Export"
    start, stop, step = well.top, well.bottom, well.step
    finite_depth = _normalize_numeric_array(well.depth) if well.depth is not None else np.array([])
    finite_depth = finite_depth[np.isfinite(finite_depth)]
    if not reproducing_benchmark() and finite_depth.size > 1 and finite_depth[0] > finite_depth[-1]:
        # LAS 2.0: STRT is the first depth row, STOP the last, STEP negative
        # for decreasing depth.
        start, stop = float(finite_depth[0]), float(finite_depth[-1])
        step = -step if step is not None else None

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
        handle.write(_format_header_line("STRT", well.depth_unit, _format_value(start), "First reference value") + "\n")
        handle.write(_format_header_line("STOP", well.depth_unit, _format_value(stop), "Last reference value") + "\n")
        handle.write(_format_header_line("STEP", well.depth_unit, _format_value(step), "Step increment") + "\n")
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

    curve = well.get("BAD_HOLE")
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
    if details.get("no_bit_size_points"):
        calculation_thresholds += (
            f" Where no bit size is known ({details['no_bit_size_points']} samples), WASHOUT is "
            f"|DRHO| > {thresholds['drho_washout_threshold_g_per_cc']:.2f} g/cc alone."
        )
    summary = {
        "bad_hole_percentage": details["bad_hole_percentage"],
        "counts": {
            "washout": details["counts"]["washout"],
            "large_hole": details["counts"]["large_hole"],
            "bad_hole": details["counts"]["bad_hole"],
            "total_points": details["counts"]["total_points"],
        },
        "bit_size_used_in": _json_safe(details["bit_size_used"]),
    }
    for key in ("no_bit_size_points", "not_assessed_points"):
        if key in details:
            summary[key] = details[key]

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
        metadata={"workflow_bad_hole_summary": summary},
    )


def _workflow_flag_analysis(well: "Well") -> Optional[CategoryAnalysisResult]:
    sections = {}

    reservoir = _binary_flag_section(
        well,
        ["RESFLAG", "NET_RES"],
        "reservoir_flag_screening",
        "Reservoir screening from the reservoir flag.",
        "Uses the reservoir flag derived from VCL and effective-porosity cutoffs.",
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
        "Uses crossover flag where PHID - PHIN > 0.05, VCL < 0.35, selected effective porosity > 0.05, and BAD_HOLE < 0.5.",
        "CROSSOVER_FLAG when PHID - PHIN > 0.05, VCL < 0.35, selected effective porosity > 0.05, and BAD_HOLE < 0.5; otherwise NO_CROSSOVER.",
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
    if upper == "TMPC":
        return "TEMP"
    if upper == "RSHAL" and _context_dataset() is not None:
        return "RSHA"
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


def _input_log_display_list(curves: set[str]) -> List[str]:
    curve_names = {str(name).upper() for name in curves}
    display_names: list[str] = []
    for name in curves:
        upper = str(name).upper()
        canonical = _canonical_delivery_curve_name(upper)
        if canonical in CORE_ONLY_CONTEXT_CURVES:
            continue
        if upper == "DCAL" and "CALI" in curve_names:
            continue
        if upper == "DT" and "DTC" in curve_names:
            continue
        display_names.append(_display_curve_name(name))
    return sorted(_ordered_unique(display_names))


def _target_curve_candidates(curve: str) -> List[str]:
    mapping = {
        "TEMP": ["TEMP", "TMPC"],
        "PHI_NORM": ["PHI_NORM", "PHI_Z"],
        "SW": ["SW", "SWA"],
        "SXO": ["SXO", "SWXOA"],
        "VCL": ["VCL", "VCLAY", "VSH"],
    }
    return mapping.get(curve, [curve])


def _has_finite_any_curve(well: "Well", candidates: Sequence[str]) -> bool:
    return any(_has_finite_curve(well, name) for name in candidates)


def _has_any_curve(well: "Well", candidates: Sequence[str]) -> bool:
    return any(well.get(name) is not None for name in candidates)


def _finite_input_status_text(well: "Well", specs: Sequence[tuple[str, Sequence[str]]]) -> str:
    parts: list[str] = []
    for label, candidates in specs:
        later = _context_dataset() is not None
        if _has_finite_any_curve(well, candidates):
            status = "present"
        elif _has_any_curve(well, candidates):
            status = "curve present but no finite samples in this segment" if later else "present/no finite samples"
        else:
            status = "missing in this segment" if later else "missing/no finite samples"
        parts.append(f"{label}={status}")
    return ", ".join(parts)


def _depthwise_curve_data_for_export(
    well: "Well",
    candidates: Sequence[str],
) -> Optional[np.ndarray]:
    """Return a depth-wise finite-first alias composite for export notes."""
    n_samples = int(getattr(well, "n_samples", 0) or 0)
    result = np.full(n_samples, np.nan, dtype=float)
    for name in candidates:
        curve = well.get(name)
        if curve is None:
            continue
        try:
            data = _normalize_numeric_array(curve.data)
        except Exception:
            continue
        if data.size != n_samples:
            continue
        data = data.reshape(n_samples)
        fill = ~np.isfinite(result) & np.isfinite(data)
        result[fill] = data[fill]
    return result if np.any(np.isfinite(result)) else None


def _segment_sxo_method_counts(well: "Well") -> dict[str, int]:
    """Estimate segment-local SXO method counts from finite inputs."""
    sxo = _depthwise_curve_data_for_export(well, ["SXO", "SWXOA", "SXO-LOG"])
    sw = _depthwise_curve_data_for_export(well, ["SW", "SWA", "SWT-LOG"])
    if sxo is None or sw is None:
        return {}
    finite_sxo = np.isfinite(sxo) & np.isfinite(sw)
    if not np.any(finite_sxo):
        return {}
    rxo = _depthwise_curve_data_for_export(well, ["RSHA", "RXO", "RESS"])
    rmf = _depthwise_curve_data_for_export(well, ["RMF"])
    phie = _depthwise_curve_data_for_export(well, ["PHIE_CC", "PHIE", "PHIT"])
    if rxo is not None and rmf is not None and phie is not None:
        archie_mask = (
            finite_sxo
            & np.isfinite(rxo)
            & np.isfinite(rmf)
            & np.isfinite(phie)
            & (rxo > 0.01)
            & (rmf > 0.0001)
            & (phie > 0.001)
        )
    else:
        archie_mask = np.zeros_like(finite_sxo, dtype=bool)
    fallback_mask = finite_sxo & ~archie_mask
    counts: dict[str, int] = {}
    if np.any(archie_mask):
        counts["archie_rxo_rmf"] = int(np.count_nonzero(archie_mask))
    if np.any(fallback_mask):
        counts["fifth_root_quicklook"] = int(np.count_nonzero(fallback_mask))
    return counts


def _segment_diagnostic_sxo_count(well: "Well") -> int:
    """Return segment-local SXO samples backed by invaded-zone inputs."""
    return int(_segment_sxo_method_counts(well).get("archie_rxo_rmf", 0))


def _phit_source_counts_text(params: Mapping[str, Any]) -> str:
    counts = params.get("phit_source_counts") or {}
    if not isinstance(counts, Mapping):
        return ""
    parts = []
    for key in (
        "density_from_missing_phin",
        "density_from_gas_crossover",
        "rms_from_gas_crossover",
        "weighted_nd",
        "neutron_from_missing_phid",
        "density_only",
        "neutron_only",
        "sonic_only",
        "existing_phit",
        "nan",
    ):
        if key in counts:
            parts.append(f"{key}={counts[key]}")
    return "; counts: " + ", ".join(parts) if parts else ""


def _phit_gas_rms(params: Mapping[str, Any]) -> bool:
    """True when gas-crossover PHIT is the root-mean-square of PHID and PHIN."""
    counts = params.get("phit_source_counts") or {}
    return isinstance(counts, Mapping) and "rms_from_gas_crossover" in counts


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


def _workflow_output_curve_written(workflow_result, curve_name: str) -> bool:
    target = curve_name.upper()
    for curve in _workflow_output_curves(workflow_result):
        if str(curve).upper() == target:
            return True
    return False


def _derived_logs_computed(
    well: "Well",
    workflow_result,
    original_curves: set[str],
) -> List[str]:
    if _reproduces_benchmark_context(workflow_result):
        derived = {
            _canonical_delivery_curve_name(curve)
            for curve in _workflow_output_curves(workflow_result)
        }
        derived.update(
            _canonical_delivery_curve_name(curve)
            for curve in (set(well.curves) - set(original_curves))
        )
    else:
        # The normalization step lists every curve of the well as output: the
        # raw logs it renamed or standardized are not derived logs.
        standardized: set[str] = set()
        derived = set()
        for step in _all_step_results(workflow_result):
            outputs = step.output_curves or []
            if step.step == "normalization":
                standardized.update(str(curve) for curve in outputs)
            else:
                derived.update(_canonical_delivery_curve_name(curve) for curve in outputs)
        derived.update(
            _canonical_delivery_curve_name(curve)
            for curve in (set(well.curves) - set(original_curves) - standardized)
        )

    # Keep the reference-style list on canonical mnemonics rather than
    # method-specific aliases that are internal compatibility artifacts.
    derived.difference_update(REFERENCE_DERIVED_LOG_EXCLUDE)
    derived.difference_update(CORE_ONLY_CONTEXT_CURVES)
    derived.discard("TMPC")
    derived.discard("PHI_Z")
    derived.difference_update({"SWA", "SWXOA", "VCLAY", "VSH", "VCLGR", "VCLSP", "VCLND", "VCL_GR", "VCL_SP", "VCL_ND"})
    if (
        interpreted_lithology_code_curve(well)
        in {_canonical_delivery_curve_name(curve) for curve in original_curves}
        and not _workflow_output_curve_written(workflow_result, "LITHO")
    ):
        derived.discard("LITHO")

    finite_derived = {
        curve
        for curve in derived
        if _has_finite_any_curve(well, _target_curve_candidates(curve))
    }

    return sorted(finite_derived)


def _context_class_tables(workflow_result) -> str:
    """The diagnostics analyzers' class tables for a context.

    ``"geolink"`` (the eval package's tables) when the workflow behind the
    context ran the "geolink" profile, else ``"current"``, the canonical tables
    of :data:`~petrophys.model.class_tables.CLASS_TABLES`.
    """
    if workflow_result is not None and _reproduces_benchmark_context(workflow_result):
        return "geolink"
    return "current"


#: The category analyzers that grade reservoir quality, with the limits of each
#: sample's lithology family under the "current" class tables.
_RESERVOIR_QUALITY_ANALYZERS = (PorosityAnalysis, PermeabilityAnalysis, HydrocarbonAnalysis, LithologyAnalysis)


def _run_lithology_families(well: "Well", workflow_result) -> np.ndarray:
    """The lithology family of the run that interpreted each sample.

    A zone's re-run inside its zone (as the zoned runner merges it), the
    whole-well run elsewhere; clastic without a workflow. Net pay gives this
    family to a sample without a lithology group.
    """
    from petrophys.evaluation.net_pay.cutoffs import cutoff_family

    def family(run) -> str:
        lithology = getattr(getattr(run, "config", None), "lithology", None)
        return cutoff_family(lithology) if lithology else "clastic"

    base = getattr(workflow_result, "base_result", workflow_result)
    families = np.full(well.n_samples, family(base), dtype=object)
    depth = getattr(well, "depth", None)
    if depth is not None and len(depth) == well.n_samples:
        depth = np.asarray(depth, dtype=float)
        zones = getattr(workflow_result, "zones", None) or []
        for zone, zone_result in zip(zones, getattr(workflow_result, "zone_results", None) or []):
            families[(depth >= zone.top) & (depth <= zone.bottom)] = family(zone_result)
    return families


def _context_lithology_families(
    well: "Well",
    workflow_result,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> np.ndarray:
    """Each sample's lithology family for the reservoir-quality table of a context.

    The family of the sample's lithology, the label a lithology group is named
    after (:func:`_build_lithology_group_segments`), so every sample of a
    group takes the family of the group's lithology; a sample without one
    takes the family of the run that interpreted it
    (:func:`_run_lithology_families`).
    """
    from petrophys.evaluation.diagnostics._tables import lithology_family

    families = _run_lithology_families(well, workflow_result)
    curve = _context_lithology_curve(well)
    if curve is None:
        return families
    mapping = _run_lithology_mapping(lithology_mapping, workflow_result)
    resolved: dict[str, Optional[str]] = {}
    for index, value in enumerate(np.asarray(curve.data, dtype=object)):
        key = str(value)
        if key not in resolved:
            label = _normalize_lithology_label(value, lithology_mapping=mapping)
            resolved[key] = lithology_family(label) if label else None
        if resolved[key] is not None:
            families[index] = resolved[key]
    return families


def _category_analysis_result(
    category: str,
    well: "Well",
    workflow_result=None,
    lithology_family=None,
) -> CategoryAnalysisResult:
    """One category's analysis of a context well.

    Under the "current" class tables the analyzers that grade reservoir quality
    take ``lithology_family`` (one family per sample), by default each sample's
    from :func:`_context_lithology_families`.
    """
    spec = CATEGORY_CONTEXT_SPECS[category]
    options: dict[str, Any] = {}
    if spec["analyzer"] is not NumericalStatisticalAnalysis:
        options["class_tables"] = _context_class_tables(workflow_result)
    if spec["analyzer"] in _RESERVOIR_QUALITY_ANALYZERS and options["class_tables"] == "current":
        options["lithology_family"] = (
            _context_lithology_families(well, workflow_result) if lithology_family is None else lithology_family
        )
    if (
        spec["analyzer"] is LithologyAnalysis
        and workflow_result is not None
        and not _reproduces_benchmark_context(workflow_result)
    ):
        # Group the lithology codes as the "current" calculations do.
        from petrophys.evaluation.diagnostics.lithology_analysis import CURRENT_GROUND_TRUTH_GROUPS

        options["ground_truth_groups"] = CURRENT_GROUND_TRUTH_GROUPS
    analyzer = spec["analyzer"](**options)
    return analyzer.analyze(well, add_curves=False)


def _category_derived_logs_computed(
    category: str,
    well: "Well",
    workflow_result,
    original_curves: set[str],
    category_result: CategoryAnalysisResult,
) -> List[str]:
    computed = set(_derived_logs_computed(well, workflow_result, original_curves))
    original_canonical = {
        _canonical_delivery_curve_name(curve)
        for curve in original_curves
    }

    for name, curve in category_result.derived_curves.items():
        canonical = _canonical_delivery_curve_name(name)
        if canonical in original_canonical:
            continue
        if canonical in CORE_ONLY_CONTEXT_CURVES or canonical in REFERENCE_DERIVED_LOG_EXCLUDE:
            continue
        if canonical in {"PHI_Z", "LITHOLOGY_GEOLINK"}:
            continue
        if curve is not None and np.any(np.isfinite(_normalize_numeric_array(curve.data))):
            computed.add(canonical)

    if category == "lithology":
        if "GT_LITHOLOGY_GROUP_CODE" in computed:
            computed.update({"LITHOLOGY_GROUP", "LITHOLOGY_NAME"})
        if "COMPOSITE_CLAY" in computed:
            computed.add("COMPOSITE_CLAY_INDICATOR")

    return sorted(computed)


def _step_lookup(workflow_result) -> tuple[dict[str, Any], set[str]]:
    step_results: dict[str, Any] = {}
    skipped_steps: set[str] = set()
    for run in [workflow_result.base_result, *workflow_result.zone_results]:
        skipped_steps.update(run.steps_skipped)
        for step in run.steps_executed:
            if step.step not in step_results or step.success:
                step_results[step.step] = step
    return step_results, skipped_steps


def _dataset_sxo_missing_reason(well: "Well") -> str:
    """Why SXO is missing from a segment, worded as the Danish and COSTA packages."""
    shallow_present = _has_finite_any_curve(well, ("RSHA", "RXO", "RESS"))
    rmf_present = _has_finite_curve(well, "RMF")
    phie_present = _has_finite_any_curve(well, ("PHIE_CC", "PHIE"))
    sw_present = _has_finite_any_curve(well, ("SW", "SWA", "SWT-LOG"))

    direct_missing: list[str] = []
    if not shallow_present:
        if _has_any_curve(well, ("RSHA", "RXO", "RESS")):
            direct_missing.append("RSHA/RXO/RESS curve present but no finite samples in this segment")
        else:
            direct_missing.append("RSHA/RXO/RESS missing in this segment")
    if not rmf_present:
        if _has_any_curve(well, ("RMF",)):
            direct_missing.append("RMF curve present but no finite samples in this segment")
        else:
            direct_missing.append("RMF missing in this segment")
    if not phie_present:
        if _has_any_curve(well, ("PHIE_CC", "PHIE")):
            direct_missing.append("PHIE_CC/PHIE curve present but no finite samples in this segment")
        else:
            direct_missing.append("PHIE_CC/PHIE missing in this segment")

    if direct_missing:
        if sw_present:
            fallback_reason = "fifth-root fallback input SW was present, but SXO was not written"
        elif _has_any_curve(well, ("SW", "SWA", "SWT-LOG")):
            fallback_reason = "fifth-root fallback also unavailable because SW had no finite samples in this segment"
        else:
            fallback_reason = "fifth-root fallback also unavailable because SW was missing in this segment"
        return "direct invaded-zone inputs unavailable: " + "; ".join(direct_missing) + "; " + fallback_reason

    return "direct invaded-zone inputs were present, but no finite SXO values were produced after screening"


def _dataset_sw_missing_reason(well: "Well") -> str:
    """Why SW is missing from a segment, worded as the Danish and COSTA packages."""
    missing_inputs: list[str] = []
    nonfinite_inputs: list[str] = []
    for label, candidates in (
        ("RDEP or RT", ("RDEP", "RT")),
        ("selected effective porosity", ("PHIE_CC", "PHIE")),
        ("RW", ("RW",)),
    ):
        if _has_finite_any_curve(well, candidates):
            continue
        if _has_any_curve(well, candidates):
            nonfinite_inputs.append(label)
        else:
            missing_inputs.append(label)
    parts = []
    if missing_inputs:
        parts.append("missing " + ", ".join(missing_inputs))
    if nonfinite_inputs:
        parts.append("present but no finite samples in this segment: " + ", ".join(nonfinite_inputs))
    if parts:
        return "; ".join(parts)
    return "no finite SW values were produced in this segment after saturation routing"


def _primary_log_present(
    well: "Well",
    workflow_result,
    name: str,
    original_curves: set[str],
) -> bool:
    """Whether a primary log counts as present for a context's ``missing_primary_logs``.

    The GeoLink main run's context counted only the input curves. The Danish
    and COSTA packages also counted a primary log the segment holds finite
    samples of (e.g. TEMP or RW the run wrote), and the Danish package counted
    BS present when the bad-hole step used a bit size.
    """
    if _context_dataset() is None:
        return name in original_curves
    canonical = _canonical_delivery_curve_name(name)
    if canonical in {_canonical_delivery_curve_name(str(curve)) for curve in original_curves}:
        return True
    if (
        canonical == "BS"
        and _context_dataset() == "danish"
        and _bit_size_used_for_bad_hole(well, workflow_result) is not None
    ):
        return True
    return _has_finite_any_curve(well, _target_curve_candidates(canonical))


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

    def missing_required(required: dict[str, bool]) -> str:
        missing_inputs = [name for name, present in required.items() if not present]
        if missing_inputs:
            return "required input curves unavailable: " + ", ".join(missing_inputs)
        return ""

    if curve in {"SXO", "MHI"}:
        # A reason without an SXO method: the saturation step skipped SXO (oil-based mud).
        saturation_params = getattr(step_results.get("saturation"), "parameters", None) or {}
        if saturation_params.get("sxo_method") is None and saturation_params.get("sxo_reason"):
            return str(saturation_params["sxo_reason"])

    if curve == "SXO" and _context_dataset() is not None:
        return _dataset_sxo_missing_reason(well)

    if curve == "SXO":
        sw_present = _has_finite_curve(well, "SW")
        shallow_present = _has_finite_any_curve(well, ("RSHA", "RXO", "RESS"))
        rmf_present = _has_finite_curve(well, "RMF")
        if not sw_present:
            return "SW unavailable or non-finite in this segment, so SXO could not be calculated"
        if not shallow_present and not rmf_present:
            return "SXO was not written; only the fifth-root fallback inputs were available"
        if not shallow_present:
            if _has_any_curve(well, ("RSHA", "RXO", "RESS")):
                return "shallow resistivity curve was present but had no finite samples"
            return "shallow resistivity curve was absent"
        if not rmf_present:
            return "shallow resistivity was available but RMF was unavailable"

    if curve == "MHI":
        if not _has_finite_curve(well, "SW"):
            return "SW unavailable or non-finite in this segment"
        if not _has_finite_curve(well, "SXO"):
            return "SXO unavailable or non-finite in this segment"
        counts = _segment_sxo_method_counts(well)
        if counts.get("archie_rxo_rmf", 0) == 0:
            if counts.get("fifth_root_quicklook", 0) > 0:
                return "SXO was computed only from SW^(1/5) fallback in this segment, so MHI was not calculated"
            return "SXO lacked diagnostic invaded-zone support from RSHA/RXO/RESS plus RMF in this segment"

    if curve == "VP_VS_RATIO":
        has_dtc = _has_finite_any_curve(well, ("DTC", "DT"))
        has_dts = _has_finite_curve(well, "DTS")
        if not has_dtc and not has_dts:
            return "DTC/DT and DTS had no finite samples in this segment"
        if not has_dtc:
            return "DTC/DT had no finite samples in this segment"
        if not has_dts:
            return "DTS had no finite samples in this segment"
        return "DTC and DTS were present but VP_VS_RATIO was not written"

    if curve == "TEMP":
        return "not calculated because TEMP was unavailable and a fallback temperature profile could not be written"

    if curve == "RMF":
        return "no RMF, RM, CLFL salinity, or RMF salinity input was available, and mud type was not provided or was unknown"

    if curve == "PHIN":
        reason = missing_required({"NPHI": _has_finite_curve(well, "NPHI")})
        return reason or "no finite PHIN values were produced in this segment"

    if curve == "PHID":
        reason = missing_required({"RHOB": _has_finite_curve(well, "RHOB")})
        return reason or "no finite PHID values were produced in this segment"

    if curve == "PHIS":
        reason = missing_required({"DTC or DT": _has_finite_any_curve(well, ("DTC", "DT"))})
        return reason or "no finite PHIS values were produced in this segment"

    if curve == "PHIT":
        reason = missing_required({
            "PHID, PHIN, PHIS, or existing PHIT": (
                _has_finite_curve(well, "PHIT")
                or _has_finite_curve(well, "PHID")
                or _has_finite_curve(well, "PHIN")
                or _has_finite_curve(well, "PHIS")
            )
        })
        return reason or "no finite PHIT values were produced in this segment"

    if curve == "PHIE":
        reason = missing_required({
            "porosity basis PHIT/PHID/PHIN/PHIS": _has_finite_any_curve(well, ("PHIT", "PHID", "PHIN", "PHIS")),
            "VCL or equivalent clay volume": _has_finite_any_curve(well, ("VCL", "VCLAY", "VSH")),
        })
        if reason:
            return reason
        if not _has_finite_curve(well, "PHIT") and _has_finite_curve(well, "PHID"):
            return "no finite PHIE values in this segment; PHID exists, but the selected PHIT/PHIE basis has no finite samples"
        return "no finite PHIE values were produced in this segment"

    if curve == "PHI_SEC":
        reason = missing_required({
            "PHIT": _has_finite_curve(well, "PHIT"),
            "PHIS": _has_finite_curve(well, "PHIS"),
        })
        return reason or "no finite PHI_SEC values were produced in this segment"

    if curve == "SW" and _context_dataset() is not None:
        return _dataset_sw_missing_reason(well)

    if curve == "SW":
        reason = missing_required({
            "RDEP or RT": _has_finite_any_curve(well, ("RDEP", "RT")),
            "selected effective porosity": _has_finite_any_curve(well, ("PHIE_CC", "PHIE")),
            "RW": _has_finite_curve(well, "RW"),
        })
        if reason:
            return reason
        return "no finite SW values were produced in this segment after saturation routing"

    if curve == "RWA":
        required = {
            "RDEP or RT": well.get("RDEP") is not None or well.get("RT") is not None,
            "PHIE or PHIT": well.get("PHIE") is not None or well.get("PHIT") is not None,
        }
        missing_inputs = [name for name, present in required.items() if not present]
        if missing_inputs:
            return "required input curves unavailable: " + ", ".join(missing_inputs)
        return "standalone apparent-Rw curve was not exported; the RW route records the RWA P5 calculation when used"

    if curve == "SAL":
        if "SAL" in original_curves or well.get("SAL") is not None:
            return "present in source data"
        return "standalone salinity curve was not exported; salinity is stored as RW-route metadata when available"

    if curve == "SALAPP":
        if well.get("RWA") is None:
            return "RWA unavailable; requires apparent Rw before apparent salinity conversion"
        return "standalone apparent-salinity curve was not exported"

    if curve == "SALMUDAPP":
        if well.get("RMFA") is None:
            return "RMFA unavailable; requires apparent mud-filtrate resistivity before mud-salinity conversion"
        return "standalone apparent mud-filtrate salinity curve was not exported"

    if curve in {"PHIE_CC", "PERM_CC"} and step is not None:
        same_well_models = step.parameters.get("same_well_core_models")
        peer_models = step.parameters.get("peer_core_models")
        if (same_well_models or 0) == 0 and (peer_models or 0) == 0:
            rejected = step.parameters.get("core_rejected_models") or []
            if rejected:
                reasons = sorted({str(model.get("reason") or "unknown") for model in rejected})
                reason_text = ", ".join(reasons)
                if "no_core_curves" in reasons:
                    if curve == "PHIE_CC":
                        return "no same-well core porosity was available for calibration"
                    return "no same-well core porosity/permeability curves were available for calibration"
                if "missing_core_porosity" in reasons or "missing_core_porosity_or_permeability" in reasons:
                    if curve == "PHIE_CC":
                        return "no usable core porosity was available; core permeability is not required for porosity calibration"
                    return "same-well core data did not contain usable paired porosity and permeability for calibration"
                if "too_few_core_points" in reasons:
                    if curve == "PHIE_CC":
                        return "same-well core data had too few paired log/core porosity samples" + _rejected_core_model_text(step.parameters)
                    return "same-well core data had too few paired core porosity/permeability samples" + _rejected_core_model_text(step.parameters)
                return f"same-well core model was not fitted; reason={reason_text}" + _rejected_core_model_text(step.parameters)
            if curve == "PHIE_CC":
                return "no usable core porosity calibration was available; requires paired log porosity and core porosity"
            return "no usable core permeability calibration was available; requires paired porosity and core permeability"

    if curve in {"PHIE_CC", "PERM_CC"}:
        if curve == "PHIE_CC":
            return "no usable core porosity calibration was available; requires paired log porosity and core porosity"
        return "no usable core permeability calibration was available; requires paired porosity and core permeability"

    if curve == "SWIRR_APP" and step is not None:
        swirr_source = step.parameters.get("swirr_source")
        if swirr_source:
            return f"Swirr source was {swirr_source}"

    if curve == "PERM":
        reason = missing_required({
            "selected effective porosity": _has_finite_any_curve(well, ("PHIE_CC", "PHIE")),
            "SWIRR_APP or SW": _has_finite_curve(well, "SWIRR_APP") or _has_finite_curve(well, "SW"),
        })
        return reason or "no finite PERM values were produced in this segment"

    if curve in {"RQI", "PHI_NORM", "FZI"}:
        reason = missing_required({
            "PERM": _has_finite_curve(well, "PERM"),
            "selected effective porosity": _has_finite_any_curve(well, ("PHIE_CC", "PHIE")),
        })
        return reason or f"no finite {curve} values were produced in this segment"

    if curve == "BVW":
        reason = missing_required({"selected effective porosity": _has_finite_any_curve(well, ("PHIE_CC", "PHIE")), "SW": _has_finite_curve(well, "SW")})
        return reason or "no finite BVW values were produced in this segment"

    if curve == "INVASION_RATIO":
        has_shallow = _has_finite_any_curve(well, ("RSHA", "RESS", "RXO"))
        has_deep = _has_finite_any_curve(well, ("RDEP", "RT"))
        has_shallow_curve = _has_any_curve(well, ("RSHA", "RESS", "RXO"))
        has_deep_curve = _has_any_curve(well, ("RDEP", "RT"))
        if not has_shallow and not has_deep:
            if has_shallow_curve or has_deep_curve:
                return "shallow or deep resistivity curves were present but had no finite samples; requires finite RSHA/RESS/RXO and RDEP/RT"
            return "shallow resistivity and deep resistivity curves were absent; requires RSHA, RESS, or RXO plus RDEP or RT"
        if not has_shallow:
            if has_shallow_curve:
                return "shallow resistivity curve was present but had no finite samples; requires finite RSHA, RESS, or RXO"
            return "shallow resistivity curve was absent; requires RSHA, RESS, or RXO"
        if not has_deep:
            if has_deep_curve:
                return "deep resistivity curve was present but had no finite samples; requires finite RDEP or RT"
            return "deep resistivity curve was absent; requires RDEP or RT"
        return "no finite INVASION_RATIO values after flushed/shallow-or-fallback/deep resistivity ratio screening"

    if curve == "PRIME_GAS_TARGET":
        required = {
            "CROSSOVER_FLAG": _has_finite_curve(well, "CROSSOVER_FLAG"),
            "VP_VS_RATIO": _has_finite_curve(well, "VP_VS_RATIO"),
            "selected effective porosity": _has_finite_any_curve(well, ("PHIE_CC", "PHIE")),
            _perm_inputs_label(): _has_finite_curve(well, "PERM") or _has_finite_curve(well, "PERM_CC"),
        }
        missing_inputs = [name for name, present in required.items() if not present]
        if missing_inputs:
            return "required input curves unavailable: " + ", ".join(missing_inputs)
        return "hydrocarbon-analysis classification only; not a dynamic-workflow curve"

    if curve == "PRIME_OIL_TARGET":
        required = {
            "NET_PAY": _has_finite_curve(well, "NET_PAY"),
            "selected effective porosity": _has_finite_any_curve(well, ("PHIE_CC", "PHIE")),
            _perm_inputs_label(): _has_finite_curve(well, "PERM") or _has_finite_curve(well, "PERM_CC"),
        }
        missing_inputs = [name for name, present in required.items() if not present]
        if missing_inputs:
            return "required input curves unavailable: " + ", ".join(missing_inputs)
        return "hydrocarbon-analysis classification only; not a dynamic-workflow curve"

    if curve == "LITHO":
        if _has_finite_curve(well, interpreted_lithology_code_curve(well)):
            return "lithology was supplied by the source lithology curve, so no derived LITHO curve was needed"
        reason = missing_required({
            "mineral volume curves VQUAR/VCALC/VDOLO/VANHY": _has_finite_any_curve(well, ("VQUAR", "VCALC", "VDOLO", "VANHY")),
        })
        return reason or "derived lithology was not written from the mineral-volume solution"

    if curve in {"VQUAR", "VCALC", "VDOLO", "VANHY"}:
        reason = missing_required({
            "VCL": _has_finite_any_curve(well, ("VCL", "VCLAY", "VSH")),
            "selected effective porosity": _has_finite_any_curve(well, ("PHIE_CC", "PHIE")),
            "RHOB": _has_finite_curve(well, "RHOB"),
            "NPHI": _has_finite_curve(well, "NPHI"),
        })
        if reason:
            return reason
        if not _has_finite_any_curve(well, ("PEF", "DTC", "DT")):
            return "PEF and DTC/DT were unavailable; mineral solver had no supplemental matrix-response constraint"
        return f"multi-mineral solver did not produce finite {curve} values in this segment"

    if curve in {"LITHOLOGY_GROUP", "LITHOLOGY_NAME", "GT_LITHOLOGY_GROUP_CODE"}:
        reason = missing_required({"lithology": _has_finite_curve(well, interpreted_lithology_code_curve(well))})
        return reason or "lithology category analysis did not write grouped lithology output"

    if curve in {"LIMESTONE_INDICATOR", "DOLOMITE_INDICATOR", "CARBONATE_INDICATOR"}:
        reason = missing_required({
            "PEF": _has_finite_curve(well, "PEF"),
            "RHOB": _has_finite_curve(well, "RHOB"),
        })
        return reason or f"lithology category analysis did not write {curve}"

    if curve == "GR_CLAY_INDICATOR":
        reason = missing_required({"GR": _has_finite_curve(well, "GR")})
        return reason or "lithology category analysis did not write GR_CLAY_INDICATOR"

    if curve == "NPHI_CLAY_INDICATOR":
        reason = missing_required({"NPHI": _has_finite_curve(well, "NPHI")})
        return reason or "lithology category analysis did not write NPHI_CLAY_INDICATOR"

    if curve == "PEF_CLAY_INDICATOR":
        reason = missing_required({"PEF": _has_finite_curve(well, "PEF")})
        return reason or "lithology category analysis did not write PEF_CLAY_INDICATOR"

    if curve in {"COMPOSITE_CLAY", "COMPOSITE_CLAY_INDICATOR"}:
        reason = missing_required({
            "GR, NPHI, or PEF": _has_finite_any_curve(well, ("GR", "NPHI", "PEF")),
        })
        return reason or "lithology category analysis did not write composite clay indicator"

    if curve == "CONSISTENCY_SCORE":
        reason = missing_required({
            "lithology": _has_finite_curve(well, interpreted_lithology_code_curve(well)),
            "GR or PEF": _has_finite_any_curve(well, ("GR", "PEF")),
        })
        return reason or "lithology category analysis did not write consistency score"

    if curve == "CROSSOVER_FLAG":
        reason = missing_required({
            "PHID": _has_finite_curve(well, "PHID"),
            "PHIN": _has_finite_curve(well, "PHIN"),
            "VCL": _has_finite_any_curve(well, ("VCL", "VCLAY", "VSH")),
            "selected effective porosity": _has_finite_any_curve(well, ("PHIE_CC", "PHIE")),
            "BAD_HOLE": _has_finite_curve(well, "BAD_HOLE"),
        })
        return reason or "no samples passed CROSSOVER_FLAG criteria: PHID-PHIN>0.05, VCL<0.35, selected effective porosity>0.05, BAD_HOLE<0.5"

    if curve == "ND_SEPARATION":
        reason = missing_required({"PHID": _has_finite_curve(well, "PHID"), "PHIN": _has_finite_curve(well, "PHIN")})
        return reason or "saturation analysis did not write ND_SEPARATION"

    if curve == "PHIT_GAS_CORR":
        reason = missing_required({"PHID": _has_finite_curve(well, "PHID"), "PHIN": _has_finite_curve(well, "PHIN")})
        return reason or "saturation analysis did not write PHIT_GAS_CORR"

    if curve == "DS_SEPARATION":
        reason = missing_required({"PHID": _has_finite_curve(well, "PHID"), "PHIS": _has_finite_curve(well, "PHIS")})
        return reason or "porosity analysis did not write DS_SEPARATION"

    if curve == "PORO_QUALITY_INDEX":
        reason = missing_required({
            "PHIT": _has_finite_curve(well, "PHIT"),
            _perm_inputs_label(): _has_finite_curve(well, "PERM") or _has_finite_curve(well, "PERM_CC"),
        })
        return reason or "porosity analysis did not write PORO_QUALITY_INDEX"

    if curve in {"RXO_RDEP_RATIO", "SEPARATION_RATIO"}:
        has_shallow = _has_finite_any_curve(well, ("RXO", "RSHA", "RESS"))
        has_deep = _has_finite_any_curve(well, ("RDEP", "RT"))
        if not has_shallow:
            if _has_any_curve(well, ("RXO", "RSHA", "RESS")):
                return "shallow resistivity curve was present but had no finite samples; requires finite RXO, RSHA, or RESS"
            return "required input curves unavailable: RXO, RSHA, or RESS"
        if not has_deep:
            if _has_any_curve(well, ("RDEP", "RT")):
                return "deep resistivity curve was present but had no finite samples; requires finite RDEP or RT"
            return "required input curves unavailable: RDEP or RT"
        return f"{curve} was not written after resistivity-ratio screening"

    if curve == "KH":
        reason = missing_required({_perm_inputs_label(): _has_finite_curve(well, "PERM") or _has_finite_curve(well, "PERM_CC")})
        return reason or "permeability analysis did not write KH"

    if curve == "PERM_QUALITY_INDEX":
        if not (
            _has_finite_curve(well, "PERM")
            or _has_finite_curve(well, "PERM_CC")
            or _has_finite_curve(well, "RQI")
            or _has_finite_curve(well, "FZI")
        ):
            return f"required input curves unavailable: {_perm_inputs_label('/')}, RQI, or FZI"
        return "permeability analysis did not write PERM_QUALITY_INDEX"

    if curve.endswith("_GRAD"):
        base = curve[:-5]
        aliases = {
            "RDEP": ("RDEP", "RT"),
            "DTC": ("DTC", "DT"),
        }.get(base, (base,))
        if not _has_finite_any_curve(well, aliases):
            return "required input curves unavailable: " + " or ".join(aliases)
        return f"{base} was available but gradient curve was not written"

    if curve == "DATA_COMPLETENESS":
        return "no finite numerical logs were available for data-completeness analysis"

    if curve == "CONSISTENCY_RATIO":
        return "no supported log pairs were available for consistency analysis"

    if curve == "ACOUSTIC_IMPEDANCE":
        reason = missing_required({"RHOB": _has_finite_curve(well, "RHOB"), "DTC or DT": _has_finite_any_curve(well, ("DTC", "DT"))})
        return reason or "no plausible DTC samples in 20-300 us/ft after screening"

    if curve in {"POISSON_RATIO", "YOUNG_MOD", "BRITTLENESS_INDEX"}:
        reason = missing_required(
            {
                "RHOB": _has_finite_curve(well, "RHOB"),
                "DTC or DT": _has_finite_any_curve(well, ("DTC", "DT")),
                "DTS": _has_finite_curve(well, "DTS"),
            }
        )
        return reason or "Vp/Vs elastic calculation had no finite plausible samples"

    if curve == "ELECTROFACIES":
        available = sum(
            int(_has_finite_any_curve(well, names))
            for names in (("RHOB",), ("NPHI", "PHIN"), ("GR",), ("RDEP", "RT"))
        )
        if available < 2:
            return "requires at least two available feature curves from RHOB, NPHI/PHIN, GR, and RDEP/RT"
        return "fewer than two valid samples remained after electrofacies screening"

    if curve == "S_V":
        reason = missing_required({"RHOB": _has_finite_curve(well, "RHOB")})
        return reason or "overburden stress calculation had no finite RHOB samples"

    if curve == "PP_HYDROSTATIC":
        return "depth array unavailable for hydrostatic pressure calculation"

    if curve == "SH_MIN":
        reason = missing_required({
            "RHOB": _has_finite_curve(well, "RHOB"),
            "DTC or DT": _has_finite_any_curve(well, ("DTC", "DT")),
        })
        return reason or "Poisson ratio, overburden stress, or hydrostatic pressure had no overlapping finite samples"

    if step_name and step_name in skipped_steps:
        return f"{step_name} step skipped"
    if step is not None and not step.success:
        return step.message or f"{step_name} step failed"

    if curve in original_curves:
        return "present in source data"

    return "not produced by the available workflow inputs"


def _computed_curve_detail(curve: str, step_results: dict[str, Any], well: Optional["Well"] = None) -> str:
    category_reasons = {
        "LITHOLOGY_GROUP": "computed by lithology category analysis from lithology codes",
        "LITHOLOGY_NAME": "computed by lithology category analysis from lithology codes",
        "GT_LITHOLOGY_GROUP_CODE": "computed by lithology category analysis from lithology codes",
        "LIMESTONE_INDICATOR": "computed by lithology category analysis from PEF and RHOB",
        "DOLOMITE_INDICATOR": "computed by lithology category analysis from PEF and RHOB",
        "CARBONATE_INDICATOR": "computed by lithology category analysis from PEF and RHOB",
        "GR_CLAY_INDICATOR": "computed by lithology category analysis from GR",
        "NPHI_CLAY_INDICATOR": "computed by lithology category analysis from NPHI",
        "PEF_CLAY_INDICATOR": "computed by lithology category analysis from PEF",
        "COMPOSITE_CLAY": "computed by lithology category analysis from available clay indicators",
        "COMPOSITE_CLAY_INDICATOR": "computed by lithology category analysis from available clay indicators",
        "CONSISTENCY_SCORE": "computed by lithology category analysis from GR and PEF checks",
        "ND_SEPARATION": "computed by saturation category analysis from PHID and PHIN",
        "PHIT_GAS_CORR": "computed by saturation category analysis from PHIN and PHID",
        "DS_SEPARATION": "computed by porosity category analysis from PHID and PHIS",
        "PORO_QUALITY_INDEX": "computed by porosity category analysis from selected porosity and selected permeability",
        "RXO_RDEP_RATIO": "computed by permeability category analysis from RXO and RDEP",
        "SEPARATION_RATIO": "computed by saturation category analysis from RXO and RDEP",
        "KH": "computed by permeability category analysis from selected permeability and depth step",
        "PERM_QUALITY_INDEX": "computed by permeability category analysis from selected permeability, RQI, and FZI",
        "DATA_COMPLETENESS": "computed by numerical category analysis from finite numerical log count",
        "CONSISTENCY_RATIO": "computed by numerical category analysis from expected log relationships",
        "ACOUSTIC_IMPEDANCE": "computed by numerical category analysis from RHOB and DTC",
        "POISSON_RATIO": "computed by numerical category analysis from Vp and Vs",
        "YOUNG_MOD": "computed by numerical category analysis from RHOB, Vp, and Vs",
        "BRITTLENESS_INDEX": "computed by numerical category analysis from Young's modulus and Poisson ratio",
        "ELECTROFACIES": "computed by numerical category analysis from clustered log responses",
        "S_V": "computed by numerical category analysis from RHOB and depth step",
        "PP_HYDROSTATIC": "computed by numerical category analysis from depth",
        "SH_MIN": "computed by numerical category analysis from Poisson ratio, overburden stress, and pore pressure",
        "PRIME_GAS_TARGET": "computed by hydrocarbon category analysis from gas crossover, Vp/Vs, and reservoir quality",
        "PRIME_OIL_TARGET": "computed by hydrocarbon category analysis from pay flag and reservoir quality",
    }
    if curve in category_reasons:
        return category_reasons[curve]
    if curve.endswith("_GRAD"):
        return "computed by numerical category analysis as a depth gradient"

    step_name = REFERENCE_CURVE_TO_STEP.get(curve)
    step = step_results.get(step_name) if step_name else None
    if step is None:
        return ""

    if curve == "SXO":
        sxo_method = step.parameters.get("sxo_method")
        sxo_reason = step.parameters.get("sxo_reason")
        if sxo_method == "archie_rxo_rmf":
            return "computed from Archie invaded-zone equation using RSHA/RXO/RESS plus RMF"
        if sxo_method == "fifth_root_quicklook":
            if sxo_reason:
                return f"computed with fifth-root quick-look fallback because {sxo_reason}"
            return "computed with fifth-root quick-look fallback SXO = SW^(1/5)"
    if curve == "RMF":
        return _method_label(step.method) if step.method else "computed"
    if curve == "RW":
        return _method_label(step.method) if step.method else "computed"
    if curve in {"RWA", "SAL", "SALAPP", "SALMUDAPP"}:
        return "computed by resistivity/salinity category analysis"
    if curve == "INVASION_RATIO":
        return "computed as flushed/shallow resistivity over deep resistivity; RMED/RDEP is used only as a fallback proxy indicator; invalid denominators and extreme ratios are written as NaN"
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
        details = ""
        if well is not None:
            diagnostic = _segment_diagnostic_sxo_count(well)
            fallback = _segment_sxo_method_counts(well).get("fifth_root_quicklook", 0)
            details = f" for {diagnostic} diagnostic SXO samples" if diagnostic else ""
            if fallback:
                details += f"; {fallback} fallback-SXO samples were left as NaN"
        return "computed as MHI = SW / SXO" + details
    if curve == "TEMP":
        return "computed as depth-wise formation temperature"
    if curve == "VP_VS_RATIO":
        return "computed as VP_VS_RATIO = DTS / DTC"
    if step.method:
        return _method_label(step.method)
    return "computed"


def _derived_status_target_list(extra_targets: Optional[Sequence[str]] = None) -> List[str]:
    targets = list(REFERENCE_DERIVED_LOG_TARGETS)
    if extra_targets:
        targets.extend(_canonical_delivery_curve_name(name) for name in extra_targets)
    return [
        target
        for target in _ordered_unique(targets)
        if target not in CORE_ONLY_CONTEXT_CURVES
    ]


def _build_reference_derived_log_status(
    well: "Well",
    workflow_result,
    original_curves: set[str],
    extra_targets: Optional[Sequence[str]] = None,
    computed_curve_names: Optional[Sequence[str]] = None,
) -> List[dict[str, str]]:
    computed_curves = set(_derived_logs_computed(well, workflow_result, original_curves))
    if computed_curve_names:
        computed_curves.update(
            _canonical_delivery_curve_name(name)
            for name in computed_curve_names
        )
    normalized_original_curves = {
        _canonical_delivery_curve_name(curve)
        for curve in original_curves
    }
    step_results, skipped_steps = _step_lookup(workflow_result)
    statuses: List[dict[str, str]] = []
    code_curve = interpreted_lithology_code_curve(well)

    for curve in _derived_status_target_list(extra_targets):
        if (
            curve == "LITHO"
            and code_curve in normalized_original_curves
            and not _workflow_output_curve_written(workflow_result, "LITHO")
        ):
            continue
        elif curve == "VP_VS_RATIO" and curve not in computed_curves:
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
        elif curve in computed_curves:
            status = "computed"
            reason = _computed_curve_detail(curve, step_results, well)
        elif curve in normalized_original_curves and _has_finite_any_curve(well, _target_curve_candidates(curve)):
            status = "already_present"
            reason = "present in source data"
        elif _has_finite_any_curve(well, _target_curve_candidates(curve)):
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
        "finite_percent": _round_stat(finite_count / max(sample_count, 1) * 100.0),
    }
    if finite_count:
        summary.update(
            {
                "min": _round_stat(np.nanmin(finite)),
                "p05": _round_stat(np.nanpercentile(finite, 5)),
                "p50": _round_stat(np.nanpercentile(finite, 50)),
                "p95": _round_stat(np.nanpercentile(finite, 95)),
                "max": _round_stat(np.nanmax(finite)),
            }
        )

    depth = getattr(well, "depth", None)
    if depth is not None and len(depth) == sample_count and np.any(nan_mask):
        intervals = _flag_intervals(np.asarray(depth, dtype=float), nan_mask)
        summary["nan_interval_count"] = len(intervals)
        summary["nan_intervals_first_25"] = intervals[:25]
    return summary


def _invasion_ratio_rejection_details(well: "Well", workflow_result=None) -> Optional[dict[str, Any]]:
    """Why INVASION_RATIO samples of a context well are NaN.

    Under the "geolink" profile the inputs are the first shallow (RSHA, RXO,
    RESS) and deep curve present; otherwise the curves the workflow's
    ``"current"`` ratio reads, sample by sample: RXO, RSHA and RESS, then the
    RMED/RESM proxy, over RDEP then RT.
    """
    from petrophys.interpretation.electrical.invasion.invasion_module import (
        MEDIUM_RESISTIVITY_CURVES,
        SHALLOW_RESISTIVITY_CURVES,
    )

    ratio_curve = well.get("INVASION_RATIO")
    if workflow_result is not None and _reproduces_benchmark_context(workflow_result):
        rxo_curve = _first_curve(well, ["RSHA", "RXO", "RESS"])
        rt_curve = _first_curve(well, ["RDEP", "RT"])
        if ratio_curve is None or rxo_curve is None or rt_curve is None:
            return None
        rxo_data, rxo_name, rt_data, rt_name = rxo_curve.data, rxo_curve.mnemonic, rt_curve.data, rt_curve.mnemonic
    else:
        rxo_data, rxo_name, _ = get_curve_data_depthwise(
            well, [*SHALLOW_RESISTIVITY_CURVES, *MEDIUM_RESISTIVITY_CURVES]
        )
        rt_data, rt_name, _ = get_curve_data_depthwise(well, ["RDEP", "RT"])
        if ratio_curve is None or rxo_data is None or rt_data is None:
            return None

    rxo, rt = np.broadcast_arrays(
        _normalize_numeric_array(rxo_data),
        _normalize_numeric_array(rt_data),
    )
    ratio = _normalize_numeric_array(ratio_curve.data)
    rt_min = 0.0
    max_ratio = 100.0
    core_valid = (
        np.isfinite(rxo)
        & np.isfinite(rt)
        & (rxo > 0.0)
        & (rt > rt_min)
    )
    raw_ratio = np.full(rxo.shape, np.nan, dtype=float)
    raw_ratio[core_valid] = rxo[core_valid] / rt[core_valid]
    finite_ratio = np.isfinite(ratio)
    nan_mask = ~finite_ratio
    details: dict[str, Any] = {
        "rxo_curve": rxo_name,
        "rt_curve": rt_name,
        "total_samples": int(ratio.size),
        "valid_samples": int(np.count_nonzero(finite_ratio)),
        "nan_samples": int(np.count_nonzero(nan_mask)),
        "rt_min_used": rt_min,
        "max_ratio_used": max_ratio,
        "nonfinite_rxo_samples": int(np.count_nonzero(~np.isfinite(rxo))),
        "nonfinite_rt_samples": int(np.count_nonzero(~np.isfinite(rt))),
        "nonpositive_rxo_samples": int(np.count_nonzero(np.isfinite(rxo) & (rxo <= 0.0))),
        "rt_below_min_samples": int(np.count_nonzero(np.isfinite(rt) & (rt <= rt_min))),
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
            rejection_details = _invasion_ratio_rejection_details(well, workflow_result)
            item["invasion_rejection_details"] = rejection_details
            if rejection_details is not None:
                item["note"] += (
                    f" Segment invasion-ratio validity: {rejection_details['valid_samples']} valid "
                    f"of {rejection_details['total_samples']} samples; "
                    f"{rejection_details['nan_samples']} samples written as NaN/rejected."
                )
                reasons = []
                if rejection_details["rt_below_min_samples"]:
                    reasons.append(f"deep resistivity <= 0 ohm.m={rejection_details['rt_below_min_samples']}")
                if rejection_details["ratio_above_max_samples"]:
                    reasons.append(f"raw ratio > 100={rejection_details['ratio_above_max_samples']}")
                if rejection_details["nonfinite_rxo_samples"]:
                    reasons.append(f"non-finite flushed/shallow or fallback resistivity={rejection_details['nonfinite_rxo_samples']}")
                if rejection_details["nonfinite_rt_samples"]:
                    reasons.append(f"non-finite deep resistivity={rejection_details['nonfinite_rt_samples']}")
                if rejection_details["nonpositive_rxo_samples"]:
                    reasons.append(f"flushed/shallow or fallback resistivity <= 0={rejection_details['nonpositive_rxo_samples']}")
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


def _net_pay_cutoff_text(params: Mapping[str, Any], key: str) -> Any:
    """A net-pay cutoff from the step parameters, by lithology family where they differ."""
    value = params.get(key)
    by_family = params.get("cutoffs_by_family")
    if by_family and not isinstance(value, (int, float)):
        return ", ".join(f"{item[key]:g} {family}" for family, item in by_family.items())
    return value


def _context_workflow_config(
    config: Optional["DynamicWorkflowConfig"] = None,
    *,
    well: Optional["Well"] = None,
    lithology_file: Optional[str | Path] = None,
) -> "DynamicWorkflowConfig":
    """Build the workflow configuration from LAS and geology inputs."""
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig
    from petrophys.workflow.zoned_batch import (
        _extract_global_parameter_overrides,
        lithology_domain_from_metadata,
    )

    options = {}
    if lithology_file is not None:
        options.update(lithology_intervals=lithology_file, lithology_source="external")
    if config is None:
        if well is not None:
            metadata_options, _ = _extract_global_parameter_overrides(well)
            options.update(metadata_options)
        if "lithology_domain" not in options and well is not None:
            metadata_domain = lithology_domain_from_metadata(well)
            if metadata_domain is not None:
                options["lithology_domain"] = metadata_domain
        defaulted = {name for name in ("matrix_type", "lithology_domain") if name not in options}
        options.setdefault("matrix_type", "sandstone")
        options.setdefault("lithology_domain", "clastic")
        config = DynamicWorkflowConfig(**options)
        # The two defaults filled here are not the caller's settings.
        config._explicit_settings = config._explicit_settings - defaulted
        return config
    if not options:
        return config
    from petrophys.workflow.dynamic_workflow import _copy_config

    return _copy_config(config, **options)


def _attach_formations_for_interpretation(well: "Well", segments, config) -> None:
    """Attach a well's formation tops before its context interpretation, as ``petrophys.read`` does.

    The interpretation then uses them as a direct ``petrophys.interpret`` of
    the well read with ``formations=`` does (per-formation core calibration),
    so a context package and a direct interpretation agree. The benchmark
    context profiles keep their packages' interpretation, which had no tops.
    """
    import pandas as pd

    from petrophys.io import read_formations
    from petrophys.workflow.zoned_batch import _reproduces_benchmark

    if not segments or _reproduces_benchmark(config) or reproducing_benchmark():
        return
    context = dict(well.get_parameter("_geology_context") or {})
    if context.get("formations"):
        return
    records = [segment["record"] for segment in segments if segment.get("record")]
    if not records:
        return
    attached = read_formations(pd.DataFrame(records), well=well)
    for key in ("depth_unit", "depth_reference", "datum"):
        if attached.attrs.get(key) is not None:
            attached[key] = attached.attrs[key]
    context["formations"] = attached.to_dict("records")
    context["formations_import_report"] = attached.attrs.get("import_report", {})
    well.set_parameter("_geology_context", context)


def _read_context_input_well(
    file_path: Path,
    *,
    lithology_file: Optional[str | Path] = None,
    workflow_config: Optional["DynamicWorkflowConfig"] = None,
    lithology_domain: Optional[str] = None,
    reader_options: Optional[dict[str, Any]] = None,
):
    """Read one well file and build its workflow configuration.

    The configuration is ``workflow_config`` (default: the workflow defaults)
    with the optional lithology intervals and domain applied; LAS header
    parameters are applied by the zoned workflow runner.
    """
    import dataclasses

    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig
    from petrophys.api import read

    well = read(file_path, **(reader_options or {}))
    well = standardize_curves(well, rename_curves=True, convert_units=True)
    config = workflow_config if workflow_config is not None else DynamicWorkflowConfig()
    options: dict[str, Any] = {}
    if lithology_file is not None:
        options.update(lithology_intervals=lithology_file, lithology_source="external")
    if lithology_domain is not None:
        options.update(lithology_domain=lithology_domain, lithology=None)
    if options:
        from petrophys.workflow.dynamic_workflow import _copy_config

        config = _copy_config(config, **options)
    return well, config


def _formation_run_config(
    config: "DynamicWorkflowConfig", whole_well: "Well", formation_well: "Well"
) -> "DynamicWorkflowConfig":
    """The configuration of a formation's own run: the whole well's lithology code curve by name.

    Under ``"current"`` a lithology code curve other than ``LITHOLOGY_GEOLINK``
    is read once its codes pass the code-log checks; a thin formation may hold
    too few samples to pass them on its own, so its run names the curve the
    whole-well run read (as the zoned runner's zones do). The configuration is
    returned unchanged otherwise.
    """
    import dataclasses

    from petrophys.workflow.zoned_batch import _well_lithology_code_curve

    name = _well_lithology_code_curve(config, whole_well)
    if name is None or get_curve(formation_well, [name]) is None:
        return config
    return dataclasses.replace(config, lithology_curve=name)


def _run_context_workflow(well: "Well", config: "DynamicWorkflowConfig"):
    """Interpret a well for a context with the zoned workflow runner.

    Under the "geolink" profile every well gets the calculations of the eval
    package's re-run, which wrote the package's context (see
    :attr:`~petrophys.workflow.DynamicWorkflowConfig.calculation_profile`).
    """
    from petrophys._internal.analysis_helpers import DEPTH_UNIT_LABEL_PARAMETER
    from petrophys.workflow.dynamic_workflow import _rerun_calculations
    from petrophys.workflow.zoned_batch import run_dynamic_workflow_with_zones

    with _rerun_calculations():
        result = run_dynamic_workflow_with_zones(well, config)
    label = getattr(well, "_source_depth_unit_label", None)
    if label and _reproduces_benchmark_context(result):
        # The published packages wrote depth intervals with the source's label.
        result.well.set_parameter(DEPTH_UNIT_LABEL_PARAMETER, label)
    return result


def _write_context_workflow_las(
    file_path: Path,
    input_path: Path,
    processed_las_dir: str | Path,
    source_well: "Well",
    workflow_result,
    input_curves: set[str],
    warnings: Sequence[str],
    source_file: Optional[Path] = None,
    preparatory_steps: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Path:
    """Save the interpretation a context is computed from, with its workflow trace.

    The saved copy gets the context's value-range QC and the recomputation of
    the curves that depend on its inputs (BVW, PHI_NORM, RQI, FZI), as every
    group does before its statistics, so the context statistics are the
    statistics of exactly these curves over each group's samples. The trace
    records both as ``post_workflow_steps``, its ``final_curve_summaries``
    describe the saved curves, ``source_file`` names the well's raw LAS and
    ``input_file`` the LAS this workflow read. The GeoLink training profile
    writes the curve values at full precision (17 significant digits), the
    others at five decimals.
    """
    import dataclasses

    from petrophys.io import export_las
    from petrophys.workflow.zoned_batch import BatchLasResult, _write_json_log

    root = Path(processed_las_dir)
    try:
        relative = file_path.relative_to(input_path)
    except ValueError:
        relative = Path(file_path.name)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    checked, post_workflow_steps = context_checked_well(workflow_result)
    precision = {}
    if _context_dataset() == "geolink_train":
        # The GeoLink training package's statistics are those of the curves
        # at full precision: 17 significant digits read back as the same
        # float64 values; the depths keep the package's five decimals.
        precision = {"fmt": "%.17g", "column_fmt": {0: "%.5f"}}
    export_las(
        checked,
        target,
        depth_unit_label=getattr(source_well, "_source_depth_unit_label", None),
        pad_header_by_bytes=_reproduces_benchmark_context(workflow_result),
        **precision,
    )
    output_curves = sorted(checked.curves)
    trace_path = _write_json_log(
        BatchLasResult(
            source_file=str(source_file or file_path),
            output_file=str(target),
            success=workflow_result.base_result.success
            and all(zone.success for zone in workflow_result.zone_results),
            warnings=list(warnings),
            input_curves=sorted(input_curves),
            output_curves=output_curves,
            computed_curves=sorted(set(output_curves) - set(input_curves)),
            zones_applied=len(workflow_result.zones),
            # The final summaries describe the saved curves.
            result=dataclasses.replace(workflow_result, well=checked),
        ),
        relative,
        root,
    )
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    trace["input_file"] = str(file_path)
    if preparatory_steps:
        trace["preparatory_steps"] = _json_safe(list(preparatory_steps))
    config = getattr(workflow_result.base_result, "config", None)
    if not preparatory_steps and getattr(config, "reuse_input_resflag", False) and "RESFLAG" in input_curves:
        trace["input_note"] = (
            "The input LAS is an interpretation of source_file; its RESFLAG selects "
            "the SWIRR_APP samples (reuse_input_resflag)."
        )
    trace["post_workflow_steps"] = post_workflow_steps
    Path(trace_path).write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _input_reading_record(file_path: Path, well: "Well") -> dict[str, Any]:
    """How reading ``file_path`` for a context changed its curves.

    Compares the LAS as written with ``well`` as the context read it: curves
    renamed to standard mnemonics, unit conversions and value-range screening
    (``changed``), new names (``created``) and curves set aside
    (``removed``).
    """
    import lasio

    raw = {item.mnemonic: np.asarray(item.data, dtype=float) for item in lasio.read(str(file_path)).curves[1:]}
    raw_upper = {name.upper(): values for name, values in raw.items()}
    created, changed = [], []
    for name in well.curves:
        values = np.asarray(well.get(name).data, dtype=float)
        source = raw_upper.get(name.upper())
        if source is None:
            created.append(name)
        elif source.shape != values.shape or not np.array_equal(source, values, equal_nan=True):
            changed.append(name)
    present = {name.upper() for name in well.curves}
    return {
        "created": created,
        "changed": changed,
        "removed": [name for name in raw if name.upper() not in present],
        "rename_map": dict(well.get_parameter("_rename_map") or {}),
        "range_screening": {
            name: {key: well.get(name).metadata.custom["range_screening"].get(key) for key in ("masked", "low", "high")}
            for name in well.curves
            if well.get(name).metadata.custom.get("range_screening")
        },
    }


def context_group_statistics_for_well(group_well: "Well", workflow_result) -> dict[str, dict[str, Any]]:
    """The statistics of each category file of one context group, as the context computes them.

    ``group_well`` is the group's samples of ``workflow_result.well`` (the
    interpretation before the context's QC); it is modified as the context
    modifies it. Per category, in the context's order: the category's
    analysis, whose derived curves (indicators, gradients, MHI or BVW where
    the group has none, ...) join the statistics, then the statistics of the
    group's curves. Returns ``{category: statistics}``.
    """
    benchmark = _reproduces_benchmark_context(workflow_result)
    rules = _context_value_range_rules(workflow_result)
    apply_value_range_qc(group_well, rules=rules)
    if not benchmark:
        _synchronize_post_qc_dependent_curves(group_well)
    families = None if benchmark else _context_lithology_families(group_well, workflow_result, None)
    statistics: dict[str, dict[str, Any]] = {}
    for category_name in CATEGORY_CONTEXT_SPECS:
        category_result = _category_analysis_result(
            category_name, group_well, workflow_result, lithology_family=families
        )
        extra = category_result.derived_curves
        _apply_value_range_qc_to_extra_curves(extra, rules)
        _synchronize_post_qc_dependent_curves(group_well, benchmark=benchmark)
        statistics[category_name] = _build_reference_statistics(group_well, extra)
    return statistics


def context_checked_well(workflow_result) -> tuple["Well", list[dict[str, Any]]]:
    """The interpreted well as a context computes its statistics from, and how it got there.

    A copy of ``workflow_result.well`` gets the context's value-range QC and
    the recomputation of the curves that depend on its inputs (BVW, PHI_NORM,
    PHI_Z, RQI, FZI), as every context group does before its statistics.
    Returns the copy and the trace entries of both steps
    (``post_workflow_steps``).
    """
    benchmark = _reproduces_benchmark_context(workflow_result)
    checked = copy.deepcopy(workflow_result.well)
    apply_value_range_qc(checked, rules=_context_value_range_rules(workflow_result))
    qc_report = list(checked.get_parameter(VALUE_RANGE_QC_PARAMETER, []) or [])
    dependent = ("BVW", "PHI_NORM", "RQI", "FZI", "PHI_Z")
    before = {
        name: np.array(checked.get(name).data, dtype=float)
        for name in dependent
        if checked.get(name) is not None
    }
    _synchronize_post_qc_dependent_curves(checked, benchmark=benchmark)
    recomputed = {}
    for name in dependent:
        curve = checked.get(name)
        if curve is None:
            continue
        after = np.asarray(curve.data, dtype=float)
        old = before.get(name)
        if old is None:
            recomputed[name] = int(np.isfinite(after).sum())
        else:
            same = (old == after) | (np.isnan(old) & np.isnan(after))
            recomputed[name] = int((~same).sum())
    recompute_text = (
        "BVW = porosity x SW (porosity PHIE_CC, then PHIE, then PHIT; SW, then SWA); "
        "PHI_NORM = porosity/(1-porosity); RQI = 0.0314*sqrt(permeability/porosity) "
        "(permeability PERM_CC, then PERM); FZI = RQI/PHI_NORM; NaN where an input is missing"
    )
    if benchmark:
        recompute_text += "; without porosity or permeability the workflow's PHI_NORM, RQI and FZI are kept"
    steps = [
        {
            "step": "value_range_qc",
            "method": "context value-range rules",
            "description": "Samples outside each curve's range are set to NaN, or clipped where the rule says so.",
            "changes": [_json_safe(item) for item in qc_report if item.get("changed")],
        },
        {
            "step": "dependent_curve_recompute",
            "method": "post-QC recomputation of dependent curves",
            "description": recompute_text,
            "samples_changed": recomputed,
        },
    ]
    return checked, steps


def _method_label(name: str) -> str:
    mapping = {
        "mud_filtrate_salinity_conversion": "CLFL-to-RMF salinity conversion",
        "mud_filtrate_chlorides_conversion": "CLFL chlorides-to-RMF conversion via NaCl equivalent",
        "las_metadata": "direct LAS header value",
        "las_metadata_arps_temperature_correction": "LAS header RMF with RMFT temperature correction",
        "las_metadata_profile_temperature_correction": "LAS header RMF with RMFT depth-wise temperature correction",
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
        "modified_simandoux": "Modified Simandoux: the sw_model setting",
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
        "modified_simandoux": "Modified Simandoux equation: 1/Rt = (PHIE^m / (a * Rw * (1 - VCL))) * Sw^n + (VCL / RCL) * Sw.",
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


def _phi_cl_source_label(source: str) -> str:
    mapping = {
        "LAS PNSH parameter": "LAS PNSH parameter",
        "zoned LAS PNSH fallback": "zoned LAS PNSH fallback",
        "LAS PNSH and RBSH shale point": (
            "PHIT of the LAS neutron-density shale point (PNSH, RBSH) on the reservoir endpoints"
        ),
        "zoned LAS PNSH and RBSH shale point": (
            "PHIT of the zoned LAS neutron-density shale point (PNSH, RBSH) on the reservoir endpoints"
        ),
        "lithology_shale_average_phit": "average PHIT in shale lithology samples",
        "vcl_gt_0.9_average_phit": "average PHIT where VCL > 0.9",
        "lithology_shale_average_phit_nd": "average neutron-density PHIT in shale lithology samples",
        "lithology_shale_average_phit_nd_on_reservoir_matrix": (
            "average neutron-density PHIT in shale lithology samples on the reservoir matrix endpoints"
        ),
        "vcl_gt_0.9_average_phit_nd": "average neutron-density PHIT where VCL > 0.9",
        "lithology_shale_average_phid": "average PHID in shale lithology samples",
        "vcl_gt_0.9_average_phid": "average PHID where VCL > 0.9",
        "lithology_shale_average_phis": "average PHIS in shale lithology samples",
        "vcl_gt_0.9_average_phis": "average PHIS where VCL > 0.9",
        "lithology_shale_average_phin": "average PHIN in shale lithology samples",
        "vcl_gt_0.9_average_phin": "average PHIN where VCL > 0.9",
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


def _representative_bit_size(values: np.ndarray) -> Optional[float]:
    finite = _normalize_numeric_array(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    rounded = np.round(finite, 4)
    unique, counts = np.unique(rounded, return_counts=True)
    if unique.size:
        return float(unique[np.argmax(counts)])
    return float(np.nanmedian(finite))


def _bit_size_curve_data(well: "Well") -> Optional[np.ndarray]:
    bs_curve = well.get("BS")
    if bs_curve is not None:
        bs_data = _normalize_numeric_array(bs_curve.data)
        if np.any(np.isfinite(bs_data)):
            return bs_data
    return None


def _bit_size_used_for_bad_hole(well: "Well", workflow_result) -> Optional[float]:
    bs_data = _bit_size_curve_data(well)
    if bs_data is not None:
        return _representative_bit_size(bs_data)

    values = _step_parameter_values(_all_step_results(workflow_result, "bad_hole"), "bit_size_used")
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric

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
    """The bad-hole flags of a context segment, recomputed as its workflow flags them.

    Under ``"current"`` no bit size is assumed: the caliper washout and
    large-hole tests run where the segment records a bit size (BS curve, else
    a 3-36 in header value), ``|DRHO|`` is tested wherever DRHO is logged, and
    ``no_bit_size_points`` and ``not_assessed_points`` count the samples
    without a bit size and those that neither test reaches. The "geolink"
    profile takes the bit size from the segment's BS curve, else the bad-hole
    step's trace, else the header.
    """
    from petrophys.model.constants import DefaultParameters
    from petrophys.preprocessing.quality.bad_hole.flag_generator import (
        _known_bit_size,
        generate_bad_hole_flags,
    )

    current = not _reproduces_benchmark_context(workflow_result)
    if current:
        bit_size_for_flags = _known_bit_size(well)
        known = np.isfinite(bit_size_for_flags)
        bit_size = _representative_bit_size(bit_size_for_flags)
        bs_data = bit_size_for_flags if known.any() else None
    else:
        bit_size = _bit_size_used_for_bad_hole(well, workflow_result)
        bs_data = _bit_size_curve_data(well)
        bit_size_for_flags = bs_data if bs_data is not None else bit_size
    if well.get("CALI") is None and well.get("DRHO") is None:
        return None

    flags = generate_bad_hole_flags(
        well,
        bit_size=bit_size_for_flags,
        cali_threshold=DefaultParameters.CALI_BS_THRESHOLD,
        drho_threshold=DefaultParameters.DRHO_THRESHOLD,
        pef_max=DefaultParameters.PEF_MAX,
        pef_min=DefaultParameters.PEF_MIN,
        large_hole_threshold=DefaultParameters.LARGE_HOLE_BS,
        # The GeoLink training package flags |DRHO| alone where no bit size is known.
        drho_alone=current or (bit_size_for_flags is None and _context_dataset() == "geolink_train"),
    )

    depth = np.asarray(well.depth, dtype=float) if well.depth is not None else np.array([], dtype=float)
    washout = np.asarray(flags.get("washout", np.zeros(well.n_samples, dtype=bool)), dtype=bool)
    large_hole = np.asarray(flags.get("large_hole", np.zeros(well.n_samples, dtype=bool)), dtype=bool)
    pef_exclusion = np.asarray(flags.get("pef_exclusion", np.zeros(well.n_samples, dtype=bool)), dtype=bool)
    bad_hole = np.asarray(flags.get("bad_hole", np.zeros(well.n_samples, dtype=bool)), dtype=bool)

    total_points = int(well.n_samples)
    bad_points = int(np.count_nonzero(bad_hole))
    bad_percentage = round((bad_points / total_points) * 100.0, 4) if total_points else 0.0

    details = {
        "bit_size_used": bit_size,
        "bit_size_unique_values": (
            sorted({float(value) for value in np.round(bs_data[np.isfinite(bs_data)], 4)})
            if bs_data is not None
            else ([float(bit_size)] if bit_size is not None else [])
        ),
        "bit_size_depth_dependent": bool(
            bs_data is not None and np.unique(np.round(bs_data[np.isfinite(bs_data)], 4)).size > 1
        ),
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
    if current:
        cali, _, _ = get_curve_data_depthwise(well, ["CALI", "CAL"])
        drho, _, _ = get_curve_data_depthwise(well, ["DRHO"])
        tested = large_hole.copy()
        if cali is not None:
            tested |= np.isfinite(cali) & known
        if drho is not None:
            tested |= np.isfinite(drho)
        details["log_availability"]["BS"] = bool(known.any())
        details["no_bit_size_points"] = int(np.count_nonzero(~known))
        details["not_assessed_points"] = int(np.count_nonzero(~tested))
    return details


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


def _porosity_phi_cl_text(params: dict[str, Any]) -> str:
    phi_cl = params.get("phi_cl")
    if phi_cl is None:
        return ""
    source = str(params.get("phi_cl_source") or "")
    source_text = _phi_cl_source_label(source) if source else "configured/default clay-porosity value"
    text = f"phi_cl={_format_numeric(phi_cl, 3)} v/v from {source_text}"
    count = params.get("phi_cl_basis_count")
    if count is not None:
        text += f" using {count} control samples"
    reason = params.get("phi_cl_fallback_reason")
    if source == "config_or_default" and reason:
        text += f" because {reason}"
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
        counts_text = _phit_source_counts_text(params)
        if method == "adaptive_nd":
            gas_rule = (
                "PHIT = PHID where PHIN is missing; PHIT = sqrt((PHID^2 + PHIN^2) / 2) where PHID > PHIN + 0.05; "
                if _phit_gas_rms(params)
                else "PHIT = PHID where PHIN is missing or PHID > PHIN + 0.05; "
            )
            return (
                "PHIT calculated from PHIN and PHID by the adaptive neutron-density rule: "
                + gas_rule
                + "PHIT = PHIN where PHID is missing; "
                "otherwise PHIT = 0.6*PHID + 0.4*PHIN; clipped to 0-0.5"
                + counts_text
            )
        if method == "existing_phit":
            return "PHIT reused from the existing PHIT curve because no new PHIN/PHID/PHIS route replaced it" + counts_text
        if method == "density_only":
            return "PHIT calculated as PHIT = PHID because PHIN was unavailable" + counts_text
        if method == "neutron_only":
            return "PHIT calculated as PHIT = PHIN because PHID was unavailable" + counts_text
        if method == "sonic_only":
            return "PHIT calculated as PHIT = PHIS because PHIN and PHID were unavailable" + counts_text
        return "PHIT route was evaluated by the porosity step"

    if curve == "PHIE":
        basis = params.get("phie_basis") or "selected porosity"
        clay_text = _porosity_phi_cl_text(params)
        vcl_inputs = [name for name in inputs if name in {"VCL", "VCLAY", "VSH"}]
        if vcl_inputs:
            text = (
                f"PHIE calculated as clip({basis} - {vcl_inputs[0]} * phi_cl, 0, {basis}) "
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
        from petrophys.model.constants import DefaultParameters

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
        if method == "mud_filtrate_chlorides_conversion":
            return (
                "CLFL route selected because mud-filtrate chlorides were available; "
                f"CLFL={_format_numeric(params.get('mud_filtrate_chlorides_mg_l'), 0)} mg/l Cl, "
                f"NaCl equivalent={_format_numeric(params.get('mud_filtrate_salinity_ppm'), 0)} ppm, "
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
        return (
            "MHI calculated only where SXO had invaded-zone support from RSHA/RXO/RESS plus RMF; "
            f"diagnostic_samples={params.get('diagnostic_sxo_samples')}, total_samples={params.get('total_samples')}"
        )

    if step_name == "net_pay":
        return (
            f"flag curve calculated because VCL, PHIE, and SW were available; "
            f"VCL_cutoff={_net_pay_cutoff_text(params, 'vcl_cutoff')}, "
            f"PHIE_cutoff={_net_pay_cutoff_text(params, 'phie_cutoff')}, "
            f"SW_cutoff={_net_pay_cutoff_text(params, 'sw_cutoff')}, seal_VCL_cutoff={params.get('seal_vcl_cutoff')}"
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
    from petrophys.model.constants import ArchieParameterDefaults

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
    if "source" in info:
        notes.append(_COMPUTED_LITHOLOGY_NOTE)
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
    extra_targets: Optional[Sequence[str]] = None,
    computed_curve_names: Optional[Sequence[str]] = None,
) -> tuple[list[str], list[str]]:
    if original_curves is None:
        return [], []
    statuses = _build_reference_derived_log_status(
        well,
        workflow_result,
        original_curves,
        extra_targets=extra_targets,
        computed_curve_names=computed_curve_names,
    )
    computed = [
        item["mnemonic"]
        for item in statuses
        if item.get("status") == "computed" and item["mnemonic"] not in CONTEXT_NOTE_OMIT_CURVES
    ]
    missing = [
        item["mnemonic"]
        for item in statuses
        if item.get("status") == "missing" and item["mnemonic"] not in CONTEXT_NOTE_OMIT_CURVES
    ]
    return computed, missing


def _compact_lithology_notes(
    well: "Well",
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> list[str]:
    info = _build_reference_lithology_info(well, lithology_mapping=lithology_mapping)
    unique = ", ".join(info["unique_lithologies"]) if info["unique_lithologies"] else "none"
    notes = [
        (
            "Lithology: "
            f"valid={info['valid_lithology_count']}/{info['total_rows']} "
            f"({info['valid_lithology_percentage']:.0f}%), NaN={info['nan_count']}, "
            f"sentinel={info['sentinel_count']}, unique={unique}."
        ),
        *([_COMPUTED_LITHOLOGY_NOTE] if "source" in info else []),
        "Lithology fallback: "
        + (
            "generic clean-rock defaults used for invalid/missing rows."
            if info["nan_count"] or info["sentinel_count"] or not info["has_valid_lithology"]
            else "not used."
        ),
    ]

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

    selected = (well.get_parameter("_geology_context") or {}).get("selected_lithology_source")
    if selected == "external":
        curve = well.get("LITHO_CLASS")
        mapping = curve.metadata.custom.get("category_groups", {}) if curve is not None else {}
        groups = _groups_from_curve("LITHO_CLASS", {int(code): group for code, group in mapping.items()})
        if groups is not None:
            return groups
        return np.full(int(well.n_samples), fallback_text, dtype=object)
    code_curve = interpreted_lithology_code_curve(well)
    geolink = well.get(code_curve)
    recorded = geolink.metadata.custom.get("category_groups") if geolink is not None else None
    if recorded:
        # Codes the workflow left without a group take the fallback lithology.
        geolink_mapping = {int(code): group or fallback_text for code, group in recorded.items()}
    else:
        geolink_mapping = GROUND_TRUTH_CODE_TO_GROUP
    geolink_groups = _groups_from_curve(code_curve, geolink_mapping)
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

    vcl_curve = _first_curve(well, ["VCL", "VCLAY", "VSH"])
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
    # The current profile runs Dual Water only with a supplied Rwb; shaly
    # carbonate without one takes the clastic routing.
    carbonate_dual_water = not (
        params.get("shaly_sand_equations") == "published" and params.get("rwb") is None
    )
    counts: dict[str, int] = {}
    for idx in range(n_samples):
        if sw_curve is not None and idx < finite_sw.size and not bool(finite_sw[idx]):
            continue
        vcl_value = float(vcl[idx]) if np.isfinite(vcl[idx]) else 0.0
        if vcl_value < cutoff:
            model = "archie"
        else:
            group = str(lithology[idx] or "").strip().lower()
            if carbonate_dual_water and group in {"carbonate", "limestone", "dolomite", "calcite"}:
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
        "modified_simandoux": "Modified Simandoux",
        "dual_water": "Dual Water",
        "waxman_smits": "Waxman-Smits",
        "north_sea": "North Sea",
        "nigeria_schlumberger": "Nigeria-Schlumberger",
    }.get(str(model), str(model).replace("_", " ").title())


def _sw_equation_text(models: Iterable[str], params: dict[str, Any]) -> str:
    parts: list[str] = []
    model_set = {str(model) for model in models}
    published = params.get("shaly_sand_equations") == "published"
    if "archie" in model_set:
        parts.append("Archie Sw=(a*RW/(PHIE^m*Rt))^(1/n)")
    if "indonesia" in model_set and published:
        parts.append(
            "Indonesia (Poupon-Leveaux) 1/sqrt(Rt)=[VCL^(1-VCL/2)/sqrt(RCL)+PHIE^(m/2)/sqrt(a*RW)]*Sw^(n/2)"
        )
    elif "indonesia" in model_set:
        parts.append(
            "Indonesia 1/sqrt(Rt)=VCL^(1-VCL/2)/sqrt(RCL)+PHIE^(m/2)*Sw^(n/2)/sqrt(a*RW)"
        )
    if "simandoux" in model_set:
        parts.append("Simandoux 1/Rt=(PHIE^m/(a*RW))*Sw^n+(VCL/RCL)*Sw")
    if "modified_simandoux" in model_set:
        parts.append("Modified Simandoux 1/Rt=(PHIE^m/(a*RW*(1-VCL)))*Sw^n+(VCL/RCL)*Sw")
    if "dual_water" in model_set:
        rwb = params.get("rwb")
        phi_cl = params.get("phi_cl")
        phi_cl_source = params.get("phi_cl_source")
        rwb_text = f", Rwb={_format_numeric(rwb, 4)} ohm.m" if rwb is not None else ", Rwb=0.3*RW"
        phi_cl_text = f", phi_cl={_format_numeric(phi_cl, 3)}" if phi_cl is not None else ""
        if phi_cl_text and phi_cl_source:
            phi_cl_text += f" from {_phi_cl_source_label(str(phi_cl_source))}"
        if published:
            parts.append(
                "Dual Water (wet-zone form) phi_bw=VCL*phi_cl, Swb=min(phi_bw/PHIT,0.999), "
                "1/Rwt=Swb/Rwb+(1-Swb)/RW, Swt=[a*Rwt/(PHIT^m*Rt)]^(1/n), "
                f"Sw=(Swt-Swb)/(1-Swb){rwb_text}{phi_cl_text}"
            )
        else:
            parts.append(
                "Dual Water phi_bw=VCL*phi_cl, Swb=phi_bw/PHIT, "
                f"1/Rwt=Swb/Rwb+(1-Swb)/RW, Sw=[a*Rwt/(PHIT^m*Rt)]^(1/n){rwb_text}{phi_cl_text}"
            )
    if "waxman_smits" in model_set and published:
        parts.append(
            "Waxman-Smits Ct=(Swt^n/F*)*(Cw+B*Qv/Swt), solved iteratively with CEC in meq/g, "
            "Sw=1-(PHIT/PHIE)*(1-Swt)"
        )
    elif "waxman_smits" in model_set:
        parts.append("Waxman-Smits Ct=(Sw^n/F*)*(Cw+B*Qv/Sw), solved iteratively")
    if "north_sea" in model_set:
        parts.append("North Sea 1/Rt=(PHIE^m/(a*RW))*Sw^n+VCL^vsh_exp/RTSH")
    if "nigeria_schlumberger" in model_set:
        parts.append("Nigeria-Schlumberger 1/sqrt(Rt)=VCL^shale_exp/sqrt(RSH)+PHIE^(m/2)*Sw^(n/2)/sqrt(a*RW)")
    return "; ".join(parts)


def _archie_coefficients_text(params: Mapping[str, Any], well: Optional["Well"] = None) -> str:
    if well is not None:
        try:
            from petrophys.model.constants import ArchieParameterDefaults

            groups = _segment_lithology_groups_for_saturation(
                well,
                fallback=params.get("lithology_type") or "clastic",
            )
            unique_groups = _ordered_unique(
                str(group or "generic_clean").lower()
                for group in np.asarray(groups, dtype=object).tolist()
            )
            parts: list[str] = []
            override_a = params.get("a") if params.get("archie_a_source") == "config_or_las" else None
            override_m = params.get("m") if params.get("archie_m_source") == "config_or_las" else None
            override_n = params.get("n") if params.get("archie_n_source") == "config_or_las" else None
            for group in unique_groups:
                coeff = ArchieParameterDefaults.get(group)
                parts.append(
                    f"{str(coeff.lithology or group).upper()}: "
                    f"a={_format_numeric(override_a if override_a is not None else coeff.a, 3)}, "
                    f"m={_format_numeric(override_m if override_m is not None else coeff.m, 3)}, "
                    f"n={_format_numeric(override_n if override_n is not None else coeff.n, 3)}"
                )
            if parts:
                return "; Archie coefficients: " + "; ".join(parts)
        except Exception:
            pass

    coeffs = params.get("archie_coefficients") or {}
    if isinstance(coeffs, Mapping) and coeffs:
        parts: list[str] = []
        for lithology, item in coeffs.items():
            if not isinstance(item, Mapping):
                continue
            parts.append(
                f"{str(lithology).upper()}: a={_format_numeric(item.get('a'), 3)}, "
                f"m={_format_numeric(item.get('m'), 3)}, n={_format_numeric(item.get('n'), 3)}"
            )
        if parts:
            return "; Archie coefficients: " + "; ".join(parts)

    a_value = params.get("a")
    m_value = params.get("m")
    n_value = params.get("n")
    if a_value is not None or m_value is not None or n_value is not None:
        return (
            f"; Archie coefficients: a={_format_numeric(a_value, 3)}, "
            f"m={_format_numeric(m_value, 3)}, n={_format_numeric(n_value, 3)}"
        )
    return ""


def _sw_route_reason_text(model: str, params: dict[str, Any]) -> str:
    cutoff = params.get("vcl_cutoff_for_routing")
    salinity = params.get("salinity_ppm")
    cec_present = bool(params.get("cec_present") or params.get("cec") is not None)
    b_coeff_present = bool(params.get("b_coeff_present") or params.get("b_coeff") is not None)
    published = params.get("shaly_sand_equations") == "published"
    requested = str(params.get("sw_model_requested") or "auto").strip().lower()
    if published and requested != "auto":
        # A requested model runs on every sample, whatever its VCL.
        return f"requested sw_model='{requested}'"
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
            if published and params.get("rwb") is None:
                # The current profile runs Dual Water only with a supplied Rwb;
                # shaly carbonate without one takes the clastic routing.
                return f"VCL>={cutoff}, no CEC/B, {sal_text}; non-carbonate, or carbonate without Rwb"
            return f"VCL>={cutoff}, non-carbonate, no CEC/B, {sal_text}"
        return f"VCL>={cutoff}, non-carbonate"
    return "configured saturation model"


def _permeability_equation_text(method: str, params: dict[str, Any]) -> str:
    method_norm = str(method or "").strip().lower()
    if method_norm == "timur":
        return "Timur k=0.136*(100*PHIE)^4.4/(100*SWIRR)^2"
    if method_norm in {"coates", "coates_denoo"}:
        return "Coates-Denoo k=10000*PHIE^4*((1-SWIRR)/SWIRR)^2"
    if method_norm == "wyllie_rose":
        return "Wyllie-Rose (Morris-Biggs oil) k=62500*PHIE^6/SWIRR^2"
    if method_norm == "lith_vcl":
        metadata = params.get("empirical_method_metadata") or {}
        groups = metadata.get("lithology_parameter_groups") or []
        group_text = f"; parameter groups={', '.join(map(str, groups))}" if groups else ""
        return "lith_vcl k=10^min(4.5, slope*PHIT*(1-VCL*1.2)+intercept); valid VCL>=0 and PHIT>0.001" + group_text
    return f"{method} empirical equation"


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
    if params.get("gr_bounded") is not None:
        # The endpoints the run used (the bound holds for picked ones only).
        gr_clean, gr_shale = params.get("gr_min_used", gr_clean), params.get("gr_max_used", gr_shale)

    formulas = [
        f"IGR=clip((GR-{_format_numeric(gr_clean, 1)})/({_format_numeric(gr_shale, 1)}-{_format_numeric(gr_clean, 1)}),0,1)",
        _vcl_gr_equation(params.get("gr_method")),
    ]
    used = ["GR"]
    if params.get("use_sp"):
        sp_in_vcl = params.get("sp_in_vcl", True)
        if sp_in_vcl:
            used.append("SP")
        formulas.append(
            "VCL_SP=clip((SPclean-SP)/(SPclean-SPshale),0,1); "
            f"SPclean={_format_numeric(params.get('sp_max'), 1)} mV, "
            f"SPshale={_format_numeric(params.get('sp_min'), 1)} mV"
            + ("" if sp_in_vcl else " (reported only)")
        )
    if params.get("use_nd"):
        nd_in_vcl = params.get("nd_in_vcl", True)
        if nd_in_vcl:
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
            + ("" if nd_in_vcl else " (reported only)")
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
    if method == "mud_filtrate_chlorides_conversion":
        return (
            "RMF: CLFL chlorides as NaCl; NaCl ppm=CLFL*58.44/35.45/brine density, "
            f"Rmf75=0.0123+3647.5/NaCl^0.955, then {arps}; "
            f"CLFL={_format_numeric(params.get('mud_filtrate_chlorides_mg_l'), 0)} mg/l, "
            f"brine density={_format_numeric(params.get('brine_density_g_cm3'), 4)} g/cm3, "
            f"NaCl={_format_numeric(params.get('mud_filtrate_salinity_ppm'), 0)} ppm, "
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


def _rwa_rank_text(params: Mapping[str, Any]) -> str:
    """How the clean-wet Rwa route picked its value: P5 of Rwa, with or without a temperature correction."""
    if not params.get("rwa_temperature_corrected"):
        return "P5"
    return f"P5 of each sample's RWA corrected to {_format_numeric(params.get('rwa_reference_temp_f'), 0)} F"


def _dataset_rw_curve_note(step: Any, well: "Well") -> str:
    """The RW note of the Danish and COSTA packages: Rw corrected directly with Arps."""
    params = getattr(step, "parameters", {}) or {}
    rw_range = _depth_profile_range_text([step], "rw_min", "rw_max", 4, "ohm.m")
    method = getattr(step, "method", "")
    arps_formula = "RW(T)=RWref*(Tref+6.77)/(TEMP+6.77)"
    salinity_formula = _rw_temperature_formula_text()

    if method == "rwa_clean_wet_p5":
        coeff_text = _archie_coefficients_text(params, well).lstrip("; ")
        return (
            f"RW: RWA clean-wet {_rwa_rank_text(params)}; "
            "RWA=RDEP*PHIE^m/a, "
            f"mask=VCL<{params.get('vcl_cutoff')} and PHIE>{params.get('phie_cutoff')}; "
            f"{'P5' if not params.get('rwa_temperature_corrected') else 'RW'} at reference temperature={_format_numeric(params.get('rw_from_rwa'), 4)} ohm.m, "
            f"Tref={_format_numeric(params.get('rwa_reference_temp_f'), 1)} F, "
            f"samples={params.get('rwa_clean_wet_samples')}, "
            + (f"{coeff_text}; " if coeff_text else "")
            + f"{arps_formula}; RW(T)={rw_range}."
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
            "RW: clean-water Rxo/Rt; RW_clean=RMF(T)*RT/RXO, then direct Arps temperature correction; "
            f"samples={params.get('n_points')}, RWref={_format_numeric(params.get('rw_at_reference_temp'), 4)} ohm.m "
            f"at Tref={_format_numeric(params.get('rw_reference_temp_f'), 1)} F, RW(T)={rw_range}."
        )
    if method == "salinity_conversion":
        return (
            f"RW: salinity conversion; salinity={_format_numeric(params.get('salinity_ppm'), 0)} ppm; "
            f"{salinity_formula}; RW(T)={rw_range}."
        )
    if method in {"user_provided_temperature_profile_corrected", "las_metadata_temperature_profile_corrected"}:
        source = "user input" if method.startswith("user") else "LAS metadata"
        return (
            f"RW: {source}; measured RW={_format_numeric(params.get('rw_at_measured_temp'), 4)} ohm.m "
            f"at { _format_numeric(params.get('measured_temp_f'), 1)} F; {arps_formula}; RW(T)={rw_range}."
        )
    if method in {"user_provided_assumed_reference_temperature", "las_metadata_assumed_reference_temperature"}:
        source = "user input" if method.startswith("user") else "LAS metadata"
        return (
            f"RW: {source}; RW={_format_numeric(params.get('rw'), 4)} ohm.m at reference "
            f"{_format_numeric(params.get('reference_temp_f'), 1)} F; {arps_formula}; RW(T)={rw_range}."
        )
    if method == "default_fallback_profile":
        return (
            f"RW: default fallback; RW={_format_numeric(params.get('rw_default'), 4)} ohm.m at "
            f"{_format_numeric(params.get('reference_temp_f'), 1)} F; {arps_formula}; RW(T)={rw_range}."
        )
    return f"RW: {method}; {arps_formula}; RW(T)={rw_range or 'not recorded'}."


def _rw_curve_note(step: Any, well: "Well") -> str:
    if _context_dataset() is not None:
        return _dataset_rw_curve_note(step, well)
    params = getattr(step, "parameters", {}) or {}
    rw_range = _depth_profile_range_text([step], "rw_min", "rw_max", 4, "ohm.m")
    method = getattr(step, "method", "")
    formula = _rw_temperature_formula_text()

    if method == "rwa_clean_wet_p5":
        coeff_text = _archie_coefficients_text(params, well).lstrip("; ")
        return (
            f"RW: RWA clean-wet {_rwa_rank_text(params)}; "
            "RWA=RDEP*PHIE^m/a, "
            f"mask=VCL<{params.get('vcl_cutoff')} and PHIE>{params.get('phie_cutoff')}; "
            f"{'P5' if not params.get('rwa_temperature_corrected') else 'RW at 75 F'}={_format_numeric(params.get('rw_from_rwa'), 4)} ohm.m, "
            f"samples={params.get('rwa_clean_wet_samples')}, "
            f"salinity={_format_numeric(params.get('salinity_ppm_estimated'), 0)} ppm; "
            + (f"{coeff_text}; " if coeff_text else "")
            + f"{formula}; RW(T)={rw_range}."
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


def _core_model_text(params: dict[str, Any]) -> str:
    models = params.get("core_models") or []
    if not models:
        return ""
    same_well_models = [
        model for model in models
        if str(model.get("source_type") or "").startswith("same_well_core")
    ]
    model = same_well_models[0] if same_well_models else models[0]
    por = model.get("porosity_correction") or {}
    perm = model.get("perm_transform") or {}
    if not perm:
        return (
            f" core_phi={model.get('core_porosity_curve') or 'unknown'}, "
            f"A={_format_numeric(por.get('A'), 4)}, B={_format_numeric(por.get('B'), 4)}, "
            f"n={model.get('n_core_points')}; porosity-only calibration, "
            "no usable core permeability transform."
        )
    return (
        f" core_phi={model.get('core_porosity_curve') or 'unknown'}, "
        f"core_perm={model.get('core_permeability_curve') or 'unknown'}, "
        f"A={_format_numeric(por.get('A'), 4)}, B={_format_numeric(por.get('B'), 4)}, "
        f"A2={_format_numeric(perm.get('A'), 4)}, B2={_format_numeric(perm.get('B'), 4)}, "
        f"n={model.get('n_core_points')}."
    )


def _core_porosity_fitted_to_phit(params: Mapping[str, Any]) -> bool:
    """True when the core porosity correction was fitted to PHIT (the "current" profile)."""
    fits = params.get("core_porosity_fits") or []
    return bool(fits) and all(fit.get("input_curve") == "PHIT" for fit in fits)


def _rejected_core_model_text(params: Mapping[str, Any]) -> str:
    rejected = params.get("core_rejected_models") or []
    if not rejected:
        return ""
    model = rejected[0]
    parts = []
    if model.get("n_core_points") is not None:
        parts.append(f"n={model.get('n_core_points')}")
    if model.get("core_porosity_curve"):
        parts.append(f"core_phi={model.get('core_porosity_curve')}")
    if model.get("core_permeability_curve"):
        parts.append(f"core_perm={model.get('core_permeability_curve')}")
    if model.get("reason"):
        parts.append(f"reason={model.get('reason')}")
    return "; rejected model " + ", ".join(parts) if parts else ""


def _category_status_targets(category: str, well_stem: Optional[str] = None) -> tuple:
    """The status targets of ``category``."""
    return tuple(CATEGORY_CONTEXT_SPECS[category]["targets"])


def _perm_inputs_label(separator: str = " or ") -> str:
    """How the context notes name the permeability inputs: the GeoLink training
    package names PERM_CC first ("PERM_CC or PERM"), the others PERM first."""
    names = ("PERM_CC", "PERM") if _context_dataset() == "geolink_train" else ("PERM", "PERM_CC")
    return separator.join(names)


def _context_dataset() -> Optional[str]:
    """The published package being reproduced besides GeoLink's eval package, else None.

    "danish", "costa" or "geolink_train", set by those context paths (see
    :func:`petrophys.workflow.dynamic_workflow._dataset_calculations`). Those
    packages were written by a later version of this exporter than the
    GeoLink main run's context, whose notes differ where this module tests it.
    """
    from petrophys.workflow.dynamic_workflow import _DATASET_CALCULATIONS_REQUESTED

    return _DATASET_CALCULATIONS_REQUESTED.get()


def _reproduces_benchmark_context(workflow_result) -> bool:
    """True when the workflow behind a context ran the "geolink" calculation profile."""
    from petrophys.workflow.zoned_batch import _reproduces_benchmark

    run = getattr(workflow_result, "base_result", workflow_result)
    return _reproduces_benchmark(getattr(run, "config", None))


def _run_lithology_mapping(
    lithology_mapping: Optional[dict[int, dict[str, str]]],
    workflow_result,
) -> Optional[dict[int, Optional[dict[str, str]]]]:
    """Lithology code names for the calculation profile of a context's workflow.

    A "current" run names code 10 "Porous Limestone" and gives code 26, "Gap",
    no lithology (``None``), as its calculations treat them
    (``CURRENT_PROFILE_CODE_NAMES``); the "geolink" profile keeps the names of
    the lithology workbook or ``GROUND_TRUTH_NAMES``.
    """
    from petrophys.evaluation.diagnostics.lithology_analysis import CURRENT_PROFILE_CODE_NAMES

    if workflow_result is None or _reproduces_benchmark_context(workflow_result):
        return lithology_mapping
    current = {
        code: None if name is None else {"lithology_attribute": name}
        for code, name in CURRENT_PROFILE_CODE_NAMES.items()
    }
    return {**(lithology_mapping or {}), **current}


def _top_reservoir_grade_text() -> str:
    """The highest grade of the reservoir-quality table and its limits, as note text."""
    from petrophys.model.class_tables import CLASS_TABLES

    table = CLASS_TABLES["RESERVOIR_QUALITY"]
    grade = table.grades[-1]
    limits = {family: grade.limits(family) for family in table.families}

    def text(phi: float, k: float, clay: float) -> str:
        return f"selected effective porosity>{phi:g}, selected permeability>{k:g} mD and VCL<{clay:g}"

    if len(set(limits.values())) == 1:
        return f"{grade.label} ({text(*next(iter(limits.values())))})"
    return f"{grade.label} (" + "; ".join(f"{family}: {text(*value)}" for family, value in limits.items()) + ")"


def _compact_curve_note(
    curve: str,
    status: str,
    reason: str,
    step: Any,
    well: "Well",
    benchmark: bool = False,
) -> str:
    params = getattr(step, "parameters", {}) or {}

    if status == "missing":
        if curve == "SW":
            inputs = _finite_input_status_text(
                well,
                [
                    ("RDEP/RT", ("RDEP", "RT")),
                    ("PHIE_CC/PHIE", ("PHIE_CC", "PHIE")),
                    ("RW", ("RW",)),
                    ("VCL/VSH", ("VCL", "VCLAY", "VSH")),
                ],
            )
            return (
                f"SW: skipped because {reason}. Required inputs/status: {inputs}. "
                "Routing also uses lithology/RCL for shaly-sand models."
            )
        if curve == "SXO":
            inputs = _finite_input_status_text(
                well,
                [
                    ("RSHA/RXO/RESS", ("RSHA", "RXO", "RESS")),
                    ("RMF", ("RMF",)),
                    ("PHIE_CC/PHIE", ("PHIE_CC", "PHIE")),
                    ("SW", ("SW", "SWA", "SWT-LOG")),
                ]
                if _context_dataset() is not None
                else [
                    ("SW", ("SW",)),
                    ("RSHA/RXO/RESS", ("RSHA", "RXO", "RESS")),
                    ("RMF", ("RMF",)),
                ],
            )
            return f"SXO: skipped because {reason}. Inputs/status: {inputs}."
        if curve == "MHI":
            inputs = _finite_input_status_text(
                well,
                [
                    ("SW", ("SW", "SWA", "SWT-LOG")),
                    ("SXO", ("SXO", "SWXOA", "SXO-LOG")),
                    ("RSHA/RXO/RESS", ("RSHA", "RXO", "RESS")),
                    ("RMF", ("RMF",)),
                ],
            )
            counts = _segment_sxo_method_counts(well)
            counts_text = (
                "; SXO route samples: " + ", ".join(f"{name}={count}" for name, count in counts.items())
                if counts else ""
            )
            return f"MHI: skipped because {reason}. Required inputs/status: {inputs}{counts_text}."
        if curve == "PERM":
            inputs = _finite_input_status_text(
                well,
                [
                    ("PHIE_CC/PHIE", ("PHIE_CC", "PHIE")),
                    ("SWIRR_APP", ("SWIRR_APP",)),
                    ("SW", ("SW",)),
                    ("RDEP/RT", ("RDEP", "RT")),
                ],
            )
            return f"PERM: skipped because {reason}. Required inputs/status: {inputs}."
        if curve == "PHIE":
            inputs = _finite_input_status_text(
                well,
                [
                    ("PHIT", ("PHIT",)),
                    ("PHID", ("PHID",)),
                    ("PHIN", ("PHIN",)),
                    ("PHIS", ("PHIS",)),
                    ("VCL/VSH", ("VCL", "VCLAY", "VSH")),
                ],
            )
            return f"PHIE: skipped because {reason}. Inputs/status: {inputs}."

    if curve == "BAD_HOLE" and step is not None:
        from petrophys.model.constants import DefaultParameters

        bad_curve = well.get("BAD_HOLE")
        bad_count = int(np.nansum(_normalize_numeric_array(bad_curve.data) > 0.5)) if bad_curve is not None else 0
        bad_pct = bad_count / max(int(well.n_samples), 1) * 100.0
        no_bit_size = 0
        if benchmark:
            bs_data = _bit_size_curve_data(well)
            bit_size = _representative_bit_size(bs_data) if bs_data is not None else params.get("bit_size_used")
        else:
            # "current" assumes no bit size: the segment's BS curve, else a
            # plausible header bit size.
            from petrophys.preprocessing.quality.bad_hole.flag_generator import _known_bit_size

            known_bit_size = _known_bit_size(well)
            no_bit_size = int(np.count_nonzero(~np.isfinite(known_bit_size)))
            bs_data = known_bit_size if no_bit_size < known_bit_size.size else None
            bit_size = _representative_bit_size(known_bit_size)
        bit_values = (
            sorted({float(value) for value in np.round(bs_data[np.isfinite(bs_data)], 4)})
            if bs_data is not None
            else None
        )
        bit_text = f"; BS={_format_numeric(bit_size, 2)} in" if bit_size is not None else ""
        if bit_values and len(bit_values) > 1:
            bit_text = "; BS values=" + ", ".join(_format_numeric(value, 2) for value in bit_values) + " in"
        if no_bit_size:
            drho_alone = f"washout=|DRHO|>{DefaultParameters.DRHO_THRESHOLD:.2f} g/cc alone"
            bit_text += (
                f"; no bit size (BS curve or header) for {no_bit_size} samples, where {drho_alone}"
                if bit_size is not None
                else f"; no bit size (BS curve or header), so {drho_alone}"
            )
        return (
            f"{curve}: washout=(CALI-BS)>{DefaultParameters.CALI_BS_THRESHOLD:.2f} in or "
            f"|DRHO|>{DefaultParameters.DRHO_THRESHOLD:.2f} g/cc; "
            f"large_hole=BS>{DefaultParameters.LARGE_HOLE_BS:.2f} in; rule=washout OR large_hole"
            + bit_text
            + f"; flagged={bad_count}/{well.n_samples} ({bad_pct:.0f}%)."
        )

    if curve == "TEMP" and step is not None:
        method = getattr(step, "method", "")
        temp_range = _curve_range_text(well, ["TEMP", "TMPC"], 1, "F")
        # TemperatureProfile.get_temperature: depth in ft measured from 0.
        formula = (
            "TEMP=surface+gradient*(DEPT-minDEPT)/100"
            if benchmark
            else "TEMP=surface+gradient*DEPT_ft/100 (DEPT in ft from depth 0)"
        )
        if method == "linear_gradient_from_BHT":
            return (
                f"TEMP: {formula}; BHT-derived gradient; "
                f"surface={_format_numeric(params.get('surface_temp_f'), 1)} F, "
                f"BHT={_format_numeric(params.get('bht_f'), 1)} F, "
                f"gradient={_format_numeric(params.get('gradient_f_per_100ft'), 4)} F/100ft"
                + (f"; TEMP={temp_range}." if temp_range else ".")
            )
        return (
            f"TEMP: {formula}; fallback geothermal gradient; "
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
    if curve == "RWA":
        return (
            f"RWA: skipped because {reason}."
            if status == "missing"
            else "RWA: RDEP*PHIE^m/a using Archie m and a; finite only where RDEP>0 and PHIE>0."
        )
    if curve == "SAL":
        return (
            f"SAL: skipped because {reason}."
            if status == "missing"
            else "SAL: formation-water salinity in ppm used for RW conversion; Rw75=0.0123+3647.5/SAL^0.955."
        )
    if curve == "SALAPP":
        return (
            f"SALAPP: skipped because {reason}."
            if status == "missing"
            else "SALAPP: apparent salinity from RWA corrected to 75 F; SALAPP=(3647.5/(RWA75-0.0123))^(1/0.955)."
        )
    if curve == "SALMUDAPP":
        return (
            f"SALMUDAPP: skipped because {reason}."
            if status == "missing"
            else "SALMUDAPP: apparent mud-filtrate salinity from RMFA corrected to 75 F; SALMUDAPP=(3647.5/(RMFA75-0.0123))^(1/0.955)."
        )

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
        method = params.get("phit_method")
        counts_text = _phit_source_counts_text(params)
        if method == "density_only":
            return "PHIT: PHIT=PHID because PHIN was unavailable; clipped 0-0.5" + counts_text + "."
        if method == "neutron_only":
            return "PHIT: PHIT=PHIN because PHID was unavailable; clipped 0-0.5" + counts_text + "."
        if method == "sonic_only":
            return "PHIT: PHIT=PHIS because PHIN and PHID were unavailable; clipped 0-0.5" + counts_text + "."
        if method == "existing_phit":
            return "PHIT: reused existing PHIT curve" + counts_text + "."
        gas_rule = (
            "PHID where PHIN is missing; sqrt((PHID^2+PHIN^2)/2) where PHID>PHIN+0.05"
            if _phit_gas_rms(params)
            else "PHID where PHIN is missing or PHID>PHIN+0.05"
        )
        return "PHIT: " + gas_rule + "; PHIN where PHID is missing; otherwise 0.6*PHID+0.4*PHIN; clipped 0-0.5" + counts_text + "."
    if curve == "PHIE" and step is not None:
        return "PHIE: clip(PHIT-VCL*phi_cl,0,PHIT); " + _porosity_phi_cl_text(params) + "."
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
        inputs = _finite_input_status_text(
            well,
            [
                ("RDEP/RT", ("RDEP", "RT")),
                ("PHIE_CC/PHIE", ("PHIE_CC", "PHIE")),
                ("RW", ("RW",)),
                ("VCL/VSH", ("VCL", "VCLAY", "VSH")),
            ],
        )
        return (
            f"SW: {route_text}; inputs/status: {inputs}; {counts_text}; "
            f"equation(s): {equations}"
            + _archie_coefficients_text(params, well)
            + (f"; RW(T)={rw_range}" if rw_range else "")
            + rcl_text
            + "."
        )
    if curve == "SXO" and step is not None:
        method = params.get("sxo_method")
        counts = _segment_sxo_method_counts(well) or params.get("sxo_method_counts") or {}
        if counts:
            if "archie_rxo_rmf" in counts and "fifth_root_quicklook" in counts:
                method = "archie_rxo_rmf+fifth_root_quicklook"
            elif "archie_rxo_rmf" in counts:
                method = "archie_rxo_rmf"
            elif "fifth_root_quicklook" in counts:
                method = "fifth_root_quicklook"
        counts_text = (
            "; samples: " + ", ".join(f"{name}={count}" for name, count in counts.items())
            if counts else ""
        )
        has_shallow = _has_finite_any_curve(well, ("RSHA", "RXO", "RESS"))
        has_rmf = _has_finite_curve(well, "RMF")
        has_sw = _has_finite_curve(well, "SW")
        inputs = _finite_input_status_text(
            well,
            [
                ("SW", ("SW",)),
                ("RSHA/RXO/RESS", ("RSHA", "RXO", "RESS")),
                ("RMF", ("RMF",)),
            ],
        )
        if has_sw and not has_shallow:
            return f"SXO: fallback SXO=SW^(1/5){counts_text}; inputs/status: {inputs}."
        if method == "archie_rxo_rmf" and has_shallow and has_rmf:
            rmf_range = _depth_profile_range_text([step], "rmf_min", "rmf_max", 4, "ohm.m")
            return (
                "SXO: invaded-zone Archie; SXO=(a*RMF/(PHIE^m*Rxo))^(1/n), clipped 0-1"
                + ("; " if benchmark else ", then SXO=max(SXO,SW); ")
                + _archie_coefficients_text(params, well).lstrip("; ")
                + counts_text
                + (f"; RMF(T)={rmf_range}." if rmf_range else ".")
            )
        if method == "fifth_root_quicklook":
            return f"SXO: fallback SXO=SW^(1/5){counts_text}; inputs/status: {inputs}."
        if method == "archie_rxo_rmf+fifth_root_quicklook":
            return (
                "SXO: invaded-zone Archie where RSHA/RXO/RESS and RMF are finite; "
                f"fallback SXO=SW^(1/5) where invaded-zone inputs are non-finite{counts_text}; inputs/status: {inputs}."
            )
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
            if _core_porosity_fitted_to_phit(params):
                return (
                    "PERM_CC: same-well core-calibrated transform; PHIT_CC=A*PHIT+B, "
                    "PERM_CC=10^(A2*PHIT_CC+B2), clipped 0.001-10000 mD."
                    + _core_model_text(params)
                )
            return (
                "PERM_CC: same-well core-calibrated transform; PHIE_CC=A*PHIE+B, "
                "PERM_CC=10^(A2*PHIE_CC+B2), clipped 0.001-10000 mD."
                + _core_model_text(params)
            )
        return f"PERM_CC: skipped because {reason}."
    if curve == "PHIE_CC":
        if status == "computed" and step is not None:
            if _core_porosity_fitted_to_phit(params):
                return (
                    "PHIE_CC: same-well core porosity correction fitted to PHIT; "
                    "PHIT_CC=A*PHIT+B, PHIE_CC=clip(PHIT_CC-(PHIT-PHIE),0,PHIT_CC)."
                    + _core_model_text(params)
                )
            return (
                "PHIE_CC: same-well core porosity correction; PHIE_CC=A*PHIE+B, "
                "clipped 0-0.5."
                + _core_model_text(params)
            )
        return f"PHIE_CC: skipped because {reason}."
    if curve == "SWIRR_APP" and step is not None:
        metadata = params.get("swirr_metadata") or {}
        return (
            "SWIRR_APP: P5(SW) over RESFLAG or NET_RES and deep resistivity (RDEP or RT)>5 ohm.m; "
            f"percentile={metadata.get('percentile', 5)}, min_samples={metadata.get('min_samples', 10)}, "
            f"hc_samples={metadata.get('hc_mask_samples')}."
        )
    if curve == "MHI" and status != "missing":
        counts = _segment_sxo_method_counts(well)
        diagnostic = counts.get("archie_rxo_rmf", 0)
        fallback = counts.get("fifth_root_quicklook", 0)
        fallback_text = f"; fallback-SXO rows set to NaN={fallback}" if fallback else ""
        return (
            "MHI: SW/SXO only where SXO has invaded-zone support from RSHA/RXO/RESS plus RMF; "
            f"diagnostic samples={diagnostic}{fallback_text}."
        )

    simple = {
        "RQI": "RQI: 0.0314*sqrt(selected permeability / selected effective porosity).",
        "PHI_NORM": "PHI_NORM: phi/(1-phi), where phi is the selected effective porosity.",
        "FZI": "FZI: RQI/PHI_NORM.",
        "BVW": "BVW: selected effective porosity * SW.",
        "CROSSOVER_FLAG": "CROSSOVER_FLAG: PHID-PHIN>0.05, VCL<0.35, selected effective porosity>0.05, BAD_HOLE<0.5.",
        "VP_VS_RATIO": "VP_VS_RATIO: DTS divided by DTC where DTC=20-300 us/ft, DTS=20-600 us/ft, and ratio=0.5-5.0.",
    }
    if curve in simple and status != "missing":
        return simple[curve]
    if curve == "VP_VS_RATIO" and status == "missing":
        inputs = _finite_input_status_text(
            well,
            [
                ("DTC/DT", ("DTC", "DT")),
                ("DTS", ("DTS",)),
            ],
        )
        return f"VP_VS_RATIO: skipped because {reason}. Required inputs/status: {inputs}."

    category_simple = {
        "LITHOLOGY_GROUP": (
            "LITHOLOGY_GROUP: grouped class mapped from lithology codes; numeric code is GT_LITHOLOGY_GROUP_CODE."
        ),
        "LITHOLOGY_NAME": (
            "LITHOLOGY_NAME: grouped lithology label mapped from lithology codes."
        ),
        "GT_LITHOLOGY_GROUP_CODE": (
            "GT_LITHOLOGY_GROUP_CODE: numeric class from lithology codes; limestone=1, dolomite=2, anhydrite=3, "
            "coal=4, calcareous_shale=5, halite=6, sandstone=7, shale=8, silt=9, coarse_clastic=10, "
            "evaporite=11, igneous=12, chert=13."
        ),
        "LIMESTONE_INDICATOR": "LIMESTONE_INDICATOR: 1 where PEF>4.5 and RHOB>2.7; else 0.",
        "DOLOMITE_INDICATOR": "DOLOMITE_INDICATOR: 1 where PEF is 2.8-3.2 and RHOB is 2.8-2.9; else 0.",
        "CARBONATE_INDICATOR": "CARBONATE_INDICATOR: 1 where PEF>2.8 and RHOB>2.6; else 0.",
        "GR_CLAY_INDICATOR": "GR_CLAY_INDICATOR: 1 where GR>100 API; else 0.",
        "NPHI_CLAY_INDICATOR": "NPHI_CLAY_INDICATOR: 1 where NPHI>0.30 v/v; else 0.",
        "PEF_CLAY_INDICATOR": "PEF_CLAY_INDICATOR: 1 where PEF<2.5; else 0.",
        "COMPOSITE_CLAY": "COMPOSITE_CLAY: mean of available GR, NPHI, and PEF clay indicators.",
        "COMPOSITE_CLAY_INDICATOR": "COMPOSITE_CLAY_INDICATOR: alias of COMPOSITE_CLAY; mean of available GR, NPHI, and PEF clay indicators.",
        "CONSISTENCY_SCORE": (
            "CONSISTENCY_SCORE: mean of available lithology-log checks; GR and PEF are compared against expected lithology response."
        ),
        "ND_SEPARATION": "ND_SEPARATION: PHID-PHIN.",
        "PHIT_GAS_CORR": "PHIT_GAS_CORR: clip(sqrt((PHIN^2+PHID^2)/2),0,0.45).",
        "DS_SEPARATION": "DS_SEPARATION: PHID-PHIS.",
        "PORO_QUALITY_INDEX": "PORO_QUALITY_INDEX: sqrt(max(PHIT*log10(PERM),1e-6)); clipped 0.001-5.",
        "RXO_RDEP_RATIO": "RXO_RDEP_RATIO: RXO/RDEP where both are finite and RDEP>0.",
        "SEPARATION_RATIO": "SEPARATION_RATIO: RXO/RDEP where both are finite and RDEP>0.1.",
        "KH": "KH: PERM*depth_step.",
        "PERM_QUALITY_INDEX": "PERM_QUALITY_INDEX: weighted mean of PERM(0.5), RQI*10(0.3), and FZI*5(0.2), using available inputs.",
        "DATA_COMPLETENESS": "DATA_COMPLETENESS: finite numerical logs divided by total numerical logs at each depth.",
        "CONSISTENCY_RATIO": (
            "CONSISTENCY_RATIO: fraction of expected log relationships matched: GR-VCL positive, RHOB-porosity negative, "
            "NPHI/PHIN-porosity positive, RDEP-SW negative, PHIT/PHIE-PERM positive."
        ),
        "ACOUSTIC_IMPEDANCE": "ACOUSTIC_IMPEDANCE: RHOB*1000*Vp, Vp=304800/DTC_us_per_ft; clipped 1e6-30e6 kg/m2/s.",
        "POISSON_RATIO": "POISSON_RATIO: (Vp^2-2Vs^2)/(2*(Vp^2-Vs^2)), with Vp=304800/DTC and Vs=304800/DTS; clipped 0-0.49.",
        "YOUNG_MOD": "YOUNG_MOD: RHOB*1000*Vs^2*(3Vp^2-4Vs^2)/(Vp^2-Vs^2)/1e9 GPa.",
        "BRITTLENESS_INDEX": "BRITTLENESS_INDEX: mean of normalized Young's modulus and inverse Poisson-ratio term; clipped 0-1.",
        "ELECTROFACIES": "ELECTROFACIES: k-means cluster label from available RHOB, NPHI/PHIN, GR, and RDEP/RT features.",
        "S_V": "S_V: cumulative overburden stress from RHOB*1000*9.81*depth_step.",
        "PP_HYDROSTATIC": "PP_HYDROSTATIC: hydrostatic pore pressure from water gradient plus seawater head where applicable.",
        "SH_MIN": "SH_MIN: ((nu/(1-nu))*(S_V-PP_HYDROSTATIC))+PP_HYDROSTATIC.",
    }
    if not benchmark:
        category_simple["S_V"] = (
            "S_V: overburden stress increment from the top of the analysed interval, "
            "cumulative RHOB*1000*9.81*depth_step; not the overburden from surface."
        )
        category_simple["PP_HYDROSTATIC"] = (
            "PP_HYDROSTATIC: 1000*9.81*|depth| along the depth index (not corrected to TVD), "
            "plus a fixed 100 m seawater head (1025 kg/m3) when the deepest sample is below 200 m."
        )
        category_simple["SH_MIN"] = (
            "SH_MIN: ((nu/(1-nu))*(S_V-PP_HYDROSTATIC))+PP_HYDROSTATIC with S_V the increment "
            "over the analysed interval, so not an absolute minimum horizontal stress."
        )
        category_simple["KH"] = f"KH: PERM*depth_step, in mD*{well.depth_unit}."
        category_simple["GT_LITHOLOGY_GROUP_CODE"] = (
            "GT_LITHOLOGY_GROUP_CODE: numeric class from lithology codes; no_lithology_data=0, limestone=1, "
            "dolomite=2, anhydrite=3, coal=4, calcareous_shale=5, halite=6, sandstone=7, shale=8, silt=9, "
            "clastic_coarse=10, evaporite=11, igneous=12, chert=13, special_lithology=14."
        )
    if curve in category_simple and status != "missing":
        return category_simple[curve]
    if curve.endswith("_GRAD") and status != "missing":
        base = curve[:-5]
        return f"{curve}: numerical gradient d{base}/dDEPT."

    if curve == "INVASION_RATIO":
        if status != "missing" and step is not None:
            stats = params or {}
            ratio_curve = well.get("INVASION_RATIO")
            if ratio_curve is not None:
                ratio = _normalize_numeric_array(ratio_curve.data)
                finite = np.isfinite(ratio)
                shallow, _, _ = get_curve_data_depthwise(well, ["RSHA", "RXO", "RESS"])
                medium, _, _ = get_curve_data_depthwise(well, ["RMED", "RESM"])
                shallow_finite = (
                    np.isfinite(np.asarray(shallow, dtype=float))
                    if shallow is not None else np.zeros(ratio.shape, dtype=bool)
                )
                medium_finite = (
                    np.isfinite(np.asarray(medium, dtype=float))
                    if medium is not None else np.zeros(ratio.shape, dtype=bool)
                )
                true_count = int(np.count_nonzero(finite & shallow_finite))
                proxy_count = int(np.count_nonzero(finite & ~shallow_finite & medium_finite))
                source_parts = []
                if true_count:
                    source_parts.append(f"RXO/RSHA/RESS={true_count}")
                if proxy_count:
                    source_parts.append(f"RMED={proxy_count}")
                source_text = "; sources: " + ", ".join(source_parts) if source_parts else ""
                proxy_text = (
                    f"; RMED/RDEP fallback proxy indicator used because RXO/RSHA/RESS were unavailable or non-finite at those samples, samples={proxy_count}"
                    if proxy_count else ""
                )
                return (
                    "INVASION_RATIO: flushed/shallow resistivity over deep resistivity depth-wise; "
                    f"finite={int(np.count_nonzero(finite))}/{well.n_samples}, "
                    f"NaN={int(ratio.size - np.count_nonzero(finite))}; screened where deep resistivity<="
                    f"{stats.get('rt_min_used', 0.0)} ohm.m or ratio>"
                    f"{stats.get('max_ratio_used', 100.0)}"
                    + source_text
                    + proxy_text
                    + "."
                )
            source_counts = stats.get("rxo_source_counts") or {}
            source_text = (
                "; flushed/shallow sources: " + ", ".join(f"{name}={count}" for name, count in source_counts.items())
                if source_counts else ""
            )
            proxy_samples = int(stats.get("medium_deep_proxy_samples") or 0)
            proxy_text = (
                f"; RMED/RDEP fallback proxy indicator used because RXO/RSHA/RESS were unavailable or non-finite at those samples, samples={proxy_samples}"
                if proxy_samples else ""
            )
            return (
                "INVASION_RATIO: flushed/shallow resistivity over deep resistivity depth-wise; "
                f"finite={stats.get('valid_samples')}/{stats.get('total_samples')}, "
                f"NaN={stats.get('nan_samples')}; screened where deep resistivity<="
                f"{stats.get('rt_min_used', 0.0)} ohm.m or ratio>"
                f"{stats.get('max_ratio_used', 100.0)}"
                + source_text
                + proxy_text
                + "."
            )
        return (
            f"INVASION_RATIO: skipped because {reason}."
            if status == "missing"
            else "INVASION_RATIO: flushed/shallow resistivity over deep resistivity; RMED/RDEP is fallback proxy indicator only; NaN where deep resistivity<=0 ohm.m or ratio>100."
        )
    if curve == "LITHO" and status == "already_present":
        return "LITHO: not inferred because lithology input was present; mineral volumes were still computed."
    if curve in {"VQUAR", "VCALC", "VDOLO", "VANHY", "LITHO"}:
        return (
            f"{curve}: multi-mineral output; sum(Vi)=1-VCL-PHIE, "
            "RHOB=sum(Vi*rho_i)+fluids, NPHI/U/DTC mixing equations where inputs are available."
        )
    if curve == "NET_ROCK" and step is not None:
        return f"NET_ROCK: 1 where VCL<{_net_pay_cutoff_text(params, 'vcl_cutoff')}; else 0."
    if curve in {"NET_RES", "RESFLAG"} and step is not None:
        cutoff = _net_pay_cutoff_text(params, "phie_cutoff")
        cutoff_text = cutoff if cutoff is not None else "lithology cutoff: 0.08 clastic, 0.045 carbonate"
        return f"{curve}: 1 where NET_ROCK=1 and selected effective porosity>{cutoff_text}; else 0."
    if curve == "NET_PAY" and step is not None:
        return f"NET_PAY: 1 where NET_RES=1 and SW<{_net_pay_cutoff_text(params, 'sw_cutoff')}; else 0."
    if curve == "SEALFLAG" and step is not None:
        return f"SEALFLAG: 1 where VCL>={params.get('seal_vcl_cutoff', 0.8)}; else 0."
    if curve == "PRIME_GAS_TARGET" and status != "missing":
        return "PRIME_GAS_TARGET: 1 where CROSSOVER_FLAG=1, VP_VS_RATIO<1.75, and reservoir quality is GOOD/EXCELLENT."
    if curve == "PRIME_OIL_TARGET" and status != "missing":
        if benchmark:
            return "PRIME_OIL_TARGET: 1 where pay flag=1 and reservoir quality is EXCELLENT (selected permeability>10 mD and selected effective porosity>0.15)."
        return f"PRIME_OIL_TARGET: 1 where pay flag=1 and reservoir quality is {_top_reservoir_grade_text()}."
    if status == "missing":
        return f"{curve}: skipped because {reason}."
    if status == "already_present":
        return f"{curve}: raw input curve; not recalculated."
    return f"{curve}: computed by {getattr(step, 'step', 'workflow') if step is not None else 'workflow'}."


def _build_compact_curve_routing_notes(
    well: "Well",
    workflow_result,
    original_curves: Optional[set[str]],
    extra_targets: Optional[Sequence[str]] = None,
    computed_curve_names: Optional[Sequence[str]] = None,
) -> List[str]:
    if original_curves is None:
        return []
    statuses = _build_reference_derived_log_status(
        well,
        workflow_result,
        original_curves,
        extra_targets=extra_targets,
        computed_curve_names=computed_curve_names,
    )
    step_results, _ = _step_lookup(workflow_result)
    benchmark = _reproduces_benchmark_context(workflow_result)
    notes: List[str] = []
    for item in statuses:
        curve = item["mnemonic"]
        if curve in CONTEXT_NOTE_OMIT_CURVES:
            continue
        if curve in {"VQUAR", "VCALC", "VDOLO", "VANHY"}:
            continue
        step_name = REFERENCE_CURVE_TO_STEP.get(curve)
        step = step_results.get(step_name) if step_name else None
        note = _compact_curve_note(
            curve, item["status"], item.get("reason", ""), step, well, benchmark=benchmark
        )
        if note:
            notes.append(note)
    return notes


def _build_concise_delivery_computation_notes(
    well: "Well",
    workflow_result,
    original_curves: Optional[set[str]] = None,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
    derived_logs_override: Optional[Sequence[str]] = None,
    status_targets_override: Optional[Sequence[str]] = None,
) -> List[str]:
    notes: List[str] = [
        "Analysis started: dynamic petrophysical workflow with per-well routing, temperature correction, and NaN handling."
    ]
    status_targets = list(
        _ordered_unique(
            [
                *(status_targets_override or []),
                *(derived_logs_override or []),
            ]
        )
    )
    computed, missing = _compact_status_lists(
        well,
        workflow_result,
        original_curves,
        extra_targets=status_targets,
        computed_curve_names=derived_logs_override,
    )
    if computed:
        notes.append(f"Computed {len(computed)} derived logs: {', '.join(computed)}.")
    if missing:
        notes.append(f"Missing derived logs ({len(missing)}): {', '.join(missing)}.")
    qc_report = well.get_parameter(VALUE_RANGE_QC_PARAMETER) if hasattr(well, "get_parameter") else None
    if qc_report is not None:
        notes.append(value_range_qc_note(qc_report))

    step_results, skipped_steps = _step_lookup(workflow_result)
    benchmark = _reproduces_benchmark_context(workflow_result)
    computed_curves = {
        _canonical_delivery_curve_name(name)
        for name in (derived_logs_override or computed)
    }

    def segment_curve_note(curve: str, step: Any) -> str:
        canonical = _canonical_delivery_curve_name(curve)
        status = "computed" if _has_finite_any_curve(well, _target_curve_candidates(canonical)) else "missing"
        reason = ""
        if status == "missing":
            reason = _missing_curve_reason(
                canonical,
                well,
                workflow_result,
                step_results,
                skipped_steps,
                computed_curves,
                original_curves or set(),
            )
        return _compact_curve_note(canonical, status, reason, step, well, benchmark=benchmark)

    notes.extend(
        _compact_lithology_notes(
            well, lithology_mapping=_run_lithology_mapping(lithology_mapping, workflow_result)
        )
    )

    for curve, step_name in (
        ("TEMP", "temperature"),
        ("BAD_HOLE", "bad_hole"),
        ("VCL", "vclay"),
        ("RW", "rw"),
    ):
        step = _first_step_result(workflow_result, step_name)
        if step is not None:
            notes.append(_compact_curve_note(curve, "computed", "", step, well, benchmark=benchmark))

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
        phi_cl = _porosity_phi_cl_text(por_step.parameters)
        porosity_parts: list[str] = []
        if _has_finite_curve(well, "PHIN"):
            porosity_parts.append("PHIN=clip(NPHI-nphi_ma,0,0.6)")
        if _has_finite_curve(well, "PHID"):
            porosity_parts.append("PHID=clip((rho_ma-RHOB)/(rho_ma-rho_fl),0,0.5)")
        if _has_finite_curve(well, "PHIS"):
            porosity_parts.append("PHIS=clip((DTC-dt_ma)/(dt_fl-dt_ma),0,0.5)")
        if _has_finite_curve(well, "PHIT"):
            method = (getattr(por_step, "parameters", {}) or {}).get("phit_method")
            counts_text = _phit_source_counts_text((getattr(por_step, "parameters", {}) or {}))
            if method == "density_only":
                porosity_parts.append("PHIT=PHID because PHIN unavailable" + counts_text)
            elif method == "neutron_only":
                porosity_parts.append("PHIT=PHIN because PHID unavailable" + counts_text)
            elif method == "sonic_only":
                porosity_parts.append("PHIT=PHIS because PHIN and PHID unavailable" + counts_text)
            else:
                gas_rule = (
                    "PHIT=PHID where PHIN missing; sqrt((PHID^2+PHIN^2)/2) where PHID>PHIN+0.05"
                    if _phit_gas_rms(getattr(por_step, "parameters", {}) or {})
                    else "PHIT=PHID where PHIN missing or PHID>PHIN+0.05"
                )
                porosity_parts.append(gas_rule + "; PHIN where PHID missing; else 0.6*PHID+0.4*PHIN" + counts_text)
        if _has_finite_curve(well, "PHIE"):
            porosity_parts.append("PHIE=clip(PHIT-VCL*phi_cl,0,PHIT)")
        if _has_finite_curve(well, "PHI_SEC"):
            porosity_parts.append("PHI_SEC=max(PHIT-PHIS,0)")
        if porosity_parts:
            notes.append(
                "Porosity: "
                + "; ".join(porosity_parts)
                + (f"; {phi_cl}." if phi_cl and _has_finite_curve(well, "PHIE") else ".")
            )

    sat_step = _first_step_result(workflow_result, "saturation")
    if sat_step is not None:
        notes.append(segment_curve_note("SW", sat_step))
        if well.get("SXO") is not None:
            notes.append(segment_curve_note("SXO", sat_step))

    lith_step = _first_step_result(workflow_result, "lithology")
    if lith_step is not None and any(well.get(name) is not None for name in ("VQUAR", "VCALC", "VDOLO", "VANHY")):
        litho_written = bool((getattr(lith_step, "parameters", {}) or {}).get("lithology_curve_written"))
        suffix = (
            "no input lithology was available, so LITHO was inferred."
            if litho_written
            else "input lithology was available, so LITHO was not inferred or overwritten."
        )
        notes.append(
            "Mineral volumes (VQUAR, VCALC, VDOLO, VANHY): solved with sum(Vi)=1-VCL-PHIE plus RHOB, "
            f"NPHI, U=PEF*RHOB, and DTC mixing; {suffix}"
        )

    perm_step = _first_step_result(workflow_result, "permeability")
    if perm_step is not None:
        notes.append(segment_curve_note("PERM", perm_step))
        if _has_finite_curve(well, "SWIRR_APP"):
            notes.append(segment_curve_note("SWIRR_APP", perm_step))

    if any(_has_finite_curve(well, name) for name in ("RQI", "PHI_NORM", "FZI")):
        notes.append(
            "Rock quality: RQI=0.0314*sqrt(selected permeability / selected effective porosity); "
            "PHI_NORM=phi/(1-phi); FZI=RQI/PHI_NORM."
        )
    if _has_finite_curve(well, "BVW"):
        notes.append("BVW: selected effective porosity * SW.")
    if _has_finite_curve(well, "MHI"):
        diagnostic = _segment_diagnostic_sxo_count(well)
        fallback = _segment_sxo_method_counts(well).get("fifth_root_quicklook", 0)
        notes.append(
            f"MHI: SW/SXO only where SXO has invaded-zone support; diagnostic samples={diagnostic}"
            + (f", fallback-SXO rows set to NaN={fallback}." if fallback else ".")
        )

    net_step = _first_step_result(workflow_result, "net_pay")
    if net_step is not None:
        flag_parts = [f"NET_ROCK=VCL<{_net_pay_cutoff_text(net_step.parameters, 'vcl_cutoff')}"]
        phie_cutoff = _net_pay_cutoff_text(net_step.parameters, "phie_cutoff")
        if phie_cutoff is not None:
            flag_parts.append(f"NET_RES and RESFLAG add selected effective porosity>{phie_cutoff}")
        else:
            flag_parts.append("NET_RES and RESFLAG use reservoir screening after NET_ROCK")
        flag_parts.extend(
            [
                f"NET_PAY adds SW<{_net_pay_cutoff_text(net_step.parameters, 'sw_cutoff')}",
                f"SEALFLAG=VCL>={net_step.parameters.get('seal_vcl_cutoff', 0.8)}",
                "CROSSOVER_FLAG=PHID-PHIN>0.05 and VCL<0.35 and selected effective porosity>0.05 and BAD_HOLE<0.5",
            ]
        )
        if _has_finite_curve(well, "VP_VS_RATIO"):
            flag_parts.append("VP_VS_RATIO=DTS divided by DTC with plausibility screens")
        notes.append("Flags: " + "; ".join(flag_parts) + ".")

    notes.extend(
        _build_compact_curve_routing_notes(
            well,
            workflow_result,
            original_curves,
            extra_targets=status_targets,
            computed_curve_names=derived_logs_override,
        )
    )
    notes.append("Analysis completed: compact routing notes recorded.")
    return list(dict.fromkeys(note for note in notes if note))


def _build_delivery_computation_notes(
    well: "Well",
    workflow_result,
    original_curves: Optional[set[str]] = None,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
    derived_logs_override: Optional[Sequence[str]] = None,
    status_targets_override: Optional[Sequence[str]] = None,
) -> List[str]:
    return _build_concise_delivery_computation_notes(
        well,
        workflow_result,
        original_curves=original_curves,
        lithology_mapping=lithology_mapping,
        derived_logs_override=derived_logs_override,
        status_targets_override=status_targets_override,
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
                bit_values = details.get("bit_size_unique_values") or []
                if details.get("bit_size_depth_dependent") and len(bit_values) > 1:
                    values_text = ", ".join(_format_numeric(value, 2) for value in bit_values)
                    note += f" Bit sizes in this segment: {values_text} in."
                else:
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
            archie_text = _archie_coefficients_text(getattr(rw_steps[0], "parameters", {}) or {}).lstrip("; ")
            salinity = _format_numeric_summary(_step_parameter_values(rw_steps, "salinity_ppm_estimated"), 0, "ppm")
            rank_text = _rwa_rank_text(getattr(rw_steps[0], "parameters", {}) or {})
            notes.append(
                "RW: RW was derived from apparent Rw because direct RW metadata/configuration, salinity input, RMF+SP, and RMF+Rxo/Rt routes did not produce usable RW. "
                f"Formula used: RWA = RDEP * PHIE^m / a. Mask used: VCL < 0.30 and PHIE > 0.05. Percentile used: {rank_text}. "
                "The selected RWA value was converted to equivalent salinity and rewritten depth-by-depth as RW(T)."
                + (f" {archie_text}." if archie_text else "")
                + (f" Samples used: {n_samples}." if n_samples else "")
                + (f" {'P5 RWA' if rank_text == 'P5' else 'RW'} reference value: {rw_from_rwa}." if rw_from_rwa else "")
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
        phi_cl_values = _format_numeric_summary(_step_parameter_values(porosity_steps, "phi_cl"), 3)
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
            str(step.parameters.get("phi_cl_source"))
            for step in porosity_steps
            if step.parameters.get("phi_cl_source")
        )
        phi_cl_basis_count = _format_numeric_summary(_step_parameter_values(porosity_steps, "phi_cl_basis_count"), 0)
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
            counts_text = _phit_source_counts_text(porosity_steps[0].parameters if porosity_steps else {})
            details.append(
                "PHIT from the library's adaptive neutron-density combination: "
                "PHIT = PHID where PHIN is missing or PHID > PHIN + 0.05; "
                "PHIT = PHIN where PHID is missing; PHIT = 0.6*PHID + 0.4*PHIN for the remaining samples"
                + counts_text
                + "."
            )
        elif phit_methods == ["density_only"] or any("Density-only porosity" in decision for decision in decisions):
            counts_text = _phit_source_counts_text(porosity_steps[0].parameters if porosity_steps else {})
            details.append("PHIT = PHID because PHIN was unavailable" + counts_text + ".")
        elif phit_methods == ["neutron_only"] or any("Neutron-only porosity" in decision for decision in decisions):
            counts_text = _phit_source_counts_text(porosity_steps[0].parameters if porosity_steps else {})
            details.append("PHIT = PHIN because PHID was unavailable" + counts_text + ".")
        elif phit_methods == ["sonic_only"] or any("Sonic-only porosity" in decision for decision in decisions):
            counts_text = _phit_source_counts_text(porosity_steps[0].parameters if porosity_steps else {})
            details.append("PHIT = PHIS because PHIN and PHID were unavailable" + counts_text + ".")
        elif phit_methods == ["existing_phit"] or any("PHIT reused from existing well curve" in decision for decision in decisions):
            details.append("PHIT was taken directly from the existing PHIT curve in the well.")
        if any("PHIE =" in decision for decision in decisions):
            if phi_cl_values:
                phi_cl_source = _friendly_join([_phi_cl_source_label(source) for source in phi_cl_sources])
                phi_cl_fallback_reasons = _ordered_unique(
                    str(reason)
                    for reason in _step_parameter_values(porosity_steps, "phi_cl_fallback_reason")
                    if reason
                )
                details.append(
                    f"PHIE = selected porosity basis - VCL * phi_cl, with phi_cl={phi_cl_values} v/v "
                    f"from {phi_cl_source if phi_cl_source else 'available shale-porosity input'}"
                    + (f" using {phi_cl_basis_count} shale-control samples." if phi_cl_basis_count else "")
                    + (f" Fallback reason: {_friendly_join(phi_cl_fallback_reasons)}." if phi_cl_fallback_reasons else "")
                    + "."
                )
            else:
                details.append("PHIE was calculated from the selected porosity basis and VCL clay correction.")
        if well.get("PHI_SEC") is not None:
            details.append("PHI_SEC = max(PHIT - PHIS, 0).")
        if well.get("PHIE_CC") is not None:
            details.append("PHIE_CC was written separately where a valid core-calibration model existed.")
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
            if any(model in {"north_sea", "indonesia", "simandoux", "modified_simandoux", "nigeria_schlumberger"} for model in models) and rcl_used:
                note += f" RCL used in shaly-sand saturation calculations: {rcl_used}."
            rtsh_used = _format_numeric_summary(_step_parameter_values(saturation_steps, "rtsh"), 4, "ohm.m")
            if any(model in {"north_sea", "indonesia", "simandoux", "modified_simandoux", "nigeria_schlumberger"} for model in models) and rtsh_used:
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
                equations.append("Timur equation: k = 0.136 * (100 * PHIE)^4.4 / (100 * Swirr)^2")
            if "coates" in empirical:
                equations.append("Coates-Denoo equation: k = 10000 * PHIE^4 * ((1 - Swirr) / Swirr)^2")
            if "wyllie_rose" in empirical:
                equations.append("Wyllie-Rose equation (Morris-Biggs oil): k = 62500 * PHIE^6 / Swirr^2")
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
        payflag_curve = well.get("NET_PAY")
        payflag_count = int(np.nansum(_normalize_numeric_array(payflag_curve.data) > 0.5)) if payflag_curve is not None else 0
        seal_count = int(np.nansum(_normalize_numeric_array(well.get("SEALFLAG").data) > 0.5)) if well.get("SEALFLAG") is not None else 0
        note = (
            "WORKFLOW FLAGS: NET_ROCK is defined where VCL is below the clean-rock cutoff, "
            "RESFLAG / NET_RES additionally require selected effective porosity above the reservoir cut-off, and "
            "NET_PAY additionally requires water saturation below the pay cut-off."
        )
        if vcl_cutoff and phie_cutoff and sw_cutoff:
            note += f" Cutoffs used in this well: VCL < {vcl_cutoff}, selected effective porosity > {phie_cutoff}, SW < {sw_cutoff}."
        if seal_cutoff:
            note += f" SEALFLAG = 1 where VCL >= {seal_cutoff}."
        note += (
            f" Positive sample counts in this well: NET_ROCK={net_rock_count}, RESFLAG={resflag_count}, "
            f"NET_PAY={payflag_count}, SEALFLAG={seal_count}."
        )
        if well.get("CROSSOVER_FLAG") is not None:
            note += " CROSSOVER_FLAG = 1 where PHID - PHIN > 0.05, VCL < 0.35, selected effective porosity > 0.05, and BAD_HOLE < 0.5."
        if _has_finite_curve(well, "VP_VS_RATIO"):
            note += " VP_VS_RATIO = DTS / DTC."
        if well.get("MHI") is not None:
            diagnostic = _segment_diagnostic_sxo_count(well)
            fallback = _segment_sxo_method_counts(well).get("fifth_root_quicklook", 0)
            note += f" MHI = SW / SXO only where SXO had invaded-zone support ({diagnostic} samples)."
            if fallback:
                note += f" Fallback-SXO samples were left as NaN for MHI ({fallback} samples)."
        notes.append(note)

    invasion_steps = _all_step_results(workflow_result, "invasion")
    if well.get("INVASION_RATIO") is not None:
        ratio_details = _invasion_ratio_rejection_details(well, workflow_result)
        note = (
            "INVASION: INVASION_RATIO = flushed/shallow resistivity / deep resistivity "
            "(RXO/RSHA/RESS; RDEP preferred, then RT). RMED/RDEP is used only as a fallback proxy indicator. "
            "Values are written as NaN where inputs are non-finite, flushed/shallow or fallback resistivity is non-positive, "
            "deep resistivity is non-positive, or the raw ratio exceeds 100."
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
                reasons.append(f"flushed/shallow or fallback resistivity <= 0={ratio_details['nonpositive_rxo_samples']}")
            if ratio_details["rt_below_min_samples"]:
                reasons.append(f"deep resistivity <= 0 ohm.m={ratio_details['rt_below_min_samples']}")
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


def _context_hydrocarbon_analysis(
    well: "Well",
    workflow_result,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> CategoryAnalysisResult:
    """The hydrocarbon analysis of a context well, with the profile's class tables
    and, under "current", each sample's lithology family."""
    class_tables = _context_class_tables(workflow_result)
    options: dict[str, Any] = {"class_tables": class_tables}
    if class_tables == "current":
        options["lithology_family"] = _context_lithology_families(well, workflow_result, lithology_mapping)
    return HydrocarbonAnalysis(**options).analyze(well, add_curves=False)


def _build_analysis_summary(
    well: "Well",
    workflow_result=None,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    hydrocarbon = _context_hydrocarbon_analysis(well, workflow_result, lithology_mapping)
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


def _formation_file_key(display_name: str, benchmark: bool) -> str:
    """The key under which formations of one well share a folder and file stem.

    Names such as "LKC/G" and "LKC_G" sanitize to the same path (and Windows
    paths ignore case), so the current calculations count them as one name
    and give each a depth suffix; the benchmark packages counted raw names.
    """
    if benchmark:
        return display_name
    return _sanitize_path_component(display_name).casefold()


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
            value = float(row[abbreviation_col])
            code = int(round(value))
        except (TypeError, ValueError):
            continue
        entry = {
            "attribute": str(row[attribute_col]).strip(),
            "lithology_attribute": str(row[lithology_attr_col]).strip(),
        }
        if not reproducing_benchmark():
            # A fractional code or a code given twice would silently rename groups.
            if value != code:
                raise ValueError(f"Lithology code {value:g} in {mapping_path.name} is not an integer")
            if code in mapping and mapping[code] != entry:
                raise ValueError(f"Lithology code {code} is defined twice in {mapping_path.name}")
        mapping[code] = entry
    return mapping


def _strip_top_suffix(name: str) -> str:
    text = str(name or "").strip()
    if text.lower().endswith(" top"):
        return text[:-4].rstrip()
    return text


def _load_excel_formation_segments(
    tops_file: str | Path,
    *, rank: Optional[str] = None, sheet_name=None, columns=None, sep=None,
) -> dict[str, list[dict[str, Any]]]:
    from petrophys.io.geology_reader import CONTEXT_COLUMNS, _key, _read_table, read_formations

    dataframe = _read_table(tops_file, sheet_name=sheet_name, sep=sep)
    normalized_columns = {_key(column): column for column in dataframe.columns}
    if len(normalized_columns) != len(dataframe.columns):
        raise ValueError("Duplicate formation column names")
    well_col = (columns or {}).get("well") or next(
        (normalized_columns[name] for name in CONTEXT_COLUMNS["well"] if name in normalized_columns), None)
    single_well_key = "__SINGLE_WELL_TOPS__"
    if well_col is None:
        well_col = "__petrophys_well__"
        dataframe[well_col] = single_well_key

    def _is_formation_unit(name: str) -> bool:
        text = str(name).upper()
        return " FM." in text or " GP." in text or " MB." in text

    segments_by_well: dict[str, list[dict[str, Any]]] = {}
    if dataframe[well_col].isna().any() or dataframe[well_col].astype(str).str.strip().eq("").any():
        raise ValueError("Formation table contains missing well identifiers")
    if not reproducing_benchmark():
        spellings: dict[str, set[str]] = {}
        for name in dataframe[well_col].astype(str).unique():
            spellings.setdefault(_normalize_tops_well_name(name), set()).add(name)
        clashes = sorted(sorted(names) for names in spellings.values() if len(names) > 1)
        if clashes:
            # One well's picks would be stored under one name and overwrite each other.
            raise ValueError(
                "Formation table spells one well in several ways: "
                + "; ".join(", ".join(repr(name) for name in names) for names in clashes)
                + ". Use one well name per well."
            )
    for well_name, raw_group in dataframe.groupby(well_col, sort=False):
        read_columns = dict(columns or {})
        read_columns["well"] = well_col
        group = read_formations(raw_group, columns=read_columns, well_name=str(well_name))
        report = group.attrs["import_report"]
        formation_col = next(key for key, value in report["column_mapping"].items() if value == "formation_name")
        top_col = next(key for key, value in report["column_mapping"].items() if value == "top_depth")
        display_names = {(_strip_top_suffix(str(row[formation_col])), float(row[top_col])): str(row[formation_col]).strip().upper()
                         for row in raw_group.to_dict("records")}
        if rank is not None:
            if "rank" not in group:
                raise ValueError("Supply stratigraphic rank metadata before selecting a tops rank")
            group = group.loc[group["rank"].astype(str).str.lower() == rank.lower()].copy()
            if group.empty:
                raise ValueError(f"No formation picks for rank={rank!r} in well {well_name!r}")
        group["_formation_name"] = group["formation_name"]
        group["_formation_display_name"] = [display_names.get((row.formation_name, row.top_depth), row.formation_name.upper())
                                            for row in group.itertuples()]
        group["_md_top"] = group["top_depth"]
        group = group.sort_values(["_md_top", "_formation_display_name"], kind="stable")
        explicit_units = group["_formation_display_name"].map(_is_formation_unit)
        if explicit_units.any():
            group = group[explicit_units]

        unique_tops = sorted(float(value) for value in group["_md_top"].dropna().unique())
        next_top_by_top = {
            unique_tops[index]: (unique_tops[index + 1] if index + 1 < len(unique_tops) else None)
            for index in range(len(unique_tops))
        }

        segments: list[dict[str, Any]] = []
        seen: set[tuple[str, float]] = set()
        for row in group.to_dict("records"):
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
                    "bottom_depth": row.get("bottom_depth", next_top_by_top.get(top_depth)),
                    **{key: group.attrs[key] for key in ("depth_unit", "depth_reference", "datum") if group.attrs.get(key)},
                    **{key: row[key] for key in ("rank", "parent") if row.get(key)},
                    # The pick as read_formations reads it, for attaching the tops to the well.
                    "record": {
                        key: value for key, value in row.items()
                        if not str(key).startswith("_") and key not in ("well", well_col)
                    },
                }
            )

        segments_by_well[_normalize_tops_well_name(str(well_name))] = segments

    return segments_by_well


def _formation_segments_for_well(
    source_file: Path,
    well: "Well",
    tops_by_well: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    from petrophys.model.units import normalize_depth_unit

    def aligned(segments):
        converted = []
        for segment in segments:
            item = dict(segment)
            for field in ("depth_reference", "datum"):
                if item.get(field) and str(item[field]).upper() != str(well.get_parameter(field) or "").upper():
                    raise ValueError(f"Formation {field} does not match the well; align depth coordinates before export")
            if item.get("depth_unit"):
                source_unit = normalize_depth_unit(item["depth_unit"])
                target_unit = normalize_depth_unit(well.depth_unit)
                factor = 1.0 if source_unit == target_unit else (0.3048 if source_unit == "ft" else 1.0 / 0.3048)
                for field in ("top_depth", "bottom_depth"):
                    if item.get(field) is not None:
                        item[field] *= factor
                item["depth_unit"] = target_unit
            converted.append(item)
        return converted

    single_well_segments = tops_by_well.get(
        _normalize_tops_well_name("__SINGLE_WELL_TOPS__")
    )
    if single_well_segments is not None:
        return aligned(single_well_segments)

    candidates = [
        _normalize_tops_well_name(source_file.stem),
        _normalize_tops_well_name(well.name),
        _normalize_tops_well_name(getattr(well.header, "well_name", "")),
    ]
    for candidate in candidates:
        if candidate and candidate in tops_by_well:
            return aligned(tops_by_well[candidate])

    prefix_candidates: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        base_candidate = candidate.split(" ", 1)[0].strip()
        if not base_candidate:
            continue
        matches = [name for name in tops_by_well if name.startswith(base_candidate)]
        if not reproducing_benchmark():
            # "15/9-1" names a different well than "15/9-19 S" or "15/9-1A":
            # the tops name must continue after a separator.
            matches = [
                name for name in matches
                if len(name) == len(base_candidate) or not name[len(base_candidate)].isalnum()
            ]
        if len(matches) == 1:
            prefix_candidates.append(matches[0])

    unique_prefix_candidates = sorted(set(prefix_candidates))
    if len(unique_prefix_candidates) == 1:
        return aligned(tops_by_well[unique_prefix_candidates[0]])
    return []


def _validate_single_well_tops_scope(
    tops_by_well: dict[str, list[dict[str, Any]]],
    source_files: Sequence[Path],
) -> None:
    single_well_key = _normalize_tops_well_name("__SINGLE_WELL_TOPS__")
    if single_well_key in tops_by_well and len(source_files) > 1:
        raise ValueError(
            "Formation tops table has no 'well' column; select one LAS file "
            "or add a well column before running a multiwell export."
        )


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
    from petrophys.model.well import Well

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
    lithology_curve = _context_lithology_curve(well)
    if lithology_curve is None:
        labels = np.asarray([MISSING_LITHOLOGY_LABEL] * well.n_samples, dtype=object)
    else:
        labels_list = []
        resolved: dict[str, str] = {}
        for value in lithology_curve.data:
            key = str(value)
            if key not in resolved:
                if _is_null_like_lithology(value) or _is_sentinel_lithology(value):
                    resolved[key] = MISSING_LITHOLOGY_LABEL
                else:
                    resolved[key] = (
                        _normalize_lithology_label(value, lithology_mapping=lithology_mapping)
                        or MISSING_LITHOLOGY_LABEL
                    )
            labels_list.append(resolved[key])
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
        entry = lithology_mapping[numeric]
        if entry is None:
            # The run's calculation profile records no lithology for this code.
            return None
        mapped = entry.get("lithology_attribute") or entry.get("attribute") or text
        return _display_lithology_text(mapped)

    try:
        from petrophys.evaluation.diagnostics.lithology_analysis import (
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


def _context_lithology_curve(well: "Well"):
    """The lithology code curve, or the lithology intervals the interpretation selected.

    The code curve is the one the interpretation read: ``LITHOLOGY_GEOLINK``,
    or another lithology-code curve the workflow recorded on the well (see
    :func:`~petrophys.interpretation.lithology.matrix_inference.interpreted_lithology_code_curve`).
    Interval labels are returned as text (NaN where the intervals leave a gap),
    in an object with the ``data`` and ``metadata`` of a curve.
    """
    selected = (well.get_parameter("_geology_context") or {}).get("selected_lithology_source")
    external = well.get("LITHO_CLASS") if selected == "external" else None
    if external is None:
        code_curve = well.get(interpreted_lithology_code_curve(well))
        # Only "current" runs record their code curve; the benchmark profiles keep their groups.
        if code_curve is None and selected == "computed" and _recorded_code_curve(well)[0]:
            return _computed_lithology_labels(well)
        return code_curve
    categories = external.metadata.custom.get("category_labels", {})
    labels = []
    for value in np.asarray(external.data, dtype=object):
        label = None
        if not _is_null_like_lithology(value):
            label = categories.get(value, categories.get(str(value)))
        labels.append(np.nan if label is None else str(label))
    return SimpleNamespace(data=np.asarray(labels, dtype=object), metadata=external.metadata)


#: ``metadata.custom["lithology_source"]`` of the lithology the interpretation
#: computed from the logs, used where a well has no input lithology.
COMPUTED_LITHOLOGY_SOURCE = "computed"


def _computed_lithology_labels(well: "Well"):
    """The per-sample lithology the interpretation computed (LITHO), as text labels.

    Used when the well has no input lithology (no lithology-code curve and no
    lithology intervals). Unclassified samples (code 0) are NaN. None when the
    run wrote no LITHO curve.
    """
    from petrophys.interpretation.lithology.lithology_module import LithologyModule

    curve = well.get("LITHO")
    if curve is None:
        return None
    names = {code: name for name, code in LithologyModule.LITHOLOGY_CODES.items() if code != 0}
    labels = []
    for value in np.asarray(curve.data, dtype=float):
        name = names.get(int(round(value))) if np.isfinite(value) else None
        labels.append(np.nan if name is None else name)
    metadata = SimpleNamespace(custom={"lithology_source": COMPUTED_LITHOLOGY_SOURCE})
    return SimpleNamespace(data=np.asarray(labels, dtype=object), metadata=metadata)


_COMPUTED_LITHOLOGY_NOTE = (
    "Lithology source: computed from the logs by the interpretation (LITHO); "
    "no input lithology was supplied."
)


def _computed_lithology(curve) -> bool:
    """Whether ``curve`` (from :func:`_context_lithology_curve`) is the computed lithology."""
    custom = getattr(getattr(curve, "metadata", None), "custom", None) or {}
    return custom.get("lithology_source") == COMPUTED_LITHOLOGY_SOURCE


def _valid_lithology_labels(
    well: "Well",
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
) -> List[str]:
    curve = _context_lithology_curve(well)
    if curve is None:
        return []
    labels: List[str] = []
    resolved: dict[str, Optional[str]] = {}
    for value in np.asarray(curve.data, dtype=object):
        key = str(value)
        if key not in resolved:
            resolved[key] = _normalize_lithology_label(value, lithology_mapping=lithology_mapping)
        label = resolved[key]
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
    curve = _context_lithology_curve(well)
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
    resolved: dict[str, tuple[str, Optional[str]]] = {}
    for value in values:
        key = str(value)
        if key not in resolved:
            if _is_null_like_lithology(value):
                resolved[key] = ("nan", None)
            elif _is_sentinel_lithology(value):
                resolved[key] = ("sentinel", None)
            else:
                label = _normalize_lithology_label(value, lithology_mapping=lithology_mapping)
                # None: a code the run's calculation profile reads as no lithology.
                resolved[key] = ("nan", None) if label is None else ("label", label)
        kind, label = resolved[key]
        if kind == "nan":
            nan_count += 1
        elif kind == "sentinel":
            sentinel_count += 1
        else:
            valid_labels.append(label)

    valid_count = len(valid_labels)
    unique_lithologies = sorted(_ordered_unique(valid_labels))
    source = {"source": "computed from logs"} if _computed_lithology(curve) else {}
    return {
        **source,
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


def _round_stat(value: float, decimals: int = 4) -> int | float:
    rounded = round(float(value), decimals)
    return int(rounded) if float(rounded).is_integer() else rounded


#: Curve statistics by array content: the six category files of a group share
#: their curves, so each curve's statistics are computed once.
_CURVE_STATISTICS_CACHE: "OrderedDict[tuple, Optional[dict[str, float]]]" = OrderedDict()
_CURVE_STATISTICS_CACHE_SIZE = 4096


def _curve_statistics_payload(data: np.ndarray) -> Optional[dict[str, float]]:
    from petrophys.evaluation.summaries.statistics import STATISTICS, curve_statistics

    values = _normalize_numeric_array(data)
    key = (values.shape, hashlib.blake2b(np.ascontiguousarray(values).tobytes(), digest_size=16).digest())
    if key in _CURVE_STATISTICS_CACHE:
        _CURVE_STATISTICS_CACHE.move_to_end(key)
        cached = _CURVE_STATISTICS_CACHE[key]
        return dict(cached) if cached is not None else None
    stats = curve_statistics(values)
    payload = None if stats is None else {name: _round_stat(stats[name]) for name in STATISTICS}
    _CURVE_STATISTICS_CACHE[key] = payload
    if len(_CURVE_STATISTICS_CACHE) > _CURVE_STATISTICS_CACHE_SIZE:
        _CURVE_STATISTICS_CACHE.popitem(last=False)
    return dict(payload) if payload is not None else None


def _reference_statistics_curve_key(mnemonic: str, well: "Well") -> Optional[str]:
    upper = str(mnemonic).upper()
    canonical = _canonical_delivery_curve_name(upper)
    if upper in {"DEPT", "DEPTH"}:
        return None
    if canonical == "BADHOLEFLAG":
        return None
    if canonical in CORE_ONLY_CONTEXT_CURVES:
        return None
    if upper == "LITHOLOGY_GEOLINK" or upper == interpreted_lithology_code_curve(well):
        return None
    if upper == "LITHO" and _recorded_code_curve(well)[0]:
        # A lithology code; its distribution is reported, not its statistics.
        return None
    if upper == "DCAL" and well.get("CALI") is not None:
        return None
    if upper == "DT" and well.get("DTC") is not None:
        return None
    return canonical


def _add_reference_statistics_payload(
    stats: dict[str, Any],
    key: str,
    data: Any,
    *,
    prefer_existing: bool = True,
) -> None:
    try:
        payload = _curve_statistics_payload(data)
    except Exception:
        payload = None
    if payload is None:
        return
    stat_key = f"{key}_overall"
    if prefer_existing and stat_key in stats:
        return
    stats[stat_key] = payload


def _dataset_context_value_range_rules() -> dict[str, ValueRangeRule]:
    """The context value-range rules of the Danish and COSTA packages.

    Those packages were written after the GeoLink main run's context: RMF is
    kept down to 0.001 ohm.m and SO is clipped to 0-1 (after SW, in the
    order of the rules then).
    """
    rules: dict[str, ValueRangeRule] = {}
    for mnemonic, rule in _CONTEXT_VALUE_RANGE_RULES.items():
        if mnemonic == "RMF":
            rule = ValueRangeRule(0.001, 10.0, "nan", ("RMF",))
        rules[mnemonic] = rule
        if mnemonic == "SW":
            rules["SO"] = ValueRangeRule(0.0, 1.0, "clip", ("SO",))
    return rules


_DATASET_CONTEXT_VALUE_RANGE_RULES = _dataset_context_value_range_rules()


def _context_value_range_rules(workflow_result) -> dict[str, ValueRangeRule]:
    """Value-range rules for a context of ``workflow_result``'s run."""
    if _context_dataset() is not None and _reproduces_benchmark_context(workflow_result):
        return _DATASET_CONTEXT_VALUE_RANGE_RULES
    return _CONTEXT_VALUE_RANGE_RULES


def _apply_value_range_qc_to_extra_curves(
    extra_derived_curves: Optional[dict[str, Any]],
    rules: Optional[dict[str, ValueRangeRule]] = None,
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for mnemonic, curve in (extra_derived_curves or {}).items():
        try:
            data = getattr(curve, "data", curve)
            cleaned, item = clean_curve_values(
                mnemonic, data, rules=_CONTEXT_VALUE_RANGE_RULES if rules is None else rules
            )
        except (TypeError, ValueError):
            continue
        if item is None:
            continue
        if item["changed"] > 0 and hasattr(curve, "data"):
            curve.data = cleaned
        report.append(item)
    return report


def _has_finite_extra_curve(curve: Any) -> bool:
    try:
        data = _normalize_numeric_array(getattr(curve, "data", curve))
    except Exception:
        return False
    return bool(np.any(np.isfinite(data)))


def _build_reference_statistics(
    well: "Well",
    extra_derived_curves: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for mnemonic in well.curves:
        key = _reference_statistics_curve_key(mnemonic, well)
        if key is None:
            continue
        curve = well.get(mnemonic)
        if curve is None:
            continue
        prefer_existing = str(mnemonic).upper() != key
        data = curve.data
        if key == "TEMP":
            data = _convert_temperature_to_fahrenheit(curve.data, curve.unit)
        _add_reference_statistics_payload(
            stats,
            key,
            data,
            prefer_existing=prefer_existing,
        )

    for mnemonic, curve in (extra_derived_curves or {}).items():
        key = _reference_statistics_curve_key(mnemonic, well)
        if key is None:
            continue
        data = getattr(curve, "data", curve)
        _add_reference_statistics_payload(stats, key, data, prefer_existing=True)
        if key == "COMPOSITE_CLAY":
            _add_reference_statistics_payload(
                stats,
                "COMPOSITE_CLAY_INDICATOR",
                data,
                prefer_existing=True,
            )
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
    no_bit_size = details.get("no_bit_size_points", 0)
    not_assessed = details.get("not_assessed_points", 0)
    drho_text = f"|DRHO| > {thresholds['drho_washout_threshold_g_per_cc']:.2f} g/cc"
    if not_assessed and not_assessed == counts["total_points"]:
        return {
            "total_points": counts["total_points"],
            "bad_hole_points": 0,
            "bad_hole_percentage": 0.0,
            "bad_hole_sequences": [],
            "log_availability": details["log_availability"],
            "qc_method": "washout_or_large_hole",
            "summary_text": (
                "Bad-hole screening could not be evaluated: no DRHO reading and no caliper "
                "reading with a bit size (BS curve or header)."
            ),
            "parameters_used": {},
            "no_bit_size_points": no_bit_size,
            "not_assessed_points": not_assessed,
        }
    summary = (
        "Bad-hole screening uses washout OR large-hole logic: "
        f"washout={(counts['washout'])}, large_hole={counts['large_hole']}, "
        f"final BAD_HOLE={counts['bad_hole']} of "
        f"{counts['total_points']} samples ({details['bad_hole_percentage']:.2f}%)."
    )
    if no_bit_size:
        summary += (
            f" No bit size (BS curve or header) for {no_bit_size} samples: washout there is "
            f"{drho_text} alone."
        )
    if not_assessed:
        summary += (
            f" {not_assessed} samples have neither a DRHO reading nor a caliper reading with "
            "a bit size and are not assessed."
        )
    sequences = details["bad_hole_intervals"]
    if not _reproduces_benchmark_context(workflow_result):
        sequences = [f"{interval} {well.depth_unit}" for interval in sequences]
    qc = {
        "total_points": counts["total_points"],
        "bad_hole_points": counts["bad_hole"],
        "bad_hole_percentage": details["bad_hole_percentage"],
        "bad_hole_sequences": sequences,
        "log_availability": details["log_availability"],
        "qc_method": "washout_or_large_hole",
        "summary_text": summary,
        "parameters_used": {
            "bit_size_used_in": _json_safe(details["bit_size_used"]),
            "washout_formula": (
                f"(CALI - BS) > {thresholds['cali_minus_bs_washout_threshold_in']:.2f} in"
                + (
                    f" OR |DRHO| > {thresholds['drho_washout_threshold_g_per_cc']:.2f} g/cc"
                    if details["log_availability"].get("DRHO")
                    else ""
                )
                + (f"; {drho_text} alone where no bit size is known" if no_bit_size else "")
            ),
            "large_hole_formula": f"BS > {thresholds['large_hole_bit_size_threshold_in']:.2f} in",
            "pef_exclusion_formula": (
                f"PEF < {thresholds['pef_exclusion_min']:.2f} or "
                f"PEF > {thresholds['pef_exclusion_max']:.2f}"
            ),
            "combined_logic": "BAD_HOLE = washout OR large_hole",
        },
    }
    if "no_bit_size_points" in details:
        qc["no_bit_size_points"] = no_bit_size
        qc["not_assessed_points"] = not_assessed
    return qc


def _set_or_add_curve(
    well: "Well",
    mnemonic: str,
    data: np.ndarray,
    unit: str,
    description: str,
) -> None:
    curve = well.get(mnemonic) if hasattr(well, "get") else None
    if curve is not None:
        curve._data = np.asarray(data, dtype=float)
        return
    well.add_curve(mnemonic, np.asarray(data, dtype=float), unit, description)


def _synchronize_post_qc_dependent_curves(well: "Well", *, benchmark: bool = False) -> None:
    """Recompute simple dependent curves after segment value-range QC.

    Context JSONs are built from already-processed LAS files, so a segment-level
    QC pass can NaN PHIE_CC/PHIE/PHIT/PERM_CC/PERM samples after BVW/RQI/FZI were originally
    written. Recomputing these direct formula curves keeps statistics aligned
    with the cleaned segment inputs.

    PHI_NORM, RQI and FZI are recomputed from the post-QC porosity (PHIE_CC,
    then PHIE, then PHIT) and permeability (PERM_CC, then PERM), and are NaN
    wherever an input is missing, also in a segment that QC left without any
    porosity or permeability; a curve the well lacks is added only when both
    inputs remain. PHI_Z is overwritten with the recomputed PHI_NORM, and BVW
    is NaN wherever no porosity remains, so no value computed from a rejected
    sample reaches the statistics, the category analyzers or
    DATA_COMPLETENESS.

    With ``benchmark`` (the eval package's calculations), a segment left
    without porosity or without permeability by QC keeps its PHI_NORM, RQI and
    FZI, and its BVW too when no porosity is left; PHI_Z always keeps the
    workflow's values.
    """

    phie, _, _ = get_curve_data_depthwise(well, ["PHIE_CC", "PHIE", "PHIT"])
    sw, _, _ = get_curve_data_depthwise(well, ["SW", "SWA", "SWT-LOG"])
    if phie is not None and sw is not None:
        finite = np.isfinite(phie) & np.isfinite(sw)
        bvw = np.full_like(phie, np.nan, dtype=float)
        bvw[finite] = phie[finite] * sw[finite]
        _set_or_add_curve(well, "BVW", bvw, "v/v", "Bulk Volume Water")
    elif not benchmark and well.get("BVW") is not None:
        bvw = np.array(well.get("BVW").data, dtype=float)
        porosity_left = np.isfinite(phie) if phie is not None else np.zeros(bvw.shape, dtype=bool)
        bvw[~porosity_left] = np.nan
        _set_or_add_curve(well, "BVW", bvw, "v/v", "Bulk Volume Water")

    perm, _, _ = get_curve_data_depthwise(well, ["PERM_CC", "PERM"])
    both_left = phie is not None and perm is not None
    if benchmark and not both_left:
        return
    if not both_left:
        missing = np.full(well.n_samples, np.nan, dtype=float)
        phie = missing if phie is None else phie
        perm = missing if perm is None else perm

    phi_norm = np.full_like(phie, np.nan, dtype=float)
    phi_norm_mask = np.isfinite(phie) & (phie > 0.0) & (phie < 1.0)
    phi_norm[phi_norm_mask] = phie[phi_norm_mask] / (1.0 - phie[phi_norm_mask])

    rqi = np.full_like(phie, np.nan, dtype=float)
    rqi_mask = np.isfinite(perm) & np.isfinite(phie) & (perm > 0.0) & (phie > 0.0)
    rqi[rqi_mask] = 0.0314 * np.sqrt(perm[rqi_mask] / phie[rqi_mask])

    fzi = np.full_like(phie, np.nan, dtype=float)
    fzi_mask = np.isfinite(rqi) & np.isfinite(phi_norm) & (phi_norm > 0.0)
    fzi[fzi_mask] = rqi[fzi_mask] / phi_norm[fzi_mask]

    for mnemonic, data, unit, description in (
        ("PHI_NORM", phi_norm, "fraction", "Normalized Porosity"),
        ("RQI", rqi, "um", "Reservoir Quality Index"),
        ("FZI", fzi, "um", "Flow Zone Indicator"),
    ):
        if both_left or well.get(mnemonic) is not None:
            _set_or_add_curve(well, mnemonic, data, unit, description)
    if not benchmark and well.get("PHI_Z") is not None:
        _set_or_add_curve(well, "PHI_Z", phi_norm.copy(), "fraction", "Normalized Porosity")


#: Curves the GeoLink training package's context neither lists as input or
#: derived logs nor gives statistics: the bit size and the oil saturation.
_GEOLINK_TRAIN_UNLISTED_CURVES = ("BS", "SO")


def _geolink_train_bad_hole_qc(qc: dict[str, Any], well: "Well") -> dict[str, Any]:
    """BADHOLE_QC_CHECK as the GeoLink training package wrote it.

    The bit size is named in words ("caliper-minus-bit-size", "bit size");
    without a bit size the caliper and large-hole tests are written as
    skipped, with a null ``bit_size_source``; and a segment without caliper
    or DRHO reports BS available where the well has a BS curve.
    """
    params = qc.get("parameters_used") or {}
    if not params:
        qc["log_availability"] = {**qc.get("log_availability", {}), "BS": well.get("BS") is not None}
        return qc
    no_bit_size = params.get("bit_size_used_in") is None
    washout = str(params.get("washout_formula", ""))
    if no_bit_size:
        washout = re.sub(
            r"^\(CALI - BS\) > [0-9.]+ in", "caliper-minus-bit-size skipped because bit size unavailable", washout
        )
        large_hole = "skipped because bit size unavailable"
    else:
        washout = washout.replace("(CALI - BS)", "(caliper-minus-bit-size)")
        large_hole = str(params.get("large_hole_formula", "")).replace("BS > ", "bit size > ")
    presented: dict[str, Any] = {}
    for key, value in params.items():
        presented[key] = value
        if key == "bit_size_used_in" and no_bit_size:
            presented["bit_size_source"] = None
    presented["washout_formula"] = washout
    presented["large_hole_formula"] = large_hole
    qc["parameters_used"] = presented
    return qc


def _geolink_train_bad_hole_note(note: str, qc: Mapping[str, Any]) -> str:
    """The BAD_HOLE note in the GeoLink training package's bit-size wording."""
    if not note.startswith("BAD_HOLE: washout=(CALI-BS)>"):
        return note
    params = qc.get("parameters_used") or {}
    if params and params.get("bit_size_used_in") is None:
        drho = re.search(r" or \|DRHO\|>[0-9.]+ g/cc", note)
        head = "BAD_HOLE: washout=caliper-minus-bit-size skipped because bit size unavailable"
        if drho and qc.get("log_availability", {}).get("DRHO"):
            head += drho.group(0)
        return head + "; large_hole skipped because bit size unavailable; rule=" + note.split("; rule=", 1)[1]
    return (
        note.replace("washout=(CALI-BS)>", "washout=(caliper-minus-bit-size)>")
        .replace("large_hole=BS>", "large_hole=bit_size>")
        .replace("; BS values=", "; bit-size values=")
        .replace("; BS=", "; bit size=")
    )


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
    status_targets_override: Optional[Sequence[str]] = None,
    include_derived_log_status: bool = False,
    extra_statistics_curves: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    well_name = _reference_well_name(source_file, well)
    context_rules = _context_value_range_rules(workflow_result)
    extra_qc_report = _apply_value_range_qc_to_extra_curves(extra_statistics_curves, context_rules)
    if hasattr(well, "get_parameter") and well.get_parameter(VALUE_RANGE_QC_PARAMETER) is None:
        apply_value_range_qc(well, rules=context_rules)
    benchmark = _reproduces_benchmark_context(workflow_result)
    if hasattr(well, "get_parameter") and hasattr(well, "set_parameter"):
        base_report = list(well.get_parameter(VALUE_RANGE_QC_PARAMETER, []) or [])
        if not benchmark:
            # Keep the well's own curves: extra-curve items of an earlier
            # context built from this well belong to that context's notes.
            base_report = [item for item in base_report if well.get(str(item.get("curve", ""))) is not None]
        if extra_qc_report or not benchmark:
            well.set_parameter(VALUE_RANGE_QC_PARAMETER, [*base_report, *extra_qc_report])
    _synchronize_post_qc_dependent_curves(well, benchmark=benchmark)
    depth_info = _depth_summary(well)
    formation_info = formation_depth_info or depth_info
    lithology_mapping = _run_lithology_mapping(lithology_mapping, workflow_result)
    lithology_info = _build_reference_lithology_info(well, lithology_mapping=lithology_mapping)
    dominant_lithology = _dominant_lithology(well, lithology_mapping=lithology_mapping)
    derived_logs = list(derived_logs_override) if derived_logs_override is not None else _derived_logs_computed(well, workflow_result, original_curves)
    if derived_logs_override is not None:
        derived_logs = [
            name for name in derived_logs
            if _has_finite_any_curve(well, _target_curve_candidates(name))
            or _has_finite_extra_curve((extra_statistics_curves or {}).get(name))
        ]
    train = _context_dataset() == "geolink_train"
    if train:
        derived_logs = [name for name in derived_logs if name not in _GEOLINK_TRAIN_UNLISTED_CURVES]
    status_targets = list(
        _ordered_unique(
            [
                *(status_targets_override or []),
                *derived_logs,
            ]
        )
    )
    computation_notes = _build_delivery_computation_notes(
        well,
        workflow_result,
        original_curves=original_curves,
        lithology_mapping=lithology_mapping,
        derived_logs_override=derived_logs,
        status_targets_override=status_targets,
    )
    derived_log_set = set(derived_logs)
    core_phi_used = "PHIE_CC" in derived_log_set and _has_finite_any_curve(well, _target_curve_candidates("PHIE_CC"))
    core_k_used = "PERM_CC" in derived_log_set and _has_finite_any_curve(well, _target_curve_candidates("PERM_CC"))

    hydrocarbon_summary = _context_hydrocarbon_analysis(well, workflow_result, lithology_mapping).to_dict()
    full_analysis_summary = _build_analysis_summary(well, workflow_result, lithology_mapping)
    selected_analysis_summary = analysis_summary_override or full_analysis_summary or hydrocarbon_summary
    data_completeness = {
        "missing_primary_logs": [
            _display_curve_name(name)
            for name in PRIMARY_LOGS_FOR_COMPLETENESS
            if _canonical_delivery_curve_name(name) not in CORE_ONLY_CONTEXT_CURVES
            and not _primary_log_present(well, workflow_result, name, original_curves)
            and not (train and name in _GEOLINK_TRAIN_UNLISTED_CURVES)
        ],
        "derived_log_computation_notes": computation_notes,
    }
    if include_derived_log_status:
        data_completeness["derived_log_status"] = _build_reference_derived_log_status(
            well,
            workflow_result,
            original_curves,
            extra_targets=status_targets,
            computed_curve_names=derived_logs,
        )

    payload = {
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
            "ground_truth_lithology_present": (
                lithology_info["has_valid_lithology"] and "source" not in lithology_info
            ),
            "core_phi_present": core_phi_used,
            "core_k_present": core_k_used,
            "derived_logs_computed": derived_logs,
            "lithology_distribution": _build_lithology_distribution(well, lithology_mapping=lithology_mapping),
            "statistics": _build_reference_statistics(well, extra_statistics_curves),
            "analysis_summary": selected_analysis_summary,
            "data_completeness": data_completeness,
            "lithology_info": lithology_info,
            "input_logs_present": _input_log_display_list(original_curves),
            "BADHOLE_QC_CHECK": _build_reference_bad_hole_qc(well, workflow_result),
        },
        "raw_log_traces_in_segment": {},
        "include_raw_traces_in_context": False,
    }
    if not benchmark:
        # Depths and thicknesses are in the well's depth unit, as the interval strings say.
        payload["formation_info"]["depth_unit"] = well.depth_unit
        payload["analysis_results"]["lithology_group_info"]["depth_unit"] = well.depth_unit
    if train:
        results = payload["analysis_results"]
        for name in _GEOLINK_TRAIN_UNLISTED_CURVES:
            results["statistics"].pop(f"{name}_overall", None)
        results["input_logs_present"] = [
            name for name in results["input_logs_present"] if name not in _GEOLINK_TRAIN_UNLISTED_CURVES
        ]
        qc = _geolink_train_bad_hole_qc(results["BADHOLE_QC_CHECK"], well)
        results["BADHOLE_QC_CHECK"] = qc
        data_completeness["derived_log_computation_notes"] = [
            _geolink_train_bad_hole_note(note, qc) for note in computation_notes
        ]
    return payload


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
    status_targets_override: Optional[Sequence[str]] = None,
    include_derived_log_status: bool = False,
    extra_statistics_curves: Optional[dict[str, Any]] = None,
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
        status_targets_override=status_targets_override,
        include_derived_log_status=include_derived_log_status,
        extra_statistics_curves=extra_statistics_curves,
    )
    return _write_payload(json_path, payload)


def _write_payload(json_path: Path, payload: dict[str, Any]) -> Path:
    """Write one context document as the packages store it."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)
    return json_path


def iter_context_groups(
    workflow_result,
    formation_segments,
    *,
    well_top: float,
    well_bottom: float,
    minimum_group_thickness: float = 2.0,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
    layout: Optional[Sequence[Mapping[str, Any]]] = None,
    fallback_warnings: Optional[list[str]] = None,
):
    """Yield the context groups of an interpreted well, as the context package splits it.

    Each formation of ``formation_segments`` (``name``, ``display_name``,
    ``top_depth``, ``bottom_depth``; the whole well without any) is split by
    its lithology into groups thicker than ``minimum_group_thickness``, or
    ``layout`` gives the groups. Yields dicts with the group's ``well`` (its
    samples of ``workflow_result.well``), ``formation_name``,
    ``formation_depth_info``, ``group_name``, ``lithology`` and ``file_stem``.
    """
    if layout is not None:
        for group in layout:
            group_well = workflow_result.well.depth_slice(float(group["start_depth"]), float(group["end_depth"]))
            if group_well.n_samples == 0:
                continue
            formation_display_name = str(group["formation_name"])
            group_name = str(group["group_name"])
            yield {
                "well": group_well,
                "formation_name": formation_display_name,
                "formation_depth_info": dict(group["formation_depth_info"]),
                "group_name": group_name,
                "lithology": str(group["lithology"]),
                "file_stem": str(group.get("file_stem") or f"{formation_display_name}_{group_name}"),
            }
        return

    if not formation_segments:
        formation_segments = [
            {
                "name": "WHOLE_WELL",
                "display_name": "WHOLE_WELL",
                "top_depth": well_top,
                "bottom_depth": None,
            }
        ]
        if fallback_warnings is not None:
            fallback_warnings.append(
                "No formation tops matched this well in the tops table. Falling back to WHOLE_WELL segmentation."
            )

    formation_name_counts: dict[str, int] = {}
    for segment in formation_segments:
        display_name = _formation_file_key(
            str(segment.get("display_name") or segment["name"]), reproducing_benchmark()
        )
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

        formation_depth_info = _depth_summary(formation_well)
        lithology_segments = _build_lithology_group_segments(
            formation_well,
            minimum_group_thickness=minimum_group_thickness,
            lithology_mapping=_run_lithology_mapping(lithology_mapping, workflow_result),
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
            if raw_lithology_name == MISSING_LITHOLOGY_LABEL:
                json_group_name = "No lithology information available"
                json_lithology_name = "No lithology information available"

            file_stem = f"{formation_display_name}_{file_group_name}"
            if formation_name_counts.get(
                _formation_file_key(formation_display_name, reproducing_benchmark()), 0
            ) > 1:
                file_stem = (
                    f"{file_stem}"
                    f"__{formation_depth_info['min_depth']:.1f}_{formation_depth_info['max_depth']:.1f}"
                )
            yield {
                "well": group_well,
                "formation_name": formation_display_name,
                "formation_depth_info": formation_depth_info,
                "group_name": json_group_name,
                "lithology": json_lithology_name,
                "file_stem": file_stem,
            }


def iter_context_group_payloads(
    group_well: "Well",
    workflow_result,
    source_file: Path,
    raw_curves,
    *,
    formation_name: str,
    formation_depth_info: Mapping[str, Any],
    group_name: str,
    lithology_name: str,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
):
    """Yield ``(category, document)`` for the six context files of one group, in the context's order.

    ``group_well`` is the group's samples of ``workflow_result.well`` (the
    interpretation before the context's QC) and is modified as the context
    modifies it: its value-range QC, then per category the category's
    analysis and the document built from it. Write or copy each document
    before taking the next.
    """
    apply_value_range_qc(group_well, rules=_context_value_range_rules(workflow_result))
    if not _reproduces_benchmark_context(workflow_result):
        # The category analyzers read RQI, FZI and BVW and count
        # every curve's samples: recompute them after the QC.
        _synchronize_post_qc_dependent_curves(group_well)

    # Every sample of the group shares its lithology, and with it the
    # lithology family whose reservoir-quality limits apply.
    group_families = (
        None
        if _reproduces_benchmark_context(workflow_result)
        else _context_lithology_families(group_well, workflow_result, lithology_mapping)
    )
    stem = Path(source_file).stem
    for category_name in CATEGORY_CONTEXT_SPECS:
        category_result = _category_analysis_result(
            category_name, group_well, workflow_result, lithology_family=group_families
        )
        derived_logs = _category_derived_logs_computed(
            category_name,
            group_well,
            workflow_result,
            raw_curves,
            category_result,
        )
        yield category_name, _build_reference_json_payload(
            group_well,
            workflow_result,
            Path(source_file),
            raw_curves,
            formation_name=formation_name,
            formation_depth_info=formation_depth_info,
            lithology_group_name=group_name,
            lithology_name=lithology_name,
            lithology_mapping=lithology_mapping,
            analysis_summary_override=category_result.to_dict(),
            derived_logs_override=derived_logs,
            status_targets_override=_category_status_targets(category_name, stem),
            include_derived_log_status=False,
            extra_statistics_curves=category_result.derived_curves,
        )


def _trace_safe_value(value: Any) -> Any:
    """Return compact JSON-safe trace metadata, including ndarray summaries."""
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        payload: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(arr.shape),
        }
        if np.issubdtype(arr.dtype, np.number):
            numeric = _normalize_numeric_array(arr)
            finite = numeric[np.isfinite(numeric)]
            payload["finite_count"] = int(finite.size)
            payload["nan_count"] = int(np.count_nonzero(~np.isfinite(numeric)))
            if finite.size:
                payload.update(
                    {
                        "min": _round_stat(np.nanmin(finite)),
                        "p05": _round_stat(np.nanpercentile(finite, 5)),
                        "p50": _round_stat(np.nanpercentile(finite, 50)),
                        "p95": _round_stat(np.nanpercentile(finite, 95)),
                        "max": _round_stat(np.nanmax(finite)),
                    }
                )
            return payload
        flat = arr.ravel()
        payload["sample_values"] = [str(item) for item in flat[:10]]
        return payload
    if isinstance(value, Mapping):
        return {str(key): _trace_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_trace_safe_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return _round_stat(numeric)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, str):
        return _display_trace_text(value)
    return value


def _trace_curve_summary(well: "Well", curve: str) -> dict[str, Any]:
    """Summarize a curve inside the current lithology-group slice."""
    candidates = list(_target_curve_candidates(_canonical_delivery_curve_name(str(curve))))
    if str(curve).upper() not in {name.upper() for name in candidates}:
        candidates.insert(0, str(curve))

    summary = _curve_value_summary(well, candidates[0])
    if summary.get("present"):
        return summary

    for candidate in candidates[1:]:
        summary = _curve_value_summary(well, candidate)
        if summary.get("present"):
            return summary
    return {"present": False}


def _trace_extra_curve_summary(curve: Any) -> dict[str, Any]:
    data = getattr(curve, "data", curve)
    try:
        arr = _normalize_numeric_array(data)
    except Exception:
        return {"present": False}

    finite = arr[np.isfinite(arr)]
    summary: dict[str, Any] = {
        "present": True,
        "unit": str(getattr(curve, "unit", "") or ""),
        "sample_count": int(arr.size),
        "finite_count": int(finite.size),
        "nan_count": int(np.count_nonzero(~np.isfinite(arr))),
        "finite_percent": _round_stat(finite.size / max(arr.size, 1) * 100.0),
    }
    if finite.size:
        summary.update(
            {
                "min": _round_stat(np.nanmin(finite)),
                "p05": _round_stat(np.nanpercentile(finite, 5)),
                "p50": _round_stat(np.nanpercentile(finite, 50)),
                "p95": _round_stat(np.nanpercentile(finite, 95)),
                "max": _round_stat(np.nanmax(finite)),
            }
        )
    return summary


def _trace_step_payload(step: Any, group_well: "Well") -> dict[str, Any]:
    input_curves = list(step.input_curves or [])
    output_curves = list(step.output_curves or [])
    return {
        "step": str(step.step or ""),
        "success": bool(step.success),
        "method": _display_trace_text(step.method),
        "message": _display_trace_text(step.message),
        "input_curves": _display_curve_list(input_curves),
        "output_curves": _display_curve_list(output_curves),
        "parameters": _trace_safe_value(step.parameters or {}),
        "decisions": [_display_trace_text(item) for item in (step.decisions or [])],
        "data": _trace_safe_value(step.data or {}),
        "input_curve_summaries_in_group": {
            _display_curve_name(curve): _trace_curve_summary(group_well, str(curve))
            for curve in input_curves
        },
        "output_curve_summaries_in_group": {
            _display_curve_name(curve): _trace_curve_summary(group_well, str(curve))
            for curve in output_curves
        },
    }


def _trace_group_overlaps_zone(depth_info: Mapping[str, Any], zone: Any) -> bool:
    try:
        group_top = float(depth_info["min_depth"])
        group_bottom = float(depth_info["max_depth"])
        zone_top = float(zone.top)
        zone_bottom = float(zone.bottom)
    except (KeyError, TypeError, ValueError):
        return True
    return group_bottom >= zone_top and group_top <= zone_bottom


def _trace_workflow_run_payload(
    run: Any,
    group_well: "Well",
    *,
    scope: str,
    zone: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scope": scope,
        "success": bool(getattr(run, "success", False)),
        "executed_steps": [str(step.step or "") for step in getattr(run, "steps_executed", [])],
        "skipped_steps": list(getattr(run, "steps_skipped", []) or []),
        "specific_steps_skipped": _trace_safe_value(getattr(run, "specific_steps_skipped", {}) or {}),
        "warnings": [_display_trace_text(item) for item in (getattr(run, "warnings", []) or [])],
        "errors": [_display_trace_text(item) for item in (getattr(run, "errors", []) or [])],
        "computed": {
            "rw": _trace_safe_value(getattr(run, "rw_computed", None)),
            "rmf": _trace_safe_value(getattr(run, "rmf_computed", None)),
            "rcl": _trace_safe_value(getattr(run, "rcl_computed", None)),
            "temperature_gradient": _trace_safe_value(getattr(run, "temperature_gradient", None)),
        },
        "config": _trace_safe_value(
            {name: value for name, value in vars(run.config).items() if not name.startswith("_")}
            if getattr(run, "config", None) is not None else {}
        ),
        "steps": [
            _trace_step_payload(step, group_well)
            for step in getattr(run, "steps_executed", [])
        ],
    }
    if zone is not None:
        payload["zone"] = {
            "index": int(getattr(zone, "index", 0) or 0),
            "top": _round_stat(getattr(zone, "top", np.nan)),
            "bottom": _round_stat(getattr(zone, "bottom", np.nan)),
            "top_unit": str(getattr(zone, "top_unit", "") or ""),
            "bottom_unit": str(getattr(zone, "bottom_unit", "") or ""),
            "config_overrides": _trace_safe_value(getattr(zone, "config_overrides", {}) or {}),
            "injected_parameters": _trace_safe_value(getattr(zone, "injected_parameters", {}) or {}),
            "unmapped_parameters": list(getattr(zone, "unmapped_parameters", []) or []),
        }
    return payload


def _trace_workflow_runs_for_group(
    workflow_result: Any,
    group_well: "Well",
    depth_info: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = [
        _trace_workflow_run_payload(
            workflow_result.base_result,
            group_well,
            scope="base_well",
        )
    ]
    for zone, zone_result in zip(
        getattr(workflow_result, "zones", []) or [],
        getattr(workflow_result, "zone_results", []) or [],
    ):
        if not _trace_group_overlaps_zone(depth_info, zone):
            continue
        runs.append(
            _trace_workflow_run_payload(
                zone_result,
                group_well,
                scope=f"zone_Z{getattr(zone, 'index', '')}",
                zone=zone,
            )
        )
    return runs


def _write_lithology_group_trace_json(
    trace_path: Path,
    *,
    context_json_path: Path,
    well: "Well",
    workflow_result,
    source_file: Path,
    original_curves: set[str],
    category_name: str,
    category_result: CategoryAnalysisResult,
    derived_logs: Sequence[str],
    target_curves: Sequence[str],
    formation_name: str,
    formation_depth_info: Optional[dict[str, Any]],
    lithology_group_name: str,
    lithology_name: str,
    lithology_mapping: Optional[dict[int, dict[str, str]]] = None,
    warnings: Optional[Sequence[str]] = None,
    errors: Optional[Sequence[str]] = None,
) -> Path:
    """Write a full sidecar trace for one category/formation/lithology group."""
    depth_info = _depth_summary(well)
    formation_info = formation_depth_info or depth_info
    status_targets = list(_ordered_unique([*target_curves, *derived_logs]))
    derived_status = _build_reference_derived_log_status(
        well,
        workflow_result,
        original_curves,
        extra_targets=status_targets,
        computed_curve_names=derived_logs,
    )
    value_qc_report = []
    if hasattr(well, "get_parameter"):
        value_qc_report = list(well.get_parameter(VALUE_RANGE_QC_PARAMETER, []) or [])

    payload = {
        "schema": "petrophys_lithology_group_trace",
        "schema_version": 1,
        "source_file": str(source_file),
        "context_json_file": str(context_json_path),
        "well_name": _reference_well_name(source_file, well),
        "category": category_name,
        "formation": {
            "name": formation_name,
            "min_depth": formation_info.get("min_depth"),
            "max_depth": formation_info.get("max_depth"),
            "thickness": formation_info.get("thickness"),
        },
        "lithology_group": {
            "name": lithology_group_name,
            "lithology": lithology_name,
            "depth_interval": {
                "start_depth": depth_info["min_depth"],
                "end_depth": depth_info["max_depth"],
                "thickness": depth_info["thickness"],
                "data_points": depth_info["data_points"],
                "median_dz": depth_info["median_dz"],
            },
            "distribution": _build_lithology_distribution(well, lithology_mapping=lithology_mapping),
            "info": _build_reference_lithology_info(well, lithology_mapping=lithology_mapping),
        },
        "raw_input_logs_present": _input_log_display_list(original_curves),
        "target_curves_for_category": sorted(str(curve) for curve in target_curves),
        "derived_logs_computed": list(derived_logs),
        "derived_log_status": derived_status,
        "workflow_metadata": {
            "global_overrides": _trace_safe_value(getattr(workflow_result, "global_overrides", {}) or {}),
            "global_unmapped": list(getattr(workflow_result, "global_unmapped", []) or []),
            "zoned_default_overrides": _trace_safe_value(getattr(workflow_result, "zoned_default_overrides", {}) or {}),
            "zoned_default_injected": _trace_safe_value(getattr(workflow_result, "zoned_default_injected", {}) or {}),
            "warnings": [_display_trace_text(item) for item in (warnings or [])],
            "errors": [_display_trace_text(item) for item in (errors or [])],
        },
        "value_range_qc": {
            "parameter": VALUE_RANGE_QC_PARAMETER,
            "report": _trace_safe_value(value_qc_report),
            "note": value_range_qc_note(value_qc_report) if value_qc_report else "",
        },
        "workflow_runs": _trace_workflow_runs_for_group(workflow_result, well, depth_info),
        "category_analysis": _trace_safe_value(category_result.to_dict()),
        "category_derived_curve_summaries": {
            str(name): _trace_extra_curve_summary(curve)
            for name, curve in (category_result.derived_curves or {}).items()
        },
        "final_curve_summaries_in_group": {
            str(curve): _trace_curve_summary(well, str(curve))
            for curve in sorted(_ordered_unique([*target_curves, *derived_logs]))
        },
    }

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)
    return trace_path


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
    """Rewrite every LAS file in a folder in the example delivery format.

    Each file is read and rewritten by :func:`export_example_format_las`,
    without running an interpretation. The output folder
    mirrors the input folder's layout. A file that cannot be read or written is
    recorded as failed and the others still run.

    Parameters
    ----------
    input_dir : str or Path
        Folder of LAS files.
    output_dir : str or Path
        Folder for the rewritten files; created if needed.
    pattern : str, optional
        Glob pattern selecting the input files; ``"*.las"`` by default.
    recursive : bool, default False
        Search subfolders of ``input_dir`` as well.

    Returns
    -------
    list of ExampleFormatExportResult
        One result per input file, in sorted path order.

    See Also
    --------
    export_example_work_package_directory : Interpret each file before writing it.

    Examples
    --------
    >>> from petrophys.reporting import (
    ...     export_example_format_directory, write_example_format_report)
    >>> results = export_example_format_directory("logs", "delivery")
    >>> write_example_format_report(results, "delivery/report.csv")
    PosixPath('delivery/report.csv')
    """
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
    """Interpret every LAS file in a folder and write a delivery work package.

    For each file, curve names and units are standardized and the dynamic
    workflow runs with the depth-zoned settings found in the LAS header (see
    :func:`~petrophys.workflow.run_dynamic_workflow_with_zones`). The
    interpreted well is written by :func:`export_example_format_las`, and a
    JSON file with the same name holds an analysis summary and notes on how
    each derived log was computed. The output folder mirrors the input
    folder's layout. A file that fails is recorded and the others still run.

    Parameters
    ----------
    input_dir : str or Path
        Folder of LAS files.
    output_dir : str or Path
        Folder for the LAS and JSON files; created if needed.
    pattern : str, optional
        Glob pattern selecting the input files; ``"*.las"`` by default.
    recursive : bool, default False
        Search subfolders of ``input_dir`` as well.

    Returns
    -------
    list of ExampleFormatExportResult
        One result per input file. ``success`` is False when the workflow
        reported errors; workflow warnings and errors are kept on the result.

    See Also
    --------
    export_reference_json_work_package_directory : The same package with
        reference-format JSON context files.
    """
    from petrophys.workflow.zoned_batch import _finalize_export_curves

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

            workflow_result = _run_context_workflow(well, _context_workflow_config())
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
    """Interpret every LAS file in a folder and write reference-format contexts.

    Like :func:`export_example_work_package_directory`, each file is
    standardized, interpreted with its depth-zoned LAS header settings and
    written as a delivery LAS file. The JSON file written next to it uses the
    reference context format instead: well and interval information, input
    logs present, curve statistics, lithology distribution, data completeness
    and the analysis summary for the whole well.

    Parameters
    ----------
    input_dir : str or Path
        Folder of LAS files.
    output_dir : str or Path
        Folder for the LAS and JSON files; created if needed. It mirrors the
        input folder's layout.
    pattern : str, optional
        Glob pattern selecting the input files; ``"*.las"`` by default.
    recursive : bool, default False
        Search subfolders of ``input_dir`` as well.

    Returns
    -------
    list of ExampleFormatExportResult
        One result per input file. ``success`` is False when the workflow
        reported errors.

    See Also
    --------
    export_reference_json_work_package_by_formation : One context per formation.
    """
    from petrophys.workflow.zoned_batch import _finalize_export_curves

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

            workflow_result = _run_context_workflow(well, _context_workflow_config())
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
    """Write reference-format contexts for every formation of every well.

    Each LAS file in ``input_dir`` is standardized and interpreted once as a
    whole well, written to ``<output_dir>/<well>/<well>.las``, and then split
    at the formation tops listed for it in ``tops_file``. Each formation slice
    is interpreted on its own and written to
    ``<output_dir>/<well>/<formation>/<well>__<formation>__<top>_<base>.las``
    with a reference-format JSON context next to it. A well without matching
    tops is exported as one ``WHOLE_WELL`` segment, with a warning.

    Parameters
    ----------
    input_dir : str or Path
        Folder of LAS files.
    output_dir : str or Path
        Folder for the per-well packages; created if needed.
    tops_file : str or Path
        Formation tops as CSV or Excel. Without a well column, the table
        applies to a single well and ``input_dir`` must hold one LAS file.
    pattern : str, optional
        Glob pattern selecting the input files; ``"*.las"`` by default.
    recursive : bool, default False
        Search subfolders of ``input_dir`` as well.

    Returns
    -------
    list of ExampleFormatExportResult
        One result per formation written, plus one failed result for each
        well that could not be read or sliced.

    Raises
    ------
    ValueError
        If the tops table has no well column and ``input_dir`` holds more
        than one LAS file.
    """
    from petrophys.workflow.zoned_batch import _finalize_export_curves

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))
    tops_by_well = _load_excel_formation_segments(tops_file)
    _validate_single_well_tops_scope(tops_by_well, files)

    results: List[ExampleFormatExportResult] = []
    for file_path in files:
        try:
            full_well = read_las(file_path)
            full_well = standardize_curves(full_well, rename_curves=True, convert_units=True)
            formation_segments = _formation_segments_for_well(file_path, full_well, tops_by_well)
            _attach_formations_for_interpretation(full_well, formation_segments, _context_workflow_config())

            well_folder = output_path / _sanitize_path_component(file_path.stem)
            well_folder.mkdir(parents=True, exist_ok=True)
            whole_well_original_curves = set(full_well.curves)
            whole_well_result = _run_context_workflow(copy.deepcopy(full_well), _context_workflow_config())
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
                    "No formation tops matched this well in the tops table. Falling back to WHOLE_WELL segmentation."
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
                workflow_result = _run_context_workflow(
                    sliced_well,
                    _formation_run_config(_context_workflow_config(), whole_well_result.well, sliced_well),
                )
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
    tops_file: Optional[str | Path] = None,
    pattern: str = "*.las",
    recursive: bool = False,
    minimum_group_thickness: float = 2.0,
    well_names: Optional[Sequence[str]] = None,
    lithology_mapping_file: Optional[str | Path] = None,
    tops_options: Optional[dict] = None,
    workflow_config: Optional["DynamicWorkflowConfig"] = None,
    lithology_file: Optional[str | Path] = None,
    lithology_domain: Optional[str] = None,
    reader_options: Optional[dict[str, Any]] = None,
) -> List[ExampleFormatExportResult]:
    """Write reference-format contexts per formation and lithology group.

    Each well is read with its metadata and optional lithology intervals,
    interpreted as a whole well and written to
    ``<output_dir>/<well>/<well>.las``. Each formation is
    then interpreted on its own and split into lithology groups: runs of
    consecutive samples with the same lithology code, taken from the
    lithology code curve the interpretation read (``LITHOLOGY_GEOLINK``, or
    another lithology-code curve under the ``"current"`` profile; see
    :attr:`~petrophys.workflow.DynamicWorkflowConfig.lithology_curve`), or
    from the lithology intervals it selected. Every group
    thicker than ``minimum_group_thickness`` gets a JSON context at
    ``<output_dir>/<well>/<formation>/<formation>_<group>.json``. Without
    tops, or when no tops match a well, the whole well is one ``WHOLE_WELL``
    formation.

    Parameters
    ----------
    input_dir : str or Path
        Folder of well files (LAS or DLIS; tabular logs need reader_options).
    output_dir : str or Path
        Folder for the per-well packages; created if needed.
    tops_file : str or Path, optional
        Formation tops as CSV or Excel.
    pattern : str, optional
        Glob pattern selecting the input files; ``"*.las"`` by default.
    recursive : bool, default False
        Search subfolders of ``input_dir`` as well.
    minimum_group_thickness : float, default 2.0
        Lithology groups this thick or thinner (in the well depth unit) are
        skipped.
    well_names : sequence of str, optional
        Only export LAS files whose name (without extension) matches one of
        these well names.
    lithology_mapping_file : str or Path, optional
        Excel lithology codebook with columns ``Attribute``, ``Lithology
        Attribute`` and ``Abbreviation`` (the numeric code), used to name the
        groups from ``Lithology Attribute``. This is not the
        :func:`~petrophys.io.read_lithology_codes` format; rows whose code is
        not a number are skipped.
    tops_options : dict, optional
        Options for reading ``tops_file``: ``rank``, ``sheet_name``,
        ``columns`` or ``sep``.
    workflow_config : DynamicWorkflowConfig, optional
        Interpretation settings (default: the workflow defaults). LAS header
        parameters override them. ``calculation_profile="geolink"`` selects
        the GeoLink evaluation configuration for context export.
    lithology_file : str or Path, optional
        Per-depth lithology intervals for the wells.
    lithology_domain : str, optional
        Override the broad domain while retaining per-well LAS settings.
    reader_options : dict, optional
        Options passed to petrophys.read, e.g. {"step": 0.1524} for DLIS.
        Select DLIS files with ``pattern="*.dlis"``. Defaults preserve the reader's
        native settings; tabular inputs require an explicit depth_unit.

    Returns
    -------
    list of ExampleFormatExportResult
        One result per lithology group written, each pointing at the
        whole-well LAS file and the group's JSON file, plus one failed result
        for each well that could not be processed.

    Raises
    ------
    ValueError
        If the tops table has no well column and more than one LAS file is
        selected, spells one well in several ways, or the lithology codebook
        lacks its columns or gives one code a fractional value or two names.
    """
    from petrophys.workflow.zoned_batch import _finalize_export_curves

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))
    files = _filter_input_files_by_well_names(files, well_names)
    tops_by_well = (
        _load_excel_formation_segments(tops_file, **(tops_options or {}))
        if tops_file is not None
        else {}
    )
    _validate_single_well_tops_scope(tops_by_well, files)
    lithology_mapping = (
        _load_lithology_mapping_workbook(lithology_mapping_file)
        if lithology_mapping_file
        else None
    )

    results: List[ExampleFormatExportResult] = []
    for file_path in files:
        try:
            full_well, well_config = _read_context_input_well(
                file_path,
                lithology_file=lithology_file,
                workflow_config=workflow_config,
                lithology_domain=lithology_domain,
                reader_options=reader_options,
            )
            formation_segments = _formation_segments_for_well(file_path, full_well, tops_by_well)
            _attach_formations_for_interpretation(full_well, formation_segments, well_config)

            well_folder = output_path / _sanitize_path_component(file_path.stem)
            well_folder.mkdir(parents=True, exist_ok=True)
            whole_well_original_curves = set(full_well.curves)
            whole_well_result = _run_context_workflow(copy.deepcopy(full_well), well_config)
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
                    "No formation tops matched this well in the tops table. Falling back to WHOLE_WELL segmentation."
                )

            for segment in formation_segments:
                display_name = _formation_file_key(
                    str(segment.get("display_name") or segment["name"]), reproducing_benchmark()
                )
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
                workflow_result = _run_context_workflow(
                    formation_input_well,
                    _formation_run_config(well_config, whole_well_result.well, formation_input_well),
                )
                warnings = list(fallback_warnings)
                warnings.extend(workflow_result.warnings)
                warnings.extend(workflow_result.base_result.warnings)
                for zone_result in workflow_result.zone_results:
                    warnings.extend(zone_result.warnings)

                errors = list(workflow_result.base_result.errors)
                for zone_result in workflow_result.zone_results:
                    errors.extend(zone_result.errors)

                _finalize_export_curves(
                    workflow_result.well,
                    formation_original_curves,
                    retain_computed_lithology=True,
                )
                formation_depth_info = _depth_summary(formation_input_well)
                lithology_segments = _build_lithology_group_segments(
                    workflow_result.well,
                    minimum_group_thickness=minimum_group_thickness,
                    lithology_mapping=_run_lithology_mapping(lithology_mapping, workflow_result),
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
                    if raw_lithology_name == MISSING_LITHOLOGY_LABEL:
                        file_group_name = raw_group_name
                        json_group_name = "No lithology information available"
                        json_lithology_name = "No lithology information available"

                    file_stem = f"{formation_display_name}_{file_group_name}"
                    if formation_name_counts.get(
                        _formation_file_key(formation_display_name, reproducing_benchmark()), 0
                    ) > 1:
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


def _group_layout_for_file(
    group_layout: Optional[Mapping[str, Sequence[Mapping[str, Any]]]],
    file_path: Path,
) -> Optional[Sequence[Mapping[str, Any]]]:
    """The groups ``group_layout`` gives the well read from ``file_path``, else None."""
    if not group_layout:
        return None
    wanted = file_path.stem.strip().casefold()
    for well_key, groups in group_layout.items():
        if str(well_key).strip().casefold() == wanted:
            return groups
    return None


def _benchmark_profile_standardization(function):
    """Under the "geolink" calculation profile, resolve duplicate curves as the benchmark packages did."""
    import functools
    import inspect

    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        from petrophys._internal.benchmark import historical_benchmark

        config = signature.bind_partial(*args, **kwargs).arguments.get("workflow_config")
        if getattr(config, "calculation_profile", None) == "geolink":
            with historical_benchmark():
                return function(*args, **kwargs)
        return function(*args, **kwargs)

    return wrapper


@_benchmark_profile_standardization
def export_reference_json_work_package_by_category_formation_lithology(
    input_dir: str | Path,
    output_dir: str | Path,
    tops_file: Optional[str | Path] = None,
    pattern: str = "*.las",
    recursive: bool = False,
    minimum_group_thickness: float = 2.0,
    well_names: Optional[Sequence[str]] = None,
    lithology_mapping_file: Optional[str | Path] = None,
    workflow_config: Optional["DynamicWorkflowConfig"] = None,
    tops_options: Optional[dict] = None,
    lithology_file: Optional[str | Path] = None,
    lithology_domain: Optional[str] = None,
    raw_input_dir: Optional[str | Path] = None,
    reader_options: Optional[dict[str, Any]] = None,
    group_layout: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    processed_las_dir: Optional[str | Path] = None,
    prepare_input: Optional[Any] = None,
    processed_las_only: bool = False,
    on_interpreted: Optional[Any] = None,
) -> List[ExampleFormatExportResult]:
    """Generate category contexts from well files and optional geology inputs.

    Use ``pattern="*.dlis"`` for DLIS. reader_options are passed to petrophys.read
    for both context inputs and optional raw inputs (e.g. DLIS step/channels).

    ``group_layout`` gives, per well (keyed by file stem, compared without
    case), the formation/lithology groups to export instead of splitting the
    well by its tops and lithology codes. Each group is a mapping with
    ``formation_name``, ``formation_depth_info`` (``min_depth``,
    ``max_depth``, ``thickness``), ``group_name``, ``lithology``,
    ``start_depth`` and ``end_depth``, and optionally ``file_stem`` (default
    ``"<formation_name>_<group_name>"``); the group's samples are those from
    ``start_depth`` to ``end_depth``. A well missing from the layout is split
    as usual.

    ``processed_las_dir`` saves, per well, the interpretation the context is
    computed from (LAS plus ``workflow_json`` trace). Each context statistic
    is then the statistic of that LAS over the group's depth window. With
    ``processed_las_only`` only those LAS and traces are written: no context
    JSON, and one result per well whose ``output_file`` is its LAS.

    ``on_interpreted(file_path, workflow_result, preparatory_steps, raw_curves)``
    receives each well's interpretation, the one the context statistics are
    computed from, and the names of the raw LAS curves. With ``processed_las_only`` and no ``processed_las_dir`` nothing is
    written: the wells are only interpreted.

    ``prepare_input(file_path, well)`` runs on each well after it is read and
    before the workflow; it may add input curves and returns the trace entries
    of what it computed (a list, recorded as ``preparatory_steps``), or a
    mapping with those entries under ``preparatory_steps`` and an optional
    ``after_workflow(well)`` callable applied to the interpreted well before
    the context and LAS are written. The mapping may also replace the input:
    ``well`` and ``well_config`` (the well and workflow configuration to
    interpret) and ``raw_curves`` (the curve names the context classifies as
    raw input logs).

    ``scripts/generate_benchmark_context.py`` runs it to reproduce the GeoLink
    eval context benchmark with the GeoLink evaluation configuration.
    Input curves are classified against the raw LAS files in
    ``raw_input_dir`` (default: the sibling of a ``*_processed`` input
    directory), falling back to the curves of the LAS being read.
    """
    from petrophys.workflow.zoned_batch import _finalize_export_curves

    if processed_las_only and processed_las_dir is None and on_interpreted is None:
        raise ValueError("processed_las_only requires processed_las_dir")
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not processed_las_only:
        output_path.mkdir(parents=True, exist_ok=True)
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))
    files = _filter_input_files_by_well_names(files, well_names)
    raw_context_dir = Path(raw_input_dir) if raw_input_dir is not None else _raw_input_dir_for_processed(input_path)
    raw_curves_by_stem = _raw_curve_lookup(raw_context_dir, pattern, recursive, reader_options)
    tops_by_well = (
        _load_excel_formation_segments(tops_file, **(tops_options or {}))
        if tops_file is not None
        else {}
    )
    _validate_single_well_tops_scope(tops_by_well, files)
    lithology_mapping = (
        _load_lithology_mapping_workbook(lithology_mapping_file)
        if lithology_mapping_file
        else None
    )

    results: List[ExampleFormatExportResult] = []
    for file_path in files:
        try:
            full_well, well_config = _read_context_input_well(
                file_path,
                lithology_file=lithology_file,
                workflow_config=workflow_config,
                lithology_domain=lithology_domain,
                reader_options=reader_options,
            )
            input_reading = _input_reading_record(file_path, full_well) if on_interpreted is not None else None
            preparation = prepare_input(file_path, full_well) if prepare_input is not None else None
            after_workflow = None
            preparatory_steps = preparation
            if isinstance(preparation, Mapping):
                preparatory_steps = preparation.get("preparatory_steps")
                after_workflow = preparation.get("after_workflow")
                full_well = preparation.get("well", full_well)
                well_config = preparation.get("well_config", well_config)
            formation_segments = _formation_segments_for_well(file_path, full_well, tops_by_well)
            _attach_formations_for_interpretation(full_well, formation_segments, well_config)
            fallback_warnings: list[str] = []
            processed_original_curves = set(full_well.curves)
            raw_original_curves = _raw_original_curves_for_file(
                file_path,
                processed_original_curves,
                raw_curves_by_stem,
            )
            if isinstance(preparation, Mapping) and preparation.get("raw_curves") is not None:
                raw_original_curves = set(preparation["raw_curves"])
            workflow_result = _run_context_workflow(full_well, well_config)
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
            from petrophys.workflow.dynamic_workflow import _curve_snapshot, _same_curve_values

            before_finalize = _curve_snapshot(workflow_result.well)
            _finalize_export_curves(
                workflow_result.well,
                processed_original_curves,
                # The computed lithology names the groups of a well without input lithology.
                retain_computed_lithology=not _reproduces_benchmark_context(workflow_result),
            )
            # What the export finalization wrote and removed, for the step record
            # of a context-profile interpretation (not written to the context).
            workflow_result.export_finalization = {
                "written": [
                    name for name, curve in workflow_result.well._curves.items()
                    if name not in before_finalize or not _same_curve_values(before_finalize[name], curve.data)
                ],
                "removed": [name for name in before_finalize if workflow_result.well.get(name) is None],
            }
            if after_workflow is not None:
                after_workflow(workflow_result.well)
            if on_interpreted is not None:
                workflow_result.input_reading = input_reading
                on_interpreted(file_path, workflow_result, preparatory_steps, raw_original_curves)
                if processed_las_only and processed_las_dir is None:
                    results.append(
                        ExampleFormatExportResult(
                            source_file=str(file_path),
                            output_file="",
                            success=not errors,
                            warnings=warnings,
                            errors=errors,
                        )
                    )
                    continue
            if processed_las_dir is not None:
                raw_source = (
                    raw_context_dir / file_path.name
                    if raw_context_dir is not None and (raw_context_dir / file_path.name).exists()
                    else None
                )
                processed_las = _write_context_workflow_las(
                    file_path,
                    input_path,
                    processed_las_dir,
                    full_well,
                    workflow_result,
                    processed_original_curves,
                    warnings,
                    source_file=raw_source,
                    preparatory_steps=preparatory_steps,
                )
                if processed_las_only:
                    results.append(
                        ExampleFormatExportResult(
                            source_file=str(file_path),
                            output_file=str(processed_las),
                            success=not errors,
                            warnings=warnings,
                            errors=errors,
                        )
                    )
                    continue

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

            def export_group(
                group_well: "Well",
                formation_display_name: str,
                formation_depth_info: Mapping[str, Any],
                json_group_name: str,
                json_lithology_name: str,
                file_stem: str,
            ) -> None:
                """Write the category files of one formation/lithology group."""
                for category_name, payload in iter_context_group_payloads(
                    group_well,
                    workflow_result,
                    file_path,
                    raw_original_curves,
                    formation_name=formation_display_name,
                    formation_depth_info=formation_depth_info,
                    group_name=json_group_name,
                    lithology_name=json_lithology_name,
                    lithology_mapping=lithology_mapping,
                ):
                    json_path = (
                        output_path
                        / category_name
                        / _sanitize_path_component(file_path.stem)
                        / _sanitize_path_component(formation_display_name)
                        / f"{_sanitize_path_component(file_stem)}_{category_name}.json"
                    )
                    json_file = _write_payload(json_path, payload)
                    result_warnings = [*warnings, *fallback_warnings]

                    results.append(
                        ExampleFormatExportResult(
                            source_file=str(file_path),
                            output_file=str(json_file),
                            success=not errors,
                            json_file=str(json_file),
                            curves_exported=0,
                            warnings=result_warnings,
                            errors=errors,
                        )
                    )

            for group in iter_context_groups(
                workflow_result,
                formation_segments,
                well_top=well_top,
                well_bottom=well_bottom,
                minimum_group_thickness=minimum_group_thickness,
                lithology_mapping=lithology_mapping,
                layout=_group_layout_for_file(group_layout, file_path),
                fallback_warnings=fallback_warnings,
            ):
                export_group(
                    group["well"],
                    group["formation_name"],
                    group["formation_depth_info"],
                    group["group_name"],
                    group["lithology"],
                    group["file_stem"],
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

    if processed_las_only:
        return results
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
    """Write a CSV summary of a batch export, one row per result.

    Columns are ``source_file``, ``output_file``, ``json_file``, ``success``,
    ``curves_exported``, ``warnings`` and ``errors``; several warnings or
    errors in one cell are joined with ``" | "``.

    Parameters
    ----------
    results : iterable of ExampleFormatExportResult
        Results returned by one of the batch exporters.
    report_path : str or Path
        Output CSV path. Missing parent folders are created.

    Returns
    -------
    Path
        The file written.
    """
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


#: The description the benchmark packages carry in curve_name_mapping.json.
_BENCHMARK_MAPPING_DESCRIPTION = (
    "Mapping between the current LAS/JSON export mnemonics and the older/example naming "
    "conventions used in prior reference files."
)


def write_reference_curve_name_mapping(
    output_dir: str | Path,
    file_name: str = "curve_name_mapping.json",
) -> Path:
    """Write the mapping between the export curve names and the reference delivery names.

    The JSON file lists the curves whose exported mnemonic differs from the
    reference deliveries (for example ``SXO``, ``SWXO`` there), the
    method-specific reference names, explicit replacements and the key curves
    whose names are the same, so that consumers of the reference deliveries
    can match curves.

    Parameters
    ----------
    output_dir : str or Path
        Folder for the file; created if needed.
    file_name : str, default "curve_name_mapping.json"
        Name of the file.

    Returns
    -------
    Path
        The file written.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mapping_path = output_path / file_name
    payload = {
        "description": (
            _BENCHMARK_MAPPING_DESCRIPTION if reproducing_benchmark() else
            "Mapping between the LAS/JSON export mnemonics and the example naming "
            "conventions used in the reference files."
        ),
        "mapping": REFERENCE_CURVE_NAME_MAPPING,
    }
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)
    return mapping_path


# Raw-LAS lookup for the input-curve classification of the context packages.

def _raw_curve_lookup(
    raw_input_dir: Optional[str | Path],
    pattern: str,
    recursive: bool,
    reader_options: Optional[dict[str, Any]] = None,
) -> dict[str, set[str]]:
    """Return standardized raw curve names keyed by well-file stem."""
    from petrophys.api import read
    if raw_input_dir is None:
        return {}

    raw_path = Path(raw_input_dir)
    if not raw_path.exists() or not raw_path.is_dir():
        return {}

    files = sorted(raw_path.rglob(pattern) if recursive else raw_path.glob(pattern))
    lookup: dict[str, set[str]] = {}
    for file_path in files:
        try:
            raw_well = read(file_path, **(reader_options or {}))
            raw_well = standardize_curves(raw_well, rename_curves=True, convert_units=True)
            lookup[file_path.stem.lower()] = set(raw_well.curves)
        except Exception:
            continue
    return lookup


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


def _raw_original_curves_for_file(
    file_path: Path,
    processed_original_curves: set[str],
    raw_curves_by_stem: dict[str, set[str]],
) -> set[str]:
    """Use raw LAS curves for context provenance, falling back to processed input."""
    return set(raw_curves_by_stem.get(file_path.stem.lower()) or processed_original_curves)


def _rw_temperature_formula_text() -> str:
    return "Rw75=0.0123+3647.5/salinity^0.955; RW(T)=Rw75*(75+6.77)/(TEMP+6.77)"


