"""
Curve alias definitions for standardization.

This module provides mapping of common curve name variations
to standard mnemonics used throughout the library.
"""

from typing import Dict, List, Optional

from petrophys._internal.analysis_helpers import base_curve_mnemonic
from petrophys._internal.benchmark import reproducing_benchmark

# Standard mnemonic to aliases mapping
ALIAS_MAP: Dict[str, List[str]] = {
    # Core measurements are distinct from calculated fluid saturations.
    "SWC": ["SW_CORE", "CORE_SW"],
    "SOC": ["SO_CORE", "CORE_SO"],
    "SGC": ["SG_CORE", "CORE_SG"],
    # Deep Resistivity (RDEP): the 90 in array curve is the deepest reading,
    # so it ranks above the 60 in curve.
    "RDEP": [
        "RT",
        "ILD",
        "RILD",
        "LLD",
        "AT90",
        "AT60",
        "HRLT",
        "RESD",
        "RD",
        "RDEEP",
        "AO60",
        "AO90",
        "RLA5",
        "ILS",
        "M2RX",
        "RESDEP",
        "RVDR",
    ],
    # Medium Resistivity (RMED)
    # NOTE: AT10 (10in array induction) and RLLS (laterolog shallow) belong to RSHA,
    # not here: each alias maps to one standard curve. AT30 (30in) is the medium curve.
    # RESS and SFLU are RSHA aliases: the rest of the library (analysis, workflow
    # gating, log viewer) treats RESS as shallow/flushed-zone, and SFL-family
    # curves are the shallow member of the classic DIL-SFL suite.
    "RMED": [
        "ILM",
        "AT30",
        "RM",
        "RMED",
        "RLA3",
        "RESMED",
    ],
    # Shallow Resistivity (RSHA)
    # AT30 removed (it is the medium array-induction spacing -> RMED).
    "RSHA": [
        "LLS",
        "RLLS",
        "AT10",
        "AT20",
        "SFL",
        "SFLU",
        "RSFL",
        "RESS",
        "RLA1",
        "RS",
        "RSHAL",
        "RSHALLOW",
        "RVSR",
    ],
    # Flushed zone Resistivity (RXO)
    "RXO": [
        "MSFL",
        "MCFL",
        "MLL",
        "RXOZ",
        "RXO",
        "RMLL",
        "PL",
        "RESMIC",
        "RMIC",
        "RXO8",
        "RXO82",
        "RXOI",
        "RXR",
    ],
    # Bulk Density (RHOB)
    "RHOB": [
        "RHOB",
        "ZDEN",
        "RHOZ",
        "DEN",
        "DENSITY",
        "DENS",
        "RHOBN",
        "RHOM",
        "DENB",
        "BULK_DEN",
        "RHO8",
        "RHOI",
    ],
    # Density Correction (DRHO)
    "DRHO": ["DRHO", "DCOR", "HDRA", "DENC", "RHOC"],
    # Neutron Porosity (NPHI)
    "NPHI": [
        "NPHI",
        "TNPH",
        "CNL",
        "CNLS",
        "NPLS",
        "NEUT",
        "NPOR",
        "CNCF",
        "NPRL",
        "NEU",
        "APLC",
        "PHIS_NEU",
        "NPHI_LS",
        # Sandstone-referenced neutron: see NEUTRON_MATRIX_REFERENCE.
        "NPSS",
        "CNSS",
        "NPHI_SS",
    ],
    # Compressional Sonic (DTC)
    "DTC": [
        "DT",
        "DTC",
        "AC",
        "SONIC",
        "DTCO",
        "DT4P",
        "DT24",
        "DTLN",
        "DTL",
        "SLOWNESS",
        "BHC",
    ],
    # Shear Sonic (DTS). Do not alias DTE: several LAS exports use DTE for
    # non-slowness telemetry, and those values can be orders of magnitude too large.
    "DTS": ["DTS", "DTSM", "DT4S", "DTRS", "SHEAR"],
    # Photoelectric Factor (PEF)
    "PEF": ["PE", "PEF", "PEFZ", "PEFL", "PEF8", "PHOTO"],
    # Caliper (CALI)
    "CALI": ["CALI", "HCAL", "CAL", "CALIPER", "C1", "C2", "HD1_PPC1", "HD2_PPC1"],
    # Gamma Ray (GR)
    "GR": [
        "GR",
        "SGR",
        "CGR",
        "GAMMA",
        "GAPI",
        "GRN",
        "GRD",
        "GR_EDTC",
        "HSGR",
        "ECGR",
        "ECGR_EDTC",
        "HGR",
        "HGR_EDTC",
    ],
    # Spectral Gamma - Total (SGR)
    "SGR": ["SGR", "GR_S", "HSGR", "SGRC"],
    # Spectral Gamma - Computed (CGR): HCGR is the HNGS uranium-free GR.
    "CGR": ["CGR", "GR_C", "GRC", "HCGR"],
    # Spontaneous Potential (SP)
    "SP": ["SP", "SPON", "SSP", "PSP"],
    # Bit Size (BS)
    "BS": ["BS", "BIT", "BITSZ", "BIT_SIZE"],
    # Depth variants. TVD/TVDSS are deliberately NOT aliased to DEPTH: they are
    # physically distinct from measured depth and must not collide with it on
    # deviated wells.
    "DEPTH": ["DEPT", "DEPTH", "MD", "DEPH"],
    # Temperature
    "TEMP": ["TEMP", "TEMPERATURE", "BHT", "TLOG", "GTMP", "CMR_TEMP"],
    # Interpreted porosity and shale/clay volume
    "PHIT": ["PHIT", "PHIT_HILT"],
    "PHIE": ["PHIE", "PHIE_HILT", "PHIE_D", "PHIE_ND", "PHIE_S"],
    "VCL": ["VCL", "VCLAY", "VSH", "VSH_GR", "VSH_ND", "VCL_HILT"],
    # Core-calibrated porosity
    "PHIE_CC": ["PHIE_CC", "PHIE_CORECAL"],
    # NMR processed outputs, including vendor-specific aliases
    "NMR_PHIT": ["MPHI", "TCMR", "CMRPHI", "NMRP", "PHINMR"],
    "NMR_FFI": ["MFFI", "CMFF", "CMR_FFI", "NMR_FFI", "FFI"],
    "NMR_BVI": ["MBVI", "CBVI", "CMR_BVI", "NMR_BVI", "BVI"],
    "NMR_T2LM": ["T2LM", "T2ML", "T2LOGMEAN"],
    "NMR_T2CUTOFF": ["T2CUTOFF", "T2_CUTOFF", "T2CUT"],
    "NMR_KSDR": ["KSDR", "K_SDR", "KSDR_CMR"],
    "NMR_KTIM": ["KTIM", "K_TIM", "KTC", "K_COATES"],
    "NMR_3MS": ["CMRP_3MS", "CMR_3MS"],
    "NMR_SWIRR": ["SWIRR_NMR", "NMR_SWIRR", "SWIRR_CMR", "CMR_SWIRR"],
}

# Reverse mapping: alias -> standard mnemonic
_REVERSE_MAP: Dict[str, str] = {}
for standard, aliases in ALIAS_MAP.items():
    for alias in aliases:
        _REVERSE_MAP[alias.upper()] = standard
    _REVERSE_MAP[standard.upper()] = standard

#: Neutron aliases of NPHI recorded on a matrix other than limestone, which
#: NPHI is read on. The curve keeps the reference in its description and in
#: ``metadata.custom["neutron_matrix_reference"]``.
NEUTRON_MATRIX_REFERENCE: Dict[str, str] = {
    "NPSS": "sandstone",
    "CNSS": "sandstone",
    "NPHI_SS": "sandstone",
}

#: NPHI aliases the benchmark context packages did not have.
_BENCHMARK_MISSING_NPHI_ALIASES = ("NPHI_LS", "NPSS", "CNSS", "NPHI_SS")

#: Alias lists the benchmark context packages were computed with, where they
#: differ from :data:`ALIAS_MAP`: RDEP ranked AT60 before AT90, HCGR was
#: a GR alias, and NPHI had no NPHI_LS or sandstone-referenced aliases. Used
#: inside ``historical_benchmark()``.
_BENCHMARK_ALIAS_MAP: Dict[str, List[str]] = {
    "RDEP": [
        name if name not in ("AT60", "AT90") else {"AT60": "AT90", "AT90": "AT60"}[name]
        for name in ALIAS_MAP["RDEP"]
    ],
    "GR": [*ALIAS_MAP["GR"][:ALIAS_MAP["GR"].index("HGR")], "HCGR", *ALIAS_MAP["GR"][ALIAS_MAP["GR"].index("HGR"):]],
    "CGR": [name for name in ALIAS_MAP["CGR"] if name != "HCGR"],
    "RXO": [name for name in ALIAS_MAP["RXO"] if name != "RMIC"],
    "NPHI": [name for name in ALIAS_MAP["NPHI"] if name not in _BENCHMARK_MISSING_NPHI_ALIASES],
}
_BENCHMARK_REVERSE_MAP: Dict[str, str] = {
    "HCGR": "GR",
    "RMIC": "RMIC",
    **{name: name for name in _BENCHMARK_MISSING_NPHI_ALIASES},
}


def get_standard_mnemonic(mnemonic: str) -> str:
    """
    Get the standard mnemonic for a curve alias.

    Parameters
    ----------
    mnemonic : str
        Curve mnemonic (may be an alias)

    Returns
    -------
    str
        Standard mnemonic, or original if not found

    Examples
    --------
    >>> get_standard_mnemonic("ILD")
    'RDEP'
    >>> get_standard_mnemonic("ZDEN")
    'RHOB'
    >>> get_standard_mnemonic("UNKNOWN")
    'UNKNOWN'
    """
    normalized = str(mnemonic).strip().upper()
    base = base_curve_mnemonic(normalized)
    if reproducing_benchmark():
        historical = _BENCHMARK_REVERSE_MAP.get(normalized, _BENCHMARK_REVERSE_MAP.get(base))
        if historical is not None:
            return historical
    return _REVERSE_MAP.get(normalized, _REVERSE_MAP.get(base, normalized))


def get_aliases(standard_mnemonic: str) -> List[str]:
    """
    Get all aliases for a standard mnemonic.

    Parameters
    ----------
    standard_mnemonic : str
        Standard mnemonic

    Returns
    -------
    list of str
        List of aliases, in ranking order (the benchmark packages' order
        inside ``historical_benchmark()``)
    """
    standard = standard_mnemonic.upper()
    if reproducing_benchmark() and standard in _BENCHMARK_ALIAS_MAP:
        return list(_BENCHMARK_ALIAS_MAP[standard])
    return ALIAS_MAP.get(standard, [])


def find_curve_type(mnemonic: str) -> Optional[str]:
    """
    Identify the curve type from mnemonic.

    Parameters
    ----------
    mnemonic : str
        Curve mnemonic

    Returns
    -------
    str or None
        Curve type description, or None if unknown
    """
    standard = get_standard_mnemonic(mnemonic)

    type_map = {
        "RDEP": "Deep Resistivity",
        "RMED": "Medium Resistivity",
        "RSHA": "Shallow Resistivity",
        "RXO": "Flushed zone Resistivity",
        "RHOB": "Bulk Density",
        "DRHO": "Density Correction",
        "NPHI": "Neutron Porosity",
        "DTC": "Compressional Sonic",
        "DTS": "Shear Sonic",
        "PEF": "Photoelectric Factor",
        "CALI": "Caliper",
        "GR": "Gamma Ray",
        "SGR": "Spectral Gamma Ray (Total)",
        "CGR": "Computed Gamma Ray",
        "SP": "Spontaneous Potential",
        "BS": "Bit Size",
        "DEPTH": "Depth",
        "TEMP": "Temperature",
        "PHIT": "Total Porosity",
        "PHIE": "Effective Porosity",
        "VCL": "Clay Volume",
        "NMR_PHIT": "NMR Total Porosity",
        "NMR_FFI": "NMR Free Fluid",
        "NMR_BVI": "NMR Bound Fluid",
        "NMR_T2LM": "NMR T2 Log Mean",
        "NMR_T2CUTOFF": "NMR T2 Cutoff",
        "NMR_KSDR": "NMR SDR Permeability",
        "NMR_KTIM": "NMR Timur/Coates Permeability",
        "NMR_3MS": "CMR 3ms Porosity",
        "NMR_SWIRR": "NMR Irreducible Water Saturation",
    }

    return type_map.get(standard)


# Standard units for each curve type
STANDARD_UNITS: Dict[str, str] = {
    "SWC": "v/v",
    "SOC": "v/v",
    "SGC": "v/v",
    "RDEP": "ohm.m",
    "RMED": "ohm.m",
    "RSHA": "ohm.m",
    "RXO": "ohm.m",
    "RHOB": "g/cm3",
    "DRHO": "g/cm3",
    "NPHI": "v/v",
    "DTC": "us/ft",
    "DTS": "us/ft",
    "PEF": "b/e",
    "CALI": "in",
    "GR": "API",
    "SGR": "API",
    "CGR": "API",
    "SP": "mV",
    "BS": "in",
    "DEPTH": "ft",
    "TEMP": "F",
    "PHIT": "v/v",
    "PHIE": "v/v",
    "VCL": "v/v",
    "NMR_PHIT": "pu",
    "NMR_FFI": "pu",
    "NMR_BVI": "pu",
    "NMR_T2LM": "ms",
    "NMR_T2CUTOFF": "ms",
    "NMR_KSDR": "mD",
    "NMR_KTIM": "mD",
    "NMR_3MS": "pu",
    "NMR_SWIRR": "v/v",
}


def get_standard_unit(mnemonic: str) -> Optional[str]:
    """
    Get the standard unit for a curve type.

    Parameters
    ----------
    mnemonic : str
        Curve mnemonic

    Returns
    -------
    str or None
        Standard unit, or None if unknown
    """
    standard = get_standard_mnemonic(mnemonic)
    return STANDARD_UNITS.get(standard)
