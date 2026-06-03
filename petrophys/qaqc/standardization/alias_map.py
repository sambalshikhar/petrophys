"""
Curve alias definitions for standardization.

This module provides mapping of common curve name variations
to standard mnemonics used throughout the library.
"""

from typing import Dict, List, Optional

# Standard mnemonic to aliases mapping
ALIAS_MAP: Dict[str, List[str]] = {
    # Deep Resistivity (RDEP)
    "RDEP": [
        "RT",
        "ILD",
        "RILD",
        "LLD",
        "AT60",
        "AT90",
        "HRLT",
        "RESD",
        "RD",
        "RDEEP",
        "AO60",
        "AO90",
        "RLA5",
        "ILS",
        "M2RX",
    ],
    # Medium Resistivity (RMED)
    "RMED": ["AT10", "RLLS", "SFLU", "RESS", "ILM", "AT30", "RM", "RMED", "RLA3"],
    # Shallow Resistivity (RSHA)
    "RSHA": [
        "LLS",
        "RLLS",
        "AT10",
        "AT20",
        "AT30",
        "SFL",
        "RSFL",
        "RLA1",
        "RS",
        "RSHAL",
        "RSHALLOW",
    ],
    # Flushed zone Resistivity (RXO)
    "RXO": ["MSFL", "MCFL", "MLL", "RXOZ", "RXO", "RMLL", "PL"],
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
    ],
    # Density Correction (DRHO)
    "DRHO": ["DRHO", "DCOR", "HDRA", "DENC"],
    # Neutron Porosity (NPHI)
    "NPHI": [
        "NPHI",
        "TNPH",
        "CNL",
        "NPLS",
        "NEUT",
        "NPOR",
        "CNCF",
        "NPRL",
        "NEU",
        "APLC",
        "PHIS_NEU",
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
    "GR": ["GR", "SGR", "CGR", "GAMMA", "GAPI", "GRN", "GRD", "GR_EDTC", "HSGR"],
    # Spectral Gamma - Total (SGR)
    "SGR": ["SGR", "GR_S", "HSGR", "SGRC"],
    # Spectral Gamma - Computed (CGR)
    "CGR": ["CGR", "GR_C", "GRC"],
    # Spontaneous Potential (SP)
    "SP": ["SP", "SPON", "SSP", "PSP"],
    # Bit Size (BS)
    "BS": ["BS", "BIT", "BITSZ", "BIT_SIZE"],
    # Depth variants
    "DEPTH": ["DEPT", "DEPTH", "MD", "TVD", "TVDSS", "DEPH"],
    # Temperature
    "TEMP": ["TEMP", "TEMPERATURE", "BHT", "TLOG", "GTMP", "CMR_TEMP"],
    # Core-calibrated porosity
    "PHIE_CC": ["PHIE_CC", "PHIE_CORECAL"],
    # NMR / CMR processed outputs
    "NMR_PHIT": ["MPHI", "TCMR", "CMRPHI", "NMRP", "PHINMR"],
    "NMR_FFI": ["MFFI", "CMFF", "CMR_FFI", "NMR_FFI", "FFI"],
    "NMR_BVI": ["MBVI", "CBVI", "CMR_BVI", "NMR_BVI", "BVI"],
    "NMR_T2LM": ["T2LM", "T2ML", "T2LOGMEAN"],
    "NMR_T2CUTOFF": ["T2CUTOFF", "T2_CUTOFF", "T2CUT"],
    "NMR_KSDR": ["KSDR", "K_SDR", "KSDR_CMR"],
    "NMR_KTIM": ["KTIM", "K_TIM", "KTC", "K_COATES"],
    "NMR_3MS": ["CMRP_3MS", "CMR_3MS"],
}

# Reverse mapping: alias -> standard mnemonic
_REVERSE_MAP: Dict[str, str] = {}
for standard, aliases in ALIAS_MAP.items():
    for alias in aliases:
        _REVERSE_MAP[alias.upper()] = standard
    _REVERSE_MAP[standard.upper()] = standard


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
    return _REVERSE_MAP.get(mnemonic.upper(), mnemonic.upper())


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
        List of aliases
    """
    return ALIAS_MAP.get(standard_mnemonic.upper(), [])


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
        "NMR_PHIT": "NMR Total Porosity",
        "NMR_FFI": "NMR Free Fluid",
        "NMR_BVI": "NMR Bound Fluid",
        "NMR_T2LM": "NMR T2 Log Mean",
        "NMR_T2CUTOFF": "NMR T2 Cutoff",
        "NMR_KSDR": "NMR SDR Permeability",
        "NMR_KTIM": "NMR Timur/Coates Permeability",
        "NMR_3MS": "CMR 3ms Porosity",
    }

    return type_map.get(standard)


# Standard units for each curve type
STANDARD_UNITS: Dict[str, str] = {
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
    "NMR_PHIT": "pu",
    "NMR_FFI": "pu",
    "NMR_BVI": "pu",
    "NMR_T2LM": "ms",
    "NMR_T2CUTOFF": "ms",
    "NMR_KSDR": "mD",
    "NMR_KTIM": "mD",
    "NMR_3MS": "pu",
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
