"""
Unit standardization and conversion for curve data.

This module provides functions to convert curve data between
different unit systems and standardize units.
"""

from typing import Union, Optional, Tuple
import numpy as np

from petrophys.base.units import UnitConverter
from petrophys.qaqc.standardization.alias_map import (
    get_standard_mnemonic,
    get_standard_unit,
)


# Unit conversion rules: (from_unit, to_unit) -> conversion_factor or function
UNIT_CONVERSIONS = {
    # Porosity
    ("pu", "v/v"): 0.01,
    ("%", "v/v"): 0.01,
    ("percent", "v/v"): 0.01,
    ("decimal", "v/v"): 1.0,
    ("m3/m3", "v/v"): 1.0,
    ("v/v", "pu"): 100.0,
    ("v/v", "%"): 100.0,

    # Density
    ("g/cc", "g/cm3"): 1.0,
    ("kg/m3", "g/cm3"): 0.001,
    ("lb/ft3", "g/cm3"): 0.016018,

    # Resistivity
    ("ohmm", "ohm.m"): 1.0,
    ("ohm-m", "ohm.m"): 1.0,
    ("ohm.ft", "ohm.m"): 0.3048,

    # Slowness
    ("us/m", "us/ft"): 0.3048,
    ("μs/m", "us/ft"): 0.3048,
    ("μs/ft", "us/ft"): 1.0,

    # Depth
    ("m", "ft"): 3.28084,
    ("ft", "m"): 0.3048,

    # Gamma Ray
    ("gapi", "api"): 1.0,

    # Caliper / bit size
    ("mm", "in"): 0.0393701,
    ("cm", "in"): 0.393701,
    ("in", "mm"): 25.4,
}


def get_conversion_factor(from_unit: str, to_unit: str) -> Optional[float]:
    """
    Get conversion factor between units.

    Parameters
    ----------
    from_unit : str
        Source unit
    to_unit : str
        Target unit

    Returns
    -------
    float or None
        Conversion factor, or None if no conversion available
    """
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    if from_unit == to_unit:
        return 1.0

    key = (from_unit, to_unit)
    if key in UNIT_CONVERSIONS:
        return UNIT_CONVERSIONS[key]

    # Try reverse
    reverse_key = (to_unit, from_unit)
    if reverse_key in UNIT_CONVERSIONS:
        return 1.0 / UNIT_CONVERSIONS[reverse_key]

    return None


def convert_units(
    data: Union[float, np.ndarray],
    from_unit: str,
    to_unit: str
) -> Union[float, np.ndarray]:
    """
    Convert data between units.

    Parameters
    ----------
    data : float or np.ndarray
        Data to convert
    from_unit : str
        Source unit
    to_unit : str
        Target unit

    Returns
    -------
    float or np.ndarray
        Converted data

    Raises
    ------
    ValueError
        If no conversion is available
    """
    factor = get_conversion_factor(from_unit, to_unit)
    if factor is None:
        raise ValueError(
            f"No conversion available from {from_unit} to {to_unit}"
        )
    return data * factor


def convert_curve_units(
    curve: "Curve",
    to_unit: str,
    inplace: bool = False
) -> "Curve":
    """
    Convert curve data to specified unit.

    Parameters
    ----------
    curve : Curve
        Curve to convert
    to_unit : str
        Target unit
    inplace : bool, optional
        Modify curve in place (default: False)

    Returns
    -------
    Curve
        Curve with converted data
    """
    from petrophys.base.curve import Curve

    if not inplace:
        curve = curve.copy()

    if curve.unit.lower() == to_unit.lower():
        return curve

    factor = get_conversion_factor(curve.unit, to_unit)
    if factor is None:
        raise ValueError(
            f"Cannot convert {curve.mnemonic} from {curve.unit} to {to_unit}"
        )

    curve._data = curve._data * factor
    curve.unit = to_unit
    return curve


def standardize_units(
    curve: "Curve",
    inplace: bool = False
) -> "Curve":
    """
    Convert curve to standard units based on curve type.

    Parameters
    ----------
    curve : Curve
        Curve to standardize
    inplace : bool, optional
        Modify curve in place (default: False)

    Returns
    -------
    Curve
        Curve with standardized units
    """
    standard_unit = get_standard_unit(curve.mnemonic)
    if standard_unit is None:
        return curve if inplace else curve.copy()

    return convert_curve_units(curve, standard_unit, inplace=inplace)


def standardize_porosity_units(
    data: Union[float, np.ndarray],
    current_unit: str
) -> Tuple[np.ndarray, str]:
    """
    Standardize porosity to fraction (v/v).

    Parameters
    ----------
    data : float or np.ndarray
        Porosity data
    current_unit : str
        Current unit

    Returns
    -------
    tuple
        (converted_data, "v/v")
    """
    data = np.asarray(data, dtype=np.float64)
    current_unit = current_unit.lower().strip()

    if current_unit in ["v/v", "fraction", "decimal"]:
        # Check if data looks like percentage
        finite = data[np.isfinite(data)]
        if finite.size and np.nanmax(finite) > 1.0:
            data = data / 100.0
    elif current_unit in ["%", "pu", "percent"]:
        data = data / 100.0
    else:
        # Unknown unit, try to guess
        finite = data[np.isfinite(data)]
        if finite.size and np.nanmax(finite) > 1.0:
            data = data / 100.0

    return data, "v/v"


def standardize_resistivity_units(
    data: Union[float, np.ndarray],
    current_unit: str
) -> Tuple[np.ndarray, str]:
    """
    Standardize resistivity to ohm.m.

    Parameters
    ----------
    data : float or np.ndarray
        Resistivity data
    current_unit : str
        Current unit

    Returns
    -------
    tuple
        (converted_data, "ohm.m")
    """
    data = np.asarray(data, dtype=np.float64)
    factor = get_conversion_factor(current_unit, "ohm.m")
    if factor:
        data = data * factor
    return data, "ohm.m"


def detect_porosity_unit(data: np.ndarray) -> str:
    """
    Detect if porosity is in fraction or percentage.

    Parameters
    ----------
    data : np.ndarray
        Porosity data

    Returns
    -------
    str
        Detected unit ("v/v" or "%")
    """
    # Filter valid data
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return "v/v"

    max_val = np.max(valid_data)
    if max_val > 1.0:
        return "%"
    return "v/v"
