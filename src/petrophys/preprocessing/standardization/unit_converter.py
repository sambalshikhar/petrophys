"""
Unit standardization and conversion for curve data.

This module provides functions to convert curve data between
different unit systems and standardize units.
"""

from typing import TYPE_CHECKING, Union, Optional, Tuple
import warnings
import numpy as np

from petrophys.model.units import UnitConverter, core_saturation_to_fraction, temperature_unit
from petrophys.preprocessing.standardization.alias_map import (
    get_standard_mnemonic,
    get_standard_unit,
)

if TYPE_CHECKING:
    from petrophys.model.curve import Curve


CORE_SATURATION_CURVES = frozenset({"SWC", "SOC", "SGC"})


def standardize_core_saturation(
    curve: "Curve",
    *,
    source_unit: Optional[str] = None,
    inplace: bool = False,
    warn: bool = True,
) -> "Curve":
    """Normalize a core saturation curve to fractions, preserving missing data.

    Explicit percent units are divided by 100. For missing or fractional units,
    infer percent only when at least two samples lie in (1, 100], at least 80%
    of positive plausible samples exceed 1, and their median is at least 2.
    Mixed or sparse ambiguous data are not rescaled; values
    outside the selected physical range are masked, never clipped. A warning
    accompanies inference or rejected values. All-null curves remain all-null.

    ``source_unit='%'`` or ``source_unit='v/v'`` overrides inference. The source
    curve is unchanged unless ``inplace=True``. Conversion evidence and warning
    messages are recorded in ``metadata.custom['core_saturation_normalization']``.
    """
    result = curve if inplace else curve.copy()
    original_unit = str(curve.unit or "")
    data, record = core_saturation_to_fraction(
        curve.data, original_unit, mnemonic=curve.mnemonic, source_unit=source_unit
    )
    factor, messages = record["factor"], record["warnings"]
    result._data = data
    result.unit = "v/v"
    previous = result.metadata.custom.get("core_saturation_normalization")
    # Keep the original conversion evidence when normalizing a second time.
    if not (previous and source_unit is None and original_unit == "v/v"
            and factor == 1.0 and not messages):
        result.metadata.custom["core_saturation_normalization"] = {
            "original_unit": original_unit,
            "source_unit": source_unit,
            "target_unit": "v/v",
            "factor": factor,
            "reason": record["reason"],
            "masked_count": record["masked_count"],
            "warnings": messages,
        }
    if warn:
        for message in messages:
            warnings.warn(message, UserWarning, stacklevel=2)
    return result


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


def _converts_temperature(from_unit: str, to_unit: str) -> bool:
    """True when both units are temperature scales and differ, e.g. degC and F.

    Temperature scales are offset from each other, so they have no
    conversion factor in :data:`UNIT_CONVERSIONS`.
    """
    source, target = temperature_unit(from_unit), temperature_unit(to_unit)
    return source is not None and target is not None and source != target


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
        if _converts_temperature(from_unit, to_unit):
            return UnitConverter.temperature(data, from_unit, to_unit)
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
    from petrophys.model.curve import Curve

    if not inplace:
        curve = curve.copy()

    if curve.unit.lower() == to_unit.lower():
        return curve

    factor = get_conversion_factor(curve.unit, to_unit)
    if factor is None:
        if not _converts_temperature(curve.unit, to_unit):
            raise ValueError(
                f"Cannot convert {curve.mnemonic} from {curve.unit} to {to_unit}"
            )
        curve._data = UnitConverter.temperature(curve._data, curve.unit, to_unit)
        curve.unit = to_unit
        return curve

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
    if get_standard_mnemonic(curve.mnemonic) in CORE_SATURATION_CURVES:
        return standardize_core_saturation(curve, inplace=inplace)
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
