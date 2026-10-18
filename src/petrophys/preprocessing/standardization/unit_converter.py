"""
Unit standardization and conversion for curve data.

This module provides functions to convert curve data between
different unit systems and standardize units.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator, Union, Optional, Tuple
import warnings
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
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

#: Other spellings of units in :data:`UNIT_CONVERSIONS` (lower case), e.g. the
#: LAS 2.0 specification's ``K/M3`` and ``US/M`` and the micro sign U+00B5 that
#: cp1252/Latin-1 files use for the Greek mu U+03BC.
UNIT_SPELLINGS = {
    "k/m3": "kg/m3",
    "g/c3": "g/cm3",
    "us/f": "us/ft",
    "usec/ft": "us/ft",
    "usec/m": "us/m",
    "µs/m": "μs/m",
    "µs/ft": "μs/ft",
    "ft3/ft3": "v/v",
    "dec": "v/v",
    "frac": "v/v",
    "fraction": "v/v",
}

_BENCHMARK_UNITS: ContextVar[bool] = ContextVar("petrophys_benchmark_units", default=False)


@contextmanager
def benchmark_unit_table() -> Iterator[None]:
    """Run the block with the unit handling of the benchmark packages.

    The workflow uses it under the "geolink" calculation profile; context
    exports get it from :func:`~petrophys._internal.benchmark.historical_benchmark`.
    """
    token = _BENCHMARK_UNITS.set(True)
    try:
        yield
    finally:
        _BENCHMARK_UNITS.reset(token)


def _benchmark_units() -> bool:
    """True when units are handled as the benchmark packages did.

    Those packages had no :data:`UNIT_SPELLINGS`.
    """
    return _BENCHMARK_UNITS.get() or reproducing_benchmark()


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
    return _conversion_factor(from_unit, to_unit, spellings=not _benchmark_units())


def _conversion_factor(from_unit: str, to_unit: str, *, spellings: bool) -> Optional[float]:
    """:func:`get_conversion_factor`, with :data:`UNIT_SPELLINGS` only if ``spellings``."""
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()
    if spellings:
        from_unit = UNIT_SPELLINGS.get(from_unit, from_unit)
        to_unit = UNIT_SPELLINGS.get(to_unit, to_unit)

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

    A curve with a blank unit, or a unit with no known conversion to the
    standard one, is returned unconverted with a ``UserWarning``.

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
    unit = str(curve.unit or "").strip()
    if unit.lower() != standard_unit.lower() and (
        get_conversion_factor(unit, standard_unit) is None
        and not _converts_temperature(unit, standard_unit)
    ):
        warnings.warn(
            f"{curve.mnemonic}: unit {unit!r} cannot be converted to {standard_unit!r}; "
            "the curve is left unconverted.",
            UserWarning,
            stacklevel=2,
        )
        return curve if inplace else curve.copy()

    return convert_curve_units(curve, standard_unit, inplace=inplace)


def standardize_porosity_units(
    data: Union[float, np.ndarray],
    current_unit: str
) -> Tuple[np.ndarray, str]:
    """
    Standardize porosity to fraction (v/v).

    Percent units are divided by 100. For fractional or unknown units the data
    are divided by 100 only when their median exceeds 1, so a few spikes above
    1 do not rescale a fractional curve.

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

    if current_unit in ["%", "pu", "percent"]:
        data = data / 100.0
    else:
        # Fractional or unknown unit: check if data looks like percentage
        finite = data[np.isfinite(data)]
        if finite.size and np.median(finite) > 1.0:
            data = data / 100.0

    return data, "v/v"


def standardize_resistivity_units(
    data: Union[float, np.ndarray],
    current_unit: str
) -> Tuple[np.ndarray, str]:
    """
    Standardize resistivity to ohm.m.

    A unit with no known conversion to ohm.m (for example a conductivity in
    mS/m) is returned unconverted, with its own label and a ``UserWarning``.

    Parameters
    ----------
    data : float or np.ndarray
        Resistivity data
    current_unit : str
        Current unit

    Returns
    -------
    tuple
        (converted_data, "ohm.m"), or (data, current_unit) when unconvertible
    """
    data = np.asarray(data, dtype=np.float64)
    factor = get_conversion_factor(current_unit, "ohm.m")
    if factor is None:
        warnings.warn(
            f"Resistivity unit {current_unit!r} cannot be converted to 'ohm.m'; "
            "the data are left unconverted.",
            UserWarning,
            stacklevel=2,
        )
        return data, current_unit
    return data * factor, "ohm.m"


def detect_porosity_unit(data: np.ndarray) -> str:
    """
    Detect if porosity is in fraction or percentage.

    Percentage is reported when the median of the valid samples exceeds 1.

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

    if np.median(valid_data) > 1.0:
        return "%"
    return "v/v"
