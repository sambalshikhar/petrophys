"""
Unit conversion utilities for petrophysical calculations.

This module provides comprehensive unit handling for common
petrophysical measurements including depth, porosity, resistivity,
temperature, and sonic measurements.
"""

from petrophys._internal.advice import Advice
import re
from typing import Union, Optional
from enum import Enum
import numpy as np


class DepthUnit(Enum):
    """Depth measurement units."""
    FEET = "ft"
    METERS = "m"


class TemperatureUnit(Enum):
    """Temperature measurement units."""
    FAHRENHEIT = "F"
    CELSIUS = "C"
    KELVIN = "K"


class DensityUnit(Enum):
    """Density measurement units."""
    G_CM3 = "g/cm3"
    KG_M3 = "kg/m3"
    LB_FT3 = "lb/ft3"


class PorosityUnit(Enum):
    """Porosity measurement units."""
    FRACTION = "v/v"
    PERCENT = "%"
    DECIMAL = "decimal"


class ResistivityUnit(Enum):
    """Resistivity measurement units."""
    OHM_M = "ohm.m"
    OHM_FT = "ohm.ft"


class SlownessUnit(Enum):
    """Sonic slowness measurement units."""
    US_FT = "us/ft"
    US_M = "us/m"


class UnitRegistry:
    """
    Registry of unit definitions and conversion factors.

    This class maintains conversion factors between units
    for various measurement types.
    """

    # Depth conversions (to meters)
    DEPTH_TO_M = {
        "ft": 0.3048,
        "feet": 0.3048,
        "m": 1.0,
        "meters": 1.0,
    }

    # Density conversions (to g/cm³)
    DENSITY_TO_G_CM3 = {
        "g/cm3": 1.0,
        "g/cc": 1.0,
        "kg/m3": 0.001,
        "lb/ft3": 0.016018463,
    }

    # Porosity conversions (to fraction)
    POROSITY_TO_FRACTION = {
        "v/v": 1.0,
        "fraction": 1.0,
        "decimal": 1.0,
        "%": 0.01,
        "percent": 0.01,
        "pu": 0.01,
    }

    # Slowness conversions (to μs/ft)
    SLOWNESS_TO_US_FT = {
        "us/ft": 1.0,
        "μs/ft": 1.0,
        "us/m": 0.3048,
        "μs/m": 0.3048,
    }

    # Resistivity conversions (to ohm.m)
    RESISTIVITY_TO_OHM_M = {
        "ohm.m": 1.0,
        "ohmm": 1.0,
        "ohm-m": 1.0,
        "ohm.ft": 0.3048,
        "ohmft": 0.3048,
        "ohm-ft": 0.3048,
    }


def _lookup(registry: dict, unit: str, quantity: str) -> float:
    """Return the conversion factor for ``unit`` or raise on unknown units.

    A silent factor of 1.0 for an unrecognized unit string would return a
    confidently wrong number, which is worse than failing loudly.
    """
    try:
        return float(registry[unit])
    except KeyError:
        supported = ", ".join(sorted(registry))
        raise ValueError(
            f"Unrecognized {quantity} unit '{unit}'. Supported units: {supported}"
        ) from None


class UnitConverter:
    """
    Comprehensive unit converter for petrophysical measurements.

    Examples
    --------
    >>> # Convert depth
    >>> depth_m = UnitConverter.depth(100, "ft", "m")
    >>> print(depth_m)
    30.48

    >>> # Convert temperature
    >>> temp_c = UnitConverter.temperature(68, "F", "C")
    >>> print(temp_c)
    20.0

    >>> # Convert porosity
    >>> phi_frac = UnitConverter.porosity(20, "%", "v/v")
    >>> print(phi_frac)
    0.2
    """

    @staticmethod
    def depth(
        value: Union[float, np.ndarray],
        from_unit: str,
        to_unit: str
    ) -> Union[float, np.ndarray]:
        """
        Convert depth between units.

        Parameters
        ----------
        value : float or np.ndarray
            Depth value(s) to convert
        from_unit : str
            Source unit ("ft", "m")
        to_unit : str
            Target unit ("ft", "m")

        Returns
        -------
        float or np.ndarray
            Converted depth value(s)
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        # Convert to meters first
        to_m = _lookup(UnitRegistry.DEPTH_TO_M, from_unit, "depth")
        from_m = _lookup(UnitRegistry.DEPTH_TO_M, to_unit, "depth")

        return value * to_m / from_m

    @staticmethod
    def temperature(
        value: Union[float, np.ndarray],
        from_unit: str,
        to_unit: str
    ) -> Union[float, np.ndarray]:
        """
        Convert temperature between units.

        Parameters
        ----------
        value : float or np.ndarray
            Temperature value(s) to convert
        from_unit : str
            Source unit: any label :func:`temperature_unit` recognizes, such
            as "F", "degC" or "K"
        to_unit : str
            Target unit, likewise

        Returns
        -------
        float or np.ndarray
            Converted temperature value(s)

        Raises
        ------
        ValueError
            If either unit is not a recognized temperature unit.
        """
        source = temperature_unit(from_unit, strict=True)
        destination = temperature_unit(to_unit, strict=True)

        # First convert to Celsius
        if source == "F":
            celsius = (value - 32) * 5 / 9
        elif source == "K":
            celsius = value - 273.15
        else:  # Celsius
            celsius = value

        # Then convert to target
        if destination == "F":
            return celsius * 9 / 5 + 32
        elif destination == "K":
            return celsius + 273.15
        else:  # Celsius
            return celsius

    @staticmethod
    def density(
        value: Union[float, np.ndarray],
        from_unit: str,
        to_unit: str
    ) -> Union[float, np.ndarray]:
        """
        Convert density between units.

        Parameters
        ----------
        value : float or np.ndarray
            Density value(s) to convert
        from_unit : str
            Source unit ("g/cm3", "kg/m3", "lb/ft3")
        to_unit : str
            Target unit ("g/cm3", "kg/m3", "lb/ft3")

        Returns
        -------
        float or np.ndarray
            Converted density value(s)
        """
        from_unit = from_unit.lower().replace(" ", "")
        to_unit = to_unit.lower().replace(" ", "")

        # Convert to g/cm³ first
        to_gcc = _lookup(UnitRegistry.DENSITY_TO_G_CM3, from_unit, "density")
        from_gcc = _lookup(UnitRegistry.DENSITY_TO_G_CM3, to_unit, "density")

        return value * to_gcc / from_gcc

    @staticmethod
    def porosity(
        value: Union[float, np.ndarray],
        from_unit: str,
        to_unit: str
    ) -> Union[float, np.ndarray]:
        """
        Convert porosity between units.

        Parameters
        ----------
        value : float or np.ndarray
            Porosity value(s) to convert
        from_unit : str
            Source unit ("v/v", "%", "pu")
        to_unit : str
            Target unit ("v/v", "%", "pu")

        Returns
        -------
        float or np.ndarray
            Converted porosity value(s)
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        # Convert to fraction first
        to_frac = _lookup(UnitRegistry.POROSITY_TO_FRACTION, from_unit, "porosity")
        from_frac = _lookup(UnitRegistry.POROSITY_TO_FRACTION, to_unit, "porosity")

        return value * to_frac / from_frac

    @staticmethod
    def slowness(
        value: Union[float, np.ndarray],
        from_unit: str,
        to_unit: str
    ) -> Union[float, np.ndarray]:
        """
        Convert sonic slowness between units.

        Parameters
        ----------
        value : float or np.ndarray
            Slowness value(s) to convert
        from_unit : str
            Source unit ("us/ft", "us/m")
        to_unit : str
            Target unit ("us/ft", "us/m")

        Returns
        -------
        float or np.ndarray
            Converted slowness value(s)
        """
        # Fold both micro glyphs: GREEK SMALL LETTER MU (U+03BC) and the
        # MICRO SIGN (U+00B5) that most keyboards and LAS files produce.
        from_unit = from_unit.lower().replace("μ", "u").replace("µ", "u")
        to_unit = to_unit.lower().replace("μ", "u").replace("µ", "u")

        to_usft = _lookup(UnitRegistry.SLOWNESS_TO_US_FT, from_unit, "slowness")
        from_usft = _lookup(UnitRegistry.SLOWNESS_TO_US_FT, to_unit, "slowness")

        return value * to_usft / from_usft

    @staticmethod
    def resistivity(
        value: Union[float, np.ndarray],
        from_unit: str,
        to_unit: str
    ) -> Union[float, np.ndarray]:
        """
        Convert resistivity between units.

        Parameters
        ----------
        value : float or np.ndarray
            Resistivity value(s) to convert
        from_unit : str
            Source unit ("ohm.m", "ohm.ft")
        to_unit : str
            Target unit ("ohm.m", "ohm.ft")

        Returns
        -------
        float or np.ndarray
            Converted resistivity value(s)
        """
        from_unit = from_unit.lower().replace(" ", "")
        to_unit = to_unit.lower().replace(" ", "")

        to_ohmm = _lookup(UnitRegistry.RESISTIVITY_TO_OHM_M, from_unit, "resistivity")
        from_ohmm = _lookup(UnitRegistry.RESISTIVITY_TO_OHM_M, to_unit, "resistivity")

        return value * to_ohmm / from_ohmm

    @staticmethod
    def feet_to_meters(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert feet to meters."""
        return value * 0.3048

    @staticmethod
    def meters_to_feet(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert meters to feet."""
        return value / 0.3048

    @staticmethod
    def fahrenheit_to_celsius(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert Fahrenheit to Celsius."""
        return (value - 32) * 5 / 9

    @staticmethod
    def celsius_to_fahrenheit(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert Celsius to Fahrenheit."""
        return value * 9 / 5 + 32


_FEET_ALIASES = frozenset({"f", "ft", "fts", "feet", "foot"})
_METRE_ALIASES = frozenset({"m", "mtr", "meter", "meters", "metre", "metres"})


def normalize_depth_unit(
    unit: Optional[str],
    *,
    strict: bool = True,
    default: Optional[str] = None,
) -> str:
    """Return the canonical depth unit, ``"m"`` or ``"ft"``.

    Parameters
    ----------
    unit : str or None
        Depth unit as written in a file or supplied by a user, e.g. ``"M"``,
        ``"metres"``, ``"F"``, ``"ft."`` or ``"Feet"``. Case, surrounding
        whitespace and dots are ignored.
    strict : bool, default True
        If True, an unrecognized unit raises ``ValueError``. If False, it is
        returned unchanged (stripped) so callers reading third-party files can
        keep the original label and report it.
    default : str, optional
        Value returned when ``unit`` is empty or None. When omitted, an empty
        unit is treated like an unrecognized one.

    Returns
    -------
    str
        ``"m"`` or ``"ft"`` (or the stripped original text when
        ``strict=False`` and the unit is not recognized).

    Raises
    ------
    ValueError
        If ``strict`` is True and the unit is not a recognized depth unit.

    Examples
    --------
    >>> normalize_depth_unit("Metres")
    'm'
    >>> normalize_depth_unit("F")
    'ft'
    """
    text = "" if unit is None else str(unit).strip()
    if not text and default is not None:
        return normalize_depth_unit(default, strict=strict)
    key = text.lower().replace(".", "").replace(" ", "")
    if key in _FEET_ALIASES:
        return "ft"
    if key in _METRE_ALIASES:
        return "m"
    if strict:
        raise ValueError(
            f"Unsupported depth unit {unit!r}; use 'm' (metres) or 'ft' (feet)."
        )
    return text


def temperature_unit(unit: Optional[str], *, strict: bool = False) -> Optional[str]:
    """Classify a temperature unit label as ``"F"``, ``"C"``, ``"K"`` or None.

    Units are matched on whole words, so labels such as ``"deg C (from log)"``
    or ``"Celsius (ref)"`` are read as Celsius even though they contain the
    letter ``f``. Labels naming more than one scale, or none, return None, or
    raise ``ValueError`` when ``strict`` is True (for unit arguments, where
    guessing a scale would silently convert with the wrong one).

    Examples
    --------
    >>> temperature_unit("deg C (from log)")
    'C'
    >>> temperature_unit("°F")
    'F'
    >>> temperature_unit("") is None
    True
    """
    text = "" if unit is None else str(unit).lower().replace("°", " ").replace("º", " ")
    # The single-character signs U+2103 (degree Celsius) and U+2109 (degree Fahrenheit)
    text = text.replace("\u2103", " degc ").replace("\u2109", " degf ")
    tokens = re.findall(r"[a-z]+", text)
    joined = ["".join(tokens[i:i + 2]) for i in range(len(tokens))]
    words = set(tokens) | set(joined)
    # "oF" is read as Fahrenheit only on its own, since "of" is also an English word
    if tokens == ["of"]:
        words.add("degf")
    scales = set()
    if words & {"f", "degf", "fahrenheit"}:
        scales.add("F")
    if words & {"c", "degc", "oc", "celsius", "centigrade"}:
        scales.add("C")
    if words & {"k", "degk", "kelvin"}:
        scales.add("K")
    scale = scales.pop() if len(scales) == 1 else None
    if scale is None and strict:
        raise ValueError(
            f"Unrecognized temperature unit {unit!r}; use F (degF), C (degC) or K."
        )
    return scale


def temperature_to_fahrenheit(value: float, unit: Optional[str]) -> float:
    """Convert a scalar temperature to degrees Fahrenheit.

    Parameters
    ----------
    value : float
        Temperature value.
    unit : str or None
        Unit label of ``value`` (see :func:`temperature_unit`). Unrecognized
        or empty labels are assumed to be Fahrenheit, the convention of LAS
        header temperatures such as BHT.

    Returns
    -------
    float
        Temperature in degrees Fahrenheit.

    Examples
    --------
    >>> temperature_to_fahrenheit(100.0, "degC")
    212.0
    >>> temperature_to_fahrenheit(150.0, "DEGF")
    150.0
    """
    scale = temperature_unit(unit)
    if scale == "C":
        return float(value) * 9.0 / 5.0 + 32.0
    if scale == "K":
        return (float(value) - 273.15) * 9.0 / 5.0 + 32.0
    return float(value)


PERCENT_SATURATION_UNITS = frozenset({"%", "percent", "percentage", "pct", "pu", "p.u."})
FRACTION_SATURATION_UNITS = frozenset(
    {"v/v", "frac", "fraction", "decimal", "m3/m3", "ft3/ft3", "1"}
)
#: Further words of fraction labels, e.g. "dec", "cfcf", "frac." or "v/v decimal".
_FRACTION_LABEL_WORDS = FRACTION_SATURATION_UNITS | {
    "dec", "unitless", "cfcf", "vol/vol", "ratio", "fract", "pv/pv",
}


def _is_fraction_label(label: str) -> bool:
    """True when every word of ``label`` (dots read as spaces) names a fraction."""
    words = label.replace(".", " ").split()
    return bool(words) and all(word in _FRACTION_LABEL_WORDS for word in words)


def core_saturation_to_fraction(
    values,
    unit: Optional[str] = None,
    *,
    mnemonic: str = "saturation",
    source_unit: Optional[str] = None,
    quantity: str = "saturation",
):
    """Convert core saturation values to fractions (v/v).

    Explicit percent units are divided by 100. For missing or fractional units,
    percent is inferred only when at least two samples lie in (1, 100], at least
    80% of positive plausible samples exceed 1, and their median is at least 2:
    core laboratories often label percentages as fractions. Mixed or sparse
    ambiguous data are not rescaled. Values outside 0-1 after conversion are
    masked (never clipped), and ``-999``-style nulls become NaN.

    Fraction labels include variants such as ``dec``, ``unitless``, ``cfcf``,
    ``vol/vol``, ``frac.`` and ``v/v decimal``. An unrecognized label is read
    like a missing one, with the scale inferred from the values and a warning.
    With ``quantity="porosity"``, a percent label on values that are
    fractions (median below 1, none above 1) is overridden with a warning
    and the values are used as fractions. Inside the benchmark reproduction
    (:func:`~petrophys._internal.benchmark.reproducing_benchmark`) only the
    labels of :data:`PERCENT_SATURATION_UNITS` and
    :data:`FRACTION_SATURATION_UNITS` are read, an unrecognized label raises
    and a percent label is always divided by 100.

    Parameters
    ----------
    values : array-like
        Core saturation samples.
    unit : str, optional
        Unit label from the source (``"%"``, ``"frac"``, ``"v/v"`` ...).
    mnemonic : str
        Name used in messages.
    source_unit : str, optional
        ``"%"`` or ``"v/v"`` to override the label and skip inference.
    quantity : str
        ``"saturation"`` or ``"porosity"``.

    Returns
    -------
    fractions : numpy.ndarray
        Saturation in v/v, with invalid samples set to NaN.
    record : dict
        ``original_unit``, ``source_unit``, ``factor``, ``reason``
        (``declared_percent``, ``fraction``, ``inferred_percent`` or
        ``explicit_override``), ``masked_count`` and ``warnings``.

    Raises
    ------
    ValueError
        If ``source_unit`` is not a recognized percent or fraction unit, or,
        inside the benchmark reproduction, the unit label is not.
    """
    from petrophys._internal.benchmark import reproducing_benchmark

    historical = reproducing_benchmark()
    original_unit = str(unit or "")
    label = str(source_unit if source_unit is not None else original_unit).strip().lower()
    known = PERCENT_SATURATION_UNITS | FRACTION_SATURATION_UNITS | {"", "-", "none", "nan"}
    unknown_label = label not in known and (historical or not _is_fraction_label(label))
    if unknown_label and (historical or source_unit is not None):
        raise ValueError(
            Advice(f"{mnemonic}: unknown core {'saturation' if historical else quantity} unit {label!r}; "
            "specify source_unit='%' or source_unit='v/v'.", ('source_unit',))
        )

    data = np.array(values, dtype=float, copy=True)
    data[data <= -900.0] = np.nan
    # The benchmark packages' messages call every core quantity a saturation.
    noun = "saturation" if historical else quantity
    factor = 0.01 if label in PERCENT_SATURATION_UNITS else 1.0
    reason = "declared_percent" if factor == 0.01 else "fraction"
    messages = []
    if unknown_label:
        reason = "unknown_unit"
        messages.append(
            f"{mnemonic}: unknown core {quantity} unit {original_unit!r}; "
            "the scale is inferred from the values. Verify source units."
        )
    if source_unit is not None:
        reason = "explicit_override"
    elif factor == 0.01 and quantity == "porosity" and not historical:
        positive = data[np.isfinite(data) & (data > 0.0)]
        if positive.size >= 2 and np.median(positive) < 1.0 and np.max(positive) <= 1.0:
            factor, reason = 1.0, "fraction_despite_percent_unit"
            messages.append(
                f"{mnemonic}: core {quantity} labelled {original_unit!r} holds fractions "
                f"(median {float(np.median(positive)):.3g}); used as v/v without dividing "
                "by 100. Verify source units."
            )
    elif factor == 1.0:
        plausible = data[np.isfinite(data) & (data >= 0.0) & (data <= 100.0)]
        above_one = np.count_nonzero(plausible > 1.0)
        fractional = np.count_nonzero((plausible > 0.0) & (plausible <= 1.0))
        positive = plausible[plausible > 0.0]
        if (above_one >= 2 and above_one >= 0.8 * (above_one + fractional)
                and np.median(positive) >= 2.0):
            factor, reason = 0.01, "inferred_percent"
            messages.append(
                f"{mnemonic}: inferred percent core {noun} despite "
                f"unit {original_unit!r}; divided by 100 to v/v. Verify source units."
            )
        elif above_one:
            messages.append(
                f"{mnemonic}: ambiguous core {noun} scale; not rescaled. "
                "Values outside 0-1 are masked. Confirm source units and use "
                "standardize_core_saturation(source_unit='%' or 'v/v') on the original curve."
            )

    data *= factor
    invalid = np.isinf(data) | (np.isfinite(data) & ((data < 0.0) | (data > 1.0)))
    invalid_count = int(np.count_nonzero(invalid))
    data[invalid] = np.nan
    if invalid_count:
        messages.append(
            f"{mnemonic}: masked {invalid_count} invalid core {noun} "
            "sample(s) outside 0-1 after normalization."
        )
    return data, {
        "original_unit": original_unit,
        "source_unit": source_unit,
        "factor": factor,
        "reason": reason,
        "masked_count": invalid_count,
        "warnings": messages,
    }

