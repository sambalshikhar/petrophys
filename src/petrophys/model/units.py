"""
Unit conversion utilities for petrophysical calculations.

This module provides comprehensive unit handling for common
petrophysical measurements including depth, porosity, resistivity,
temperature, and sonic measurements.
"""

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
        return registry[unit]
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
            Source unit ("F", "C", "K")
        to_unit : str
            Target unit ("F", "C", "K")

        Returns
        -------
        float or np.ndarray
            Converted temperature value(s)
        """
        from_unit = from_unit.upper()
        to_unit = to_unit.upper()

        for unit in (from_unit, to_unit):
            if unit not in ("F", "C", "K"):
                raise ValueError(
                    f"Unrecognized temperature unit '{unit}'. "
                    "Supported units: C, F, K"
                )

        # First convert to Celsius
        if from_unit == "F":
            celsius = (value - 32) * 5 / 9
        elif from_unit == "K":
            celsius = value - 273.15
        else:  # Celsius
            celsius = value

        # Then convert to target
        if to_unit == "F":
            return celsius * 9 / 5 + 32
        elif to_unit == "K":
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
