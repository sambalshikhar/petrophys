"""
Temperature gradient calculation.

This module provides functions for calculating geothermal gradients
from temperature measurements.
"""

from typing import Optional, Tuple, Union
import numpy as np

from petrophys.base.constants import DefaultParameters
from petrophys.base.units import UnitConverter


def calculate_temperature_gradient(
    temp1: float,
    depth1: float,
    temp2: float,
    depth2: float,
    temp_unit: str = "F",
    depth_unit: str = "ft"
) -> float:
    """
    Calculate temperature gradient between two points.

    Parameters
    ----------
    temp1 : float
        Temperature at first point
    depth1 : float
        Depth at first point
    temp2 : float
        Temperature at second point
    depth2 : float
        Depth at second point
    temp_unit : str, optional
        Temperature unit ("F" or "C")
    depth_unit : str, optional
        Depth unit ("ft" or "m")

    Returns
    -------
    float
        Temperature gradient in °F/100ft

    Examples
    --------
    >>> gradient = calculate_temperature_gradient(
    ...     temp1=68, depth1=0,
    ...     temp2=200, depth2=10000
    ... )
    >>> print(f"Gradient: {gradient:.2f} °F/100ft")
    """
    # Convert to Fahrenheit and feet
    if temp_unit.upper() == "C":
        temp1 = UnitConverter.celsius_to_fahrenheit(temp1)
        temp2 = UnitConverter.celsius_to_fahrenheit(temp2)

    if depth_unit.lower() == "m":
        depth1 = UnitConverter.meters_to_feet(depth1)
        depth2 = UnitConverter.meters_to_feet(depth2)

    # Calculate gradient
    depth_diff = depth2 - depth1
    if depth_diff == 0:
        return 0.0

    temp_diff = temp2 - temp1
    gradient = (temp_diff / depth_diff) * 100  # per 100 ft

    return gradient


def estimate_gradient_from_bht(
    bht: float,
    td: float,
    surface_temp: float = DefaultParameters.SURFACE_TEMPERATURE_F,
    bht_unit: str = "F",
    depth_unit: str = "ft"
) -> float:
    """
    Estimate temperature gradient from BHT and TD.

    Parameters
    ----------
    bht : float
        Bottom hole temperature
    td : float
        Total depth
    surface_temp : float, optional
        Surface temperature (default: 68°F)
    bht_unit : str, optional
        BHT unit ("F" or "C")
    depth_unit : str, optional
        Depth unit ("ft" or "m")

    Returns
    -------
    float
        Temperature gradient in °F/100ft
    """
    return calculate_temperature_gradient(
        temp1=surface_temp,
        depth1=0,
        temp2=bht,
        depth2=td,
        temp_unit=bht_unit,
        depth_unit=depth_unit
    )


def gradient_to_celsius_per_meter(gradient_f_per_100ft: float) -> float:
    """
    Convert gradient from °F/100ft to °C/m.

    Parameters
    ----------
    gradient_f_per_100ft : float
        Gradient in °F per 100 feet

    Returns
    -------
    float
        Gradient in °C per meter
    """
    # °F/100ft to °C/m
    # 1°F = 5/9°C, 100ft = 30.48m
    return gradient_f_per_100ft * (5 / 9) / 30.48


def gradient_to_fahrenheit_per_100ft(gradient_c_per_m: float) -> float:
    """
    Convert gradient from °C/m to °F/100ft.

    Parameters
    ----------
    gradient_c_per_m : float
        Gradient in °C per meter

    Returns
    -------
    float
        Gradient in °F per 100 feet
    """
    return gradient_c_per_m * (9 / 5) * 30.48


def typical_gradient_range() -> Tuple[float, float]:
    """
    Return typical geothermal gradient range.

    Returns
    -------
    tuple
        (min, max) gradient in °F/100ft

    Notes
    -----
    Typical gradients range from about 1.0 to 2.5 °F/100ft
    (approximately 18-45 °C/km)
    """
    return (1.0, 2.5)
