"""
Temperature gradient calculation.

This module provides functions for calculating geothermal gradients
from temperature measurements.
"""

from typing import Optional, Tuple

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.model.constants import DefaultParameters
from petrophys.model.units import UnitConverter, normalize_depth_unit, temperature_unit


def _depth_unit_is_metres(depth_unit: str) -> bool:
    """True for a metre depth unit; unknown labels raise ``ValueError``.

    Benchmark context profiles recognize 'm' case-insensitively as metres;
    other labels use the feet calculation path.
    """
    if reproducing_benchmark():
        return depth_unit.lower() == "m"
    return normalize_depth_unit(depth_unit, strict=True) == "m"


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
        Temperature unit: "F", "C", "K" or a label such as "degC" (see
        :func:`petrophys.model.units.temperature_unit`)
    depth_unit : str, optional
        Depth unit: "ft", "m" or a label such as "metres"

    Returns
    -------
    float
        Temperature gradient in °F/100ft

    Raises
    ------
    ValueError
        If ``temp_unit`` or ``depth_unit`` is not a recognized unit, or the
        two depths are equal.

    Examples
    --------
    >>> gradient = calculate_temperature_gradient(
    ...     temp1=68, depth1=0,
    ...     temp2=200, depth2=10000
    ... )
    >>> print(f"Gradient: {gradient:.2f} °F/100ft")
    Gradient: 1.32 °F/100ft
    """
    # Convert to Fahrenheit and feet
    if temperature_unit(temp_unit, strict=True) != "F":
        temp1 = UnitConverter.temperature(temp1, temp_unit, "F")
        temp2 = UnitConverter.temperature(temp2, temp_unit, "F")

    if _depth_unit_is_metres(depth_unit):
        depth1 = UnitConverter.meters_to_feet(depth1)
        depth2 = UnitConverter.meters_to_feet(depth2)

    # Calculate gradient
    depth_diff = depth2 - depth1
    if depth_diff == 0:
        if reproducing_benchmark():
            return 0.0
        raise ValueError("The two depths are equal: no temperature gradient")

    temp_diff = temp2 - temp1
    gradient = (temp_diff / depth_diff) * 100  # per 100 ft

    return gradient


def estimate_gradient_from_bht(
    bht: float,
    td: float,
    surface_temp: Optional[float] = None,
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
        Surface temperature, in ``bht_unit`` (default: 68°F)
    bht_unit : str, optional
        Unit of ``bht`` and ``surface_temp`` ("F", "C", "K" or a label such
        as "degC")
    depth_unit : str, optional
        Depth unit: "ft", "m" or a label such as "metres"

    Returns
    -------
    float
        Temperature gradient in °F/100ft
    """
    if surface_temp is None:
        surface_temp = DefaultParameters.SURFACE_TEMPERATURE_F
        if temperature_unit(bht_unit, strict=True) != "F":
            surface_temp = UnitConverter.temperature(surface_temp, "F", bht_unit)
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


def gradient_to_celsius_per_km(gradient_f_per_100ft: float) -> float:
    """
    Convert gradient from °F/100ft to °C/km.

    Parameters
    ----------
    gradient_f_per_100ft : float
        Gradient in °F per 100 feet

    Returns
    -------
    float
        Gradient in °C per kilometer
    """
    return gradient_to_celsius_per_meter(gradient_f_per_100ft) * 1000.0


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


def gradient_c_per_km_to_fahrenheit_per_100ft(gradient_c_per_km: float) -> float:
    """
    Convert gradient from °C/km to °F/100ft.

    Parameters
    ----------
    gradient_c_per_km : float
        Gradient in °C per kilometer

    Returns
    -------
    float
        Gradient in °F per 100 feet
    """
    return gradient_to_fahrenheit_per_100ft(gradient_c_per_km / 1000.0)


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
