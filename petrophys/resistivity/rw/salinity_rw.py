"""
Salinity-based Rw calculation.

This module provides formation water resistivity calculation
from salinity measurements.
"""

from typing import Union
import numpy as np


def rw_from_salinity(
    salinity_ppm: Union[float, np.ndarray],
    temperature: Union[float, np.ndarray],
    temp_unit: str = "F"
) -> Union[float, np.ndarray]:
    """
    Calculate Rw from salinity using empirical relationship.

    Uses the Arps formula variation for NaCl solutions.

    Parameters
    ----------
    salinity_ppm : float or np.ndarray
        Salinity in ppm NaCl equivalent
    temperature : float or np.ndarray
        Temperature
    temp_unit : str, optional
        Temperature unit

    Returns
    -------
    float or np.ndarray
        Formation water resistivity (ohm.m)
    """
    # Convert to Fahrenheit
    if temp_unit.upper() == "C":
        from petrophys.core.units import UnitConverter
        temperature = UnitConverter.celsius_to_fahrenheit(temperature)

    # Handle zero or negative salinity
    if isinstance(salinity_ppm, np.ndarray):
        salinity_ppm = np.where(salinity_ppm <= 0, 1, salinity_ppm)
    else:
        salinity_ppm = max(salinity_ppm, 1)

    # Empirical formula (based on Schlumberger Gen-9)
    # Rw at 75°F: Rw75 = 0.0123 + 3647.5 / salinity^0.955

    rw_75 = 0.0123 + 3647.5 / np.power(salinity_ppm, 0.955)

    # Correct to formation temperature using Arps
    rw = rw_75 * (75 + 6.77) / (temperature + 6.77)

    return rw


def salinity_from_rw(
    rw: Union[float, np.ndarray],
    temperature: Union[float, np.ndarray],
    temp_unit: str = "F"
) -> Union[float, np.ndarray]:
    """
    Calculate salinity from Rw.

    Inverse of rw_from_salinity.

    Parameters
    ----------
    rw : float or np.ndarray
        Formation water resistivity (ohm.m)
    temperature : float or np.ndarray
        Temperature
    temp_unit : str, optional
        Temperature unit

    Returns
    -------
    float or np.ndarray
        Salinity in ppm NaCl
    """
    # Convert to Fahrenheit
    if temp_unit.upper() == "C":
        from petrophys.core.units import UnitConverter
        temperature = UnitConverter.celsius_to_fahrenheit(temperature)

    # Correct Rw to 75°F
    rw_75 = rw * (temperature + 6.77) / (75 + 6.77)

    # Solve for salinity
    # Rw75 = 0.0123 + 3647.5 / salinity^0.955
    # salinity^0.955 = 3647.5 / (Rw75 - 0.0123)
    # salinity = (3647.5 / (Rw75 - 0.0123))^(1/0.955)

    if isinstance(rw_75, np.ndarray):
        denominator = np.where(rw_75 > 0.0123, rw_75 - 0.0123, 1e-10)
    else:
        denominator = max(rw_75 - 0.0123, 1e-10)

    salinity = np.power(3647.5 / denominator, 1 / 0.955)

    return salinity


def classify_water_salinity(salinity_ppm: Union[float, np.ndarray]) -> str:
    """
    Classify water based on salinity.

    Parameters
    ----------
    salinity_ppm : float or np.ndarray
        Salinity in ppm

    Returns
    -------
    str
        Classification ("fresh", "brackish", "saline", "brine")
    """
    if isinstance(salinity_ppm, np.ndarray):
        salinity_ppm = np.nanmean(salinity_ppm)

    if salinity_ppm < 1000:
        return "fresh"
    elif salinity_ppm < 10000:
        return "brackish"
    elif salinity_ppm < 100000:
        return "saline"
    else:
        return "brine"
