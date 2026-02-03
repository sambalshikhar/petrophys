"""
Temperature corrections for resistivity measurements.

This module provides functions for correcting resistivity
measurements to formation temperature.
"""

from typing import Union
import numpy as np

from petrophys.core.units import UnitConverter


def arps_equation(
    r_measured: Union[float, np.ndarray],
    t_measured: Union[float, np.ndarray],
    t_formation: Union[float, np.ndarray],
    temp_unit: str = "F"
) -> Union[float, np.ndarray]:
    """
    Correct resistivity to formation temperature using Arps equation.

    The Arps equation relates resistivity at different temperatures:
    R2 = R1 * (T1 + k) / (T2 + k)

    where k = 6.77 for temperatures in °F (or 21.5 for °C)

    Parameters
    ----------
    r_measured : float or np.ndarray
        Resistivity at measured temperature
    t_measured : float or np.ndarray
        Temperature at which resistivity was measured
    t_formation : float or np.ndarray
        Formation temperature
    temp_unit : str, optional
        Temperature unit ("F" or "C")

    Returns
    -------
    float or np.ndarray
        Resistivity corrected to formation temperature

    Examples
    --------
    >>> # Correct Rmf from surface temp to formation temp
    >>> rmf_formation = arps_equation(rmf_surface, t_surface=75, t_formation=150)
    """
    # Arps constant
    if temp_unit.upper() == "F":
        k = 6.77
    else:  # Celsius
        k = 21.5

    r_corrected = r_measured * (t_measured + k) / (t_formation + k)
    return r_corrected


def correct_resistivity_temperature(
    resistivity: Union[float, np.ndarray],
    from_temp: Union[float, np.ndarray],
    to_temp: Union[float, np.ndarray],
    temp_unit: str = "F",
    method: str = "arps"
) -> Union[float, np.ndarray]:
    """
    Correct resistivity from one temperature to another.

    Parameters
    ----------
    resistivity : float or np.ndarray
        Resistivity value(s) to correct
    from_temp : float or np.ndarray
        Original temperature
    to_temp : float or np.ndarray
        Target temperature
    temp_unit : str, optional
        Temperature unit ("F" or "C")
    method : str, optional
        Correction method ("arps" or "arp2")

    Returns
    -------
    float or np.ndarray
        Temperature-corrected resistivity

    Examples
    --------
    >>> from petrophys.environmental.temperature import correct_resistivity_temperature
    >>> rmf_75f = 0.85  # Rmf at 75°F
    >>> rmf_150f = correct_resistivity_temperature(rmf_75f, from_temp=75, to_temp=150)
    """
    if method.lower() == "arps":
        return arps_equation(resistivity, from_temp, to_temp, temp_unit)
    else:
        # Alternative Arp formula (slightly different constant)
        if temp_unit.upper() == "F":
            k = 7.0
        else:
            k = 21.0
        return resistivity * (from_temp + k) / (to_temp + k)


def convert_rw_temperature(
    rw: float,
    from_temp: float,
    to_temp: float,
    temp_unit: str = "F"
) -> float:
    """
    Convert formation water resistivity between temperatures.

    This is a specific wrapper for Rw temperature correction.

    Parameters
    ----------
    rw : float
        Formation water resistivity
    from_temp : float
        Temperature at which Rw was measured
    to_temp : float
        Target formation temperature
    temp_unit : str, optional
        Temperature unit

    Returns
    -------
    float
        Rw at target temperature
    """
    return correct_resistivity_temperature(rw, from_temp, to_temp, temp_unit)


def estimate_nacl_resistivity(
    salinity_ppm: float,
    temperature: float,
    temp_unit: str = "F"
) -> float:
    """
    Estimate NaCl solution resistivity from salinity and temperature.

    Uses empirical relationship for NaCl solutions.

    Parameters
    ----------
    salinity_ppm : float
        Salinity in ppm NaCl equivalent
    temperature : float
        Temperature
    temp_unit : str, optional
        Temperature unit

    Returns
    -------
    float
        Resistivity in ohm.m
    """
    # Convert to Fahrenheit
    if temp_unit.upper() == "C":
        temperature = UnitConverter.celsius_to_fahrenheit(temperature)

    # Empirical formula (Schlumberger Gen-9)
    # Rw = (400000 / salinity_ppm) / (temperature + 6.77)
    # This is a simplified approximation

    if salinity_ppm <= 0:
        return np.inf

    # More accurate formula based on Arps
    rw_at_75f = (0.0123 + 3647.5 / salinity_ppm ** 0.955)

    # Correct to formation temperature
    rw = arps_equation(rw_at_75f, 75, temperature, "F")

    return rw


def salinity_from_rw(
    rw: float,
    temperature: float,
    temp_unit: str = "F"
) -> float:
    """
    Estimate salinity from Rw and temperature.

    Parameters
    ----------
    rw : float
        Formation water resistivity in ohm.m
    temperature : float
        Formation temperature
    temp_unit : str, optional
        Temperature unit

    Returns
    -------
    float
        Estimated salinity in ppm NaCl
    """
    # Convert to Fahrenheit
    if temp_unit.upper() == "C":
        temperature = UnitConverter.celsius_to_fahrenheit(temperature)

    # Correct Rw to 75°F
    rw_75 = arps_equation(rw, temperature, 75, "F")

    # Inverse of the empirical formula
    # Solving for salinity from: rw_at_75f = 0.0123 + 3647.5 / salinity^0.955
    if rw_75 <= 0.0123:
        return np.inf

    salinity = (3647.5 / (rw_75 - 0.0123)) ** (1 / 0.955)

    return salinity
