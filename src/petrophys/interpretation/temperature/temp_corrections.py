"""
Temperature corrections for resistivity measurements.

This module provides functions for correcting resistivity
measurements to formation temperature.

References
----------
Arps, J.J. (1953). The effect of temperature on the density and electrical
resistivity of sodium chloride solutions. *Petroleum Transactions, AIME* 198,
Technical Note 195.

Bateman, R.M. and Konen, C.E. (1977). The log analyst and the programmable
pocket calculator. *The Log Analyst* 18.
"""

from typing import Union
import numpy as np

from petrophys.model.units import UnitConverter, temperature_unit


def _arps_scale(temp_unit: str, *temperatures):
    """Return the Arps scale ("F" or "C") of ``temp_unit`` and the temperatures on it.

    Arps' constant is defined on the Fahrenheit and Celsius scales, so Kelvin
    temperatures are converted to Celsius. Unknown labels raise ``ValueError``.
    """
    scale = temperature_unit(temp_unit, strict=True)
    if scale == "K":
        return "C", tuple(UnitConverter.temperature(t, "K", "C") for t in temperatures)
    return scale, temperatures


def arps_equation(
    r_measured: Union[float, np.ndarray],
    t_measured: Union[float, np.ndarray],
    t_formation: Union[float, np.ndarray],
    temp_unit: str = "F"
) -> Union[float, np.ndarray]:
    """
    Correct resistivity to formation temperature using Arps equation.

    The Arps (1953) equation relates resistivity at different temperatures:
    R2 = R1 * (T1 + k) / (T2 + k)

    where k = 6.77 for temperatures in °F (or 21.5 for °C); temperatures in
    kelvin are converted to °C first.

    Parameters
    ----------
    r_measured : float or np.ndarray
        Resistivity at measured temperature
    t_measured : float or np.ndarray
        Temperature at which resistivity was measured
    t_formation : float or np.ndarray
        Formation temperature
    temp_unit : str, optional
        Temperature unit: "F", "C", "K" or a label such as "degC" (see
        :func:`petrophys.model.units.temperature_unit`)

    Returns
    -------
    float or np.ndarray
        Resistivity corrected to formation temperature

    Raises
    ------
    ValueError
        If ``temp_unit`` is not a recognized temperature unit.

    Examples
    --------
    >>> # Correct Rmf from surface temp to formation temp
    >>> rmf_surface = 0.85  # ohm.m at 75 degF
    >>> round(arps_equation(rmf_surface, t_measured=75, t_formation=150), 4)
    0.4434
    """
    # Arps constant
    scale, (t_measured, t_formation) = _arps_scale(temp_unit, t_measured, t_formation)
    k = 6.77 if scale == "F" else 21.5

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
        Temperature unit: "F", "C", "K" or a label such as "degC"
    method : str, optional
        Correction method ("arps" or "arp2"); other names raise
        ``ValueError``

    Returns
    -------
    float or np.ndarray
        Temperature-corrected resistivity

    Examples
    --------
    >>> from petrophys.interpretation.temperature import correct_resistivity_temperature
    >>> rmf_75f = 0.85  # Rmf at 75°F
    >>> rmf_150f = correct_resistivity_temperature(rmf_75f, from_temp=75, to_temp=150)
    """
    if method.lower() == "arps":
        return arps_equation(resistivity, from_temp, to_temp, temp_unit)
    elif method.lower() != "arp2":
        raise ValueError(f"Unknown resistivity temperature correction {method!r}: use 'arps' or 'arp2'")
    else:
        # Arps form with rounded constants: 7 for degF, 21 for degC
        scale, (from_temp, to_temp) = _arps_scale(temp_unit, from_temp, to_temp)
        k = 7.0 if scale == "F" else 21.0
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


def nacl_resistivity(
    salinity_ppm: float,
    temperature: float,
    temp_unit: str = "F"
) -> float:
    """
    Estimate NaCl solution resistivity from salinity and temperature.

    Rw at 75 degF from the Bateman and Konen (1977) fit to the Schlumberger
    Gen-9 NaCl chart, moved to ``temperature`` by Arps (1953).

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
    if temperature_unit(temp_unit, strict=True) != "F":
        temperature = UnitConverter.temperature(temperature, temp_unit, "F")

    if salinity_ppm <= 0:
        return np.inf

    # Bateman-Konen fit to the Gen-9 chart at 75 degF
    rw_at_75f = (0.0123 + 3647.5 / salinity_ppm ** 0.955)

    # Correct to formation temperature
    rw = arps_equation(rw_at_75f, 75, temperature, "F")

    return rw
