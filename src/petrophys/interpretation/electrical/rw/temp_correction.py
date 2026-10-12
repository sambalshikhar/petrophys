"""
Temperature correction for Rw.

This module provides temperature correction utilities
specifically for formation water resistivity.
"""

from typing import Optional, Union
import numpy as np

from petrophys.interpretation.temperature.temp_corrections import arps_equation
from petrophys.model.units import UnitConverter, temperature_unit


def correct_rw_temperature(
    rw: Union[float, np.ndarray],
    from_temp: Union[float, np.ndarray],
    to_temp: Union[float, np.ndarray],
    temp_unit: str = "F"
) -> Union[float, np.ndarray]:
    """
    Correct Rw from one temperature to another.

    Parameters
    ----------
    rw : float or np.ndarray
        Formation water resistivity
    from_temp : float or np.ndarray
        Original temperature
    to_temp : float or np.ndarray
        Target temperature
    temp_unit : str, optional
        Temperature unit

    Returns
    -------
    float or np.ndarray
        Temperature-corrected Rw
    """
    return arps_equation(rw, from_temp, to_temp, temp_unit)


def rw_at_reference_temperature(
    rw: float,
    current_temp: float,
    reference_temp: Optional[float] = None,
    temp_unit: str = "F"
) -> float:
    """
    Convert Rw to a reference temperature (typically 75°F).

    Parameters
    ----------
    rw : float
        Rw at current temperature
    current_temp : float
        Current temperature
    reference_temp : float, optional
        Reference temperature, in ``temp_unit`` (default: 75°F)
    temp_unit : str, optional
        Temperature unit

    Returns
    -------
    float
        Rw at reference temperature
    """
    if reference_temp is None:
        reference_temp = 75.0
        if temperature_unit(temp_unit, strict=True) != "F":
            reference_temp = UnitConverter.temperature(reference_temp, "F", temp_unit)
    return correct_rw_temperature(rw, current_temp, reference_temp, temp_unit)
