"""
SP-based Rw calculation.

This module provides formation water resistivity calculation
from spontaneous potential (SP) log.
"""

from typing import Union
import numpy as np

from petrophys.environmental.temperature.temp_corrections import arps_equation


def calculate_ssp(
    sp: Union[float, np.ndarray],
    sp_shale: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate static spontaneous potential (SSP).

    SSP = SP - SP_shale

    Parameters
    ----------
    sp : float or np.ndarray
        Measured SP values
    sp_shale : float or np.ndarray
        Shale baseline SP

    Returns
    -------
    float or np.ndarray
        Static SP (mV)
    """
    return sp - sp_shale


def rw_from_sp(
    sp: Union[float, np.ndarray],
    sp_shale: float,
    rmf: float,
    temperature: Union[float, np.ndarray],
    temp_unit: str = "F"
) -> Union[float, np.ndarray]:
    """
    Calculate Rw from SP using the SSP equation.

    SSP = -K × log10(Rmf/Rw)
    where K = 61 + 0.133×T (T in °F)

    Solving for Rw:
    Rw = Rmf / 10^(-SSP/K)

    Parameters
    ----------
    sp : float or np.ndarray
        SP log values (mV)
    sp_shale : float
        Shale baseline SP (mV)
    rmf : float
        Mud filtrate resistivity at formation temperature
    temperature : float or np.ndarray
        Formation temperature
    temp_unit : str, optional
        Temperature unit

    Returns
    -------
    float or np.ndarray
        Formation water resistivity (ohm.m)

    Notes
    -----
    This method should not be used when:
    - Formation contains significant clay
    - KCl is in the drilling mud
    - Formation water salinity is very low
    """
    # Convert temperature to Fahrenheit
    if temp_unit.upper() == "C":
        from petrophys.core.units import UnitConverter
        temperature = UnitConverter.celsius_to_fahrenheit(temperature)

    # Calculate SSP
    ssp = calculate_ssp(sp, sp_shale)

    # Temperature coefficient
    k = 61 + 0.133 * temperature

    # Solve for Rw
    # SSP = -K × log10(Rmf/Rw)
    # -SSP/K = log10(Rmf/Rw)
    # 10^(-SSP/K) = Rmf/Rw
    # Rw = Rmf / 10^(-SSP/K)

    rw = rmf / np.power(10, -ssp / k)

    # Clip to reasonable range
    if isinstance(rw, np.ndarray):
        rw = np.clip(rw, 0.01, 10.0)
    else:
        rw = max(0.01, min(rw, 10.0))

    return rw


def rw_from_rxo_rt(
    rxo: Union[float, np.ndarray],
    rt: Union[float, np.ndarray],
    rmf: float
) -> Union[float, np.ndarray]:
    """
    Calculate Rw from Rxo/Rt ratio.

    In a clean water zone: Rxo/Rt = Rmf/Rw
    Therefore: Rw = Rmf × Rt / Rxo

    Parameters
    ----------
    rxo : float or np.ndarray
        Flushed zone resistivity
    rt : float or np.ndarray
        True formation resistivity
    rmf : float
        Mud filtrate resistivity

    Returns
    -------
    float or np.ndarray
        Formation water resistivity

    Notes
    -----
    This method is most accurate in clean water zones
    where Sw = Sxo = 1.0
    """
    # Avoid division by zero
    rxo = np.where(rxo == 0, 1e-10, rxo) if isinstance(rxo, np.ndarray) else max(rxo, 1e-10)

    rw = rmf * rt / rxo
    return rw
