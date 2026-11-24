"""
SP-based Rw calculation.

This module provides formation water resistivity calculation
from spontaneous potential (SP) log.

References
----------
Asquith, G. and Krygowski, D. (2004). *Basic Well Log Analysis*, 2nd ed. AAPG
Methods in Exploration 16.

Schlumberger (1989). *Log Interpretation Principles/Applications*.
"""

from typing import Tuple, Union

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark


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


def sp_lines(sp: Union[float, np.ndarray]) -> Tuple[float, float]:
    """
    Estimate clean-sand and shale SP lines from the SP log itself.

    Parameters
    ----------
    sp : float or np.ndarray
        SP log values (mV)

    Returns
    -------
    tuple[float, float]
        Clean-line SP and shale-line SP in mV
    """
    sp_array = np.asarray(sp, dtype=float)
    valid_sp = sp_array[~np.isnan(sp_array)]
    if valid_sp.size == 0:
        raise ValueError("SP log contains no valid values")

    sp_clean = float(np.percentile(valid_sp, 5))
    sp_shale = float(np.percentile(valid_sp, 95))
    return sp_clean, sp_shale


def rw_from_ssp(
    ssp: Union[float, np.ndarray],
    rmf: float,
    temperature: Union[float, np.ndarray],
    temp_unit: str = "F"
) -> Union[float, np.ndarray]:
    """
    Calculate Rw from static spontaneous potential (SSP).

    ``temp_unit`` is "F", "C", "K" or a label such as "degC"; unknown labels
    raise ``ValueError``. The result is not limited: a value outside
    0.01-10 ohm.m shows that the SP route does not apply (the benchmark
    packages clipped it to that range). Missing (NaN) inputs give NaN.
    """
    from petrophys.model.units import UnitConverter, temperature_unit

    if temperature_unit(temp_unit, strict=True) != "F":
        temperature = UnitConverter.temperature(temperature, temp_unit, "F")

    k = 61 + 0.133 * temperature
    rw = rmf / np.power(10, -ssp / k)

    if reproducing_benchmark():
        # Apply the selected benchmark context profile's calculation rules.
        if isinstance(rw, np.ndarray):
            return np.clip(rw, 0.01, 10.0)
        return max(0.01, min(rw, 10.0))

    return rw if isinstance(rw, np.ndarray) else float(rw)


def rw_from_sp(
    sp: Union[float, np.ndarray],
    sp_shale: float,
    rmf: float,
    temperature: Union[float, np.ndarray],
    temp_unit: str = "F"
) -> Union[float, np.ndarray]:
    """
    Calculate Rw from SP using the SSP equation.

    SSP = -K * log10(Rmf/Rw)
    where K = 61 + 0.133*T (T in degF)

    Solving for Rw:
    Rw = Rmf / 10^(-SSP/K)

    Rmf and Rw stand directly for the equivalent resistivities Rmfe and Rwe
    of the textbook relation; no Rweq-to-Rw chart conversion is applied.

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
        Temperature unit: "F", "C", "K" or a label such as "degC"

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
    ssp = calculate_ssp(sp, sp_shale)
    return rw_from_ssp(ssp, rmf, temperature, temp_unit=temp_unit)


def rw_from_rxo_rt(
    rxo: Union[float, np.ndarray],
    rt: Union[float, np.ndarray],
    rmf: float
) -> Union[float, np.ndarray]:
    """
    Calculate Rw from Rxo/Rt ratio.

    In a clean water zone: Rxo/Rt = Rmf/Rw
    Therefore: Rw = Rmf * Rt / Rxo

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
        Formation water resistivity; NaN where Rxo is zero or negative

    Notes
    -----
    This method is most accurate in clean water zones
    where Sw = Sxo = 1.0
    """
    if not reproducing_benchmark():
        # A zero or negative Rxo has no Rw. Benchmark context profiles
        # evaluate the equation without this input mask.
        rxo_arr = np.asarray(rxo, dtype=float)
        rxo_arr = np.where(rxo_arr > 0, rxo_arr, np.nan)
        rxo = rxo_arr if rxo_arr.ndim else float(rxo_arr)
        return rmf * rt / rxo
    rxo = np.where(rxo == 0, 1e-10, rxo) if isinstance(rxo, np.ndarray) else max(rxo, 1e-10)
    return rmf * rt / rxo
