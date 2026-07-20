"""
Sonic porosity calculation.

This module provides sonic-derived porosity using the
Wyllie time-average equation.
"""

from typing import Union
import numpy as np


def wyllie_time_average(
    dtc: Union[float, np.ndarray],
    dtc_matrix: float = 55.5,
    dtc_fluid: float = 189.0
) -> Union[float, np.ndarray]:
    """
    Calculate porosity using Wyllie time-average equation.

    φS = (Δt - Δtma) / (Δtfl - Δtma)

    Parameters
    ----------
    dtc : float or np.ndarray
        Compressional slowness (μs/ft)
    dtc_matrix : float, optional
        Matrix slowness (default: 55.5 for sandstone)
    dtc_fluid : float, optional
        Fluid slowness (default: 189.0 for water)

    Returns
    -------
    float or np.ndarray
        Sonic porosity (v/v)
    """
    dtc_arr = np.asarray(dtc, dtype=float)
    dtc_matrix_arr = np.asarray(dtc_matrix, dtype=float)
    dtc_fluid_arr = np.asarray(dtc_fluid, dtype=float)
    denominator = dtc_fluid_arr - dtc_matrix_arr

    with np.errstate(divide="ignore", invalid="ignore"):
        phis = np.where(
            denominator != 0,
            (dtc_arr - dtc_matrix_arr) / denominator,
            0.0,
        )

    phis = np.clip(phis, 0, 0.5)
    if np.ndim(phis) == 0:
        return float(phis)
    return phis


def sonic_porosity(
    dtc: Union[float, np.ndarray],
    matrix: str = "sandstone",
    dtc_fluid: float = 189.0,
    compaction_factor: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Calculate sonic porosity with matrix selection.

    Parameters
    ----------
    dtc : float or np.ndarray
        Compressional slowness
    matrix : str
        Matrix type
    dtc_fluid : float
        Fluid slowness
    compaction_factor : float
        Compaction correction (Cp)

    Returns
    -------
    float or np.ndarray
        Sonic porosity
    """
    matrix_slowness = {
        "sandstone": 55.5,
        "limestone": 47.6,
        "dolomite": 43.5,
        "shale": 100.0,
    }

    dtc_matrix = matrix_slowness.get(matrix.lower(), 55.5)
    phis = wyllie_time_average(dtc, dtc_matrix, dtc_fluid)

    # Apply compaction correction
    if compaction_factor != 1.0:
        phis = phis / compaction_factor

    return phis


def secondary_porosity(
    phis: Union[float, np.ndarray],
    phit: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate secondary porosity.

    Secondary porosity = Total porosity - Sonic porosity

    Parameters
    ----------
    phis : float or np.ndarray
        Sonic porosity
    phit : float or np.ndarray
        Total porosity (from N-D)

    Returns
    -------
    float or np.ndarray
        Secondary porosity
    """
    phi_sec = phit - phis
    return np.maximum(phi_sec, 0)
