"""
Density porosity calculation.

This module provides density-derived porosity from bulk density.
"""

from typing import Union
import numpy as np


def density_porosity(
    rhob: Union[float, np.ndarray],
    rhob_matrix: float = 2.65,
    rhob_fluid: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Calculate density porosity.

    φD = (ρma - ρb) / (ρma - ρfl)

    Parameters
    ----------
    rhob : float or np.ndarray
        Bulk density (g/cm³)
    rhob_matrix : float, optional
        Matrix density (default: 2.65 for sandstone)
    rhob_fluid : float, optional
        Fluid density (default: 1.0 for water)

    Returns
    -------
    float or np.ndarray
        Density porosity (v/v)

    Examples
    --------
    >>> phid = density_porosity(2.35, rhob_matrix=2.65)
    >>> print(f"PHID = {phid:.3f}")
    0.182
    """
    rhob_arr = np.asarray(rhob, dtype=float)
    rhob_matrix_arr = np.asarray(rhob_matrix, dtype=float)
    rhob_fluid_arr = np.asarray(rhob_fluid, dtype=float)
    denominator = rhob_matrix_arr - rhob_fluid_arr

    with np.errstate(divide="ignore", invalid="ignore"):
        phid = np.where(
            denominator != 0,
            (rhob_matrix_arr - rhob_arr) / denominator,
            0.0,
        )

    phid = np.clip(phid, 0, 0.5)
    if np.ndim(phid) == 0:
        return float(phid)
    return phid


def calculate_phid(
    rhob: Union[float, np.ndarray],
    matrix: str = "sandstone",
    rhob_fluid: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Calculate density porosity using matrix type.

    Parameters
    ----------
    rhob : float or np.ndarray
        Bulk density
    matrix : str
        Matrix type ("sandstone", "limestone", "dolomite")
    rhob_fluid : float
        Fluid density

    Returns
    -------
    float or np.ndarray
        Density porosity
    """
    matrix_densities = {
        "sandstone": 2.65,
        "limestone": 2.71,
        "dolomite": 2.87,
        "quartz": 2.65,
        "calcite": 2.71,
    }

    rhob_matrix = matrix_densities.get(matrix.lower(), 2.65)
    return density_porosity(rhob, rhob_matrix, rhob_fluid)
