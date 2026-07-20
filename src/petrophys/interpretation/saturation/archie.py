"""
Archie equation for water saturation.

This module provides the classic Archie equation for clean formations.
"""

from typing import Union
import numpy as np


def archie(
    rt: Union[float, np.ndarray],
    rw: float,
    phi: Union[float, np.ndarray],
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Calculate water saturation using Archie equation.

    Sw^n = a × Rw / (φ^m × Rt)

    Parameters
    ----------
    rt : float or np.ndarray
        True formation resistivity (ohm.m)
    rw : float
        Formation water resistivity (ohm.m)
    phi : float or np.ndarray
        Porosity (v/v)
    a : float, optional
        Tortuosity factor (default: 1.0)
    m : float, optional
        Cementation exponent (default: 2.0)
    n : float, optional
        Saturation exponent (default: 2.0)

    Returns
    -------
    float or np.ndarray
        Water saturation (0-1)

    Examples
    --------
    >>> sw = archie(rt=10, rw=0.05, phi=0.20)
    >>> print(f"Sw = {sw:.2f}")
    """
    # Avoid division by zero
    rt = np.where(rt <= 0, 1e-10, rt) if isinstance(rt, np.ndarray) else max(rt, 1e-10)
    phi = np.where(phi <= 0, 1e-10, phi) if isinstance(phi, np.ndarray) else max(phi, 1e-10)

    # Archie equation
    sw_n = a * rw / (np.power(phi, m) * rt)

    # Solve for Sw
    sw = np.power(sw_n, 1 / n)

    return np.clip(sw, 0, 1)
def archie_f(
    phi: Union[float, np.ndarray],
    a: float = 1.0,
    m: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Calculate formation resistivity factor (F).

    F = a / φ^m

    Parameters
    ----------
    phi : float or np.ndarray
        Porosity
    a : float
        Tortuosity factor
    m : float
        Cementation exponent

    Returns
    -------
    float or np.ndarray
        Formation resistivity factor
    """
    phi = np.where(phi <= 0, 1e-10, phi) if isinstance(phi, np.ndarray) else max(phi, 1e-10)
    return a / np.power(phi, m)


def archie_ri(
    sw: Union[float, np.ndarray],
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Calculate resistivity index (RI).

    RI = 1 / Sw^n

    Parameters
    ----------
    sw : float or np.ndarray
        Water saturation
    n : float
        Saturation exponent

    Returns
    -------
    float or np.ndarray
        Resistivity index
    """
    sw = np.where(sw <= 0, 1e-10, sw) if isinstance(sw, np.ndarray) else max(sw, 1e-10)
    return 1 / np.power(sw, n)
