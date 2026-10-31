"""
Archie equation for water saturation.

This module provides the classic Archie equation for clean formations.

References
----------
Archie, G.E. (1942). The electrical resistivity log as an aid in determining
some reservoir characteristics. *Transactions of the AIME* 146, 54-62.

Winsauer, W.O., Shearin, H.M., Masson, P.H. and Williams, M. (1952).
Resistivity of brine-saturated sands in relation to pore geometry. *AAPG
Bulletin* 36(2), 253-277.
"""

from typing import Union
import numpy as np

from petrophys.interpretation.saturation._inputs import (
    check_resistivity_parameters,
    nonpositive_resistivity,
)


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
        Water saturation (0-1): a float for scalar inputs, else an array.
        NaN where Rt or an Rw array is at or below zero.

    Raises
    ------
    ValueError
        If a scalar ``rw`` is at or below zero.

    Examples
    --------
    >>> sw = archie(rt=10, rw=0.05, phi=0.20)
    >>> print(f"Sw = {sw:.2f}")
    """
    check_resistivity_parameters(rw=rw)
    rt = np.asarray(rt, dtype=float)
    rw = np.asarray(rw, dtype=float)
    phi = np.asarray(phi, dtype=float)
    finite = np.isfinite(rt) & np.isfinite(rw) & np.isfinite(phi)
    finite &= ~nonpositive_resistivity(rt, rw)
    # Avoid division by zero
    rt = np.where(rt <= 0, 1e-10, rt)
    phi = np.where(phi <= 0, 1e-10, phi)

    with np.errstate(invalid="ignore"):
        # Archie equation
        sw_n = a * rw / (np.power(phi, m) * rt)

        # Solve for Sw
        sw = np.power(sw_n, 1 / n)

    result = np.where(finite, np.clip(sw, 0, 1), np.nan)
    return float(result) if result.ndim == 0 else result
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
