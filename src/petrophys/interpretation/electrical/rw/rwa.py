"""Apparent water resistivity (Rwa).

Archie's equation solved for Rw at Sw = 1: ``Rwa = Rt * phi**m / a``. In a
water-bearing interval Rwa reads Rw; hydrocarbons raise it, so the low,
stable values of an Rwa log over clean wet sands estimate Rw, and Rwa/Rw
screens for pay (Sw = (Rw/Rwa)**(1/n)).

References
----------
Archie, G.E. (1942). The electrical resistivity log as an aid in determining
some reservoir characteristics. *Transactions of the AIME* 146, 54-62.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def apparent_water_resistivity(
    rt: ArrayLike,
    phi: ArrayLike,
    a: float = 1.0,
    m: float = 2.0,
) -> ArrayLike:
    """Apparent water resistivity ``Rwa = Rt * phi**m / a`` (ohm.m).

    Parameters
    ----------
    rt : float or array
        True (deep) resistivity, ohm.m.
    phi : float or array
        Porosity (v/v), normally effective porosity in clean rock.
    a, m : float
        Archie tortuosity factor and cementation exponent.

    Returns
    -------
    float or array
        Rwa in ohm.m; NaN where Rt or porosity is missing or not positive.

    Examples
    --------
    >>> round(apparent_water_resistivity(5.0, 0.2), 4)
    0.2
    """
    if not np.isfinite(a) or a <= 0:
        raise ValueError("a must be positive")
    rt_arr = np.asarray(rt, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    valid = np.isfinite(rt_arr) & np.isfinite(phi_arr) & (rt_arr > 0) & (phi_arr > 0)
    with np.errstate(invalid="ignore", over="ignore"):
        rwa = np.where(valid, rt_arr * np.power(np.where(valid, phi_arr, 1.0), m) / a, np.nan)
    return float(rwa) if np.ndim(rwa) == 0 else rwa


def sw_from_rwa(rwa: ArrayLike, rw: float, n: float = 2.0) -> ArrayLike:
    """Archie water saturation from the Rwa ratio, ``Sw = (Rw/Rwa)**(1/n)``, limited to 0-1."""
    rwa_arr = np.asarray(rwa, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sw = np.where(np.isfinite(rwa_arr) & (rwa_arr > 0), np.power(rw / rwa_arr, 1.0 / n), np.nan)
    sw = np.clip(sw, 0.0, 1.0)
    return float(sw) if np.ndim(sw) == 0 else sw


__all__ = ["apparent_water_resistivity", "sw_from_rwa"]
