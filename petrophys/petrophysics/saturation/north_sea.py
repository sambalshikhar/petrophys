"""
North Sea shaly-sand water saturation equation.

This module implements a common North Sea shaly-sand form:

    Sw^n / Rt = phi^m / (a * Rw) + Vsh^exp / Rtsh

Rearranged:

    Sw^n = Rt * (phi^m / (a * Rw) + Vsh^exp / Rtsh)

Some workflows instead express the shale term on the conductivity side:

    1 / Rt = (phi^m / (a * Rw)) * Sw^n + Vsh^exp / Rtsh

For practical petrophysical use, the latter is the more common/logical form
for shaly-sand correction, so this implementation solves:

    1 / Rt = (phi^m / (a * Rw)) * Sw^n + Vsh^exp / Rtsh
"""

from typing import Union

import numpy as np


def north_sea_shaly_sand(
    rt: Union[float, np.ndarray],
    rw: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    vsh: Union[float, np.ndarray],
    rtsh: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    vsh_exp: float = 1.6,
) -> Union[float, np.ndarray]:
    """
    Calculate water saturation using a North Sea shaly-sand equation.

    Parameters
    ----------
    rt : float or np.ndarray
        True formation resistivity (ohm.m)
    rw : float
        Formation water resistivity (ohm.m)
    phi : float or np.ndarray
        Effective porosity (v/v)
    vsh : float or np.ndarray
        Shale volume (v/v)
    rtsh : float
        Shale resistivity (ohm.m)
    a : float, optional
        Tortuosity factor
    m : float, optional
        Cementation exponent
    n : float, optional
        Saturation exponent
    vsh_exp : float, optional
        Vsh exponent used to weight the shale term

    Returns
    -------
    float or np.ndarray
        Water saturation (0-1)
    """
    scalar = np.isscalar(rt) and np.isscalar(phi) and np.isscalar(vsh)

    rt = np.atleast_1d(np.asarray(rt, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    vsh = np.atleast_1d(np.asarray(vsh, dtype=float))

    rt = np.where(rt <= 0, 1e-10, rt)
    phi = np.where(phi <= 0, 1e-10, phi)
    vsh = np.clip(vsh, 0.0, 1.0)
    rw = np.asarray(rw, dtype=float)
    if rw.ndim == 0:
        rw = np.full(rt.shape, float(rw), dtype=float)
    else:
        rw = np.broadcast_to(rw, rt.shape).astype(float)
    rw = np.where(rw <= 0, 1e-10, rw)
    rtsh = max(rtsh, 1e-10)

    shale_term = np.power(vsh, vsh_exp) / rtsh
    sw_n = np.maximum(((1.0 / rt) - shale_term) * (a * rw) / np.power(phi, m), 0.0)
    sw = np.power(sw_n, 1.0 / n)
    sw = np.clip(sw, 0.0, 1.0)

    return float(sw[0]) if scalar else sw


def north_sea_shaly_sand_sw(
    rt: Union[float, np.ndarray],
    rw: float,
    phi: Union[float, np.ndarray],
    vsh: Union[float, np.ndarray],
    rtsh: float,
    a: float = 1.0,
    m: float = 1.7,
    n: float = 1.9,
    vsh_exp: float = 1.6,
) -> Union[float, np.ndarray]:
    """Alias for north_sea_shaly_sand."""
    return north_sea_shaly_sand(rt, rw, phi, vsh, rtsh, a, m, n, vsh_exp)
