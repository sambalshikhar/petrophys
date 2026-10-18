"""
North Sea shaly-sand water saturation equation.

This module implements a shaly-sand form in which the shale conducts in
parallel with the water-filled pores:

    1 / Rt = (phi^m / (a * Rw)) * Sw^n + Vsh^exp / Rtsh

solved as

    Sw^n = (1 / Rt - Vsh^exp / Rtsh) * a * Rw / phi^m

with Sw = NaN where the shale term alone exceeds 1 / Rt: the equation has
no solution there (the benchmark packages gave Sw = 0).
"""

from typing import Union

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark


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
        Water saturation (0-1); NaN where the shale term alone exceeds 1/Rt
    """
    scalar = np.isscalar(rt) and np.isscalar(phi) and np.isscalar(vsh)

    rt = np.atleast_1d(np.asarray(rt, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    vsh = np.atleast_1d(np.asarray(vsh, dtype=float))

    rt = np.where(rt <= 0, 1e-10, rt)
    # Infinite Rt (no conductivity reading) is missing, as in archie().
    if not reproducing_benchmark():
        rt = np.where(np.isinf(rt), np.nan, rt)
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
    sw_n = ((1.0 / rt) - shale_term) * (a * rw) / np.power(phi, m)
    if reproducing_benchmark():
        sw_n = np.maximum(sw_n, 0.0)
    else:
        # The shale alone conducts more than the rock: no Sw solves the equation.
        sw_n = np.where(sw_n < 0, np.nan, sw_n)
    sw = np.power(sw_n, 1.0 / n)
    sw = np.clip(sw, 0.0, 1.0)

    return float(sw[0]) if scalar else sw
