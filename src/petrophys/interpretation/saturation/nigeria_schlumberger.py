"""
Nigeria/Schlumberger equation for water saturation in shaly sands.

This module implements the square-root conductivity form:

    1 / sqrt(Rt) = Vcl^c / sqrt(Rsh) + (phi^(m/2) / sqrt(a * Rw)) * Sw^(n/2)

Rearranging:

    Sw^(n/2) = [1 / sqrt(Rt) - Vcl^c / sqrt(Rsh)] * sqrt(a * Rw) / phi^(m/2)

with Sw = NaN where the shale term alone exceeds 1 / sqrt(Rt): the equation
has no solution there. The benchmark context profiles keep the packages'
original calculation.
"""

from typing import Union

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.interpretation.saturation._inputs import (
    check_resistivity_parameters,
    nonpositive_resistivity,
)


def nigeria_schlumberger(
    rt: Union[float, np.ndarray],
    rw: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    rsh: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    shale_exp: float = 1.6,
) -> Union[float, np.ndarray]:
    """
    Calculate water saturation using the Nigeria/Schlumberger equation.

    Parameters
    ----------
    rt : float or np.ndarray
        True formation resistivity (ohm.m)
    rw : float
        Formation water resistivity (ohm.m)
    phi : float or np.ndarray
        Effective porosity (v/v)
    vcl : float or np.ndarray
        Clay volume (v/v)
    rsh : float
        Shale resistivity (ohm.m)
    a : float, optional
        Tortuosity factor
    m : float, optional
        Cementation exponent
    n : float, optional
        Saturation exponent
    shale_exp : float, optional
        Shale exponent applied to Vcl (default: 1.6)

    Returns
    -------
    float or np.ndarray
        Water saturation (0-1); NaN where the shale term alone exceeds
        1/sqrt(Rt),
        and where Rt or an Rw array is at or below zero.

    Raises
    ------
    ValueError
        If a scalar ``rw`` or ``rsh`` is at or below zero.
    """
    check_resistivity_parameters(rw=rw, rsh=rsh)
    scalar = np.isscalar(rt) and np.isscalar(phi) and np.isscalar(vcl)

    rt = np.atleast_1d(np.asarray(rt, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    vcl = np.atleast_1d(np.asarray(vcl, dtype=float))

    invalid = nonpositive_resistivity(rt, rw)
    rt = np.where(rt <= 0, 1e-10, rt)
    # Infinite Rt (no conductivity reading) is missing, as in archie().
    if not reproducing_benchmark():
        rt = np.where(np.isinf(rt), np.nan, rt)
    phi = np.where(phi <= 0, 1e-10, phi)
    vcl = np.clip(vcl, 0.0, 1.0)
    rw = np.asarray(rw, dtype=float)
    if rw.ndim == 0:
        rw = np.full(rt.shape, float(rw), dtype=float)
    else:
        rw = np.broadcast_to(rw, rt.shape).astype(float)
    rw = np.where(rw <= 0, 1e-10, rw)
    rsh = max(rsh, 1e-10)

    shale_term = np.power(vcl, shale_exp) / np.sqrt(rsh)
    sand_term_base = np.power(phi, m / 2.0) / np.sqrt(a * rw)
    lhs = (1.0 / np.sqrt(rt)) - shale_term
    if reproducing_benchmark():
        lhs = np.maximum(lhs, 0.0)
    else:
        # The shale alone conducts more than the rock: no Sw solves the equation.
        lhs = np.where(lhs < 0, np.nan, lhs)
    sand_term_base = np.where(sand_term_base <= 0, 1e-10, sand_term_base)

    sw_half_n = lhs / sand_term_base
    sw = np.power(sw_half_n, 2.0 / n)
    sw = np.where(invalid, np.nan, sw)
    sw = np.clip(sw, 0.0, 1.0)

    return float(sw[0]) if scalar else sw
