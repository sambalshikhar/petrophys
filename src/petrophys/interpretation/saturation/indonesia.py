"""
Indonesia (Poupon-Leveaux) equation for water saturation.

This module provides the Indonesia equation for shaly sands,
particularly suited for formations with lower salinity.
"""

from typing import Union
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark


def indonesia(
    rt: Union[float, np.ndarray],
    rw: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    rcl: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Calculate water saturation using Indonesia equation.

    1/sqrt(Rt) = [ Vcl^(1-Vcl/2) / sqrt(Rcl) + φ^(m/2) / sqrt(a×Rw) ] × Sw^(n/2)

    Poupon and Leveaux (1971), "Evaluation of water saturation in shaly
    formations", SPWLA 12th Annual Logging Symposium.

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
    rcl : float
        Clay resistivity (ohm.m)
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
    """
    # Track whether the caller passed scalars so we return a scalar
    scalar_input = np.isscalar(rt) and np.isscalar(phi) and np.isscalar(vcl)

    # Normalize numeric inputs to arrays (0-d for scalars) and avoid division by zero.
    # Using np.asarray keeps scalar inputs working through the array code paths below
    # (a 0-d array has .shape == () and supports np.where/np.power).
    rt = np.asarray(rt, dtype=float)
    rt = np.where(rt <= 0, 1e-10, rt)
    # Infinite Rt (no conductivity reading) is missing, as in archie().
    if not reproducing_benchmark():
        rt = np.where(np.isinf(rt), np.nan, rt)
    phi = np.asarray(phi, dtype=float)
    phi = np.where(phi <= 0, 1e-10, phi)
    rcl = max(float(rcl), 1e-10)
    rw = np.asarray(rw, dtype=float)
    rw = np.broadcast_to(rw, rt.shape).astype(float)
    rw = np.where(rw <= 0, 1e-10, rw)

    # Clip vcl to valid range
    vcl = np.clip(np.asarray(vcl, dtype=float), 0, 0.99)

    # Indonesia equation components
    # Term 1: Clay contribution
    clay_exp = 1 - vcl / 2
    clay_term = np.power(vcl, clay_exp) / np.sqrt(rcl)

    # Term 2: Sand contribution (without Sw)
    sand_term_base = np.power(phi, m / 2) / np.sqrt(a * rw)

    # Solve for Sw. In the Poupon-Leveaux Indonesia equation the saturation
    # factor Sw^(n/2) multiplies the whole bracket, so the clay and matrix
    # conduction paths scale with saturation together:
    #   1/sqrt(Rt) = (clay_term + sand_term_base) * Sw^(n/2)
    #   Sw^(n/2)   = (1/sqrt(Rt)) / (clay_term + sand_term_base)
    denom = clay_term + sand_term_base
    if isinstance(denom, np.ndarray):
        denom = np.where(denom <= 0, 1e-10, denom)
    else:
        denom = max(denom, 1e-10)

    sw_half_n = (1.0 / np.sqrt(rt)) / denom
    sw = np.power(sw_half_n, 2 / n)

    sw = np.clip(sw, 0, 1)
    return float(sw) if scalar_input else sw
