"""
Indonesia (Poupon-Leveaux) equation for water saturation.

This module provides the Indonesia equation for shaly sands,
particularly suited for formations with lower salinity.
"""

from typing import Union
import numpy as np


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

    1/sqrt(Rt) = (Vcl^(1-Vcl/2) / sqrt(Rcl)) + (φ^(m/2) × Sw^(n/2) / sqrt(a×Rw))

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
    # Avoid division by zero
    rt = np.where(rt <= 0, 1e-10, rt) if isinstance(rt, np.ndarray) else max(rt, 1e-10)
    phi = np.where(phi <= 0, 1e-10, phi) if isinstance(phi, np.ndarray) else max(phi, 1e-10)
    rcl = max(rcl, 1e-10)
    rw = np.asarray(rw, dtype=float)
    if rw.ndim == 0:
        rw = np.full(rt.shape, float(rw), dtype=float)
    else:
        rw = np.broadcast_to(rw, rt.shape).astype(float)
    rw = np.where(rw <= 0, 1e-10, rw)

    # Clip vcl to valid range
    vcl = np.clip(vcl, 0, 0.99)

    # Indonesia equation components
    # Term 1: Clay contribution
    clay_exp = 1 - vcl / 2
    clay_term = np.power(vcl, clay_exp) / np.sqrt(rcl)

    # Term 2: Sand contribution (without Sw)
    sand_term_base = np.power(phi, m / 2) / np.sqrt(a * rw)

    # Solve for Sw
    # 1/sqrt(Rt) = clay_term + sand_term_base × Sw^(n/2)
    # sand_term_base × Sw^(n/2) = 1/sqrt(Rt) - clay_term
    # Sw^(n/2) = (1/sqrt(Rt) - clay_term) / sand_term_base

    lhs = 1 / np.sqrt(rt) - clay_term

    # Handle negative values (when clay dominates)
    if isinstance(lhs, np.ndarray):
        lhs = np.maximum(lhs, 0)
    else:
        lhs = max(lhs, 0)

    # Avoid division by zero
    if isinstance(sand_term_base, np.ndarray):
        sand_term_base = np.where(sand_term_base <= 0, 1e-10, sand_term_base)
    else:
        sand_term_base = max(sand_term_base, 1e-10)

    sw_half_n = lhs / sand_term_base
    sw = np.power(sw_half_n, 2 / n)

    return np.clip(sw, 0, 1)


def indonesia_sw(
    rt: Union[float, np.ndarray],
    rw: float,
    phi: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    rcl: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """Alias for indonesia function."""
    return indonesia(rt, rw, phi, vcl, rcl, a, m, n)
