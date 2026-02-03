"""
Simandoux equation for water saturation in shaly sands.

This module provides the Simandoux shaly sand model for formations
with moderate to high clay content and higher salinity.
"""

from typing import Union
import numpy as np


def simandoux(
    rt: Union[float, np.ndarray],
    rw: float,
    phi: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    rcl: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Calculate water saturation using Simandoux equation.

    1/Rt = φ^m × Sw^n / (a × Rw) + Vcl × Sw / Rcl

    This is a quadratic equation in Sw.

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

    # Simandoux coefficients for quadratic: A*Sw^2 + B*Sw + C = 0
    # Rearranging: (φ^m/(a*Rw)) * Sw^n + (Vcl/Rcl) * Sw - 1/Rt = 0

    # For n=2 (quadratic in Sw):
    # A = φ^m / (a × Rw)
    # B = Vcl / Rcl
    # C = -1/Rt

    A = np.power(phi, m) / (a * rw)
    B = vcl / rcl
    C = -1 / rt

    # Quadratic formula: Sw = (-B + sqrt(B^2 - 4AC)) / (2A)
    discriminant = B ** 2 - 4 * A * C
    discriminant = np.maximum(discriminant, 0)  # Ensure non-negative

    sw = (-B + np.sqrt(discriminant)) / (2 * A)

    return np.clip(sw, 0, 1)


def simandoux_sw(
    rt: Union[float, np.ndarray],
    rw: float,
    phi: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    rcl: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """Alias for simandoux function."""
    return simandoux(rt, rw, phi, vcl, rcl, a, m, n)
