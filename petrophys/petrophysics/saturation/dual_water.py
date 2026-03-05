"""
Dual Water model for water saturation in shaly carbonates.

This module implements the Dual Water model which separates total pore
water into free water (Rw) and bound water associated with clay
surfaces (Rwb).
"""

from typing import Union
import numpy as np


def dual_water(
    phi_t: Union[float, np.ndarray],
    phi_e: Union[float, np.ndarray],
    rt: Union[float, np.ndarray],
    rw: float,
    rwb: float,
    vcl: Union[float, np.ndarray],
    phi_cl: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Dual Water model water saturation for shaly carbonates.

    Separates total pore water into free water (Rw) and bound water (Rwb).

    Equations
    ---------
    Bound water porosity:
        phi_bw = VCL * phi_cl

    Bound water saturation:
        Swb = phi_bw / phi_t

    Apparent total water resistivity:
        1/Rwt = Swb/Rwb + (1 - Swb)/Rw

    Total Sw (Archie on total porosity):
        Swt = [(a * Rwt) / (phi_t^m * Rt)]^(1/n)

    Effective water saturation:
        Swe = (Swt - Swb) / (1 - Swb)

    Parameters
    ----------
    phi_t : float or np.ndarray
        Total porosity (fraction)
    phi_e : float or np.ndarray
        Effective porosity (fraction)
    rt : float or np.ndarray
        True formation resistivity (ohm.m)
    rw : float
        Free water resistivity (ohm.m)
    rwb : float
        Bound water resistivity (ohm.m)
    vcl : float or np.ndarray
        Volume of clay (fraction)
    phi_cl : float
        Clay porosity (fraction)
    a : float
        Tortuosity factor (default: 1.0)
    m : float
        Cementation exponent (default: 2.0)
    n : float
        Saturation exponent (default: 2.0)

    Returns
    -------
    float or np.ndarray
        Effective water saturation Swe (fraction, clipped 0-1)
    """
    # Bound water fraction
    phi_bw = vcl * phi_cl

    # Protect against division by zero
    phi_t_safe = np.where(phi_t <= 0, 1e-10, phi_t) if isinstance(phi_t, np.ndarray) else max(phi_t, 1e-10)

    swb = phi_bw / phi_t_safe

    # Apparent total water resistivity
    rwt = 1.0 / (swb / rwb + (1.0 - swb) / rw)

    # Total Sw via Archie on total porosity
    swt = ((a * rwt) / (phi_t_safe ** m * rt)) ** (1.0 / n)

    # Convert to effective Sw
    swb_safe = np.where(swb >= 1.0, 0.999, swb) if isinstance(swb, np.ndarray) else min(swb, 0.999)
    swe = (swt - swb_safe) / (1.0 - swb_safe)

    return np.clip(swe, 0.0, 1.0)


def dual_water_sw(
    phi_t: Union[float, np.ndarray],
    phi_e: Union[float, np.ndarray],
    rt: Union[float, np.ndarray],
    rw: float,
    rwb: float,
    vcl: Union[float, np.ndarray],
    phi_cl: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """Alias for dual_water function."""
    return dual_water(phi_t, phi_e, rt, rw, rwb, vcl, phi_cl, a, m, n)
