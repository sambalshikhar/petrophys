"""
Dual Water model for water saturation in shaly formations.

This module implements the Dual Water model which separates total pore
water into free water (Rw) and bound water associated with clay
surfaces (Rwb).
"""

from typing import Union
import numpy as np


def dual_water(
    phi_t: Union[float, np.ndarray],
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
    Dual Water model water saturation.

    Separates total pore water into free water (Rw) and clay-bound
    water (Rwb).  Operates entirely on total porosity — effective
    porosity is not required and is not used in the derivation.

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

    Effective (free-water) saturation:
        Swe = (Swt - Swb) / (1 - Swb)

    Parameters
    ----------
    phi_t : float or np.ndarray
        Total porosity (fraction)
    rt : float or np.ndarray
        True formation resistivity (ohm.m)
    rw : float
        Free water resistivity (ohm.m)
    rwb : float
        Bound water resistivity (ohm.m).
        If unknown, a common approximation is rwb = 0.3 * rw.
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
    phi_bw = vcl * phi_cl

    phi_t_safe = (
        np.where(phi_t <= 0, 1e-10, phi_t)
        if isinstance(phi_t, np.ndarray)
        else max(phi_t, 1e-10)
    )

    # Guard the resistivities against zero/negative (matches the other Sw models),
    # otherwise rwt and swt divide by zero on a scalar 0 or propagate NaN.
    rw_safe = np.where(rw <= 0, 1e-10, rw) if isinstance(rw, np.ndarray) else max(rw, 1e-10)
    rwb_safe = np.where(rwb <= 0, 1e-10, rwb) if isinstance(rwb, np.ndarray) else max(rwb, 1e-10)
    rt_safe = np.where(rt <= 0, 1e-10, rt) if isinstance(rt, np.ndarray) else max(rt, 1e-10)

    swb = phi_bw / phi_t_safe

    # Bound-water saturation cannot exceed 1 (happens when Vcl*phi_cl > phi_t).
    # Clamp BEFORE it is used so the apparent-Rwt mixing term (1 - swb)/rw stays
    # physical; previously rwt was built from the raw swb, giving a negative
    # conductivity contribution and an invalid Rwt in that regime.
    swb_safe = (
        np.where(swb >= 1.0, 0.999, swb)
        if isinstance(swb, np.ndarray)
        else min(swb, 0.999)
    )

    # Apparent total water resistivity
    rwt = 1.0 / (swb_safe / rwb_safe + (1.0 - swb_safe) / rw_safe)

    # Total Sw via Archie on total porosity
    swt = ((a * rwt) / (phi_t_safe ** m * rt_safe)) ** (1.0 / n)

    # Convert to effective Sw
    swe = (swt - swb_safe) / (1.0 - swb_safe)

    return np.clip(swe, 0.0, 1.0)


def dual_water_sw(
    phi_t: Union[float, np.ndarray],
    rt: Union[float, np.ndarray],
    rw: float,
    rwb: float,
    vcl: Union[float, np.ndarray],
    phi_cl: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """Alias for dual_water."""
    return dual_water(phi_t, rt, rw, rwb, vcl, phi_cl, a, m, n)
