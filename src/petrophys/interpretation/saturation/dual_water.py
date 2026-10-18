"""
Dual Water model for water saturation in shaly formations.

This module implements the Dual Water model which separates total pore
water into free water (Rw) and bound water associated with clay
surfaces (Rwb).
"""

from typing import Union
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark


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

    This is the Dual Water model of Clavier, Coates and Dumanoir (1984),
    "Theoretical and experimental bases for the dual-water model for
    interpretation of shaly sands", SPE Journal 24(2): the two waters are
    mixed in the proportions of the water actually in the pores, Swb/Swt
    bound and 1 - Swb/Swt free, so the conductivity equation is solved for
    Swt (a quadratic for n = 2, iterated otherwise) and the bound water is
    removed to give Swe. The benchmark packages used the wet-zone (Ro) form
    of Crain's Petrophysical Handbook, which mixes the waters in the
    proportions Swb and 1 - Swb of water-filled rock; the two agree in wet
    rock (Swt = 1) and the wet-zone form reads higher in hydrocarbon-bearing
    rock, where the bound water is a larger share of the pore water.

    Equations
    ---------
    Bound water porosity:
        phi_bw = VCL * phi_cl

    Bound water saturation (capped at 0.999 where phi_bw >= phi_t):
        Swb = phi_bw / phi_t

    Conductivity, with Cw = 1/Rw and Cwb = 1/Rwb:
        1/Rt = (phi_t^m / a) * Swt^n * [Cw + (Swb/Swt) * (Cwb - Cw)]

    Total Sw, for n = 2 the positive root of
        Cw * Swt^2 + Swb * (Cwb - Cw) * Swt - a / (phi_t^m * Rt) = 0

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
        Bound water resistivity (ohm.m), from lab data or the user.
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
    # Infinite Rt (no conductivity reading) is missing, as in archie().
    if not reproducing_benchmark():
        if isinstance(rt_safe, np.ndarray):
            rt_safe = np.where(np.isinf(rt_safe), np.nan, rt_safe)
        elif np.isinf(rt_safe):
            rt_safe = np.nan

    swb = phi_bw / phi_t_safe

    # Bound-water saturation cannot exceed 1 (happens when Vcl*phi_cl > phi_t).
    # Clamp BEFORE it is used so the apparent-Rwt mixing term (1 - swb)/rw stays
    # physical: a raw swb above 1 would give a negative conductivity
    # contribution and an invalid Rwt.
    swb_safe = (
        np.where(swb >= 1.0, 0.999, swb)
        if isinstance(swb, np.ndarray)
        else min(swb, 0.999)
    )

    if reproducing_benchmark():
        # Wet-zone form: apparent water resistivity mixed from Swb, then
        # Archie on total porosity.
        rwt = 1.0 / (swb_safe / rwb_safe + (1.0 - swb_safe) / rw_safe)
        swt = ((a * rwt) / (phi_t_safe ** m * rt_safe)) ** (1.0 / n)
        swe = (swt - swb_safe) / (1.0 - swb_safe)
        return np.clip(swe, 0.0, 1.0)

    # Clavier-Coates-Dumanoir: the bound water is the fraction Swb/Swt of the
    # pore water, 1/Rt = phi_t^m/a * Swt^n * [Cw + (Swb/Swt) (Cwb - Cw)].
    cw = 1.0 / rw_safe
    cwb = 1.0 / rwb_safe
    k = a / (phi_t_safe ** m * rt_safe)  # a / (phi_t^m Rt)
    b = swb_safe * (cwb - cw)
    with np.errstate(invalid="ignore"):
        # n = 2: Cw Swt^2 + B Swt - K = 0, the positive root.
        swt = (-b + np.sqrt(b * b + 4.0 * cw * k)) / (2.0 * cw)
        if float(n) != 2.0:
            # Fixed-point iteration on the apparent water conductivity
            # Cw + (Swb/Swt) (Cwb - Cw), from the n = 2 root; Swt below Swb
            # leaves Swe = 0 either way.
            for _ in range(100):
                swt = np.power(k / (cw + b / np.maximum(swt, swb_safe)), 1.0 / n)

    # Convert to effective Sw
    swe = (swt - swb_safe) / (1.0 - swb_safe)

    return np.clip(swe, 0.0, 1.0)
