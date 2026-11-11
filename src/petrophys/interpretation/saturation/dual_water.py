"""
Dual Water model for water saturation in shaly formations.

This module implements the Dual Water model which separates total pore
water into free water (Rw) and bound water associated with clay
surfaces (Rwb).

References
----------
Clavier, C., Coates, G. and Dumanoir, J. (1984). Theoretical and
experimental bases for the dual-water model for interpretation of shaly
sands. *SPE Journal* 24(2), 153-168.

Crain, E.R. *Crain's Petrophysical Handbook*.
"""

from typing import Union
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.interpretation.saturation._inputs import (
    check_resistivity_parameters,
    nonpositive_resistivity,
)


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
    Swt (a quadratic for n = 2, solved by bisection otherwise) and the bound water is
    removed to give Swe. The benchmark packages used the wet-zone (Ro) form
    of Crain's Petrophysical Handbook, which mixes the waters in the
    proportions Swb and 1 - Swb of water-filled rock; the two agree in wet
    rock (Swt = 1). In hydrocarbon-bearing rock, where the bound water is a
    larger share of the pore water, the wet-zone form reads higher when the
    bound water is the more conductive (Rwb < Rw) and lower when it is the
    less conductive (Rwb > Rw, as with saline formation water).

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
        Effective water saturation Swe (fraction, clipped 0-1): a float for
        scalar inputs, else an array. 1 where bound water fills the whole
        pore space (Vcl x phi_cl >= phi_t, a shale without effective
        porosity). NaN where the total water saturation falls below the
        bound-water saturation (the inputs disagree: check phi_cl, Rwb and
        the porosity), and where Rt or an Rw or Rwb array is at or below zero.

    Raises
    ------
    ValueError
        If a scalar ``rw`` or ``rwb`` is at or below zero.
    """
    check_resistivity_parameters(rw=rw, rwb=rwb)
    scalar_input = all(np.ndim(value) == 0 for value in (phi_t, rt, rw, rwb, vcl, phi_cl))
    phi_t, rt, rw, rwb, vcl = (np.asarray(value, dtype=float) for value in (phi_t, rt, rw, rwb, vcl))
    invalid = nonpositive_resistivity(rt, rw, rwb)

    def _result(swe):
        swe = np.clip(swe, 0.0, 1.0)
        return float(swe) if scalar_input else swe

    phi_bw = vcl * phi_cl

    phi_t_safe = np.where(phi_t <= 0, 1e-10, phi_t)

    # Guard the resistivities against zero/negative (matches the other Sw models),
    # otherwise rwt and swt divide by zero on a scalar 0 or propagate NaN.
    rw_safe = np.where(rw <= 0, 1e-10, rw)
    rwb_safe = np.where(rwb <= 0, 1e-10, rwb)
    rt_safe = np.where(rt <= 0, 1e-10, rt)
    # Infinite Rt (no conductivity reading) is missing, as in archie().
    if not reproducing_benchmark():
        rt_safe = np.where(np.isinf(rt_safe), np.nan, rt_safe)

    swb = phi_bw / phi_t_safe

    # Bound-water saturation cannot exceed 1 (happens when Vcl*phi_cl > phi_t).
    # Clamp BEFORE it is used so the apparent-Rwt mixing term (1 - swb)/rw stays
    # physical: a raw swb above 1 would give a negative conductivity
    # contribution and an invalid Rwt.
    swb_safe = np.where(swb >= 1.0, 0.999, swb)

    if reproducing_benchmark():
        # Wet-zone form: apparent water resistivity mixed from Swb, then
        # Archie on total porosity.
        rwt = 1.0 / (swb_safe / rwb_safe + (1.0 - swb_safe) / rw_safe)
        swt = ((a * rwt) / (phi_t_safe ** m * rt_safe)) ** (1.0 / n)
        swe = (swt - swb_safe) / (1.0 - swb_safe)
        return _result(swe)

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
            # Other n: solve Cw Swt^n + B Swt^(n-1) = K by bisection on
            # [Swb, 1]. There Cw Swt + B = Cw (Swt - Swb) + Swb Cwb > 0, so the
            # left side rises with Swt and the root is unique; a root below
            # Swb gives Swe = 0 and one above 1 gives Swe = 1 after clipping.
            k_b, b_b, cw_b, swb_b = np.broadcast_arrays(k, b, cw, swb_safe)

            def excess(s):
                return cw_b * s ** n + b_b * s ** (n - 1.0) - k_b

            low, high = swb_b.astype(float).copy(), np.ones_like(swb_b, dtype=float)
            below, above = excess(low) >= 0.0, excess(high) <= 0.0
            for _ in range(60):
                middle = 0.5 * (low + high)
                rising = excess(middle) < 0.0
                low = np.where(rising, middle, low)
                high = np.where(rising, high, middle)
            swt = np.where(below, swb_b, np.where(above, 1.0, 0.5 * (low + high)))
            solvable = np.isfinite(k_b) & np.isfinite(b_b) & np.isfinite(cw_b) & np.isfinite(swb_b)
            swt = np.where(solvable, swt, np.nan)

    # Convert to effective Sw
    swe = (swt - swb_safe) / (1.0 - swb_safe)
    # Where bound water fills the whole pore space (shale, Vcl*phi_cl >= phi_t)
    # there is no effective porosity to hold hydrocarbon: water-bearing.
    # Elsewhere a total water saturation below the bound-water saturation
    # would put more hydrocarbon than effective pore space in the rock: the
    # inputs disagree (phi_cl, Rwb or the porosity), so Swe is undefined.
    with np.errstate(invalid="ignore"):
        swe = np.where(swt < swb_safe, np.nan, swe)
    swe = np.where(swb >= 1.0, 1.0, swe)
    swe = np.where(invalid, np.nan, swe)

    return _result(swe)
