"""
Waxman-Smits equation for water saturation in shaly clastics with CEC data.

This module implements the Waxman-Smits model which accounts for
clay conductivity through the cation exchange capacity (CEC).
"""

from typing import Union
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark


#: Highest plausible whole-rock cation exchange capacity, meq/100 g: that of
#: pure smectite (montmorillonite, 80-150 meq/100 g; Grim, 1968).
MAX_CEC_MEQ_PER_100G = 150.0


def _waxman_smits_scalar(
    phi_t: float,
    ct: float,
    cw: float,
    qv: float,
    f_star: float,
    b_coeff: float,
    n: float,
    max_iter: int,
    tol: float
) -> float:
    """
    Iterative Waxman-Smits solver for a single depth sample.

    Full conductivity equation:
        Ct = (Sw^n / F*) * (Cw + B * Qv / Sw)

    Rearranged for iteration:
        Sw_{k+1} = [ F* * Ct / (Cw + B * Qv / Sw_k) ]^(1/n)

    Starts from the first-order approximation (Sw_0 = Sw at Sw=1
    in the clay term), then iterates until convergence.
    """
    # First-order seed (Sw = 1 in clay term)
    denom_0 = cw + b_coeff * qv
    denom_0 = max(denom_0, 1e-10)
    sw = (f_star * ct / denom_0) ** (1.0 / n)
    sw = min(max(sw, 1e-6), 1.0)

    for _ in range(max_iter):
        denom = cw + b_coeff * qv / sw
        denom = max(denom, 1e-10)
        sw_new = (f_star * ct / denom) ** (1.0 / n)
        sw_new = min(max(sw_new, 1e-6), 1.0)
        if abs(sw_new - sw) < tol:
            return sw_new
        sw = sw_new

    return sw


def waxman_smits(
    phi_t: Union[float, np.ndarray],
    rt: Union[float, np.ndarray],
    rw: Union[float, np.ndarray],
    cec: float,
    rho_ma: float,
    b_coeff: float,
    m_star: float = 2.0,
    n: float = 2.0,
    max_iter: int = 10,
    tol: float = 1e-6
) -> Union[float, np.ndarray]:
    """
    Waxman-Smits water saturation for shaly sands with CEC data.

    Full conductivity form (iterative solution):

        Ct = (Sw^n / F*) * (Cw + B * Qv / Sw)

    where:
        Ct    = 1/Rt
        Cw    = 1/Rw
        F*    = phi_t^(-m*)           (intrinsic formation factor)
        Qv    = CEC * rho_ma * (1 - phi_t) / phi_t

    Solved iteratively:
        Sw_{k+1} = [F* * Ct / (Cw + B * Qv / Sw_k)]^(1/n)

    starting from the first-order seed Sw_0 = [F* * Ct / (Cw + B*Qv)]^(1/n).
    Convergence is typically reached in 3-5 iterations.

    Note: the first-order approximation (setting Sw=1 in the clay term)
    under-estimates the clay conductivity (B*Qv instead of B*Qv/Sw), so it
    leaves too much of Ct to the water term and gives a pessimistically high
    Sw in hydrocarbon-bearing intervals; the iterative form corrects this.

    The equation is that of Waxman and Smits (1968), "Electrical
    conductivities in oil-bearing shaly sands", SPE Journal 8(2). Sw is the
    total-porosity saturation Swt; ``Swe = 1 - (phi_t / phi_e) * (1 - Swt)``
    puts it on an effective-porosity basis.

    Parameters
    ----------
    phi_t : float or np.ndarray
        Total porosity (fraction)
    rt : float or np.ndarray
        True formation resistivity (ohm.m)
    rw : float
        Formation water resistivity (ohm.m)
    cec : float
        Cation exchange capacity (meq/g). Note this is meq per gram, not the
        meq/100g lab-reporting convention; divide a meq/100g value by 100.
    rho_ma : float
        Matrix density (g/cc)
    b_coeff : float
        Equivalent conductance of clay counter-ions B, (S/m)/(meq/cm3), i.e.
        with Rw in ohm.m and Qv in meq/cm3 of pore volume. At 25 degC,
        B = 4.6 * [1 - 0.6 * exp(-0.77 / Rw)] (Waxman and Smits, 1968), about
        1.8 in fresh water rising to 4.6 in brine; it increases with
        temperature (Waxman and Thomas, 1974).
    m_star : float
        Intrinsic cementation exponent from SCAL (default: 2.0)
    n : float
        Saturation exponent (default: 2.0)
    max_iter : int
        Maximum iterations (default: 10; typically 3-5 suffice)
    tol : float
        Convergence tolerance on Sw (default: 1e-6)

    Returns
    -------
    float or np.ndarray
        Water saturation Sw (fraction, clipped 0-1)
    """
    scalar_input = np.isscalar(phi_t) and np.isscalar(rt) and np.isscalar(rw)

    phi_t, rt = np.broadcast_arrays(
        np.asarray(phi_t, dtype=float),
        np.asarray(rt, dtype=float),
    )
    phi_t = np.atleast_1d(phi_t.astype(float, copy=False))
    rt = np.atleast_1d(rt.astype(float, copy=False))

    phi_safe = np.where(phi_t <= 0, 1e-10, phi_t)
    rt_safe  = np.where(rt    <= 0, 1e-10, rt)
    rw_array = np.asarray(rw, dtype=float)
    if rw_array.ndim == 0:
        rw_array = np.full(phi_safe.shape, float(rw_array), dtype=float)
    else:
        rw_array = np.broadcast_to(rw_array, phi_safe.shape).astype(float)
    rw_array = np.where(rw_array <= 0, 1e-10, rw_array)
    cw = 1.0 / rw_array

    f_star = phi_safe ** (-m_star)
    qv     = cec * rho_ma * (1.0 - phi_safe) / phi_safe
    ct     = 1.0 / rt_safe

    sw = np.array([
        _waxman_smits_scalar(
            phi_safe[i], ct[i], float(cw[i]), qv[i], f_star[i],
            b_coeff, n, max_iter, tol
        )
        for i in range(len(phi_t))
    ])

    sw = np.clip(sw, 0.0, 1.0)
    # Infinite Rt (no conductivity reading) is missing, as in archie().
    if not reproducing_benchmark():
        sw = np.where(np.isinf(rt), np.nan, sw)
    return float(sw[0]) if scalar_input else sw
