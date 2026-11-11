"""
Pickett plot analysis for Archie parameter determination.

References
----------
Pickett, G.R. (1966). A review of current techniques for determination of
water saturation from logs. *Journal of Petroleum Technology* 18(11).

Archie, G.E. (1942). The electrical resistivity log as an aid in determining
some reservoir characteristics. *Transactions of the AIME* 146, 54-62.

Sokal, R.R. and Rohlf, F.J. (1995). *Biometry*, 3rd ed., section 14.13
(model II regression: the reduced major axis).
"""

import warnings
from typing import Tuple, Optional
import numpy as np
from scipy import stats

from petrophys.interpretation.electrical.cementation.archie_parameters import ArchieParameters

PICKETT_FIT_METHODS = ("rma", "ols")


def pickett_samples(rt: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Mask of the samples a Pickett fit uses: finite, Rt > 0 and phi > 0.01."""
    rt = np.asarray(rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    mask: np.ndarray = np.isfinite(rt) & np.isfinite(phi) & (phi > 0.01) & (rt > 0)
    return mask


def fit_pickett_line(
    log_phi: np.ndarray,
    log_rt: np.ndarray,
    method: str = "rma",
    intercept: Optional[float] = None,
) -> dict:
    """Fit the Pickett line log10(Rt) = intercept - m * log10(phi).

    Parameters
    ----------
    log_phi, log_rt : np.ndarray
        log10 porosity and log10 resistivity of the water-bearing samples.
    method : {"rma", "ols"}
        ``"rma"``: reduced major axis (geometric-mean regression), the fit
        for scatter in both variables (Sokal and Rohlf 1995); its slope is
        ``sign(r) * sd(log Rt) / sd(log phi)``. ``"ols"``: least squares of
        log Rt on log phi, which treats porosity as exact; porosity scatter
        then flattens the slope by the factor r**2 relative to RMA.
    intercept : float, optional
        Fixed intercept log10(a * Rw): the line is fitted through it and
        only m is estimated.

    Returns
    -------
    dict
        ``m``, ``slope`` (= -m), ``intercept``, ``r_squared``, ``p_value``
        and ``std_err`` (of the slope). ``r_squared`` is the squared
        correlation of log Rt and log phi; with a fixed intercept it is
        1 - SS_res / SS_tot of the line (0 when the line explains none of
        the scatter about the mean).
    """
    if method not in PICKETT_FIT_METHODS:
        raise ValueError(f"method must be one of {PICKETT_FIT_METHODS}; got {method!r}")
    x = np.asarray(log_phi, dtype=float)
    y = np.asarray(log_rt, dtype=float)
    n = x.size
    if intercept is None:
        ols = stats.linregress(x, y)
        r = float(ols.rvalue)
        if method == "ols":
            slope, std_err = float(ols.slope), float(ols.stderr)
        else:
            sign = 1.0 if r >= 0 else -1.0
            slope = sign * float(np.std(y, ddof=1) / np.std(x, ddof=1))
            std_err = abs(slope) * np.sqrt((1.0 - r ** 2) / n)
        fitted_intercept = float(np.mean(y) - slope * np.mean(x))
        p_value = float(ols.pvalue)
    else:
        # Line through the fixed intercept: regress y - c on x without a constant.
        fitted_intercept = float(intercept)
        yc = y - fitted_intercept
        sxx, syy, sxy = float(np.sum(x * x)), float(np.sum(yc * yc)), float(np.sum(x * yc))
        r = sxy / np.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
        if method == "ols":
            slope = sxy / sxx
            residual = yc - slope * x
            std_err = float(np.sqrt(np.sum(residual ** 2) / max(n - 1, 1) / sxx))
        else:
            sign = 1.0 if sxy >= 0 else -1.0
            slope = sign * np.sqrt(syy / sxx)
            std_err = abs(slope) * np.sqrt((1.0 - r ** 2) / n)
        t = r * np.sqrt((n - 1) / max(1.0 - r ** 2, 1e-300))
        p_value = float(2.0 * stats.t.sf(abs(t), n - 1))
        # r of a line through a fixed point is uncentered and near 1 for any
        # far-off point; report the share of the scatter about the mean
        # that the line explains instead.
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum((yc - slope * x) ** 2))
        r = np.sqrt(max(1.0 - ss_res / ss_tot, 0.0)) if ss_tot > 0 else 0.0
    return {
        "m": -float(slope),
        "slope": float(slope),
        "intercept": fitted_intercept,
        "r_squared": float(r ** 2),
        "p_value": p_value,
        "std_err": float(std_err),
    }


def pickett_analysis(
    rt: np.ndarray,
    phi: np.ndarray,
    sw_water: float = 1.0,
    vcl_cutoff: float = 0.3,
    vcl: Optional[np.ndarray] = None,
    *,
    method: str = "rma",
    rw: Optional[float] = None,
    a: float = 1.0,
) -> Tuple[ArchieParameters, dict]:
    """
    Perform Pickett plot analysis to determine Archie parameters.

    Fits the water line log(Rt) = -m × log(φ) + log(a × Rw) through every
    valid sample (finite, Rt > 0, φ > 0.01, VCL below ``vcl_cutoff`` when
    ``vcl`` is given) and returns m from its slope; n is 2.
    The fit gives the cementation exponent only for water-bearing (Sw = 1)
    rock, so pass the water-zone samples only: the function does not select
    them, and hydrocarbon-bearing samples bias m.

    Both log Rt and log φ carry measurement scatter, so the default fit is
    the reduced major axis (``method="rma"``). Least squares of log Rt on
    log φ (``method="ols"``) treats porosity as exact; its slope is the RMA
    slope times r², so porosity noise flattens it and m comes out low.

    With ``rw`` given, the intercept is fixed at log10(a × Rw) and only m is
    fitted (the line through the known water point at φ = 1).

    Parameters
    ----------
    rt : np.ndarray
        True formation resistivity
    phi : np.ndarray
        Porosity
    sw_water : float
        Not used (default: 1.0)
    vcl_cutoff : float
        Maximum VCL for clean sand (default: 0.3)
    vcl : np.ndarray, optional
        Clay volume for filtering
    method : {"rma", "ols"}, default "rma"
        Reduced major axis or ordinary least squares of log Rt on log φ.
    rw : float, optional
        Formation water resistivity (ohm.m) at formation temperature; fixes
        the intercept at log10(a × rw).
    a : float, default 1.0
        Tortuosity factor; Rw is read from the intercept as 10**intercept / a.

    Returns
    -------
    tuple
        (ArchieParameters, statistics_dict); the statistics include ``m``,
        ``intercept``, ``rw`` (from the intercept and ``a``, or the given
        ``rw``), ``method``, ``r_squared``, ``std_err`` (of the slope) and
        ``n_points``.

    Warns
    -----
    UserWarning
        If m lies outside 1.3-3.
    """
    if method not in PICKETT_FIT_METHODS:
        raise ValueError(f"method must be one of {PICKETT_FIT_METHODS}; got {method!r}")
    if not (np.isfinite(a) and a > 0):
        raise ValueError(f"a must be finite and positive; got {a!r}")
    if rw is not None and not (np.isfinite(rw) and rw > 0):
        raise ValueError(f"rw must be finite and positive (ohm.m); got {rw!r}")
    rt = np.asarray(rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    valid = pickett_samples(rt, phi)

    if vcl is not None:
        valid &= np.asarray(vcl, dtype=float) < vcl_cutoff

    if np.sum(valid) < 10:
        return ArchieParameters(), {"error": "Insufficient data"}

    log_rt = np.log10(rt[valid])
    log_phi = np.log10(phi[valid])

    fixed = None if rw is None else float(np.log10(a * rw))
    fit = fit_pickett_line(log_phi, log_rt, method=method, intercept=fixed)
    m = fit["m"]

    if not 1.3 <= m <= 3.0:
        warnings.warn(
            f"Pickett fit gives m = {m:.2f}, outside the usual 1.3-3 range: the samples are "
            "probably not all clean and water-bearing (Sw = 1). Fit the water leg only, or "
            "read m from the lower envelope of the points.",
            UserWarning,
            stacklevel=2,
        )

    stats_dict = {
        "m": m,
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "rw": float(rw) if rw is not None else float(10.0 ** fit["intercept"] / a),
        "method": method,
        "r_squared": fit["r_squared"],
        "p_value": fit["p_value"],
        "std_err": fit["std_err"],
        "n_points": int(np.sum(valid)),
    }

    return ArchieParameters(a=a, m=m, n=2.0), stats_dict
