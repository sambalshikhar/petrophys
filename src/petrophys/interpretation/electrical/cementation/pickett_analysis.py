"""
Pickett plot analysis for Archie parameter determination.

References
----------
Pickett, G.R. (1966). A review of current techniques for determination of
water saturation from logs. *Journal of Petroleum Technology* 18(11).

Archie, G.E. (1942). The electrical resistivity log as an aid in determining
some reservoir characteristics. *Transactions of the AIME* 146, 54-62.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats

from petrophys.interpretation.electrical.cementation.archie_parameters import ArchieParameters


def pickett_analysis(
    rt: np.ndarray,
    phi: np.ndarray,
    sw_water: float = 1.0,
    vcl_cutoff: float = 0.3,
    vcl: Optional[np.ndarray] = None
) -> Tuple[ArchieParameters, dict]:
    """
    Perform Pickett plot analysis to determine Archie parameters.

    Regresses log(Rt) on log(phi) over every valid sample (after the
    optional VCL filter) and returns m from the slope; a is fixed at 1.0
    and n at 2.0.
    The fit gives the cementation exponent only for water-bearing (Sw = 1)
    rock, so pass the water-zone samples only: the function does not select
    them, and hydrocarbon-bearing samples bias m.

    log(Rt) = -m × log(φ) + log(a × Rw)

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

    Returns
    -------
    tuple
        (ArchieParameters, statistics_dict)
    """
    # Filter valid data
    valid = ~np.isnan(rt) & ~np.isnan(phi) & (phi > 0.01) & (rt > 0)

    if vcl is not None:
        valid &= vcl < vcl_cutoff

    if np.sum(valid) < 10:
        return ArchieParameters(), {"error": "Insufficient data"}

    log_rt = np.log10(rt[valid])
    log_phi = np.log10(phi[valid])

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_phi, log_rt)

    # m = -slope, intercept = log(a × Rw)
    m = -slope

    # a is held at 1; the intercept is log10(a * Rw) and is reported in the
    # statistics, Rw itself is not returned.
    a = 1.0

    stats_dict = {
        "m": m,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
        "n_points": np.sum(valid),
    }

    return ArchieParameters(a=a, m=m, n=2.0), stats_dict
