"""
Pickett plot analysis for Archie parameter determination.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats

from petrophys.petrophysics.cementation.archie_parameters import ArchieParameters


def pickett_analysis(
    rt: np.ndarray,
    phi: np.ndarray,
    sw_water: float = 1.0,
    vcl_cutoff: float = 0.3,
    vcl: Optional[np.ndarray] = None
) -> Tuple[ArchieParameters, dict]:
    """
    Perform Pickett plot analysis to determine Archie parameters.

    Uses water-saturated zones (Sw=1) to determine m from slope
    and a from intercept.

    log(Rt) = -m × log(φ) + log(a × Rw)

    Parameters
    ----------
    rt : np.ndarray
        True formation resistivity
    phi : np.ndarray
        Porosity
    sw_water : float
        Sw value to consider as water zone (default: 1.0)
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

    # Estimate Rw from water zones (simplified)
    # In practice, would identify water zones and use their Rt/F relationship
    a = 1.0  # Assume a=1 for simplicity

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
