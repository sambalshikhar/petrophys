"""
Flag generation for net pay determination.
"""

from typing import Dict, Union
import numpy as np


def generate_net_flags(
    vcl: np.ndarray,
    phie: np.ndarray,
    sw: np.ndarray,
    vcl_cutoff: float = 0.3,
    phie_cutoff: Union[float, np.ndarray] = 0.08,
    sw_cutoff: float = 0.7
) -> Dict[str, np.ndarray]:
    """
    Generate net rock, reservoir, and pay flags.

    Parameters
    ----------
    vcl : np.ndarray
        Clay volume
    phie : np.ndarray
        Effective porosity
    sw : np.ndarray
        Water saturation
    vcl_cutoff : float
        VCL cutoff (default: 0.3)
    phie_cutoff : float or np.ndarray
        PHIE cutoff. Scalar applies uniformly; array allows per-depth
        cutoffs (e.g. 0.08 for clastics, 0.045 for carbonates).
    sw_cutoff : float
        Sw cutoff (default: 0.7)

    Returns
    -------
    dict
        Dictionary with flag arrays
    """
    # Net Rock: VCL <= cutoff (inclusive, matching the documented contract)
    net_rock = vcl <= vcl_cutoff

    # Net Reservoir: Net Rock + PHIE >= cutoff
    # numpy broadcasts correctly for both scalar and array phie_cutoff
    net_reservoir = net_rock & (phie >= phie_cutoff)

    # Net Pay: Net Reservoir + Sw <= cutoff
    net_pay = net_reservoir & (sw <= sw_cutoff)

    return {
        "net_rock": net_rock,
        "net_reservoir": net_reservoir,
        "net_pay": net_pay,
    }
