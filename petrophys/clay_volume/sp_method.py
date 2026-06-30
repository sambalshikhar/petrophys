"""
SP-based clay volume calculation.

This module provides clay volume calculation from spontaneous potential.

Note: SP should be discarded if clay or KCl is used in the drilling mud.
"""

from typing import Union
import numpy as np


def vcl_from_sp(
    sp: Union[float, np.ndarray],
    sp_max: float,
    sp_min: float
) -> Union[float, np.ndarray]:
    """
    Calculate clay volume from spontaneous potential.

    VCL_SP = (SPmax - SP) / (SPmax - SPmin)

    Note: SP_max is the maximum SP deflection (clean sand)
    and SP_min is the shale baseline.

    Parameters
    ----------
    sp : float or np.ndarray
        Spontaneous potential values (mV)
    sp_max : float
        Clean sand SP value (maximum deflection)
    sp_min : float
        Shale baseline SP value

    Returns
    -------
    float or np.ndarray
        Clay volume from SP (0-1)

    Notes
    -----
    The SP log should not be used for VCL when:
    - Formation water salinity is very low
    - KCl mud is used
    - Formation is highly shaly

    Examples
    --------
    >>> vcl_sp = vcl_from_sp(sp_data, sp_max=-80, sp_min=-10)
    """
    if sp_max == sp_min:
        return np.zeros_like(sp) if isinstance(sp, np.ndarray) else 0.0

    vcl = (sp_max - sp) / (sp_max - sp_min)
    return np.clip(vcl, 0, 1)


def should_use_sp(
    mud_type: str = "WBM",
    has_kcl: bool = False,
    salinity_ppm: float = 50000
) -> bool:
    """
    Determine if SP should be used for VCL calculation.

    Parameters
    ----------
    mud_type : str
        Mud type ("WBM", "OBM")
    has_kcl : bool
        Whether KCl is in the mud
    salinity_ppm : float
        Formation water salinity

    Returns
    -------
    bool
        True if SP can be used for VCL
    """
    # OBM gives no SP
    if mud_type.upper() == "OBM":
        return False

    # KCl interferes with SP
    if has_kcl:
        return False

    # Very low salinity gives poor SP
    if salinity_ppm < 5000:
        return False

    return True
