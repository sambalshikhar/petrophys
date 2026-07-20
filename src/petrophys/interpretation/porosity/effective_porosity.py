"""
Effective porosity calculation.

This module provides effective porosity excluding clay-bound water.
"""

from typing import Union
import numpy as np


def effective_porosity(
    phit: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    phi_clay: float = 0.15
) -> Union[float, np.ndarray]:
    """
    Calculate effective porosity.

    PHIe = PHIt - VCL * phi_clay

    Parameters
    ----------
    phit : float or np.ndarray
        Total porosity (v/v)
    vcl : float or np.ndarray
        Clay volume (v/v)
    phi_clay : float, optional
        Clay-bound porosity (default: 0.15)

    Returns
    -------
    float or np.ndarray
        Effective porosity (v/v), clipped to [0, min(phit, 1 - VCL)]
    """
    phit = np.asarray(phit, dtype=float)
    vcl  = np.asarray(vcl,  dtype=float)
    vcl_clipped = np.clip(vcl, 0.0, 1.0)
    phie = phit - vcl_clipped * phi_clay
    upper = np.maximum(np.minimum(phit, 1.0 - vcl_clipped), 0.0)
    return np.minimum(np.maximum(phie, 0.0), upper)


def calculate_phie(
    phit: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    phi_clay: float = 0.15
) -> Union[float, np.ndarray]:
    """Alias for effective_porosity."""
    return effective_porosity(phit, vcl, phi_clay)


def bound_water_volume(
    vcl: Union[float, np.ndarray],
    phi_clay: float = 0.15
) -> Union[float, np.ndarray]:
    """
    Calculate clay-bound water volume.

    BVW_clay = VCL * phi_clay

    Parameters
    ----------
    vcl : float or np.ndarray
        Clay volume (v/v)
    phi_clay : float, optional
        Clay-bound porosity (default: 0.15)

    Returns
    -------
    float or np.ndarray
        Bound water volume (v/v)
    """
    return np.asarray(vcl, dtype=float) * phi_clay
