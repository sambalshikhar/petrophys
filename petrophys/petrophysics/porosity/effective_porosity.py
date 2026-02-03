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

    φe = φt - VCL × φCL

    Parameters
    ----------
    phit : float or np.ndarray
        Total porosity
    vcl : float or np.ndarray
        Clay volume
    phi_clay : float, optional
        Clay porosity (default: 0.15)

    Returns
    -------
    float or np.ndarray
        Effective porosity (v/v)

    Examples
    --------
    >>> phie = effective_porosity(phit, vcl, phi_clay=0.12)
    """
    phie = phit - vcl * phi_clay
    return np.clip(phie, 0, phit if isinstance(phit, np.ndarray) else 0.5)


def calculate_phie(
    nphi: Union[float, np.ndarray],
    rhob: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    rhob_matrix: float = 2.65,
    rhob_fluid: float = 1.0,
    phi_clay: float = 0.15
) -> Union[float, np.ndarray]:
    """
    Calculate effective porosity from raw logs.

    Parameters
    ----------
    nphi : float or np.ndarray
        Neutron porosity
    rhob : float or np.ndarray
        Bulk density
    vcl : float or np.ndarray
        Clay volume
    rhob_matrix : float
        Matrix density
    rhob_fluid : float
        Fluid density
    phi_clay : float
        Clay porosity

    Returns
    -------
    float or np.ndarray
        Effective porosity
    """
    from petrophys.petrophysics.porosity.density_porosity import density_porosity
    from petrophys.petrophysics.porosity.total_porosity import total_porosity

    phi_d = density_porosity(rhob, rhob_matrix, rhob_fluid)
    phit = total_porosity(nphi, phi_d)
    phie = effective_porosity(phit, vcl, phi_clay)

    return phie


def bound_water_volume(
    vcl: Union[float, np.ndarray],
    phi_clay: float = 0.15
) -> Union[float, np.ndarray]:
    """
    Calculate clay-bound water volume.

    Parameters
    ----------
    vcl : float or np.ndarray
        Clay volume
    phi_clay : float
        Clay porosity

    Returns
    -------
    float or np.ndarray
        Bound water volume (v/v)
    """
    return vcl * phi_clay
