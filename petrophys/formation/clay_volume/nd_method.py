"""
Neutron-Density clay volume calculation.

This module provides clay volume calculation from the
neutron-density crossplot separation.
"""

from typing import Union, Tuple
import numpy as np


def vcl_from_nd(
    nphi: Union[float, np.ndarray],
    rhob: Union[float, np.ndarray],
    nphi_matrix: float = -0.02,
    rhob_matrix: float = 2.65,
    nphi_fluid: float = 1.0,
    rhob_fluid: float = 1.0,
    nphi_clay: float = 0.30,
    rhob_clay: float = 2.45
) -> Union[float, np.ndarray]:
    """
    Calculate clay volume from neutron-density crossplot.

    This method uses the neutron-density separation to estimate
    clay volume, particularly useful when GR is unreliable.

    Parameters
    ----------
    nphi : float or np.ndarray
        Neutron porosity (v/v)
    rhob : float or np.ndarray
        Bulk density (g/cm³)
    nphi_matrix : float, optional
        Matrix neutron response (default: -0.02 for quartz)
    rhob_matrix : float, optional
        Matrix density (default: 2.65 g/cm³)
    nphi_fluid : float, optional
        Fluid neutron response (default: 1.0)
    rhob_fluid : float, optional
        Fluid density (default: 1.0 g/cm³)
    nphi_clay : float, optional
        Clay neutron response (default: 0.30)
    rhob_clay : float, optional
        Clay density (default: 2.45 g/cm³)

    Returns
    -------
    float or np.ndarray
        Clay volume from N-D (0-1)

    Notes
    -----
    The method solves the simultaneous equations:
    NPHI = φ*NPHIf + (1-φ)*[(1-Vcl)*NPHIma + Vcl*NPHIcl]
    RHOB = φ*RHOBf + (1-φ)*[(1-Vcl)*RHOBma + Vcl*RHOBcl]

    Examples
    --------
    >>> vcl_nd = vcl_from_nd(nphi_data, rhob_data)
    """
    # Calculate density-derived porosity (for clean formation)
    phi_d = (rhob_matrix - rhob) / (rhob_matrix - rhob_fluid)

    # Calculate neutron-derived "porosity" (actually NPHI)
    phi_n = nphi

    # N-D separation indicates clay
    # For gas: N < D (gas effect)
    # For clay: N > D (clay effect)

    # Estimate VCL from N-D separation
    # Using simplified equation:
    # VCL = (NPHI - PHI_D) / (NPHI_clay - PHI_matrix)

    # More robust calculation using both curves
    # Based on the clay line in N-D space

    # Clean line parameters
    clean_slope = (rhob_matrix - rhob_fluid) / (nphi_matrix - nphi_fluid)

    # Calculate perpendicular distance to clean line
    # and normalize by distance to clay point

    # Simplified approach using N-D separation
    separation = phi_n - phi_d

    # Normalize by expected clay separation
    clay_separation = nphi_clay - (rhob_matrix - rhob_clay) / (rhob_matrix - rhob_fluid)

    if clay_separation == 0:
        return np.zeros_like(nphi) if isinstance(nphi, np.ndarray) else 0.0

    vcl = separation / clay_separation

    return np.clip(vcl, 0, 1)


def vcl_from_nd_thomas_steiber(
    nphi: Union[float, np.ndarray],
    rhob: Union[float, np.ndarray],
    nphi_sh: float = 0.30,
    rhob_sh: float = 2.45,
    nphi_sd: float = -0.02,
    rhob_sd: float = 2.65,
    rhob_fl: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thomas-Steiber analysis for clay volume and distribution.

    Determines clay volume and porosity distribution in
    laminated shaly sands.

    Parameters
    ----------
    nphi : array-like
        Neutron porosity
    rhob : array-like
        Bulk density
    nphi_sh : float
        Shale neutron porosity
    rhob_sh : float
        Shale density
    nphi_sd : float
        Clean sand neutron response
    rhob_sd : float
        Clean sand density
    rhob_fl : float
        Fluid density

    Returns
    -------
    tuple
        (vcl, phi_total, phi_effective)
    """
    nphi = np.asarray(nphi)
    rhob = np.asarray(rhob)

    # Calculate porosity from density
    phi_d = (rhob_sd - rhob) / (rhob_sd - rhob_fl)
    phi_d = np.clip(phi_d, 0, 0.5)

    # Calculate VCL
    # Using neutron excess over density porosity
    vcl = (nphi - phi_d) / (nphi_sh - 0)
    vcl = np.clip(vcl, 0, 1)

    # Total porosity
    phi_total = phi_d

    # Effective porosity (excluding clay-bound water)
    phi_clay = nphi_sh * vcl  # Clay porosity contribution
    phi_effective = phi_total - phi_clay
    phi_effective = np.clip(phi_effective, 0, phi_total)

    return vcl, phi_total, phi_effective
