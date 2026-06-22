"""
Total porosity calculation.

This module provides total porosity from neutron-density combination.
"""

from typing import Union
import numpy as np


def nd_weighted_porosity(
    phi_n: Union[float, np.ndarray],
    phi_d: Union[float, np.ndarray],
    weight_d: float = 0.6,
    weight_n: float = 0.4
) -> Union[float, np.ndarray]:
    """
    Calculate weighted average of neutron and density porosity.

    φND = wD × φD + wN × φN

    Parameters
    ----------
    phi_n : float or np.ndarray
        Neutron porosity
    phi_d : float or np.ndarray
        Density porosity
    weight_d : float, optional
        Density weight (default: 0.6)
    weight_n : float, optional
        Neutron weight (default: 0.4)

    Returns
    -------
    float or np.ndarray
        Weighted average porosity
    """
    return weight_d * phi_d + weight_n * phi_n


def total_porosity(
    nphi: Union[float, np.ndarray],
    phi_d: Union[float, np.ndarray],
    method: str = "adaptive"
) -> Union[float, np.ndarray]:
    """
    Calculate total porosity from neutron and density.

    The adaptive method uses different weightings based on
    neutron-density separation:
    - If φD > φN + 0.05 (density reads higher than neutron => gas effect):
      wD = 1, wN = 0, i.e. PHIT = φD
    - Else: PHIT = 0.6*φD + 0.4*φN (density-weighted shale-free average)

    Parameters
    ----------
    nphi : float or np.ndarray
        Neutron porosity
    phi_d : float or np.ndarray
        Density porosity
    method : str, optional
        Method: "adaptive", "average", "rms"

    Returns
    -------
    float or np.ndarray
        Total porosity

    Examples
    --------
    >>> phit = total_porosity(nphi_data, phid_data)
    """
    if method == "average":
        return (nphi + phi_d) / 2

    elif method == "rms":
        # RMS average
        return np.sqrt((nphi ** 2 + phi_d ** 2) / 2)

    else:  # adaptive
        if isinstance(nphi, np.ndarray) or isinstance(phi_d, np.ndarray):
            nphi_arr, phid_arr = np.broadcast_arrays(
                np.asarray(nphi, dtype=float),
                np.asarray(phi_d, dtype=float),
            )
            phit = np.full(nphi_arr.shape, np.nan, dtype=float)

            both = np.isfinite(nphi_arr) & np.isfinite(phid_arr)
            density_only = ~np.isfinite(nphi_arr) & np.isfinite(phid_arr)
            neutron_only = np.isfinite(nphi_arr) & ~np.isfinite(phid_arr)

            gas_effect = both & (phid_arr > nphi_arr + 0.05)
            standard = both & ~gas_effect

            phit[gas_effect] = phid_arr[gas_effect]
            phit[standard] = 0.6 * phid_arr[standard] + 0.4 * nphi_arr[standard]
            phit[density_only] = phid_arr[density_only]
            phit[neutron_only] = nphi_arr[neutron_only]
        else:
            nphi_finite = np.isfinite(nphi)
            phid_finite = np.isfinite(phi_d)
            if nphi_finite and phid_finite:
                if phi_d > nphi + 0.05:
                    phit = phi_d
                else:
                    phit = 0.6 * phi_d + 0.4 * nphi
            elif phid_finite:
                phit = phi_d
            elif nphi_finite:
                phit = nphi
            else:
                phit = np.nan

        return np.clip(phit, 0, 0.5)


def detect_gas_effect(
    nphi: Union[float, np.ndarray],
    phi_d: Union[float, np.ndarray],
    threshold: float = 0.05
) -> Union[bool, np.ndarray]:
    """
    Detect gas effect from neutron-density crossover.

    Parameters
    ----------
    nphi : float or np.ndarray
        Neutron porosity
    phi_d : float or np.ndarray
        Density porosity
    threshold : float, optional
        Separation threshold (default: 0.05)

    Returns
    -------
    bool or np.ndarray
        True where gas effect detected
    """
    return phi_d > nphi + threshold
