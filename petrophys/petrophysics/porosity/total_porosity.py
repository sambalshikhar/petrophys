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
    - If φN < φD + 0.05: wD = 1, wN = 0 (gas effect)
    - Else: wD = 0.6, wN = 0.4

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
        # Check for gas effect
        if isinstance(nphi, np.ndarray):
            gas_effect = phi_d > nphi + 0.05

            phit = np.where(
                gas_effect,
                phi_d,  # Use density only in gas zones
                0.6 * phi_d + 0.4 * nphi  # Standard weighting
            )
        else:
            if phi_d > nphi + 0.05:
                phit = phi_d
            else:
                phit = 0.6 * phi_d + 0.4 * nphi

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
