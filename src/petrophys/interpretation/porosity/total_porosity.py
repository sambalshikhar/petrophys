"""
Total porosity calculation.

This module provides total porosity from neutron-density combination.

References
----------
Gaymard, R. and Poupon, A. (1968). Response of neutron and formation
density logs in hydrocarbon bearing formations. *The Log Analyst* 9(5).
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
    method: str = "adaptive",
    gas_method: str = "rms",
) -> Union[float, np.ndarray]:
    """
    Calculate total porosity from neutron and density.

    The adaptive method uses different weightings based on
    neutron-density separation:

    - If phiD > phiN + 0.05 (density reads higher than neutron, a gas
      effect): PHIT = sqrt((phiD**2 + phiN**2) / 2), the gas-zone
      approximation of Gaymard and Poupon (1968), which balances the
      density reading too high against the neutron reading too low. A
      negative phiN counts as 0 there, so squaring cannot turn it into
      porosity. ``gas_method="density"`` takes PHIT = phiD instead, which
      overstates porosity in gas.
    - Else: PHIT = 0.6*phiD + 0.4*phiN (density-weighted average)

    Where only one of the two has a value, the adaptive method takes it. The
    adaptive result is limited to 0-0.5; "average" and "rms" are not limited.
    "rms" counts a negative neutron or density porosity as 0, so squaring
    cannot turn it into porosity.

    Parameters
    ----------
    nphi : float or np.ndarray
        Neutron porosity
    phi_d : float or np.ndarray
        Density porosity
    method : str, optional
        Method: "adaptive", "average", "rms"
    gas_method : str, optional
        Adaptive method in gas crossover: "rms" (default) or "density"

    Returns
    -------
    float or np.ndarray
        Total porosity

    Examples
    --------
    >>> float(total_porosity(0.25, 0.20))
    0.22
    """
    if method not in ("adaptive", "average", "rms"):
        raise ValueError(f"method must be 'adaptive', 'average' or 'rms'; got {method!r}")
    if gas_method not in ("rms", "density"):
        raise ValueError(f"gas_method must be 'rms' or 'density'; got {gas_method!r}")
    if np.ndim(nphi) or np.ndim(phi_d):
        # Array-likes (pandas Series, Curve) are computed and returned as arrays.
        nphi = np.asarray(nphi, dtype=float)
        phi_d = np.asarray(phi_d, dtype=float)

    if method == "average":
        return (nphi + phi_d) / 2

    elif method == "rms":
        # RMS average; a negative neutron or density porosity counts as 0
        return np.sqrt((np.clip(nphi, 0.0, None) ** 2 + np.clip(phi_d, 0.0, None) ** 2) / 2)

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

            if gas_method == "rms":
                phit[gas_effect] = np.sqrt(
                    (phid_arr[gas_effect] ** 2 + np.clip(nphi_arr[gas_effect], 0.0, None) ** 2) / 2
                )
            else:
                phit[gas_effect] = phid_arr[gas_effect]
            phit[standard] = 0.6 * phid_arr[standard] + 0.4 * nphi_arr[standard]
            phit[density_only] = phid_arr[density_only]
            phit[neutron_only] = nphi_arr[neutron_only]
        else:
            nphi_finite = np.isfinite(nphi)
            phid_finite = np.isfinite(phi_d)
            if nphi_finite and phid_finite:
                if phi_d > nphi + 0.05:
                    phit = np.sqrt((phi_d ** 2 + max(nphi, 0.0) ** 2) / 2) if gas_method == "rms" else phi_d
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
