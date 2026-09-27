"""
Gamma ray-based clay volume methods.

This module provides multiple equations for calculating VCL
from gamma ray logs.
"""

from typing import Union
import numpy as np


#: Gamma-ray clay-volume transforms accepted by :func:`vcl_from_gr`.
GR_VCL_METHODS = ("linear", "larionov_tertiary", "larionov_older", "steiber", "clavier")


def vcl_linear(
    gr: Union[float, np.ndarray],
    gr_min: float,
    gr_max: float
) -> Union[float, np.ndarray]:
    """
    Calculate clay volume using linear GR method.

    VCL = (GR - GRmin) / (GRmax - GRmin)

    Parameters
    ----------
    gr : float or np.ndarray
        Gamma ray values (API)
    gr_min : float
        Clean sand GR baseline
    gr_max : float
        Shale GR baseline

    Returns
    -------
    float or np.ndarray
        Linear clay volume indicator (IGR)
    """
    if gr_max == gr_min:
        return np.zeros_like(gr) if isinstance(gr, np.ndarray) else 0.0

    igr = (gr - gr_min) / (gr_max - gr_min)
    return np.clip(igr, 0, 1)


def vcl_larionov_tertiary(igr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Larionov equation for Tertiary (younger) rocks.

    VCL = 0.083 * (2^(3.7 * IGR) - 1)

    Parameters
    ----------
    igr : float or np.ndarray
        Linear gamma ray index

    Returns
    -------
    float or np.ndarray
        Clay volume
    """
    vcl = 0.083 * (np.power(2, 3.7 * igr) - 1)
    return np.clip(vcl, 0, 1)


def vcl_larionov_older(igr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Larionov equation for older (consolidated) rocks.

    VCL = 0.33 * (2^(2 * IGR) - 1)

    Parameters
    ----------
    igr : float or np.ndarray
        Linear gamma ray index

    Returns
    -------
    float or np.ndarray
        Clay volume
    """
    vcl = 0.33 * (np.power(2, 2 * igr) - 1)
    return np.clip(vcl, 0, 1)


def vcl_steiber(igr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Steiber equation for clay volume.

    VCL = IGR / (3 - 2 * IGR)

    Parameters
    ----------
    igr : float or np.ndarray
        Linear gamma ray index

    Returns
    -------
    float or np.ndarray
        Clay volume
    """
    # Avoid division by zero
    denominator = 3 - 2 * igr
    if isinstance(igr, np.ndarray):
        denominator = np.where(denominator == 0, 1e-10, denominator)
    elif denominator == 0:
        denominator = 1e-10

    vcl = igr / denominator
    return np.clip(vcl, 0, 1)


def vcl_clavier(igr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Clavier equation for clay volume.

    VCL = 1.7 - sqrt(3.38 - (IGR + 0.7)^2)

    Parameters
    ----------
    igr : float or np.ndarray
        Linear gamma ray index

    Returns
    -------
    float or np.ndarray
        Clay volume
    """
    # Handle potential negative values under sqrt
    term = 3.38 - np.power(igr + 0.7, 2)
    if isinstance(term, np.ndarray):
        term = np.maximum(term, 0)
    else:
        term = max(term, 0)

    vcl = 1.7 - np.sqrt(term)
    return np.clip(vcl, 0, 1)


def vcl_from_gr(
    gr: Union[float, np.ndarray],
    gr_min: float,
    gr_max: float,
    method: str = "linear"
) -> Union[float, np.ndarray]:
    """
    Calculate clay volume from gamma ray using specified method.

    Parameters
    ----------
    gr : float or np.ndarray
        Gamma ray values (API)
    gr_min : float
        Clean formation GR (default 20 API)
    gr_max : float
        Shale GR (default 100 API)
    method : str, optional
        Method to use:
        - "linear": Linear IGR
        - "larionov_tertiary": Larionov for tertiary rocks
        - "larionov_older": Larionov for older rocks
        - "steiber": Steiber equation
        - "clavier": Clavier equation

    Returns
    -------
    float or np.ndarray
        Clay volume (0-1)

    Examples
    --------
    >>> from petrophys.interpretation.clay_volume import vcl_from_gr
    >>> vcl = vcl_from_gr(gr_data, gr_min=20, gr_max=100)
    """
    # Boundary enforcement rules from design doc:
    # If calculated GR_max < 100 → force GR_max = 100
    # If calculated GR_min > 20  → force GR_min = 20
    gr_max = max(gr_max, 100.0)
    gr_min = min(gr_min, 20.0)

    # First calculate linear index
    igr = vcl_linear(gr, gr_min, gr_max)

    # Apply non-linear transform if requested
    method = method.lower()
    if method == "linear":
        return igr
    elif method == "larionov_tertiary":
        return vcl_larionov_tertiary(igr)
    elif method == "larionov_older":
        return vcl_larionov_older(igr)
    elif method == "steiber":
        return vcl_steiber(igr)
    elif method == "clavier":
        return vcl_clavier(igr)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use one of: {', '.join(GR_VCL_METHODS)}"
        )
