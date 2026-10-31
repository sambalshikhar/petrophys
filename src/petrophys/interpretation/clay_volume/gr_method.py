"""
Gamma ray-based clay volume methods.

This module provides multiple equations for calculating VCL
from gamma ray logs.

References
----------
Larionov, V.V. (1969). *Borehole Radiometry*. Nedra, Moscow.

Stieber, S.J. (1970). Pulsed neutron capture log evaluation - Louisiana
Gulf Coast. SPE Annual Fall Meeting.

Clavier, C., Hoyle, W.R. and Meunier, D. (1971). Quantitative
interpretation of thermal neutron decay time logs: Part I. Fundamentals and
techniques. *Journal of Petroleum Technology* 23(6).

Asquith, G. and Krygowski, D. (2004). *Basic Well Log Analysis*, 2nd ed.
AAPG Methods in Exploration 16.
"""

from typing import Tuple, Union
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
        Linear clay volume indicator (IGR). Equal baselines leave the
        indicator undefined and return NaN, including for missing GR.
    """
    if gr_max == gr_min:
        result = np.full_like(np.asarray(gr, dtype=float), np.nan)
        return float(result) if result.ndim == 0 else result

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


#: Bounds of automatically picked GR baselines, API: the clean line is at most
#: GR_CLEAN_BOUND and the shale line at least GR_SHALE_BOUND.
GR_CLEAN_BOUND = 20.0
GR_SHALE_BOUND = 100.0


def bounded_gr_baselines(gr_min: float, gr_max: float) -> Tuple[float, float]:
    """Bound GR baselines picked from the log: ``(min(gr_min, 20), max(gr_max, 100))``.

    A guard for automatic picks (log percentiles or extremes), not a rule of
    the method: where the logged section holds no real shale or no real clean
    rock, its GR is nearly flat, and picks taken from it would stretch the
    gamma-ray index across noise and inflate clay volume (the COSTA files,
    reservoir only, read 9-20 API at P5/P95). Baselines a user sets are used
    as given. On a well whose clean rock is radioactive (feldspathic or
    volcanic sands) the bound overstates clay: set ``gr_min`` from a known
    clean sand there.

    Examples
    --------
    >>> bounded_gr_baselines(8.8, 20.5)
    (8.8, 100.0)
    >>> bounded_gr_baselines(64.8, 123.8)
    (20.0, 123.8)
    """
    return min(float(gr_min), GR_CLEAN_BOUND), max(float(gr_max), GR_SHALE_BOUND)


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
        Clean formation GR (API), used as given.
    gr_max : float
        Shale GR (API), used as given. Baselines picked automatically from
        the log can be bounded first with :func:`bounded_gr_baselines`.
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
    if np.ndim(gr):
        # Array-likes (pandas Series, Curve) are computed and returned as arrays.
        gr = np.asarray(gr, dtype=float)

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
