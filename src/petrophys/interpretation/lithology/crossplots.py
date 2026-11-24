"""
Crossplot data generation for lithology analysis.

This module provides functions to prepare data for
lithology crossplots (N-D, M-N, PEF).

References
----------
Burke, J.A., Campbell, R.L. and Schmidt, A.W. (1969). The litho-porosity
cross plot. SPWLA 10th Annual Logging Symposium.

Schlumberger (1989). *Log Interpretation Principles/Applications*.
"""

from typing import Dict
import numpy as np

from petrophys.interpretation.lithology.matrix_properties import get_log_density, mineral_properties


def nd_crossplot_data(
    nphi: np.ndarray,
    rhob: np.ndarray,
    minerals: list = None
) -> Dict[str, np.ndarray]:
    """
    Prepare data for neutron-density crossplot.

    Parameters
    ----------
    nphi : np.ndarray
        Neutron porosity
    rhob : np.ndarray
        Bulk density
    minerals : list, optional
        Minerals to include endpoints for

    Returns
    -------
    dict
        Data for crossplot including endpoints
    """
    if minerals is None:
        minerals = ["quartz", "calcite", "dolomite"]

    # Get mineral endpoints: the density-log response (evaporites read
    # below their grain density)
    endpoints = {}
    for mineral in minerals:
        props = mineral_properties(mineral)
        endpoints[mineral] = {
            "nphi": props["nphi"],
            "rhob": get_log_density(mineral)
        }

    # Fluid point
    endpoints["water"] = {"nphi": 1.0, "rhob": 1.0}

    return {
        "nphi": nphi,
        "rhob": rhob,
        "endpoints": endpoints
    }


def mn_crossplot_data(
    nphi: np.ndarray,
    rhob: np.ndarray,
    dtc: np.ndarray,
    rhob_fluid: float = 1.0,
    dtc_fluid: float = 189.0,
    nphi_fluid: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Prepare data for M-N crossplot.

    The M-N crossplot uses transformed coordinates:
    M = (dtf - dt) / (rhob - rhof) * 0.01
    N = (nphi_f - nphi) / (rhob - rhof)

    Parameters
    ----------
    nphi : np.ndarray
        Neutron porosity
    rhob : np.ndarray
        Bulk density
    dtc : np.ndarray
        Sonic slowness
    rhob_fluid : float, optional
        Fluid density
    dtc_fluid : float, optional
        Fluid slowness
    nphi_fluid : float, optional
        Fluid neutron porosity

    Returns
    -------
    dict
        M-N crossplot data
    """
    # M and N are undefined when RHOB reads at (or near) fluid density.
    denom = np.asarray(rhob, dtype=float) - rhob_fluid
    denom = np.where(np.abs(denom) < 0.01, np.nan, denom)

    # Calculate M and N values
    m = (dtc_fluid - dtc) / denom * 0.01
    n = (nphi_fluid - nphi) / denom

    # Calculate mineral endpoints
    minerals = ["quartz", "calcite", "dolomite"]
    endpoints = {}

    for mineral in minerals:
        props = mineral_properties(mineral)
        d = props["density"] - rhob_fluid
        if abs(d) > 0.01:
            m_val = (dtc_fluid - props["slowness"]) / d * 0.01
            n_val = (nphi_fluid - props["nphi"]) / d
        else:
            m_val = 0
            n_val = 0
        endpoints[mineral] = {"m": m_val, "n": n_val}

    return {
        "m": m,
        "n": n,
        "endpoints": endpoints
    }


def pef_crossplot_data(
    rhob: np.ndarray,
    pef: np.ndarray,
    minerals: list = None
) -> Dict[str, np.ndarray]:
    """
    Prepare data for PEF crossplot.

    Parameters
    ----------
    rhob : np.ndarray
        Bulk density
    pef : np.ndarray
        Photoelectric factor
    minerals : list, optional
        Minerals for endpoints

    Returns
    -------
    dict
        PEF crossplot data
    """
    if minerals is None:
        minerals = ["quartz", "calcite", "dolomite"]

    # U (volumetric photoelectric absorption) as PEF x RHOB, the solvers'
    # convention; chartbook U uses the electron density instead.
    u = pef * rhob

    endpoints = {}
    for mineral in minerals:
        props = mineral_properties(mineral)
        endpoints[mineral] = {
            "rhob": props["density"],
            "pef": props["pef"],
            "u": props.get("u", props["pef"] * props["density"])
        }

    return {
        "rhob": rhob,
        "pef": pef,
        "u": u,
        "endpoints": endpoints
    }
def identify_lithology_zone(
    nphi: float,
    rhob: float
) -> str:
    """
    Simple lithology identification from N-D position.

    Parameters
    ----------
    nphi : float
        Neutron porosity
    rhob : float
        Bulk density

    Returns
    -------
    str
        Identified lithology
    """
    # Simple zone identification based on crossplot position
    if rhob > 2.8:
        return "dolomite"
    elif rhob > 2.68:
        if nphi < 0.05:
            return "calcite"
        else:
            return "dolomite"
    elif rhob > 2.55:
        if nphi < 0:
            return "sandstone"
        elif nphi > 0.15:
            return "shale"
        else:
            return "calcite"
    elif rhob > 2.3:
        if nphi < 0.05:
            return "sandstone"
        elif nphi > 0.2:
            return "shale"
        else:
            return "mixed"
    else:
        if nphi < 0.3:
            return "gas_effect"
        else:
            return "coal_or_washout"
