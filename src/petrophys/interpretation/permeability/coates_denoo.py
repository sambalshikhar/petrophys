"""
Coates-Denoo and Wyllie-Rose permeability from porosity and irreducible water saturation.

References
----------
Coates, G.R. and Denoo, S. (1981). The producibility answer product.
*The Technical Review* (Schlumberger) 29(2).

Wyllie, M.R.J. and Rose, W.D. (1950). Some theoretical considerations
related to the quantitative evaluation of the physical characteristics of
reservoir rock from electrical log data. *Transactions of the AIME* 189.

Tixier, M.P. (1949). Evaluation of permeability from electric-log
resistivity gradients. *Oil and Gas Journal*.

Morris, R.L. and Biggs, W.P. (1967). Using log-derived values of water
saturation and porosity. SPWLA 8th Annual Logging Symposium.

Timur, A. (1968). An investigation of permeability, porosity, and residual
water saturation relationships for sandstone reservoirs. *The Log Analyst*
9(4), 8-17.
"""

from typing import Optional, Union

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark

#: Published constants ``(P, Q, R)`` of the Wyllie-Rose family
#: ``k = P * phi^Q / Swi^R``, with k in mD and phi and Swi as fractions (v/v).
WYLLIE_ROSE_PRESETS = {
    # Morris and Biggs (1967): sqrt(k) = C * phi^3 / Swi, with C = 250 for
    # medium-gravity oil and C = 79 for dry gas.
    "morris_biggs_oil": (62500.0, 6.0, 2.0),
    "morris_biggs_gas": (6241.0, 6.0, 2.0),
    # Tixier (1949): sqrt(k) = 250 * phi^3 / Swi.
    "tixier": (62500.0, 6.0, 2.0),
    # Timur (1968): k = 0.136 * phi^4.4 / Swi^2 with phi and Swi in percent,
    # i.e. 0.136 * 100^2.4 = 8581 with fractions.
    "timur": (8581.0, 4.4, 2.0),
}


def coates_denoo(
    phi: Union[float, np.ndarray],
    swi: Union[float, np.ndarray],
    a: float = 10000.0,
    b: float = 4.0,
    c: float = 2.0,
) -> Union[float, np.ndarray]:
    """
    Calculate permeability using the Coates-Denoo equation.

    k = a * phi^b * ((1 - Swi) / Swi)^c, with a = 10000, b = 4 and c = 2

    Parameters
    ----------
    phi : float or np.ndarray
        Effective porosity (fraction). A porosity at or below zero is
        replaced by 0.001, so such a sample gets a very small permeability,
        not NaN.
    swi : float or np.ndarray
        Irreducible water saturation (fraction). A value at or below zero
        is replaced by 0.01.
    a, b, c : float, optional
        Coefficient, porosity exponent and saturation exponent

    Returns
    -------
    float or np.ndarray
        Permeability (mD); NaN where Swi exceeds 1
    """
    swi = np.asarray(swi, dtype=float)
    phi = np.asarray(phi, dtype=float)
    swi = np.where(swi <= 0, 0.01, swi)
    phi = np.where(phi <= 0, 0.001, phi)

    k = a * np.power(phi, b) * np.power((1.0 - swi) / swi, c)
    if not reproducing_benchmark():
        # Swi > 1 is not a saturation; squaring (1 - Swi) would give it a growing k.
        k = np.where(swi > 1.0, np.nan, k)
        return k[()] if k.ndim == 0 else k
    return k


def wyllie_rose(
    phi: Union[float, np.ndarray],
    swi: Union[float, np.ndarray],
    preset: str = "morris_biggs_oil",
    *,
    p: Optional[float] = None,
    q: Optional[float] = None,
    r: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Permeability from the Wyllie and Rose (1950) family.

    k = P * phi^Q / Swi^R

    with k in mD and phi and Swi as fractions (v/v). The constants come from
    ``preset`` (``WYLLIE_ROSE_PRESETS``); ``p``, ``q`` or ``r`` replace the
    preset's value.

    ====================  ======  ===  ===  ======================================
    preset                P       Q    R    Published form (phi, Swi fractions)
    ====================  ======  ===  ===  ======================================
    ``morris_biggs_oil``  62500   6    2    sqrt(k) = 250 phi^3 / Swi, medium-gravity
                                            oil (Morris and Biggs, 1967)
    ``morris_biggs_gas``  6241    6    2    sqrt(k) = 79 phi^3 / Swi, dry gas
                                            (Morris and Biggs, 1967)
    ``tixier``            62500   6    2    sqrt(k) = 250 phi^3 / Swi (Tixier, 1949)
    ``timur``             8581    4.4  2    k = 0.136 phi^4.4 / Swi^2 with phi and
                                            Swi in percent (Timur, 1968)
    ====================  ======  ===  ===  ======================================

    The default is the Morris-Biggs oil constant, which is what
    ``PermeabilityModule(method="wyllie_rose")`` uses.

    Parameters
    ----------
    phi : float or np.ndarray
        Effective porosity (fraction). A porosity at or below zero is
        replaced by 0.001, so such a sample gets a very small permeability,
        not NaN.
    swi : float or np.ndarray
        Irreducible water saturation (fraction). A value at or below zero
        is replaced by 0.01.
    preset : str
        Name of the published constants (default: ``"morris_biggs_oil"``)
    p, q, r : float, optional
        Coefficient and porosity and Swi exponents replacing the preset's

    Returns
    -------
    float or np.ndarray
        Permeability (mD)

    Examples
    --------
    >>> round(float(wyllie_rose(0.30, 0.25)), 1)
    729.0
    >>> round(float(wyllie_rose(0.30, 0.25, "morris_biggs_gas")), 1)
    72.8
    """
    if preset not in WYLLIE_ROSE_PRESETS:
        raise ValueError(
            f"Unknown Wyllie-Rose preset {preset!r}; use one of: "
            f"{', '.join(WYLLIE_ROSE_PRESETS)}"
        )
    preset_p, preset_q, preset_r = WYLLIE_ROSE_PRESETS[preset]
    p = preset_p if p is None else float(p)
    q = preset_q if q is None else float(q)
    r = preset_r if r is None else float(r)

    swi = np.asarray(swi, dtype=float)
    phi = np.asarray(phi, dtype=float)
    swi = np.where(swi <= 0, 0.01, swi)
    phi = np.where(phi <= 0, 0.001, phi)

    return p * np.power(phi, q) / np.power(swi, r)
