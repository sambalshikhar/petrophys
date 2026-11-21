"""
Timur equation for permeability estimation.

References
----------
Timur, A. (1968). An investigation of permeability, porosity, and residual
water saturation relationships for sandstone reservoirs. *The Log Analyst*
9(4), 8-17.
"""

from typing import Union

import numpy as np
from petrophys._internal.validation import broadcast_inputs, warn_range


def timur(
    phi: Union[float, np.ndarray],
    swi: Union[float, np.ndarray],
    a: float = 0.136,
    b: float = 4.4,
    c: float = 2.0,
) -> Union[float, np.ndarray]:
    """
    Calculate permeability using the Timur equation.

    k = a * phi_pct^b / Swi_pct^c

    Library porosity and Swirr inputs are fractions. They are converted to
    percent before applying the conventional Timur coefficient.

    Parameters
    ----------
    phi : float or np.ndarray
        Porosity (fraction). A porosity at or below zero is replaced by
        0.001, so such a sample gets a very small permeability, not NaN.
    swi : float or np.ndarray
        Irreducible water saturation (fraction). A value at or below zero
        is replaced by 0.01.
    a : float
        Timur constant for percent porosity/Swirr (default: 0.136)
    b : float
        Porosity exponent (default: 4.4)
    c : float
        Swi exponent (default: 2.0)

    Returns
    -------
    float or np.ndarray
        Permeability (mD)
    """
    # Apply the same nonpositive-input fallback for scalars and arrays.
    # Positive values, including small fractions, retain their measured value.
    phi, swi = broadcast_inputs(phi=phi, swi=swi)
    warn_range(phi, "Timur porosity (fraction)", 0, 1, open_lower=True,
               action="Nonpositive porosity is replaced by 0.001; other values are retained.")
    warn_range(swi, "Timur irreducible water saturation (fraction)", 0, 1, open_lower=True,
               action="Nonpositive Swirr is replaced by 0.01; other values are retained.")
    swi = np.where(swi <= 0, 0.01, swi)
    phi = np.where(phi <= 0, 0.001, phi)

    phi_pct = phi * 100.0
    swi_pct = swi * 100.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = a * np.power(phi_pct, b) / np.power(swi_pct, c)
    warn_range(result, "Timur permeability output (mD)", 0)
    return result
