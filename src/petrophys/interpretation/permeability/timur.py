"""
Timur equation for permeability estimation.
"""

from typing import Union

import numpy as np


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
        Porosity (fraction)
    swi : float or np.ndarray
        Irreducible water saturation (fraction)
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
    swi = np.asarray(swi, dtype=float)
    phi = np.asarray(phi, dtype=float)
    swi = np.where(swi <= 0, 0.01, swi)
    phi = np.where(phi <= 0, 0.001, phi)

    phi_pct = phi * 100.0
    swi_pct = swi * 100.0
    return a * np.power(phi_pct, b) / np.power(swi_pct, c)
