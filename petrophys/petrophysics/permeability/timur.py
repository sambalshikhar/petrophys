"""
Timur equation for permeability estimation.
"""

from typing import Union
import numpy as np


def timur(
    phi: Union[float, np.ndarray],
    swi: Union[float, np.ndarray],
    a: float = 8581,
    b: float = 4.4,
    c: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Calculate permeability using Timur equation.

    k = a × φ^b / Swi^c

    Parameters
    ----------
    phi : float or np.ndarray
        Porosity (fraction)
    swi : float or np.ndarray
        Irreducible water saturation (fraction)
    a : float
        Timur constant (default: 8581)
    b : float
        Porosity exponent (default: 4.4)
    c : float
        Swi exponent (default: 2.0)

    Returns
    -------
    float or np.ndarray
        Permeability (mD)
    """
    # Avoid division by zero
    swi = np.where(swi <= 0, 0.01, swi) if isinstance(swi, np.ndarray) else max(swi, 0.01)
    phi = np.where(phi <= 0, 0.001, phi) if isinstance(phi, np.ndarray) else max(phi, 0.001)

    k = a * np.power(phi, b) / np.power(swi, c)
    return k


def timur_perm(
    phi: Union[float, np.ndarray],
    swi: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Alias for timur function with default parameters."""
    return timur(phi, swi)
