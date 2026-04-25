"""
Coates-Denoo equation for permeability estimation.
"""

from typing import Union

import numpy as np


def coates_denoo(
    phi: Union[float, np.ndarray],
    swi: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate permeability using the Coates-Denoo equation.

    k = 10000 * phi^4 * ((1 - Swi) / Swi)^2

    Parameters
    ----------
    phi : float or np.ndarray
        Effective porosity (fraction)
    swi : float or np.ndarray
        Irreducible water saturation (fraction)

    Returns
    -------
    float or np.ndarray
        Permeability (mD)
    """
    swi = np.where(swi <= 0, 0.01, swi) if isinstance(swi, np.ndarray) else max(swi, 0.01)
    phi = np.where(phi <= 0, 0.001, phi) if isinstance(phi, np.ndarray) else max(phi, 0.001)

    return 10000.0 * np.power(phi, 4.0) * np.power((1.0 - swi) / swi, 2.0)


def wyllie_rose(
    phi: Union[float, np.ndarray],
    swi: Union[float, np.ndarray],
    a: float = 250,
) -> Union[float, np.ndarray]:
    """
    Wyllie-Rose permeability equation.

    k = a * phi^3 / Swi^2

    Parameters
    ----------
    phi : float or np.ndarray
        Porosity (fraction)
    swi : float or np.ndarray
        Irreducible water saturation
    a : float
        Constant (default: 250)

    Returns
    -------
    float or np.ndarray
        Permeability (mD)
    """
    swi = np.where(swi <= 0, 0.01, swi) if isinstance(swi, np.ndarray) else max(swi, 0.01)
    phi = np.where(phi <= 0, 0.001, phi) if isinstance(phi, np.ndarray) else max(phi, 0.001)

    return a * np.power(phi, 3) / np.power(swi, 2)
