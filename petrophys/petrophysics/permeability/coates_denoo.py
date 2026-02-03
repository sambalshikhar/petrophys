"""
Coates-Denoo equation for permeability estimation.
"""

from typing import Union
import numpy as np


def coates_denoo(
    phi: Union[float, np.ndarray],
    swi: Union[float, np.ndarray],
    c: float = 100
) -> Union[float, np.ndarray]:
    """
    Calculate permeability using Coates-Denoo equation.

    k = c² × (φ × (1-Swi) / Swi)²

    Parameters
    ----------
    phi : float or np.ndarray
        Porosity (fraction)
    swi : float or np.ndarray
        Irreducible water saturation (fraction)
    c : float
        Coates constant (default: 100)

    Returns
    -------
    float or np.ndarray
        Permeability (mD)
    """
    swi = np.where(swi <= 0, 0.01, swi) if isinstance(swi, np.ndarray) else max(swi, 0.01)
    phi = np.where(phi <= 0, 0.001, phi) if isinstance(phi, np.ndarray) else max(phi, 0.001)

    k = (c ** 2) * np.power(phi * (1 - swi) / swi, 2)
    return k


def wyllie_rose(
    phi: Union[float, np.ndarray],
    swi: Union[float, np.ndarray],
    a: float = 250
) -> Union[float, np.ndarray]:
    """
    Wyllie-Rose permeability equation.

    k = a × φ³ / Swi²

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

    k = a * np.power(phi, 3) / np.power(swi, 2)
    return k
