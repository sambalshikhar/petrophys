"""
Flushed zone saturation (Sxo) calculation.

This module provides calculation of water saturation in the
flushed (invaded) zone.

References
----------
Archie, G.E. (1942). The electrical resistivity log as an aid in determining
some reservoir characteristics. *Transactions of the AIME* 146, 54-62.

Asquith, G. and Krygowski, D. (2004). *Basic Well Log Analysis*, 2nd ed. AAPG
Methods in Exploration 16.
"""

from typing import Union
import numpy as np

from petrophys.interpretation.saturation.archie import archie
from petrophys.interpretation.saturation._inputs import check_resistivity_parameters


def calculate_sxo(
    rxo: Union[float, np.ndarray],
    rmf: float,
    phi: Union[float, np.ndarray],
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Calculate flushed zone water saturation (Sxo).

    Uses Archie equation with Rxo and Rmf:
    Sxo^n = a × Rmf / (φ^m × Rxo)

    Parameters
    ----------
    rxo : float or np.ndarray
        Flushed zone resistivity (ohm.m)
    rmf : float
        Mud filtrate resistivity (ohm.m)
    phi : float or np.ndarray
        Porosity (v/v)
    a : float, optional
        Tortuosity factor
    m : float, optional
        Cementation exponent
    n : float, optional
        Saturation exponent

    Returns
    -------
    float or np.ndarray
        Flushed zone water saturation (0-1): a float for scalar inputs,
        else an array. NaN where Rxo or an Rmf array is at or below zero.

    Raises
    ------
    ValueError
        If a scalar ``rmf`` is at or below zero.
    """
    check_resistivity_parameters(rmf=rmf)
    return archie(rxo, rmf, phi, a, m, n)


def moveable_oil_saturation(
    sxo: Union[float, np.ndarray],
    sw: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate moveable oil saturation (MOS).

    MOS = Sxo - Sw

    Parameters
    ----------
    sxo : float or np.ndarray
        Flushed zone water saturation
    sw : float or np.ndarray
        Uninvaded zone water saturation

    Returns
    -------
    float or np.ndarray
        Moveable oil saturation
    """
    mos = sxo - sw
    return np.maximum(mos, 0)


def residual_oil_saturation(
    sxo: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate residual oil saturation (Sor).

    Sor = 1 - Sxo

    Parameters
    ----------
    sxo : float or np.ndarray
        Flushed zone water saturation

    Returns
    -------
    float or np.ndarray
        Residual oil saturation
    """
    return np.maximum(1 - sxo, 0)
