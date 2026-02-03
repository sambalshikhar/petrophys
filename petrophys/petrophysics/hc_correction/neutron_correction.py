"""
Neutron porosity correction for hydrocarbon effect.
"""

from typing import Union
import numpy as np


def correct_neutron_hc(
    nphi: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    sxo: Union[float, np.ndarray],
    hc_type: str = "oil"
) -> Union[float, np.ndarray]:
    """
    Correct neutron porosity for hydrocarbon effect.

    NPHI_HC = NPHI + φ × (1 - Sxo) × (1 - HI)

    where HI is hydrogen index (oil=0.8, gas=0.3)

    Parameters
    ----------
    nphi : float or np.ndarray
        Raw neutron porosity
    phi : float or np.ndarray
        Porosity
    sxo : float or np.ndarray
        Flushed zone water saturation
    hc_type : str
        Hydrocarbon type ("oil" or "gas")

    Returns
    -------
    float or np.ndarray
        HC-corrected neutron porosity
    """
    # Hydrogen index
    if hc_type.lower() == "gas":
        hi = 0.3
    else:  # oil
        hi = 0.8

    # Correction
    nphi_hc = nphi + phi * (1 - sxo) * (1 - hi)

    return nphi_hc
