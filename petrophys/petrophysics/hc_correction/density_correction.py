"""
Density correction for hydrocarbon effect.
"""

from typing import Union
import numpy as np


def correct_density_hc(
    rhob: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    sxo: Union[float, np.ndarray],
    rhob_fluid: float = 1.0,
    hc_type: str = "oil"
) -> Union[float, np.ndarray]:
    """
    Correct bulk density for hydrocarbon effect.

    RHOB_HC = RHOB + φ × (1 - Sxo) × (ρfl - ρHC)

    where ρHC is hydrocarbon density (oil=0.8, gas=0.2)

    Parameters
    ----------
    rhob : float or np.ndarray
        Raw bulk density
    phi : float or np.ndarray
        Porosity
    sxo : float or np.ndarray
        Flushed zone water saturation
    rhob_fluid : float
        Fluid density (g/cm³)
    hc_type : str
        Hydrocarbon type ("oil" or "gas")

    Returns
    -------
    float or np.ndarray
        HC-corrected bulk density
    """
    # Hydrocarbon density
    if hc_type.lower() == "gas":
        rhob_hc = 0.2
    else:  # oil
        rhob_hc = 0.8

    # Correction
    rhob_corrected = rhob + phi * (1 - sxo) * (rhob_fluid - rhob_hc)

    return rhob_corrected
