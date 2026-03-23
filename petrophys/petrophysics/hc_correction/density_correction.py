"""
Density correction for hydrocarbon effect.

Step 1 — Preliminary density porosity:
    phi_D_prelim = (rho_ma - RHOB - VCL * (rho_ma - RHOB_cl)) /
                   (rho_ma - rho_fl * Sxo - rho_hc * (1 - Sxo))

Step 2 — Corrected RHOB:
    RHOB_HC = RHOB + phi_D_prelim * (1 - Sxo) * (rho_fl - rho_hc)
"""

from typing import Union, Optional
import numpy as np


def correct_density_hc(
    rhob: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    sxo: Union[float, np.ndarray],
    rhob_fluid: float = 1.0,
    hc_type: str = "oil",
    vcl: Optional[Union[float, np.ndarray]] = None,
    rho_ma: float = 2.65
) -> Union[float, np.ndarray]:
    """
    Correct bulk density for hydrocarbon effect.

    When VCL and rho_ma are provided, uses the volumetric mixing-law
    inversion (design-doc approach).  The denominator is built from
    rho_fl (mud filtrate density), NOT from the measured rhob — using
    rhob in the denominator would make it depth-varying in a physically
    incorrect way.

    Parameters
    ----------
    rhob : float or np.ndarray
        Raw bulk density (g/cc)
    phi : float or np.ndarray
        Porosity estimate — used only in the simple fallback when
        VCL is not provided
    sxo : float or np.ndarray
        Flushed zone water saturation (v/v)
    rhob_fluid : float
        Mud-filtrate / fluid density in flushed zone (g/cc).
        Default 1.0 (fresh water).
    hc_type : str
        Hydrocarbon type: "oil" (rho_hc = 0.8) or "gas" (rho_hc = 0.2)
    vcl : float or np.ndarray, optional
        Volume of clay (fraction).  Enables the preliminary-porosity
        approach; if None the simple fallback is used.
    rho_ma : float
        Matrix density (g/cc, default 2.65 for sandstone)

    Returns
    -------
    float or np.ndarray
        HC-corrected bulk density (g/cc)
    """
    rho_hc = 0.2 if hc_type.lower() == "gas" else 0.8

    if vcl is not None:
        # ---------------------------------------------------------------
        # Design-doc approach: invert the volumetric mixing law
        #
        # RHOB = Vcl*rho_cl + phi*(Sxo*rho_fl + (1-Sxo)*rho_hc)
        #      + Vma*rho_ma   where  Vma = 1 - Vcl - phi
        #
        # Solving for phi_D_prelim (the porosity that explains the
        # observed RHOB given the known fluid mix):
        #
        #   denominator = rho_ma - rho_fl*Sxo - rho_hc*(1-Sxo)
        # ---------------------------------------------------------------
        numerator   = rho_ma - rhob - vcl * (rho_ma - rhob)
        denominator = rho_ma - rhob_fluid * sxo - rho_hc * (1.0 - sxo)

        if isinstance(denominator, np.ndarray):
            denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        else:
            denominator = denominator if abs(denominator) > 1e-10 else 1e-10

        phi_d_prelim  = numerator / denominator
        rhob_corrected = rhob + phi_d_prelim * (1.0 - sxo) * (rhob_fluid - rho_hc)
    else:
        # Simple fallback when VCL is unavailable
        rhob_corrected = rhob + phi * (1.0 - sxo) * (rhob_fluid - rho_hc)

    return rhob_corrected
