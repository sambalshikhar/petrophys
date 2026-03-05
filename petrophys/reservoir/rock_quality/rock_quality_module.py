"""
Rock Quality Index (RQI), Flow Zone Indicator (FZI), and normalized porosity.

Equations
---------
RQI  = 0.0314 * sqrt(k / phi_e)
phi_z = phi_e / (1 - phi_e)
FZI  = RQI / phi_z
"""

from typing import Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class RockQualityResult:
    """Result container for rock quality calculations."""
    rqi: np.ndarray
    phi_z: np.ndarray
    fzi: np.ndarray


def calculate_rqi(
    perm_k: Union[float, np.ndarray],
    phi_e: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate Reservoir Quality Index.

    RQI = 0.0314 * sqrt(k / phi_e)

    Parameters
    ----------
    perm_k : float or np.ndarray
        Permeability in mD
    phi_e : float or np.ndarray
        Effective porosity (fraction)

    Returns
    -------
    float or np.ndarray
        RQI (microns)
    """
    phi_safe = np.where(phi_e <= 0, 1e-10, phi_e) if isinstance(phi_e, np.ndarray) else max(phi_e, 1e-10)
    k_safe = np.where(perm_k <= 0, 1e-10, perm_k) if isinstance(perm_k, np.ndarray) else max(perm_k, 1e-10)
    return 0.0314 * np.sqrt(k_safe / phi_safe)


def calculate_phi_z(
    phi_e: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate normalized porosity (pore-to-matrix ratio).

    phi_z = phi_e / (1 - phi_e)

    Parameters
    ----------
    phi_e : float or np.ndarray
        Effective porosity (fraction)

    Returns
    -------
    float or np.ndarray
        Normalized porosity
    """
    phi_safe = np.clip(phi_e, 1e-10, 0.999)
    return phi_safe / (1.0 - phi_safe)


def calculate_fzi(
    rqi: Union[float, np.ndarray],
    phi_z: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate Flow Zone Indicator.

    FZI = RQI / phi_z

    Parameters
    ----------
    rqi : float or np.ndarray
        Reservoir Quality Index
    phi_z : float or np.ndarray
        Normalized porosity

    Returns
    -------
    float or np.ndarray
        Flow Zone Indicator
    """
    phi_z_safe = np.where(phi_z <= 0, 1e-10, phi_z) if isinstance(phi_z, np.ndarray) else max(phi_z, 1e-10)
    return rqi / phi_z_safe


def calculate_rock_quality(
    perm_k: Union[float, np.ndarray],
    phi_e: Union[float, np.ndarray]
) -> dict:
    """
    Calculate RQI, normalized porosity, and FZI.

    Parameters
    ----------
    perm_k : float or np.ndarray
        Permeability in mD
    phi_e : float or np.ndarray
        Effective porosity (fraction)

    Returns
    -------
    dict
        {'RQI': array, 'phi_z': array, 'FZI': array}
    """
    rqi = calculate_rqi(perm_k, phi_e)
    phi_z = calculate_phi_z(phi_e)
    fzi = calculate_fzi(rqi, phi_z)
    return {'RQI': rqi, 'phi_z': phi_z, 'FZI': fzi}


class RockQualityModule:
    """Rock quality calculator with RQI, FZI, and phi_z."""

    def calculate(
        self,
        well: "Well",
        perm: Optional[np.ndarray] = None,
        phi_e: Optional[np.ndarray] = None
    ) -> RockQualityResult:
        """
        Calculate rock quality indices.

        Parameters
        ----------
        well : Well
            Well object with PERM and PHIE curves
        perm : np.ndarray, optional
            Permeability array (uses well PERM if not provided)
        phi_e : np.ndarray, optional
            Effective porosity (uses well PHIE if not provided)

        Returns
        -------
        RockQualityResult
        """
        if perm is None:
            perm_curve = well.get("PERM")
            if perm_curve is None:
                raise ValueError("Permeability (PERM) required")
            perm = perm_curve.data

        if phi_e is None:
            phie_curve = well.get("PHIE")
            if phie_curve is None:
                raise ValueError("Effective porosity (PHIE) required")
            phi_e = phie_curve.data

        rqi = calculate_rqi(perm, phi_e)
        phi_z = calculate_phi_z(phi_e)
        fzi = calculate_fzi(rqi, phi_z)

        return RockQualityResult(rqi=rqi, phi_z=phi_z, fzi=fzi)

    def add_to_well(
        self,
        well: "Well",
        result: Optional[RockQualityResult] = None
    ) -> "Well":
        """Add rock quality curves to well."""
        if result is None:
            result = self.calculate(well)

        well.add_curve("RQI", result.rqi, "microns", "Reservoir Quality Index")
        well.add_curve("PHI_Z", result.phi_z, "v/v", "Normalized Porosity")
        well.add_curve("FZI", result.fzi, "microns", "Flow Zone Indicator")

        return well
