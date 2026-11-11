"""
Rock Quality Index (RQI), Flow Zone Indicator (FZI), and normalized porosity.

Equations
---------
RQI  = 0.0314 * sqrt(k / phi_e)
phi_z = phi_e / (1 - phi_e)
FZI  = RQI / phi_z

FZI is NaN where effective porosity is below ``MIN_FZI_POROSITY`` (0.01):
as porosity goes to zero phi_z does too and FZI grows without bound, so
shale and tight streaks would otherwise form spurious flow units.

References
----------
Amaefule, J.O., Altunbay, M., Tiab, D., Kersey, D.G. and Keelan, D.K.
(1993). Enhanced reservoir description: using core and log data to identify
hydraulic (flow) units and predict permeability in uncored intervals/wells.
SPE 26436, SPE Annual Technical Conference and Exhibition.

Kolodzie, S. Jr. (1980). Analysis of pore throat size and use of the
Waxman-Smits equation to determine OOIP in Spindle Field, Colorado.
SPE 9382, SPE Annual Technical Conference and Exhibition.

Martin, A.J., Solomon, S.T. and Hartmann, D.J. (1997). Characterization of
petrophysical flow units in carbonate reservoirs. *AAPG Bulletin* 81(5),
734-759.
"""

from typing import TYPE_CHECKING, Optional, Union
from dataclasses import dataclass
import numpy as np

from petrophys._internal.analysis_helpers import get_curve_data_depthwise
from petrophys._internal.benchmark import reproducing_benchmark

#: Effective porosity (v/v) below which FZI is not defined (NaN).
MIN_FZI_POROSITY = 0.01

if TYPE_CHECKING:
    from petrophys.model.well import Well


@dataclass
class RockQualityResult:
    """Result container for rock quality calculations."""
    rqi: np.ndarray
    phi_z: np.ndarray
    fzi: np.ndarray


def _maybe_scalar(result: np.ndarray, *inputs) -> Union[float, np.ndarray]:
    """Return a scalar when all inputs were scalar-like."""
    if all(np.asarray(value).ndim == 0 for value in inputs):
        return float(np.asarray(result).item())
    return result


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
    perm_arr = np.asarray(perm_k, dtype=float)
    phi_arr = np.asarray(phi_e, dtype=float)
    result = np.full(np.broadcast_shapes(perm_arr.shape, phi_arr.shape), np.nan, dtype=float)
    perm_b, phi_b = np.broadcast_arrays(perm_arr, phi_arr)
    valid = np.isfinite(perm_b) & np.isfinite(phi_b) & (perm_b > 0.0) & (phi_b > 0.0) & (phi_b < 1.0)
    result[valid] = 0.0314 * np.sqrt(perm_b[valid] / phi_b[valid])
    return _maybe_scalar(result, perm_k, phi_e)


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
    phi_arr = np.asarray(phi_e, dtype=float)
    result = np.full(phi_arr.shape, np.nan, dtype=float)
    valid = np.isfinite(phi_arr) & (phi_arr > 0.0) & (phi_arr < 1.0)
    result[valid] = phi_arr[valid] / (1.0 - phi_arr[valid])
    return _maybe_scalar(result, phi_e)


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
        Flow Zone Indicator (microns); NaN where the effective porosity is
        below :data:`MIN_FZI_POROSITY` (phi_z below 0.01 / 0.99).
    """
    rqi_arr = np.asarray(rqi, dtype=float)
    phi_z_arr = np.asarray(phi_z, dtype=float)
    result = np.full(np.broadcast_shapes(rqi_arr.shape, phi_z_arr.shape), np.nan, dtype=float)
    rqi_b, phi_z_b = np.broadcast_arrays(rqi_arr, phi_z_arr)
    valid = np.isfinite(rqi_b) & np.isfinite(phi_z_b) & (rqi_b >= 0.0) & (phi_z_b > 0.0)
    if not reproducing_benchmark():
        valid &= phi_z_b >= MIN_FZI_POROSITY / (1.0 - MIN_FZI_POROSITY)
    result[valid] = rqi_b[valid] / phi_z_b[valid]
    return _maybe_scalar(result, rqi, phi_z)


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
            Well object with PERM_CC/PERM and PHIE_CC/PHIE curves
        perm : np.ndarray, optional
            Permeability array (uses well PERM if not provided)
        phi_e : np.ndarray, optional
            Effective porosity (uses well PHIE if not provided)

        Returns
        -------
        RockQualityResult
        """
        if perm is None:
            perm, _, _ = get_curve_data_depthwise(well, ["PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"])
            if perm is None:
                raise ValueError("Permeability curve with finite samples required")

        if phi_e is None:
            phi_e, _, _ = get_curve_data_depthwise(well, ["PHIE_CC", "PHIE", "PHIT"])
            if phi_e is None:
                raise ValueError("PHIE_CC, PHIE, or PHIT with finite samples required")

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


#: Pore-throat classes by Winland R35 (microns), after Martin et al. (1997):
#: upper bound and name.
PORE_THROAT_CLASSES = (
    (0.1, "nano"),
    (0.5, "micro"),
    (2.0, "meso"),
    (10.0, "macro"),
    (np.inf, "mega"),
)


def calculate_winland_r35(
    perm_k: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Winland R35, the pore-throat radius (microns) at 35 % mercury saturation.

    ``log10(R35) = 0.732 + 0.588 log10(k) - 0.864 log10(phi_percent)``
    (Winland, published by Kolodzie, 1980), with k in mD and porosity in
    percent; ``phi`` is given as a fraction. NaN where either input is
    missing or not positive.

    >>> round(calculate_winland_r35(100.0, 0.20), 2)
    6.08
    """
    perm_arr = np.asarray(perm_k, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    finite_phi = phi_arr[np.isfinite(phi_arr)]
    if finite_phi.size and np.median(finite_phi) > 1.0:
        raise ValueError(
            "calculate_winland_r35: porosity looks like percent (median "
            f"{np.median(finite_phi):g}); give it as a fraction (read_core converts core tables)"
        )
    result = np.full(np.broadcast_shapes(perm_arr.shape, phi_arr.shape), np.nan, dtype=float)
    perm_b, phi_b = np.broadcast_arrays(perm_arr, phi_arr)
    valid = np.isfinite(perm_b) & np.isfinite(phi_b) & (perm_b > 0.0) & (phi_b > 0.0) & (phi_b < 1.0)
    result[valid] = 10.0 ** (
        0.732 + 0.588 * np.log10(perm_b[valid]) - 0.864 * np.log10(100.0 * phi_b[valid])
    )
    return _maybe_scalar(result, perm_k, phi)


def classify_pore_throat(r35: Union[float, np.ndarray]) -> Union[str, np.ndarray]:
    """Pore-throat class of R35 values: nano, micro, meso, macro or mega ('' where missing)."""
    values = np.asarray(r35, dtype=float)
    labels = np.full(values.shape, "", dtype=object)
    finite = np.isfinite(values)
    lower = 0.0
    for upper, name in PORE_THROAT_CLASSES:
        labels[finite & (values >= lower) & (values < upper)] = name
        lower = upper
    return str(labels.item()) if values.ndim == 0 else labels
