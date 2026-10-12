"""
Neutron-Density clay volume calculation.
"""
from typing import Union, Tuple
import numpy as np


# N-D clean-line endpoints per matrix: (den_cl1, neu_cl1, den_cl2, neu_cl2).
# Cl1 is the zero-porosity matrix point (from MINERAL_PROPERTIES); Cl2 is the
# same matrix at ~40 p.u. water-filled: den = den_ma*(1-0.4) + 1.0*0.4,
# neu = neu_ma + 0.4. The limestone row reproduces the historical defaults
# exactly so carbonate results are unchanged.
_ND_CLEAN_LINES = {
    "limestone": (2.71, 0.00, 2.03, 0.40),   # calcite (historical default)
    "sandstone": (2.65, -0.02, 1.99, 0.38),  # quartz
    "dolomite": (2.87, 0.02, 2.12, 0.42),    # dolomite
}

#: Default N-D clay point ``(RHOB g/cm3, NPHI v/v)`` of :func:`vcl_from_nd`.
ND_CLAY_POINT = (2.62, 0.38)

_ND_MATRIX_ALIASES = {
    "quartz": "sandstone", "clastic": "sandstone", "sand": "sandstone",
    "siliciclastic": "sandstone", "ss": "sandstone", "silica": "sandstone",
    "sio2": "sandstone",
    "calcite": "limestone", "carbonate": "limestone", "chalk": "limestone",
    "lime": "limestone", "ls": "limestone", "calcium_carbonate": "limestone",
    "dolostone": "dolomite", "dolo": "dolomite",
}


def nd_clean_line(matrix: str = "limestone") -> Tuple[float, float, float, float]:
    """Return (den_cl1, neu_cl1, den_cl2, neu_cl2) for the given matrix.

    Unknown/None matrix falls back to limestone (the historical default).
    """
    key = str(matrix or "limestone").strip().lower()
    key = _ND_MATRIX_ALIASES.get(key, key)
    return _ND_CLEAN_LINES.get(key, _ND_CLEAN_LINES["limestone"])


def nd_clay_separation(den_clay: float, neu_clay: float, matrix: str = "limestone") -> float:
    """Neutron porosity (v/v) by which a clay point lies above a clean line.

    The excess of ``neu_clay`` over the :func:`nd_clean_line` of ``matrix``
    at the density ``den_clay``. A clay point needs a clear excess to
    separate clay from clean rock: at zero or below it lies on or under the
    clean line, where the three-point formula of :func:`vcl_from_nd` has no
    meaning.

    Examples
    --------
    >>> round(nd_clay_separation(2.35, 0.40, matrix="sandstone"), 4)
    0.2382
    """
    den_cl1, neu_cl1, den_cl2, neu_cl2 = nd_clean_line(matrix)
    clean_neutron = neu_cl1 + (den_clay - den_cl1) * (neu_cl2 - neu_cl1) / (den_cl2 - den_cl1)
    return float(neu_clay - clean_neutron)


def vcl_from_nd(
    nphi: Union[float, np.ndarray],
    rhob: Union[float, np.ndarray],
    den_cl1: float = 2.71,
    neu_cl1: float = 0.00,
    den_cl2: float = 2.03,
    neu_cl2: float = 0.40,
    den_clay: float = ND_CLAY_POINT[0],
    neu_clay: float = ND_CLAY_POINT[1],
    clip: bool = True
) -> Union[float, np.ndarray]:
    """
    Clay volume from the closed-form 3-point (Cl1, Cl2, Clay)
    neutron-density formulation, evaluated deterministically per sample::

        VclND =
            ((DenCl2 - DenCl1)*(Neu - NeuCl1) - (Den - DenCl1)*(NeuCl2 - NeuCl1))
            / ((DenCl2 - DenCl1)*(NeuClay - NeuCl1) - (DenClay - DenCl1)*(NeuCl2 - NeuCl1))

    Parameters
    ----------
    nphi : float or np.ndarray
        Neutron porosity (Neu)
    rhob : float or np.ndarray
        Bulk density (Den)
    den_cl1, neu_cl1 : float
        Clean point 1 in (Den, Neu)
    den_cl2, neu_cl2 : float
        Clean point 2 in (Den, Neu)
    den_clay, neu_clay : float
        Clay point in (Den, Neu)
    clip : bool
        If True, clip result to [0, 1]

    Returns
    -------
    float or np.ndarray
        VclND in [0,1] (if clip=True)
    """
    # Preserve scalar vs array behavior
    nphi_arr = np.asarray(nphi, dtype=float)
    rhob_arr = np.asarray(rhob, dtype=float)

    num = (den_cl2 - den_cl1) * (nphi_arr - neu_cl1) - (rhob_arr - den_cl1) * (neu_cl2 - neu_cl1)
    den = (den_cl2 - den_cl1) * (neu_clay - neu_cl1) - (den_clay - den_cl1) * (neu_cl2 - neu_cl1)

    # Handle degenerate configuration (denominator == 0)
    if np.isclose(den, 0.0):
        out = np.zeros_like(nphi_arr)
    else:
        out = num / den

    if clip:
        out = np.clip(out, 0.0, 1.0)

    # Return scalar if inputs were scalar
    if np.isscalar(nphi) and np.isscalar(rhob):
        return float(out)
    return out


def nd_shale_point(
    nphi: np.ndarray,
    rhob: np.ndarray,
    shale: np.ndarray,
    matrix: str = "limestone",
    min_samples: int = 20,
    min_separation: float = 0.10,
) -> Tuple[float, float]:
    """Neutron-density clay point read from a well's own shale samples.

    The point is the median bulk density and the median neutron porosity of
    the samples flagged in ``shale``: the shale point of the crossplot, picked
    in the well's shales as the shale endpoints of the other clay indicators
    are (Crain, *The Log Analysis Handbook*, 1986: shale volume from the
    density-neutron crossplot).
    Used as the clay point of :func:`vcl_from_nd`, it puts VCL_ND on the same
    scale as a gamma-ray clay volume whose shale endpoint comes from the same
    well, so both read about 1 in that well's shales.

    Parameters
    ----------
    nphi : np.ndarray
        Neutron porosity, v/v (limestone units).
    rhob : np.ndarray
        Bulk density, g/cm3.
    shale : np.ndarray of bool
        Samples to pick the point from, normally good-hole shale samples.
    matrix : str
        Matrix whose clean line (:func:`nd_clean_line`) the point must lie
        clear of.
    min_samples : int
        Fewest flagged samples with both logs.
    min_separation : float
        Smallest neutron-porosity excess (v/v) of the point over the clean
        line at the point's density.

    Returns
    -------
    tuple of float
        ``(den_clay, neu_clay)``.

    Raises
    ------
    ValueError
        If fewer than ``min_samples`` flagged samples have both logs, or the
        point lies less than ``min_separation`` above the clean line. Such a
        point cannot be told from shaly sand within the neutron and density
        measurement error, and the three-point formula would divide by a
        near-zero distance.

    Examples
    --------
    >>> shale = np.ones(20, dtype=bool)
    >>> nd_shale_point(np.full(20, 0.40), np.full(20, 2.35), shale, matrix="sandstone")
    (2.35, 0.4)
    """
    nphi_arr = np.asarray(nphi, dtype=float)
    rhob_arr = np.asarray(rhob, dtype=float)
    picked = np.asarray(shale, dtype=bool) & np.isfinite(nphi_arr) & np.isfinite(rhob_arr)
    count = int(np.count_nonzero(picked))
    if count < min_samples:
        raise ValueError(
            f"{count} shale samples have neutron and density; {min_samples} are needed"
        )
    den_clay = float(np.median(rhob_arr[picked]))
    neu_clay = float(np.median(nphi_arr[picked]))
    separation = nd_clay_separation(den_clay, neu_clay, matrix)
    if separation < min_separation:
        raise ValueError(
            f"the shale point ({den_clay:.3f} g/cm3, {neu_clay:.3f} v/v) lies "
            f"{separation:.3f} v/v above the {matrix} clean line; at least "
            f"{min_separation:.2f} is needed"
        )
    return den_clay, neu_clay


# def vcl_from_nd_thomas_steiber(
#     nphi: Union[float, np.ndarray],
#     rhob: Union[float, np.ndarray],
#     nphi_sh: float = 0.30,
#     rhob_sh: float = 2.45,
#     nphi_sd: float = -0.02,
#     rhob_sd: float = 2.65,
#     rhob_fl: float = 1.0
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Thomas-Steiber analysis for clay volume and distribution.

#     Determines clay volume and porosity distribution in
#     laminated shaly sands.

#     Parameters
#     ----------
#     nphi : array-like
#         Neutron porosity
#     rhob : array-like
#         Bulk density
#     nphi_sh : float
#         Shale neutron porosity
#     rhob_sh : float
#         Shale density
#     nphi_sd : float
#         Clean sand neutron response
#     rhob_sd : float
#         Clean sand density
#     rhob_fl : float
#         Fluid density

#     Returns
#     -------
#     tuple
#         (vcl, phi_total, phi_effective)
#     """
#     nphi = np.asarray(nphi)
#     rhob = np.asarray(rhob)

#     # Calculate porosity from density
#     phi_d = (rhob_sd - rhob) / (rhob_sd - rhob_fl)
#     phi_d = np.clip(phi_d, 0, 0.5)

#     # Calculate VCL
#     # Using neutron excess over density porosity
#     vcl = (nphi - phi_d) / (nphi_sh - 0)
#     vcl = np.clip(vcl, 0, 1)

#     # Total porosity
#     phi_total = phi_d

#     # Effective porosity (excluding clay-bound water)
#     phi_clay = nphi_sh * vcl  # Clay porosity contribution
#     phi_effective = phi_total - phi_clay
#     phi_effective = np.clip(phi_effective, 0, phi_total)

#     return vcl, phi_total, phi_effective

