"""
Sonic porosity calculation.

Sonic-derived porosity from the Wyllie time-average equation or the
Raymer-Hunt-Gardner transform.

References
----------
Wyllie, M.R.J., Gregory, A.R. and Gardner, L.W. (1956). Elastic wave
velocities in heterogeneous and porous media. *Geophysics* 21(1), 41-70.
Raymer, L.L., Hunt, E.R. and Gardner, J.S. (1980). An improved sonic transit
time-to-porosity transform. SPWLA 21st Annual Logging Symposium.
"""

from typing import Union
import numpy as np


def wyllie_time_average(
    dtc: Union[float, np.ndarray],
    dtc_matrix: float = 55.5,
    dtc_fluid: float = 189.0
) -> Union[float, np.ndarray]:
    """
    Calculate porosity using Wyllie time-average equation.

    φS = (Δt - Δtma) / (Δtfl - Δtma)

    Parameters
    ----------
    dtc : float or np.ndarray
        Compressional slowness (μs/ft)
    dtc_matrix : float, optional
        Matrix slowness (default: 55.5 for sandstone)
    dtc_fluid : float, optional
        Fluid slowness (default: 189.0 for water)

    Returns
    -------
    float or np.ndarray
        Sonic porosity (v/v). Missing/nonfinite inputs and zero
        fluid-matrix slowness contrast return NaN.
    """
    dtc_arr = np.asarray(dtc, dtype=float)
    dtc_matrix_arr = np.asarray(dtc_matrix, dtype=float)
    dtc_fluid_arr = np.asarray(dtc_fluid, dtype=float)
    denominator = dtc_fluid_arr - dtc_matrix_arr

    with np.errstate(divide="ignore", invalid="ignore"):
        phis = np.where(
            (denominator != 0) & np.isfinite(dtc_arr)
            & np.isfinite(dtc_matrix_arr) & np.isfinite(dtc_fluid_arr),
            (dtc_arr - dtc_matrix_arr) / denominator,
            np.nan,
        )

    phis = np.clip(phis, 0, 0.5)
    if np.ndim(phis) == 0:
        return float(phis)
    return phis


def raymer_hunt_gardner(
    dtc: Union[float, np.ndarray],
    dtc_matrix: float = 55.5,
    dtc_fluid: float = 189.0,
    approximate: bool = False,
    constant: float = 0.625,
) -> Union[float, np.ndarray]:
    """
    Sonic porosity from the Raymer-Hunt-Gardner (RHG) transform.

    The transform mixes velocities rather than transit times:
    ``1/dt = (1 - phi)**2 / dt_ma + phi / dt_f``, solved here exactly for
    porosity (the root between 0 and 1 of the quadratic). Raymer et al. give
    it for porosity up to about 37 %; above that it should be blended
    towards a suspension (Wood) relation, so values above 0.37 are less
    reliable. It needs no compaction correction, unlike Wyllie in
    unconsolidated sands.

    With ``approximate`` the field form ``phi = C * (dt - dt_ma) / dt`` is
    used, with ``C`` (``constant``) 0.625 for water-filled rock (0.6-0.7 in
    the literature; lower for gas).

    Parameters
    ----------
    dtc : float or np.ndarray
        Compressional slowness (us/ft).
    dtc_matrix, dtc_fluid : float
        Matrix and fluid slowness (us/ft); 55.5 and 189 for a water-filled
        sandstone.
    approximate : bool
        Use the linear field approximation.
    constant : float
        ``C`` of the approximation.

    Returns
    -------
    float or np.ndarray
        Porosity (v/v), limited to 0-0.5 like :func:`wyllie_time_average`;
        NaN for missing input.

    Examples
    --------
    >>> round(raymer_hunt_gardner(80.0), 4)
    0.2038
    """
    dt = np.asarray(dtc, dtype=float)
    valid = np.isfinite(dt) & (dt > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if approximate:
            phi = np.where(valid, constant * (dt - dtc_matrix) / dt, np.nan)
        else:
            a = 1.0 / dtc_matrix
            b = 1.0 / dtc_fluid
            y = np.where(valid, 1.0 / dt, np.nan)
            # a*phi^2 + (b - 2a)*phi + (a - y) = 0; the smaller root is the one in 0-1.
            disc = (b - 2.0 * a) ** 2 - 4.0 * a * (a - y)
            phi = np.where(valid & (disc >= 0), ((2.0 * a - b) - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a), np.nan)
    phi = np.clip(phi, 0.0, 0.5)
    if np.ndim(phi) == 0:
        return float(phi)
    return phi


def sonic_porosity(
    dtc: Union[float, np.ndarray],
    matrix: str = "sandstone",
    dtc_fluid: float = 189.0,
    compaction_factor: float = 1.0,
    transform: str = "wyllie",
) -> Union[float, np.ndarray]:
    """
    Calculate sonic porosity with matrix selection.

    Parameters
    ----------
    dtc : float or np.ndarray
        Compressional slowness
    matrix : str
        Matrix type
    dtc_fluid : float
        Fluid slowness
    compaction_factor : float
        Compaction correction (Cp), Wyllie only
    transform : str
        ``"wyllie"`` (time average, the default) or ``"raymer"``
        (:func:`raymer_hunt_gardner`, exact form).

    Returns
    -------
    float or np.ndarray
        Sonic porosity, limited to 0-0.5: (Wyllie porosity) / Cp, or the RHG porosity
    """
    if transform not in {"wyllie", "raymer"}:
        raise ValueError("transform must be 'wyllie' or 'raymer'")
    matrix_slowness = {
        "sandstone": 55.5,
        "limestone": 47.6,
        "dolomite": 43.5,
        "shale": 100.0,
    }

    dtc_matrix = matrix_slowness.get(matrix.lower())
    if dtc_matrix is None:
        # Other minerals and aliases (calcite, quartz, anhydrite, ...) from
        # the matrix table; an unknown name is an error, not sandstone.
        from petrophys.model.constants import MatrixProperties
        try:
            dtc_matrix = MatrixProperties.get(matrix).slowness
        except KeyError:
            raise ValueError(f"Unknown matrix {matrix!r} for sonic porosity") from None
    if transform == "raymer":
        if compaction_factor != 1.0:
            raise ValueError("The Raymer-Hunt-Gardner transform takes no compaction factor")
        return raymer_hunt_gardner(dtc, dtc_matrix, dtc_fluid)
    # The compaction factor Cp divides the time-average porosity (Hilchie,
    # 1978). Scaling the fluid-matrix slowness contrast by Cp does that before
    # the 0-0.5 limits, so uncompacted rock is not capped at 0.5 / Cp.
    phis = wyllie_time_average(
        dtc, dtc_matrix, dtc_matrix + compaction_factor * (dtc_fluid - dtc_matrix)
    )

    return phis


def secondary_porosity(
    phis: Union[float, np.ndarray],
    phit: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate secondary porosity.

    Secondary porosity = Total porosity - Sonic porosity

    Parameters
    ----------
    phis : float or np.ndarray
        Sonic porosity
    phit : float or np.ndarray
        Total porosity (from N-D)

    Returns
    -------
    float or np.ndarray
        Secondary porosity
    """
    phi_sec = phit - phis
    return np.maximum(phi_sec, 0)
