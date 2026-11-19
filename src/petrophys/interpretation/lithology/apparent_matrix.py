"""Apparent matrix parameters: RHOmaa, DTmaa and Umaa.

Removing the pore fluid from a log reading leaves the reading of the rock
matrix, which identifies minerals independently of porosity:

- ``RHOmaa = (RHOB - phi * rho_f) / (1 - phi)``
- ``DTmaa = (DT - phi * dt_f) / (1 - phi)``
- ``Umaa = (PEF * rho_e - phi * U_f) / (1 - phi)`` with the electron density
  ``rho_e = (RHOB + 0.1883) / 1.0704``

Supply independently determined total porosity, or the linear clean-model
estimate from :func:`nd_crossplot_porosity`. The limestone-reference mean
in :func:`apparent_total_porosity` is approximate for other minerals.
Quartz reads about 2.65 g/cc,
55.5 us/ft and 4.8 b/cc; calcite 2.71, 47.6, 13.8; dolomite 2.87, 43.5, 9.0
(Schlumberger chartbook). These transforms underlie the MID (DTmaa-RHOmaa)
and RHOmaa-Umaa plots.

References
----------
Clavier, C. and Rust, D.H. (1976). MID plot: a new lithology technique.
*The Log Analyst* 17(6).
Schlumberger (2009). *Log Interpretation Charts*, CP-21.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from petrophys._internal.validation import broadcast_inputs, numeric_array, warn_invalid

ArrayLike = Union[float, np.ndarray]

#: Fresh-water end points: density (g/cc), slowness (us/ft), volumetric PE (b/cc).
WATER_DENSITY = 1.0
WATER_SLOWNESS = 189.0
WATER_U = 0.398


def _out(values: np.ndarray) -> ArrayLike:
    return float(values) if np.ndim(values) == 0 else values


def _remove_fluid(reading: ArrayLike, phi: ArrayLike, fluid: float) -> ArrayLike:
    reading_arr = np.asarray(reading, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    valid = np.isfinite(reading_arr) & np.isfinite(phi_arr) & (phi_arr >= 0) & (phi_arr < 0.95)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(valid, (reading_arr - phi_arr * fluid) / (1.0 - phi_arr), np.nan)
    return _out(values)


def apparent_total_porosity(
    rhob: ArrayLike,
    nphi: ArrayLike,
    rho_matrix: float = 2.71,
    rho_fluid: float = WATER_DENSITY,
) -> ArrayLike:
    """Neutron-density crossplot porosity, the mean of NPHI and the density porosity.

    Both on the limestone reference (``rho_matrix`` 2.71 g/cc) by default, as
    the MID and RHOmaa-Umaa methods use them; limited to 0-0.6.
    """
    rhob_arr = np.asarray(rhob, dtype=float)
    nphi_arr = np.asarray(nphi, dtype=float)
    phid = (rho_matrix - rhob_arr) / (rho_matrix - rho_fluid)
    return _out(np.clip(0.5 * (phid + nphi_arr), 0.0, 0.6))


def apparent_matrix_density(rhob: ArrayLike, phi: ArrayLike, rho_fluid: float = WATER_DENSITY) -> ArrayLike:
    """RHOmaa (g/cc): bulk density with the pore fluid removed.

    >>> round(apparent_matrix_density(2.32, 0.2), 3)
    2.65
    """
    return _remove_fluid(rhob, phi, rho_fluid)


def nd_crossplot_porosity(rhob, nphi, *, rho_fluid=1.0, nphi_fluid=1.0):
    """Total porosity from linear clean-mineral neutron-density triangles.

    Uses quartz/calcite/water and calcite/dolomite/water volume balance,
    with limestone-referenced neutron endpoints -0.02, 0, 0.02 and matrix
    densities 2.65, 2.71, 2.87 g/cm3. This is a linear clean binary-mineral
    model, not a vendor neutron-tool calibration. Gas, clay and samples
    outside these triangles need another porosity estimate. Such samples
    return NaN and warn. Independently determined total porosity is preferred
    for matrix identification in complex lithologies.
    """
    rhob, nphi = broadcast_inputs(rhob=rhob, nphi=nphi)
    rho_fluid = numeric_array(rho_fluid, "rho_fluid")
    nphi_fluid = numeric_array(nphi_fluid, "nphi_fluid")
    if rho_fluid.ndim != 0 or not np.isfinite(rho_fluid) or rho_fluid <= 0:
        raise ValueError("rho_fluid must be a finite positive density (g/cm3)")
    if nphi_fluid.ndim != 0 or not np.isfinite(nphi_fluid):
        raise ValueError("nphi_fluid must be a finite neutron endpoint (v/v)")
    observations = np.stack([nphi - nphi_fluid, rhob - rho_fluid]).reshape(2, -1)
    out = np.full(rhob.size, np.nan)
    minerals = [(-.02, 2.65), (0., 2.71), (.02, 2.87)]
    for first, second in zip(minerals[:-1], minerals[1:]):
        matrix = np.array([[first[0]-nphi_fluid, second[0]-nphi_fluid],
                           [first[1]-rho_fluid, second[1]-rho_fluid]])
        if abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("Fluid endpoints must not be collinear with the clean mineral endpoints")
        weights = np.linalg.solve(matrix, observations)
        phi = 1.0 - weights.sum(axis=0)
        valid = (np.isfinite(observations).all(axis=0) & (weights >= -1e-9).all(axis=0)
                 & (phi >= -1e-9) & (phi < .95) & ~np.isfinite(out))
        out[valid] = np.maximum(phi[valid], 0.)
    out = out.reshape(rhob.shape)
    warn_invalid(
        np.isfinite(rhob) & np.isfinite(nphi) & ~np.isfinite(out),
        "Neutron-density crossplot porosity", "fall outside the clean binary-mineral model",
        "These porosities are NaN. Supply independently determined total porosity for complex lithology, gas or clay.",
    )
    return _out(out)


def apparent_matrix_slowness(dt: ArrayLike, phi: ArrayLike, dt_fluid: float = WATER_SLOWNESS) -> ArrayLike:
    """DTmaa (us/ft): compressional slowness with the pore fluid removed (time-average mixing)."""
    return _remove_fluid(dt, phi, dt_fluid)


def electron_density(rhob: ArrayLike) -> ArrayLike:
    """Electron density index ``rho_e = (RHOB + 0.1883) / 1.0704`` (g/cc)."""
    return _out((np.asarray(rhob, dtype=float) + 0.1883) / 1.0704)


def volumetric_photoelectric(pef: ArrayLike, rhob: ArrayLike) -> ArrayLike:
    """Volumetric photoelectric absorption ``U = PEF * rho_e`` (b/cc)."""
    return _out(np.asarray(pef, dtype=float) * np.asarray(electron_density(rhob), dtype=float))


def apparent_matrix_volumetric_pe(
    pef: ArrayLike,
    rhob: ArrayLike,
    phi: ArrayLike,
    u_fluid: float = WATER_U,
) -> ArrayLike:
    """Umaa (b/cc): volumetric photoelectric absorption with the pore fluid removed."""
    return _remove_fluid(volumetric_photoelectric(pef, rhob), phi, u_fluid)


__all__ = [
    "WATER_DENSITY",
    "WATER_SLOWNESS",
    "WATER_U",
    "apparent_matrix_density",
    "apparent_matrix_slowness",
    "apparent_matrix_volumetric_pe",
    "apparent_total_porosity",
    "nd_crossplot_porosity",
    "electron_density",
    "volumetric_photoelectric",
]
