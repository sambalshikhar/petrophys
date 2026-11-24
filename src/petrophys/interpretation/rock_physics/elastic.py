"""
Elastic rock-property calculations from density and sonic logs.

References
----------
Mavko, G., Mukerji, T. and Dvorkin, J. (2009). *The Rock Physics Handbook*,
2nd ed. Cambridge University Press.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


US_PER_FOOT_TO_M_PER_SECOND = 304800.0


@dataclass(frozen=True)
class ElasticProperties:
    """Elastic properties calculated from RHOB, DTC, and DTS.

    Each field is an array, or a float when every input was a scalar.
    """

    vp_m_s: np.ndarray
    vs_m_s: np.ndarray
    acoustic_impedance: np.ndarray
    acoustic_impedance_mrayl: np.ndarray
    poisson_ratio: np.ndarray
    shear_mod_gpa: np.ndarray
    bulk_mod_gpa: np.ndarray
    young_mod_gpa: np.ndarray


def velocity_from_slowness_us_per_ft(slowness_us_per_ft: np.ndarray | float) -> np.ndarray:
    """Convert sonic slowness in us/ft to velocity in m/s."""

    slowness = np.asarray(slowness_us_per_ft, dtype=float)
    velocity = np.full(slowness.shape, np.nan, dtype=float)
    valid = np.isfinite(slowness) & (slowness > 0.0)
    velocity[valid] = US_PER_FOOT_TO_M_PER_SECOND / slowness[valid]
    return velocity


def elastic_properties(
    rhob_gcc: np.ndarray | float,
    dtc_us_per_ft: np.ndarray | float,
    dts_us_per_ft: np.ndarray | float | None = None,
) -> ElasticProperties:
    """
    Calculate dynamic elastic properties from density and sonic slowness.

    RHOB is expected in g/cc, DTC/DTS in us/ft. No shear-sonic fallback is
    applied; Vs-dependent properties remain NaN when DTS is unavailable.
    Inputs broadcast against each other; scalar inputs return scalar
    properties.

    DTC outside 20-300 us/ft and DTS outside 20-600 us/ft are treated as
    invalid (NaN velocity).  Where Vp/Vs < sqrt(2) Poisson's ratio would be
    negative, which logged rocks practically never show (it points to a bad
    DTC or DTS pick), so Poisson's ratio, the bulk modulus and Young's modulus
    are NaN there; the shear modulus, from Vs alone, is kept. Where Vp <= Vs
    every Vs-dependent property is NaN. Benchmark context profiles clip
    Poisson's ratio to [0, 0.49] and the moduli to a minimum of 0.
    """
    from petrophys._internal.benchmark import reproducing_benchmark

    historical = reproducing_benchmark()

    rhob = np.asarray(rhob_gcc, dtype=float)
    dtc = np.asarray(dtc_us_per_ft, dtype=float)
    scalar_input = rhob.ndim == 0 and dtc.ndim == 0 and np.ndim(dts_us_per_ft) == 0
    if dts_us_per_ft is None:
        rhob, dtc = np.broadcast_arrays(np.atleast_1d(rhob), np.atleast_1d(dtc))
        dts = None
    else:
        rhob, dtc, dts = np.broadcast_arrays(
            np.atleast_1d(rhob),
            np.atleast_1d(dtc),
            np.atleast_1d(np.asarray(dts_us_per_ft, dtype=float)),
        )

    vp = np.full(rhob.shape, np.nan, dtype=float)
    valid_vp = np.isfinite(dtc) & (dtc >= 20.0) & (dtc <= 300.0)
    vp[valid_vp] = US_PER_FOOT_TO_M_PER_SECOND / dtc[valid_vp]

    vs = np.full(rhob.shape, np.nan, dtype=float)
    if dts is not None:
        valid_vs = np.isfinite(dts) & (dts >= 20.0) & (dts <= 600.0)
        vs[valid_vs] = US_PER_FOOT_TO_M_PER_SECOND / dts[valid_vs]

    acoustic_impedance = np.full(rhob.shape, np.nan, dtype=float)
    acoustic_impedance_mrayl = np.full(rhob.shape, np.nan, dtype=float)
    valid_ai = np.isfinite(rhob) & (rhob > 0.0) & np.isfinite(vp)
    acoustic_impedance[valid_ai] = rhob[valid_ai] * 1000.0 * vp[valid_ai]
    acoustic_impedance_mrayl[valid_ai] = rhob[valid_ai] * (vp[valid_ai] / 1000.0)

    poisson_ratio = np.full(rhob.shape, np.nan, dtype=float)
    shear_mod_gpa = np.full(rhob.shape, np.nan, dtype=float)
    bulk_mod_gpa = np.full(rhob.shape, np.nan, dtype=float)
    young_mod_gpa = np.full(rhob.shape, np.nan, dtype=float)

    valid_elastic = (
        np.isfinite(rhob)
        & (rhob > 0.0)
        & np.isfinite(vp)
        & np.isfinite(vs)
        & (vp > vs)
        & (vs > 0.0)
    )
    if np.any(valid_elastic):
        rho_si = rhob[valid_elastic] * 1000.0
        vp_v = vp[valid_elastic]
        vs_v = vs[valid_elastic]
        vp2 = vp_v**2
        vs2 = vs_v**2
        denom = vp2 - vs2
        valid_denom = denom > 0.0

        idx = np.flatnonzero(valid_elastic)[valid_denom]
        rho_v = rho_si[valid_denom]
        vp2_v = vp2[valid_denom]
        vs2_v = vs2[valid_denom]
        denom_v = denom[valid_denom]

        nu = (vp2_v - (2.0 * vs2_v)) / (2.0 * denom_v)
        bulk = rho_v * (vp2_v - ((4.0 / 3.0) * vs2_v)) / 1e9
        young = rho_v * vs2_v * ((3.0 * vp2_v) - (4.0 * vs2_v)) / denom_v / 1e9
        if historical:
            nu = np.clip(nu, 0.0, 0.49)
            bulk = np.clip(bulk, 0.0, None)
            young = np.clip(young, 0.0, None)
        else:
            possible = nu >= 0.0
            nu = np.where(possible, nu, np.nan)
            bulk = np.where(possible, bulk, np.nan)
            young = np.where(possible, young, np.nan)
        poisson_ratio[idx] = nu
        shear_mod_gpa[idx] = rho_v * vs2_v / 1e9
        bulk_mod_gpa[idx] = bulk
        young_mod_gpa[idx] = young

    properties = dict(
        vp_m_s=vp,
        vs_m_s=vs,
        acoustic_impedance=acoustic_impedance,
        acoustic_impedance_mrayl=acoustic_impedance_mrayl,
        poisson_ratio=poisson_ratio,
        shear_mod_gpa=shear_mod_gpa,
        bulk_mod_gpa=bulk_mod_gpa,
        young_mod_gpa=young_mod_gpa,
    )
    if scalar_input:
        properties = {name: float(values[0]) for name, values in properties.items()}
    return ElasticProperties(**properties)
