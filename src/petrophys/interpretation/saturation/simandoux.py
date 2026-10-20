"""
Simandoux equation for water saturation in shaly sands.

This module provides the Simandoux shaly sand model for formations
with moderate to high clay content and higher salinity (>= 20 kppm).
"""

from typing import Union
import numpy as np
from scipy.optimize import brentq

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.interpretation.saturation._inputs import (
    check_resistivity_parameters,
    nonpositive_resistivity,
)


def simandoux(
    rt: Union[float, np.ndarray],
    rw: float,
    phi: Union[float, np.ndarray],
    vcl: Union[float, np.ndarray],
    rcl: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0
) -> Union[float, np.ndarray]:
    """
    Calculate water saturation using the Simandoux equation.

    1/Rt = (phi^m / (a * Rw)) * Sw^n  +  (Vcl / Rcl) * Sw

    For n=2 an exact quadratic solution is used.
    For n != 2 the equation is solved numerically via Brent's method.

    Parameters
    ----------
    rt : float or np.ndarray
        True formation resistivity (ohm.m)
    rw : float
        Formation water resistivity (ohm.m)
    phi : float or np.ndarray
        Effective porosity (v/v)
    vcl : float or np.ndarray
        Clay volume (v/v)
    rcl : float
        Clay resistivity (ohm.m)
    a : float, optional
        Tortuosity factor (default: 1.0)
    m : float, optional
        Cementation exponent (default: 2.0)
    n : float, optional
        Saturation exponent (default: 2.0)

    Returns
    -------
    float or np.ndarray
        Water saturation (0-1).
        NaN where Rt or an Rw array is at or below zero.

    Raises
    ------
    ValueError
        If a scalar ``rw`` or ``rcl`` is at or below zero.
    """
    check_resistivity_parameters(rw=rw, rcl=rcl)
    scalar = np.isscalar(rt) and np.isscalar(phi) and np.isscalar(vcl)

    rt, phi, vcl = np.broadcast_arrays(
        np.asarray(rt, dtype=float),
        np.asarray(phi, dtype=float),
        np.asarray(vcl, dtype=float),
    )
    rt = np.atleast_1d(rt.astype(float, copy=False))
    phi = np.atleast_1d(phi.astype(float, copy=False))
    vcl = np.atleast_1d(vcl.astype(float, copy=False))

    invalid = nonpositive_resistivity(rt, rw)
    rt  = np.where(rt  <= 0, 1e-10, rt)
    # Infinite Rt (no conductivity reading) is missing, as in archie().
    historical = reproducing_benchmark()
    if not historical:
        rt = np.where(np.isinf(rt), np.nan, rt)
    phi = np.where(phi <= 0, 1e-10, phi)
    rcl = max(rcl, 1e-10)

    A = np.power(phi, m) / (a * rw)   # coefficient on Sw^n
    B = vcl / rcl                      # coefficient on Sw
    C = 1.0 / rt                       # RHS

    if abs(n - 2.0) < 1e-9:
        # Exact quadratic solution for n = 2:
        #   A * Sw^2 + B * Sw - C = 0  =>  Sw = (-B + sqrt(B^2 + 4AC)) / (2A),
        # evaluated as 2C / (B + sqrt(B^2 + 4AC)), which stays accurate as
        # A -> 0 (non-porous shale) and tends to C/B = Rcl / (Vcl * Rt).
        discriminant = np.maximum(B ** 2 + 4 * A * C, 0.0)
        if historical:
            # The benchmark context profiles keep the packages' original calculation.
            sw = (-B + np.sqrt(discriminant)) / (2 * A)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                sw = 2 * C / (B + np.sqrt(discriminant))
    else:
        # General n: solve A*Sw^n + B*Sw - C = 0 numerically per sample
        sw = np.empty_like(rt)
        for i in range(len(rt)):
            a_i, b_i, c_i = float(A[i]), float(B[i]), float(C[i])

            def f(s):
                return a_i * s ** n + b_i * s - c_i

            if not historical and not (
                np.isfinite(a_i) and np.isfinite(b_i) and np.isfinite(c_i)
            ):
                # Missing Rt, porosity or Vcl leaves Sw missing, as for n = 2.
                sw[i] = np.nan
                continue
            try:
                # f(0) = -c_i < 0,  f(1) = a_i + b_i - c_i (usually >= 0)
                if f(1.0) < 0:
                    sw[i] = 1.0
                elif f(1e-8) > 0:
                    sw[i] = 0.0
                else:
                    sw[i] = brentq(f, 1e-8, 1.0, xtol=1e-8)
            except ValueError:
                sw[i] = 1.0

    sw = np.where(invalid, np.nan, sw)
    sw = np.clip(sw, 0, 1)
    return float(sw[0]) if scalar else sw
