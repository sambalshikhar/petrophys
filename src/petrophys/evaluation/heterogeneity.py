"""Permeability heterogeneity: Lorenz coefficient and Dykstra-Parsons coefficient.

Both measure how unevenly flow capacity is spread through an interval,
from 0 (homogeneous) towards 1 (a few layers carry all the flow).

- :func:`lorenz_coefficient` orders samples by k/phi, accumulates flow
  capacity (k h) against storage capacity (phi h) and measures how far that
  curve bows from the diagonal (Schmalz and Rahme, 1950).
- :func:`dykstra_parsons` fits log-permeability on a probability scale:
  ``V = (k50 - k84.1) / k50`` (Dykstra and Parsons, 1950). It assumes a
  log-normal permeability distribution; check the fitted line.

Use core permeability, or log permeability over one flow unit; mixing a
shale with a reservoir measures the shale. Samples are weighted by their
thickness (sample spacing for a log).

References
----------
Schmalz, J.P. and Rahme, H.D. (1950). The variation of waterflood
performance with variation in permeability profile. *Producers Monthly* 15.

Dykstra, H. and Parsons, R.L. (1950). The prediction of oil recovery by
waterflooding. In *Secondary Recovery of Oil in the United States*, API.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Optional

import numpy as np


@dataclass
class LorenzResult:
    """Lorenz coefficient and the curve behind it."""

    coefficient: float
    #: Cumulative storage capacity (phi h) fraction, 0-1, best layers first.
    storage: np.ndarray
    #: Cumulative flow capacity (k h) fraction, 0-1.
    flow: np.ndarray
    n_samples: int

    def __repr__(self) -> str:
        return f"LorenzResult(coefficient {self.coefficient:.3f}, {self.n_samples} samples)"


@dataclass
class DykstraParsonsResult:
    """Dykstra-Parsons coefficient and its probability-plot fit."""

    coefficient: float
    k50: float
    k84: float
    #: R-squared of log k against the normal quantile (the log-normal check).
    r_squared: float
    n_samples: int


def _clean(perm: Any, other: Optional[Any], thickness: Optional[Any]):
    k = np.asarray(perm, dtype=float).reshape(-1)
    o = None if other is None else np.asarray(other, dtype=float).reshape(-1)
    h = np.ones_like(k) if thickness is None else np.broadcast_to(np.asarray(thickness, dtype=float), k.shape)
    use = np.isfinite(k) & (k > 0) & np.isfinite(h) & (h > 0)
    if o is not None:
        use &= np.isfinite(o) & (o > 0)
    return k[use], (o[use] if o is not None else None), h[use]


def lorenz_coefficient(perm: Any, phi: Any, thickness: Optional[Any] = None) -> LorenzResult:
    """Lorenz coefficient of permeability (mD) and porosity (v/v) samples.

    Parameters
    ----------
    perm, phi : array-like
        Permeability and porosity of each sample (layer or plug).
    thickness : float or array-like, optional
        Thickness each sample represents; equal weights by default.

    Returns
    -------
    LorenzResult
        ``coefficient`` is twice the area between the flow-storage curve and
        the diagonal: 0 homogeneous, near 1 extremely heterogeneous.

    Examples
    --------
    >>> lorenz_coefficient([100, 100, 100], [0.2, 0.2, 0.2]).coefficient
    0.0
    """
    k, p, h = _clean(perm, phi, thickness)
    if k.size < 2:
        raise ValueError("The Lorenz coefficient needs at least 2 samples with k > 0 and phi > 0")
    order = np.argsort(-(k / p), kind="stable")
    flow = np.concatenate([[0.0], np.cumsum((k * h)[order])])
    storage = np.concatenate([[0.0], np.cumsum((p * h)[order])])
    flow /= flow[-1]
    storage /= storage[-1]
    integrate = getattr(np, "trapezoid", None) or np.trapz  # trapz is deprecated in numpy 2
    area = integrate(flow, storage)
    coefficient = float(max(0.0, 2.0 * (area - 0.5)))
    return LorenzResult(coefficient=round(coefficient, 12), storage=storage, flow=flow, n_samples=int(k.size))


def dykstra_parsons(perm: Any, thickness: Optional[Any] = None) -> DykstraParsonsResult:
    """Dykstra-Parsons permeability variation coefficient of permeability samples (mD).

    Samples are sorted from high to low permeability; the share of thickness
    with greater permeability is plotted on a normal-probability scale
    against log k, and a straight line fitted. ``V = (k50 - k84.1) / k50``
    from that line: 0 homogeneous, 1 extremely heterogeneous (0.5-0.9 is
    typical of reservoirs).

    Examples
    --------
    >>> dykstra_parsons([50, 50, 50, 50]).coefficient
    0.0
    """
    k, _, h = _clean(perm, None, thickness)
    if k.size < 3:
        raise ValueError("The Dykstra-Parsons coefficient needs at least 3 samples with k > 0")
    order = np.argsort(-k, kind="stable")
    k, h = k[order], h[order]
    # Plotting position: thickness share above each sample's midpoint.
    share = (np.cumsum(h) - 0.5 * h) / h.sum()
    z = np.array([NormalDist().inv_cdf(float(s)) for s in share])
    log_k = np.log(k)
    if np.ptp(log_k) == 0:
        return DykstraParsonsResult(0.0, float(k[0]), float(k[0]), 1.0, int(k.size))
    slope, intercept = np.polyfit(z, log_k, 1)
    fitted = slope * z + intercept
    total = np.sum((log_k - log_k.mean()) ** 2)
    r_squared = float(1.0 - np.sum((log_k - fitted) ** 2) / total) if total > 0 else 1.0
    k50 = float(np.exp(intercept))
    k84 = float(np.exp(intercept + slope * NormalDist().inv_cdf(0.841)))
    return DykstraParsonsResult(
        coefficient=float((k50 - k84) / k50), k50=k50, k84=k84, r_squared=r_squared, n_samples=int(k.size),
    )


__all__ = ["DykstraParsonsResult", "LorenzResult", "dykstra_parsons", "lorenz_coefficient"]
