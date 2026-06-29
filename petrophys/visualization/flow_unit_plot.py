"""Hydraulic flow-unit diagnostic plot.

The classic Amaefule plot: Reservoir Quality Index (RQI) vs normalized porosity
(phi_z) on log-log axes. Each hydraulic flow unit (HFU) plots along a unit-slope
line whose intercept is the unit's mean FZI, so well-separated parallel trends
confirm the unit partitioning.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

from petrophys.reservoir.flow_units import calculate_flow_units, FlowUnitResult


def create_flow_unit_plot(
    result: Optional[FlowUnitResult] = None,
    phi_e=None,
    perm=None,
    figsize: Tuple[float, float] = (7.5, 6.5),
    cmap: str = "turbo",
    title: str = "Hydraulic Flow Units (RQI vs φz)",
    **kwargs,
):
    """Render the RQI vs phi_z flow-unit crossplot.

    Provide either a pre-computed ``result`` or ``phi_e`` + ``perm`` arrays
    (``**kwargs`` are forwarded to :func:`calculate_flow_units`).
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for visualization")
    if result is None:
        if phi_e is None or perm is None:
            raise ValueError("provide either result or both phi_e and perm")
        result = calculate_flow_units(perm, phi_e, **kwargs)

    rqi = np.asarray(result.rqi, dtype=float)
    phi_z = np.asarray(result.phi_z, dtype=float)
    hfu = np.asarray(result.hfu)
    valid = np.isfinite(rqi) & np.isfinite(phi_z) & (rqi > 0) & (phi_z > 0) & (hfu > 0)

    fig, ax = plt.subplots(figsize=figsize)
    n = max(int(result.n_units), 1)
    colors = plt.get_cmap(cmap)(np.linspace(0.06, 0.94, n))

    pz_line = np.array([np.nanmin(phi_z[valid]) * 0.8, np.nanmax(phi_z[valid]) * 1.2]) \
        if np.any(valid) else np.array([0.05, 1.0])

    for unit in range(1, n + 1):
        sel = valid & (hfu == unit)
        color = colors[unit - 1]
        fzi_mean = result.fzi_unit_mean[unit - 1] if unit - 1 < result.fzi_unit_mean.size else np.nan
        label = f"HFU {unit} (FZI≈{fzi_mean:.2f})" if np.isfinite(fzi_mean) else f"HFU {unit}"
        if np.any(sel):
            ax.scatter(phi_z[sel], rqi[sel], s=10, alpha=0.6, color=color, label=label)
        # unit-slope reference line: RQI = FZI_mean * phi_z
        if np.isfinite(fzi_mean):
            ax.plot(pz_line, fzi_mean * pz_line, color=color, lw=1.3, ls="--", alpha=0.9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Normalized porosity  φz = φe/(1-φe)")
    ax.set_ylabel("RQI = 0.0314·√(k/φe)  [microns]")
    ax.set_title(title)
    ax.grid(True, which="both", color="0.9", lw=0.5)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig
