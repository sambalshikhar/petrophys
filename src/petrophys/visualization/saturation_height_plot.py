"""Saturation-height diagnostic plot.

Two panels:
  (left)  the fitted J-function (Sw = a·J^b, or Sw = Swirr + (1 - Swirr)·a·J^b)
          over the (J, Sw) data;
  (right) Sw vs height above the free-water level for a few (k, phi) rock classes.
"""

from __future__ import annotations

from petrophys.visualization.style import _styled_plot

from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

from petrophys.interpretation.saturation.height.saturation_height_module import JFunctionModel


@_styled_plot
def create_saturation_height_plot(
    model: JFunctionModel,
    j: Optional[Sequence[float]] = None,
    sw: Optional[Sequence[float]] = None,
    rock_classes: Optional[Sequence[Tuple[float, float]]] = None,
    max_height_ft: float = 300.0,
    figsize: Tuple[float, float] = (12.0, 5.2),
    title: str = "Saturation-Height Model",
    data_label: str = "data",
):
    """Render the J-function fit and the resulting Sw-vs-height type curves.

    ``rock_classes`` is a list of (perm_mD, phi) pairs to draw height curves for;
    defaults to a poor / moderate / good rock triplet. ``data_label`` names the
    source of the (J, Sw) points in the legend (e.g. ``"core data"`` or a
    log-derived Sw curve); the points are labelled ``"data"`` without it.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for visualization")
    if rock_classes is None:
        rock_classes = [(1.0, 0.10), (10.0, 0.15), (100.0, 0.22)]

    fig, (axj, axh) = plt.subplots(1, 2, figsize=figsize)

    # --- Panel A: Sw vs J ---
    if j is not None and sw is not None and np.any(
        np.isfinite(np.asarray(j, dtype=float)) & np.isfinite(np.asarray(sw, dtype=float))
        & (np.asarray(j, dtype=float) > 0)
    ):
        j = np.asarray(j, dtype=float)
        sw = np.asarray(sw, dtype=float)
        ok = np.isfinite(j) & np.isfinite(sw) & (j > 0)
        axj.scatter(j[ok], sw[ok], s=12, alpha=0.55, color="0.35", label=data_label)
        jline = np.geomspace(max(np.nanmin(j[ok]), 1e-3), np.nanmax(j[ok]), 100)
    else:
        jline = np.geomspace(0.05, 50.0, 100)
    if model.form == "power_swir":
        fit_label = (f"Sw = {model.swir:.3g} + {1.0 - model.swir:.3g}·"
                     f"{model.a:.3g}·J^{model.b:.3g}")
    else:
        fit_label = f"Sw = {model.a:.3g}·J^{model.b:.3g}"
    axj.plot(jline, model.sw_from_j(jline), color="crimson", lw=2.0, label=fit_label)
    axj.set_xscale("log")
    axj.set_xlabel("Leverett J  (dimensionless)")
    axj.set_ylabel("Sw [frac]")
    axj.set_ylim(0, 1.02)
    r2 = f"  (R²={model.r2:.3f})" if model.r2 is not None else ""
    axj.set_title("J-function fit" + r2)
    axj.grid(True, which="both", color="0.92", lw=0.5)
    axj.legend(fontsize=8)

    # --- Panel B: Sw vs height above FWL ---
    h = np.linspace(0.0, max_height_ft, 200)
    cmap = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(rock_classes)))
    for (k, phi), color in zip(rock_classes, cmap):
        sw_h = model.sw_from_height(h, np.full_like(h, k), np.full_like(h, phi))
        axh.plot(sw_h, h, color=color, lw=1.8, label=f"k={k:g} mD, φ={phi:.2f}")
    axh.set_xlabel("Sw [frac]")
    axh.set_ylabel("Height above FWL [ft]")
    axh.set_xlim(0, 1.02)
    axh.set_title("Saturation-height type curves")
    axh.grid(True, color="0.92", lw=0.5)
    axh.legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
