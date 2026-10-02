"""XRF geochemistry panel: element concentrations + mineral proxies vs depth.

Element tracks are grouped by mineralogical affinity (clastic / clay / carbonate /
redox), with a final mineral-proxy track — the layout an elemental-spectroscopy
display uses in Techlog / Geolog.
"""

from __future__ import annotations

from petrophys.visualization.style import _styled_plot

from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False


# (track title, [(element, color), ...])
_ELEMENT_GROUPS = [
    ("Clastic", [("SI", "green")]),
    ("Clay / detrital", [("AL", "firebrick"), ("K", "darkorange"), ("TI", "purple")]),
    ("Carbonate", [("CA", "royalblue"), ("MG", "teal")]),
    ("Redox / diagenetic", [("FE", "saddlebrown"), ("MN", "magenta"), ("S", "goldenrod")]),
]

_PROXY_COLORS = [
    ("quartz_index", "green"), ("carbonate_index", "royalblue"),
    ("clay_index", "firebrick"), ("pyrite_index", "goldenrod"),
    ("dolomite_index", "teal"),
]


@_styled_plot
def create_xrf_panel(
    xrf,
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (14, 11),
):
    """Build an XRF geochemistry panel from an :class:`XRFData` object."""
    depth = np.asarray(xrf.depth, dtype=float)

    def _sel(arr):
        arr = np.asarray(arr, dtype=float)
        if depth_range is not None:
            lo, hi = depth_range
            return np.where((depth >= lo) & (depth <= hi), arr, np.nan)
        return arr

    proxies = {}
    try:
        proxies = xrf.mineral_proxies()
    except Exception:
        proxies = {}

    n_tracks = len(_ELEMENT_GROUPS) + (1 if proxies else 0)
    fig, axes = plt.subplots(1, n_tracks, sharey=True, figsize=figsize)
    if n_tracks == 1:
        axes = [axes]

    for ax, (title, elems) in zip(axes, _ELEMENT_GROUPS):
        plotted = False
        plotted_elements = []
        for el, color in elems:
            v = xrf.get(el)
            if v is None:
                continue
            ax.plot(_sel(v), depth, color=color, lw=0.6, label=el)
            plotted = True
            plotted_elements.append(el)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(_concentration_label(xrf, plotted_elements), fontsize=8)
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=7, loc="upper right")

    if proxies:
        axm = axes[len(_ELEMENT_GROUPS)]
        for key, color in _PROXY_COLORS:
            if key in proxies:
                axm.plot(_sel(proxies[key]), depth, color=color, lw=0.7,
                         label=key.replace("_index", ""))
        axm.set_title("Mineral proxies", fontsize=9)
        axm.set_xlabel("index (0-1)", fontsize=8)
        axm.set_xlim(0, 1)
        axm.grid(alpha=0.3)
        axm.legend(fontsize=7, loc="upper right")

    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth")
    name = getattr(getattr(xrf, "header", None), "well_name", None)
    fig.suptitle(f"XRF geochemistry{(' — ' + name) if name else ''}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _concentration_label(xrf, elements) -> str:
    units = getattr(xrf, "units", {}) or {}
    resolved = {
        str(value).strip()
        for element in elements
        for key, value in units.items()
        if str(key).casefold() == str(element).casefold() and str(value).strip()
    }
    if len(resolved) == 1:
        return f"Concentration [{resolved.pop()}]"
    return "Concentration"
