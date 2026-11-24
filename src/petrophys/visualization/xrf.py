"""XRF geochemistry panel: element concentrations + mineral proxies vs depth.

Element tracks are grouped by mineralogical affinity (clastic / clay / carbonate /
redox), with a final mineral-proxy track — the layout an elemental-spectroscopy
display uses in Techlog / Geolog.
"""

from __future__ import annotations

from petrophys.visualization.style import _styled_plot

from typing import Optional, Tuple

import numpy as np
import warnings
from petrophys.exceptions import DataQualityWarning
from petrophys.visualization._data import depth_limits, depth_window
from petrophys.visualization.base import Plot
from petrophys._internal.validation import numeric_array

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
def xrf_panel(
    xrf,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 11),
    *, concentration_unit: str = "%",
):
    """Plot concentrations in one mass unit (% by default, or ppm).

    A display copy is converted before curves and proxies are calculated;
    source measurements and their units are retained on the input object.
    Missing units warn and assume percent; unrecognized units are rejected.
    """
    from petrophys.interpretation.geochemistry.xrf import XRFData

    factors = {"%": .01, "percent": .01, "pct": .01, "wt%": .01, "wt.%": .01,
               "wt_pct": .01, "wtpct": .01, "ppm": 1e-6, "mg/kg": 1e-6, "ppb": 1e-9,
               "fraction": 1., "frac": 1., "g/g": 1., "w/w": 1., "wt_frac": 1.,
               "wtfrac": 1., "dec": 1., "decimal": 1.}
    if concentration_unit not in ("%", "ppm"):
        raise ValueError("concentration_unit must be '%' or 'ppm'")
    elements = {}
    units = getattr(xrf, "units", {}) or {}
    depth = numeric_array(xrf.depth, "XRF depth")
    if depth.ndim != 1:
        raise ValueError("XRF depth must be one-dimensional")
    for name, values in xrf.elements.items():
        unit = next((str(v).strip().lower() for k, v in units.items() if str(k).casefold() == str(name).casefold()), "")
        if not unit:
            warnings.warn(f"XRF {name}: missing concentration unit; assuming % for display and proxies.",
                          DataQualityWarning, stacklevel=2)
            unit = "%"
        if unit not in factors:
            raise ValueError(f"XRF {name}: unsupported concentration unit {unit!r}; declare a mass concentration such as % or ppm")
        values = numeric_array(values, f"XRF {name}")
        if values.shape != depth.shape:
            raise ValueError(f"XRF {name} values must match the depth array")
        elements[name] = values * (factors[unit] / factors[concentration_unit])
    xrf = XRFData(depth=depth, elements=elements,
                  units={name: concentration_unit for name in elements}, metadata=getattr(xrf, "metadata", {}))
    depth = np.asarray(xrf.depth, dtype=float)
    window = depth_window(depth, depth_limits(depth, top, bottom))

    def _sel(arr):
        arr = np.asarray(arr, dtype=float)
        return np.where(window, arr, np.nan)

    proxies = xrf.mineral_proxies()

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
    name = (xrf.metadata or {}).get("well")
    fig.suptitle(f"XRF geochemistry{(' - ' + name) if name else ''}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return Plot(fig)


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
