"""
Zone flag plotting utilities.
"""

from petrophys.visualization.base import Plot
from petrophys.visualization.style import _styled_plot

from typing import TYPE_CHECKING, Optional, Tuple, List, Dict

if TYPE_CHECKING:
    from petrophys.model.well import Well

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np

from petrophys._internal.analysis_helpers import resolve_well_depth_range, well_depth_array


def _depth_label(well: "Well") -> str:
    unit = getattr(well, "depth_unit", "") or ""
    return f"Depth [{unit}]" if unit else "Depth"


def _cell_fill(depth: np.ndarray, mask: np.ndarray):
    """Shade each flagged sample over its own depth cell (see log viewer)."""
    from petrophys.visualization.log_viewer import _sample_cell_fill
    return _sample_cell_fill(depth, mask)


#: Flag names as the log plot's FLAGS track shows them.
_FLAG_NAMES = {"NET_ROCK": "Clean Rock", "NET_RES": "Res", "NET_PAY": "Pay", "SEALFLAG": "Seal"}


@_styled_plot
def zone_flags_plot(
    well: "Well",
    flag_curves: List[str],
    colors: Optional[Dict[str, str]] = None,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    figsize: Tuple[float, float] = (4, 12)
) -> Plot:
    """
    Plot zone flags as colored columns.

    Parameters
    ----------
    well : Well
        Well object
    flag_curves : list
        List of flag curve names to plot
    colors : dict, optional
        Dictionary mapping curve names to colors. Flags the log plot knows
        (NET_ROCK, NET_RES, NET_PAY, SEALFLAG) default to its colours.
    top, bottom : float, optional
        Depth window, in the depth unit of the data; an end not given is the
        data's own.
    figsize : tuple
        Figure size

    Returns
    -------
    Plot
        ``.fig`` and ``.ax``; ``.save(path)``. Shows itself at the end of a
        notebook cell.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    from petrophys.visualization.log_viewer import FLAG_COLORS

    default_colors = ['#8c510a', '#762a83', '#d73027', '#4d4d4d', '#2166ac', '#e7298a']
    defaults = {
        name: FLAG_COLORS.get(str(name).upper(), default_colors[i % len(default_colors)])
        for i, name in enumerate(flag_curves)
    }
    colors = {**defaults, **(colors or {})}

    fig, ax = plt.subplots(figsize=figsize)

    depth = well_depth_array(well)

    # Set depth range
    logged_top, logged_bottom = resolve_well_depth_range(well)
    top = logged_top if top is None else float(top)
    bottom = logged_bottom if bottom is None else float(bottom)

    n_flags = len(flag_curves)
    if n_flags == 0:
        raise ValueError("At least one flag curve is required")
    width = 1.0 / n_flags

    for i, curve_name in enumerate(flag_curves):
        curve = well.get(curve_name)
        if curve is None:
            continue

        data = curve.data
        x_left = i * width

        # Fill where flag is True/1
        mask = data > 0.5
        if np.any(mask):
            cell_depth, cell_mask = _cell_fill(depth, mask)
            ax.fill_betweenx(
                cell_depth, x_left, x_left + width,
                where=cell_mask,
                color=colors.get(curve_name, 'gray'),
                alpha=0.85,
                linewidth=0.0,
            )
        if i:
            ax.axvline(x_left, color="0.75", linewidth=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom, top)
    # Each column is named above it, so no legend covers the flags.
    ax.set_xticks([(i + 0.5) * width for i in range(n_flags)])
    shown = [_FLAG_NAMES.get(str(name).upper(), name) for name in flag_curves]
    ax.set_xticklabels(shown, rotation=90 if n_flags > 3 else 0, fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(_depth_label(well))

    well_name = well.name or "Unknown Well"
    ax.set_title(f"Zone Flags - {well_name}")

    plt.tight_layout()

    return Plot(fig, ax)


def add_zone_shading(
    ax: "Axes",
    zones: List[Dict],
    alpha: float = 0.15
) -> None:
    """
    Add zone shading to an existing plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    zones : list
        List of zone dictionaries with:
        - top: float
        - bottom: float
        - name: str (optional)
        - color: str (optional)
    alpha : float
        Shading transparency
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Pale colours, behind the curves, so the curves and their fills stay legible.
    default_colors = ["#fdd49e", "#c6dbef", "#d9f0d3", "#fbb4ae", "#decbe4"]

    for i, zone in enumerate(zones):
        top = zone.get("top", 0)
        bottom = zone.get("bottom", 0)
        # "label" is accepted as an alias of "name".
        name = zone.get("name", zone.get("label", f"Zone {i+1}"))
        color = zone.get("color", default_colors[i % len(default_colors)])

        # Add horizontal span
        ax.axhspan(top, bottom, alpha=alpha, color=color, label=name, zorder=0, linewidth=0)

        # The label at the track's right edge, at the top of the zone, clear of the curves.
        ax.text(
            0.98, min(top, bottom), name,
            transform=ax.get_yaxis_transform(),
            fontsize=8, ha="right", va="top" if ax.yaxis_inverted() else "bottom",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
            zorder=8,
        )
