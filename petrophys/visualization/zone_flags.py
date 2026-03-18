"""
Zone flag plotting utilities.
"""

from typing import Optional, Tuple, List, Dict

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np


def plot_zone_flags(
    well: "Well",
    flag_curves: List[str],
    colors: Optional[Dict[str, str]] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (4, 12)
) -> Tuple["Figure", "Axes"]:
    """
    Plot zone flags as colored columns.

    Parameters
    ----------
    well : Well
        Well object
    flag_curves : list
        List of flag curve names to plot
    colors : dict, optional
        Dictionary mapping curve names to colors
    depth_range : tuple, optional
        (top, bottom) depth range
    figsize : tuple
        Figure size

    Returns
    -------
    tuple
        (Figure, Axes)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if colors is None:
        default_colors = ['green', 'yellow', 'orange', 'red', 'blue', 'purple']
        colors = {name: default_colors[i % len(default_colors)]
                 for i, name in enumerate(flag_curves)}

    fig, ax = plt.subplots(figsize=figsize)

    depth = well.depth

    # Set depth range
    if depth_range:
        top, bottom = depth_range
    else:
        top = well.start_depth or np.nanmin(depth)
        bottom = well.stop_depth or np.nanmax(depth)

    n_flags = len(flag_curves)
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
            ax.fill_betweenx(
                depth, x_left, x_left + width,
                where=mask,
                color=colors.get(curve_name, 'gray'),
                alpha=0.7,
                label=curve_name
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom, top)
    ax.set_xticks([])
    ax.set_ylabel("Depth")
    ax.legend(loc='upper right', fontsize=8)

    well_name = well.name or "Unknown Well"
    ax.set_title(f"Zone Flags - {well_name}")

    plt.tight_layout()

    return fig, ax


def add_zone_shading(
    ax: "Axes",
    zones: List[Dict],
    alpha: float = 0.2
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

    default_colors = ['yellow', 'cyan', 'magenta', 'lightgreen', 'lightsalmon']

    for i, zone in enumerate(zones):
        top = zone.get("top", 0)
        bottom = zone.get("bottom", 0)
        name = zone.get("name", f"Zone {i+1}")
        color = zone.get("color", default_colors[i % len(default_colors)])

        # Add horizontal span
        ax.axhspan(top, bottom, alpha=alpha, color=color, label=name)

        # Add zone label
        mid_depth = (top + bottom) / 2
        ax.text(
            0.02, mid_depth, name,
            transform=ax.get_yaxis_transform(),
            fontsize=8,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )


def create_net_pay_summary_plot(
    well: "Well",
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (6, 12)
) -> Tuple["Figure", List["Axes"]]:
    """
    Create a summary plot showing net rock/reservoir/pay.

    Parameters
    ----------
    well : Well
        Well object with NET_ROCK, NET_RES, NET_PAY curves
    depth_range : tuple, optional
        (top, bottom) depth range
    figsize : tuple
        Figure size

    Returns
    -------
    tuple
        (Figure, list of Axes)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    depth = well.depth

    # Set depth range
    if depth_range:
        top, bottom = depth_range
    else:
        top = well.start_depth or np.nanmin(depth)
        bottom = well.stop_depth or np.nanmax(depth)

    # Track 1: GR for reference
    ax1 = axes[0]
    gr = well.get("GR")
    if gr is not None:
        ax1.plot(gr.data, depth, 'g-', linewidth=0.5)
        ax1.fill_betweenx(depth, 0, gr.data, alpha=0.3, color='green')
        ax1.set_xlim(0, 150)
    ax1.set_xlabel("GR [API]")
    ax1.set_ylabel("Depth")
    ax1.grid(True, alpha=0.3)

    # Track 2: Flags
    ax2 = axes[1]

    flag_config = [
        ("NET_ROCK", "orange", 0.0, 0.33),
        ("NET_RES", "yellow", 0.33, 0.66),
        ("NET_PAY", "green", 0.66, 1.0),
    ]

    for curve_name, color, x_left, x_right in flag_config:
        curve = well.get(curve_name)
        if curve is not None:
            mask = curve.data > 0.5
            ax2.fill_betweenx(
                depth, x_left, x_right,
                where=mask,
                color=color,
                alpha=0.7,
                label=curve_name
            )

    ax2.set_xlim(0, 1)
    ax2.set_xticks([0.165, 0.5, 0.83])
    ax2.set_xticklabels(['Rock', 'Res', 'Pay'])
    ax2.set_xlabel("Net Flags")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Common settings
    for ax in axes:
        ax.set_ylim(bottom, top)

    well_name = well.name or "Unknown Well"
    fig.suptitle(f"Net Pay Summary - {well_name}", fontsize=12)

    plt.tight_layout()

    return fig, axes
