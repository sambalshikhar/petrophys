"""
Core data visualization utilities.

Plotting functions for core-to-log calibration:
- Porosity calibration crossplot
- Permeability calibration crossplot
- Poro-perm crossplot with fitted transform
- Core data on log depth plot
"""

from typing import Optional, Tuple, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from petrophys.core.core_data import CoreData
from petrophys.core.calibration import (
    CalibrationResult,
    PoroPermTransform,
)


def plot_calibration(
    result: CalibrationResult,
    ax: Optional["Axes"] = None,
    title: Optional[str] = None,
    show_fit: bool = True,
    figsize: Tuple[float, float] = (7, 6),
) -> Tuple["Figure", "Axes"]:
    """
    Plot a core-vs-log calibration crossplot.

    Parameters
    ----------
    result : CalibrationResult
        Output from ``calibrate_porosity`` or ``calibrate_permeability``.
    ax : Axes, optional
        Existing axes to plot on.
    title : str, optional
        Custom title.
    show_fit : bool
        Show the linear fit line.
    figsize : tuple
        Figure size (only used if ax is None).

    Returns
    -------
    tuple
        (Figure, Axes)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(
        result.core_values,
        result.log_values,
        s=15,
        alpha=0.6,
        edgecolors="none",
        label="Data",
    )

    vmin = min(np.nanmin(result.core_values), np.nanmin(result.log_values))
    vmax = max(np.nanmax(result.core_values), np.nanmax(result.log_values))
    ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5, label="1:1 line")

    if show_fit and not np.isnan(result.slope):
        x_fit = np.linspace(vmin, vmax, 100)
        y_fit = result.slope * x_fit + result.intercept
        ax.plot(
            x_fit,
            y_fit,
            "b-",
            linewidth=1,
            label=f"Fit: y={result.slope:.2f}x+{result.intercept:.3f}",
        )

    ax.set_xlabel("Core")
    ax.set_ylabel("Log")
    ax.set_title(
        title
        or (
            f"{result.property_name}\n"
            f"R^2={result.r_squared:.3f}, Bias={result.bias:.4f}, "
            f"RMSE={result.rmse:.4f} (n={result.n_points})"
        )
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_poro_perm(
    core: CoreData,
    transform: Optional[PoroPermTransform] = None,
    ax: Optional["Axes"] = None,
    title: str = "Core Poro-Perm Crossplot",
    figsize: Tuple[float, float] = (8, 6),
) -> Tuple["Figure", "Axes"]:
    """
    Plot a porosity-permeability crossplot from core data.

    Parameters
    ----------
    core : CoreData
        Core data with porosity and permeability.
    transform : PoroPermTransform, optional
        Fitted transform to overlay.
    ax : Axes, optional
        Existing axes.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    tuple
        (Figure, Axes)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required")

    if core.porosity is None or core.permeability is None:
        raise ValueError("CoreData needs both porosity and permeability")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    valid = (
        ~np.isnan(core.porosity)
        & ~np.isnan(core.permeability)
        & (core.permeability > 0)
    )
    ax.scatter(
        core.porosity[valid],
        core.permeability[valid],
        c=core.depth[valid],
        cmap="viridis",
        s=25,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.3,
    )

    if transform is not None:
        phi_range = np.linspace(0.001, 0.40, 200)
        k_pred = transform.predict(phi_range)
        ax.plot(
            phi_range,
            k_pred,
            "r-",
            linewidth=2,
            label=(
                f"Fit: log10(K) = {transform.A:.1f}*phi + ({transform.B:.1f})\n"
                f"R^2 = {transform.r_squared:.3f}"
            ),
        )
        ax.legend(fontsize=9)

    ax.set_yscale("log")
    ax.set_xlabel("Core Porosity (v/v)")
    ax.set_ylabel("Core Permeability (mD)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")

    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Depth")

    return fig, ax


def plot_core_on_logs(
    well,
    core: CoreData,
    curves: Optional[List[str]] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (20, 12),
) -> Tuple["Figure", List["Axes"]]:
    """
    Plot core data points overlaid on log curves.

    Parameters
    ----------
    well : Well
        Well object with computed curves.
    core : CoreData
        Core data to overlay.
    curves : list, optional
        Log curves to show. Default: ["GR", "PHIE", "PERM", "SW"].
    depth_range : tuple, optional
        (top, bottom) depth range. Default: core interval +/- 50m.
    figsize : tuple
        Figure size.

    Returns
    -------
    tuple
        (Figure, list of Axes)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required")

    if curves is None:
        curves = ["GR", "PHIE", "PERM", "SW"]

    core_map = {
        "PHIE": ("porosity", "Core PHI"),
        "PHIE_CORECAL": ("porosity", "Core PHI"),
        "PHIT": ("porosity", "Core PHI"),
        "PHID": ("porosity", "Core PHI"),
        "PHID_CORE": ("porosity", "Core PHI"),
        "PERM": ("permeability", "Core K"),
        "PERM_CORE": ("permeability", "Core K"),
        "PERM_CORECAL": ("permeability", "Core K"),
        "SW": ("sw", "Core Sw"),
        "SO": ("so", "Core So"),
    }

    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    depth = well.depth
    if depth_range is None:
        depth_range = (core.top - 50, core.bottom + 50)
    d_mask = (depth >= depth_range[0]) & (depth <= depth_range[1])

    for i, curve_name in enumerate(curves):
        ax = axes[i]
        curve = well.get(curve_name)

        if curve is not None:
            is_perm = "PERM" in curve_name.upper()
            if is_perm:
                ax.semilogx(
                    curve.data[d_mask],
                    depth[d_mask],
                    "b",
                    linewidth=0.8,
                    label=curve_name,
                )
            else:
                ax.plot(
                    curve.data[d_mask],
                    depth[d_mask],
                    "b",
                    linewidth=0.8,
                    label=curve_name,
                )

        if curve_name.upper() in core_map:
            attr, label = core_map[curve_name.upper()]
            core_vals = getattr(core, attr, None)
            if core_vals is not None:
                c_mask = (core.depth >= depth_range[0]) & (core.depth <= depth_range[1])
                ax.scatter(
                    core_vals[c_mask],
                    core.depth[c_mask],
                    c="red",
                    s=20,
                    zorder=5,
                    label=label,
                    edgecolors="black",
                    linewidths=0.5,
                )

        ax.set_xlabel(f"{curve_name} [{curve.unit if curve else ''}]")
        if i == 0:
            ax.set_ylabel("Depth")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_ylim(depth_range[1], depth_range[0])

    well_name = well.name or "Unknown"
    fig.suptitle(
        f"{well_name} - Core Data on Logs ({core.n_plugs} plugs, "
        f"{core.top:.0f}-{core.bottom:.0f}m)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig, axes
