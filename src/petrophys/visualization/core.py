"""
Core data visualization utilities.

Plotting functions for core-to-log calibration:
- Porosity calibration crossplot
- Permeability calibration crossplot
- Poro-perm crossplot with fitted transform
- Core data on log depth plot
"""

from petrophys.visualization.base import Plot
from petrophys.visualization.style import _styled_plot, display_unit

from typing import Any, Callable, List, Mapping, Optional, Tuple
import re

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from petrophys.model.core import CoreData
from petrophys.interpretation.core_analysis.calibration import (
    CalibrationResult,
    PoroPermTransform,
)


def _side_by_side(items: Mapping[str, Any], draw: Callable[[Any, "Axes", str], Any],
                  figsize: Optional[Tuple[float, float]]) -> Tuple["Figure", List["Axes"]]:
    """One panel per entry of ``items``, titled by its key, on shared axes."""
    if not items:
        raise ValueError("Give at least one entry to plot")
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=figsize or (6.0 * n, 5.5), sharex=True, sharey=True, squeeze=False)
    for ax, (name, item) in zip(axes[0], items.items()):
        draw(item, ax, str(name))
    fig.tight_layout()
    return fig, list(axes[0])


@_styled_plot
def calibration_plot(
    result: CalibrationResult,
    ax: Optional["Axes"] = None,
    title: Optional[str] = None,
    show_fit: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
) -> Plot:
    """
    Plot a core-vs-log calibration crossplot.

    ``result`` may also be a mapping of titles to results, e.g.
    ``{"as stored": cal, "shifted +2.89 m": cal_tied}``: one panel each, side
    by side on shared axes, and the axes are returned as a list.

    Parameters
    ----------
    result : CalibrationResult
        Output from ``calibrate_porosity``, ``calibrate_permeability``, or
        ``calibrate_saturation``.
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
    Plot
        ``.fig`` and ``.ax``; ``.save(path)``. Shows itself at the end of a
        notebook cell.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required")
    if isinstance(result, Mapping):
        return Plot(*_side_by_side(result, lambda item, axis, name: calibration_plot(
            item, ax=axis, title=name, show_fit=show_fit), figsize))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (7, 6))
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

    finite = np.concatenate(
        [
            np.asarray(result.core_values, dtype=float)[np.isfinite(result.core_values)],
            np.asarray(result.log_values, dtype=float)[np.isfinite(result.log_values)],
        ]
    )
    if finite.size:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        pad = abs(vmax) * 0.1 if vmax else 1.0
        vmin -= pad
        vmax += pad
    ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5, label="1:1 line")

    if show_fit and not np.isnan(result.slope):
        x_fit = np.linspace(vmin, vmax, 100)
        y_fit = result.slope * x_fit + result.intercept
        ax.plot(
            x_fit,
            y_fit,
            "b-",
            linewidth=1,
            label=f"Fit: y = {result.slope:.2f}x {result.intercept:+.3f}",
        )

    # "Porosity (PHIE)" names the log curve; the core axis is the measured property.
    core_property = re.sub(r"\s*\([^)]*\)$", "", result.property_name).lower()
    inside = re.search(r"\(([^)]*)\)$", result.property_name)
    parts = [part.strip() for part in inside.group(1).split(",")] if inside else []
    log_curve = parts[0] if parts else "log"
    unit = {"porosity": "v/v", "water saturation": "frac", "permeability": "mD"}.get(core_property, "")
    if "log10" in parts:
        unit = f"log10 {unit}"
    shown = f" [{unit}]" if unit else ""
    ax.set_xlabel(f"Core {core_property}{shown}")
    ax.set_ylabel(f"{log_curve}{shown}")
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

    return Plot(fig, ax)


@_styled_plot
def _poro_perm_chart(
    core: CoreData,
    transform: Optional[PoroPermTransform] = None,
    ax: Optional["Axes"] = None,
    title: str = "Core Poro-Perm Crossplot",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple["Figure", Any]:
    """
    Plot a porosity-permeability crossplot from core data.

    ``core`` may also be a mapping of titles to ``CoreData`` or to
    ``(CoreData, transform)`` pairs, e.g. ``{"30/6-5": (core, t), "25/5-1":
    (core25, t25)}``: one panel each, side by side on shared axes, and the
    axes are returned as a list.

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
    if isinstance(core, Mapping):
        def draw(item, axis, name):
            plugs, fitted = item if isinstance(item, tuple) else (item, transform)
            return _poro_perm_chart(plugs, transform=fitted, ax=axis, title=name)
        return _side_by_side(core, draw, figsize)

    if core.porosity is None or core.permeability is None:
        raise ValueError("CoreData needs both porosity and permeability")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (8, 6))
    else:
        fig = ax.figure

    valid = (
        ~np.isnan(core.porosity)
        & ~np.isnan(core.permeability)
        & (core.permeability > 0)
    )
    scatter = ax.scatter(
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
        # The fit is drawn over the plugs' porosity range, where it is supported.
        phi_valid = core.porosity[valid]
        low, high = (float(np.min(phi_valid)), float(np.max(phi_valid))) if phi_valid.size else (0.001, 0.40)
        phi_range = np.linspace(max(low, 0.001), max(high, low + 0.01), 200)
        k_pred = transform.predict(phi_range)
        ax.plot(
            phi_range,
            k_pred,
            "r-",
            linewidth=2,
            label=(
                f"Fit: log10(K) = {transform.A:.1f}*phi {'-' if transform.B < 0 else '+'} {abs(transform.B):.1f}\n"
                f"R^2 = {transform.r_squared:.3f}"
            ),
        )
        ax.legend(fontsize=9)

    ax.set_yscale("log")
    ax.set_xlabel("Core porosity [v/v]")
    ax.set_ylabel("Core permeability [mD]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")

    # Use this call's own scatter handle: ax.collections[0] would grab a
    # previous dataset's points when the caller reuses the axes.
    cbar = fig.colorbar(scatter, ax=ax)
    if cbar.solids is not None:  # opaque, whatever the points' transparency
        cbar.solids.set_alpha(1.0)
    unit = getattr(core, "depth_unit", None) or (core.metadata.get("depth_unit") if isinstance(
        getattr(core, "metadata", None), dict) else None)
    cbar.set_label(f"Depth [{unit}]" if unit else "Depth")

    return fig, ax


def _core_family(curve_name: str):
    """Core measurement to overlay on a log curve named by its family (PHI*, PERM*/K*, SW*, SO*)."""
    name = curve_name.upper()
    if name.startswith("PHI"):
        return "porosity", "Core PHI"
    if name.startswith(("PERM", "KH", "KLOG", "K_")):
        return "permeability", "Core K"
    if name.startswith("SW"):
        return "sw", "Core Sw"
    if name.startswith("SO"):
        return "so", "Core So"
    return None


#: Unit of each core measurement, drawn on the axis when the log curve is absent.
_CORE_UNITS = {"porosity": "v/v", "permeability": "mD", "sw": "v/v", "so": "v/v"}


@_styled_plot
def core_on_logs(
    well,
    core: CoreData,
    curves: Optional[List[str]] = None,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    figsize: Tuple[float, float] = (20, 12),
) -> Plot:
    """
    Plot core data points overlaid on log curves.

    Curves are found as :func:`petrophys.plot` finds them (a standard name
    such as ``RDEP`` draws ``RT`` or ``ILD``) and drawn in standard units.
    Permeability curves (``PERM*``, ``K*``) are drawn on a log scale. A curve
    the well does not hold is reported with a ``UserWarning``; its track still
    shows the matching core points.

    Parameters
    ----------
    well : Well
        Well object with computed curves.
    core : CoreData
        Core data to overlay.
    curves : list, optional
        Log curves to show. Default: ["GR", "PHIE", "PERM", "SW"].
    top, bottom : float, optional
        Depth window. Default: the core interval +/- 50 depth units (m or
        ft, as the well).
    figsize : tuple
        Figure size.

    Returns
    -------
    Plot
        ``.fig`` and ``.axes`` (one per curve); ``.save(path)``. Shows itself at the end of a
        notebook cell.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required")
    import warnings

    from matplotlib.ticker import NullFormatter

    from petrophys.visualization.log_viewer import (
        _display_well,
        _resolve_curve_name_for_well,
        auto_template,
        DEFAULT_CURVE_DISPLAY_UNITS,
    )

    if curves is None:
        curves = ["GR", "PHIE", "PERM", "SW"]

    core_map = {
        "PHIE": ("porosity", "Core PHI"),
        "PHIE_CC": ("porosity", "Core PHI"),
        "PHIT": ("porosity", "Core PHI"),
        "PHID": ("porosity", "Core PHI"),
        "PHID_CORE": ("porosity", "Core PHI"),
        "PHIE_CORECAL": ("porosity", "Core PHI"),
        "PERM": ("permeability", "Core K"),
        "PERM_CC": ("permeability", "Core K"),
        "PERM_CORE": ("permeability", "Core K"),
        "PERM_CORECAL": ("permeability", "Core K"),
        "SW": ("sw", "Core Sw"),
        "SO": ("so", "Core So"),
    }

    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    # Units converted as the log plot converts them; values are not screened.
    shown = _display_well(well, interpreted=True)
    depth = np.asarray(shown.depth, dtype=float)
    if top is None or bottom is None:
        if core.top is None or core.bottom is None:
            raise ValueError("core has no finite top/bottom depth; pass top= and bottom=")
    depth_range = (core.top - 50 if top is None else float(top), core.bottom + 50 if bottom is None else float(bottom))
    d_mask = (depth >= depth_range[0]) & (depth <= depth_range[1])
    missing = []
    # Each curve's colour, line and scale as the log plot draws it.
    layout = {}
    for track in auto_template(shown, include_on_request=True).tracks:
        for spec in track.curves:
            found = _resolve_curve_name_for_well(shown, spec.curve_name, spec.candidates)
            if found is not None:
                layout.setdefault(found.upper(), (spec, str(track.scale).lower()))

    for i, curve_name in enumerate(curves):
        ax = axes[i]
        resolved = _resolve_curve_name_for_well(shown, curve_name)
        curve = shown.get(resolved) if resolved is not None else None
        match = core_map.get(curve_name.upper()) or _core_family(curve_name)
        is_perm = match is not None and match[0] == "permeability"
        if is_perm:
            ax.set_xscale("log")

        spec, scale = layout.get(str(resolved or "").upper(), (None, "linear"))
        if curve is not None:
            values = np.asarray(curve.data, dtype=float)[d_mask]
            if is_perm:
                values = np.where(values > 0, values, np.nan)
            label = curve_name if resolved.upper() == curve_name.upper() else f"{resolved} ({curve_name})"
            # Solid, in the log plot's colour: dashes break up on a busy curve beside plugs.
            ax.plot(values, depth[d_mask], color=spec.color if spec is not None else "#000000",
                    linestyle="-", linewidth=0.8, label=label)
        else:
            missing.append(curve_name)
        if scale == "log" and not is_perm:
            ax.set_xscale("log")

        if match is not None:
            attr, label = match
            core_vals = getattr(core, attr, None)
            if core_vals is not None:
                c_mask = (core.depth >= depth_range[0]) & (core.depth <= depth_range[1])
                if is_perm:
                    c_mask &= np.asarray(core_vals, dtype=float) > 0
                ax.scatter(core_vals[c_mask], core.depth[c_mask], marker="x", color="#000000",
                           s=18, linewidths=0.8, zorder=5, label=label)

        unit = str(getattr(curve, "unit", "") or "").strip() if curve is not None else ""
        if not unit:
            unit = (
                DEFAULT_CURVE_DISPLAY_UNITS.get(curve_name.upper())
                or (_CORE_UNITS.get(match[0]) if match is not None else "")
                or ""
            )
        name = resolved if curve is not None else curve_name
        unit = display_unit(unit)
        xlabel = f"{name} [{unit}]" if unit else name
        if curve is None:
            xlabel += " (no log curve)"
        # The header of a log-plot track: name and unit at the top, with the scale ends.
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        if spec is not None and spec.left is not None and spec.right is not None:
            left, right = float(spec.left), float(spec.right)
            ax.set_xlim(left, right)
            ax.set_xticks([left, right])
            ax.set_xticklabels([f"{left:g}", f"{right:g}"], fontsize=8)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel, fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_ylabel(f"Depth [{well.depth_unit}]" if well.depth_unit else "Depth")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
        ax.grid(True, alpha=0.3, which="both" if is_perm else "major")

    if missing:
        warnings.warn(
            f"core_on_logs: the well has no {', '.join(missing)} curve "
            f"(by its own name or an alias); those tracks show core points only.",
            UserWarning,
            stacklevel=3,
        )

    for ax in axes:
        ax.set_ylim(depth_range[1], depth_range[0])

    well_name = well.name or "Unknown"
    interval = (
        f", {core.top:.0f}-{core.bottom:.0f} {well.depth_unit or 'm'}"
        if core.top is not None and core.bottom is not None else ""
    )
    fig.suptitle(
        f"{well_name} - Core Data on Logs ({core.n_plugs} plugs{interval})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    return Plot(fig, axes)
