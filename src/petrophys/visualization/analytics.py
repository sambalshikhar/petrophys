"""General analytics charts for petrophysical studies.

These complement the log/crossplot/panel plotters with the handful of non-log
chart types that recur across multi-well studies, model comparisons and
sensitivity analysis: multi-well curve overlays, grouped bar comparisons,
sensitivity tornadoes, histogram comparisons and generic line comparisons.

They stay native to the ``petrophys.model.well.Well`` data model where relevant
(``create_multiwell_overlay`` takes Well objects) and otherwise accept plain
arrays so notebooks never need to hand-roll matplotlib.
"""

from __future__ import annotations

from petrophys.visualization.style import _styled_plot

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - exercised by import environment
    HAS_MATPLOTLIB = False


def _require_mpl() -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")


def _depth_axis_down(ax: "Axes") -> None:
    """Make y increase downwards once; safe on axes that are already inverted or shared."""
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def _cell_edges(coords: Any, n: int, name: str) -> np.ndarray:
    """Cell edges for ``n`` cells from ``n`` centres or ``n + 1`` edges.

    Centres that are evenly spaced on a log scale (e.g. NMR T2 bins) get
    geometric midpoints, so cells stay symmetric on a log axis; otherwise
    arithmetic midpoints are used. The first and last cells extend by half of
    their neighbouring spacing.
    """
    values = np.asarray(coords, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a one-dimensional array of finite values")
    if values.size == n + 1:
        edges = values
    elif values.size == n:
        if n == 1:
            return np.array([values[0] - 0.5, values[0] + 0.5])
        steps = np.diff(values)
        geometric = False
        if np.all(values > 0):
            ratios = values[1:] / values[:-1]
            geometric = n > 2 and np.allclose(ratios, ratios[0], rtol=1e-3) and not np.allclose(
                steps, steps[0], rtol=1e-3
            )
        if geometric:
            logs = np.log(values)
            mid = np.exp((logs[:-1] + logs[1:]) / 2)
            first = np.exp(logs[0] - (logs[1] - logs[0]) / 2)
            last = np.exp(logs[-1] + (logs[-1] - logs[-2]) / 2)
        else:
            mid = (values[:-1] + values[1:]) / 2
            first = values[0] - steps[0] / 2
            last = values[-1] + steps[-1] / 2
        edges = np.concatenate([[first], mid, [last]])
    else:
        raise ValueError(
            f"{name} has {values.size} values; give {n} cell centres or {n + 1} cell edges"
        )
    steps = np.diff(edges)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError(f"{name} must be strictly increasing or decreasing")
    return edges


def _minmax_normalize(values: np.ndarray, low: float = 5.0, high: float = 95.0) -> np.ndarray:
    """Scale to 0-1 using robust (percentile) endpoints; NaNs preserved."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return values
    lo = np.nanpercentile(values[finite], low)
    hi = np.nanpercentile(values[finite], high)
    if hi <= lo:
        return values
    out = (values - lo) / (hi - lo)
    return out


@_styled_plot
def create_multiwell_overlay(
    wells: Sequence[Any],
    curve: str,
    depth_range: Optional[Tuple[float, float]] = None,
    normalize: bool = False,
    labels: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (7, 10),
    title: Optional[str] = None,
    ax: Optional["Axes"] = None,
) -> Tuple["Figure", "Axes"]:
    """Overlay one curve versus depth from several wells on a shared depth axis.

    Ideal for cross-well QC and normalization "before/after" displays. When
    ``normalize`` is True each well's curve is scaled so its 5th and 95th
    percentiles map to 0 and 1, making distributions directly comparable.

    Every well is drawn in the depth unit of the first well (feet and metres
    are converted with 0.3048 m/ft), and each curve in its standard unit
    (percent porosity as v/v, kg/m3 density as g/cc), as :func:`petrophys.plot`
    draws it, so the wells share one depth axis and one value axis.

    Parameters
    ----------
    wells : sequence of Well
        Wells to overlay. Wells without ``curve`` are skipped.
    curve : str
        Curve to plot: a curve of the well or a standard name (``RDEP`` finds
        ``RT`` or ``ILD``), found as :func:`petrophys.plot` finds it.
    depth_range : tuple of float, optional
        Depth window, in the depth unit of the first well.
    normalize : bool
        Scale each curve so its 5th-95th percentiles span 0-1 before plotting.
    labels : sequence of str, optional
        Legend labels (defaults to each ``well.name``).
    figsize : (float, float)
        Figure size in inches, used when ``ax`` is not given.
    title : str, optional
        Axes title.
    ax : matplotlib Axes, optional
        Axes to draw on.

    Returns
    -------
    (Figure, Axes)
    """
    _require_mpl()
    from petrophys.model.units import normalize_depth_unit
    from petrophys.visualization.log_viewer import _display_well, _resolve_curve_name_for_well

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    labels = list(labels) if labels is not None else [getattr(w, "name", f"well {i}") for i, w in enumerate(wells)]
    cmap = plt.get_cmap("tab20") if len(wells) > 10 else plt.get_cmap("tab10")

    def depth_unit(well: Any) -> str:
        return normalize_depth_unit(getattr(well, "depth_unit", None), strict=False, default="")

    plot_unit = depth_unit(wells[0]) if len(wells) else ""
    value_units = []
    plotted = 0
    for i, well in enumerate(wells):
        resolved = _resolve_curve_name_for_well(well, curve)
        if resolved is None:
            continue
        shown = _display_well(well)
        found = shown.get(resolved)
        if found is None:
            continue
        depth = np.asarray(shown.depth, dtype=float)
        unit = depth_unit(well)
        if (unit, plot_unit) == ("ft", "m"):
            depth = depth * 0.3048
        elif (unit, plot_unit) == ("m", "ft"):
            depth = depth / 0.3048
        data = np.asarray(found.data, dtype=float)
        if depth_range is not None:
            lo, hi = sorted(depth_range)
            sel = (depth >= lo) & (depth <= hi)
            depth, data = depth[sel], data[sel]
        if normalize:
            data = _minmax_normalize(data)
        value_units.append(str(found.unit or "").strip())
        name = labels[i] if resolved.upper() == str(curve).upper() else f"{labels[i]} ({resolved})"
        ax.plot(data, depth, lw=0.6, color=cmap(i % cmap.N), label=name)
        plotted += 1

    _depth_axis_down(ax)
    ax.grid(True, alpha=0.3)
    shared_unit = value_units[0] if value_units and len(set(value_units)) == 1 else ""
    if normalize:
        ax.set_xlabel(f"normalized {curve}")
    else:
        ax.set_xlabel(f"{curve} [{shared_unit}]" if shared_unit else curve)
    ax.set_ylabel(f"Depth [{plot_unit}]" if plot_unit else "Depth")
    if plotted:
        legend = ax.legend(fontsize=7, loc="best")
        for handle in legend.legend_handles if hasattr(legend, "legend_handles") else legend.legendHandles:
            handle.set_linewidth(2.0)
    if title:
        ax.set_title(title)
    return fig, ax


@_styled_plot
def create_bar_comparison(
    categories: Sequence[str],
    values: Any,
    series_labels: Optional[Sequence[str]] = None,
    horizontal: bool = False,
    sort_by: Optional[int] = None,
    descending: bool = True,
    colors: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    value_labels: bool = False,
    ax: Optional["Axes"] = None,
) -> Tuple["Figure", "Axes"]:
    """Single or grouped bar chart for rankings and category comparisons.

    ``values`` is either a 1-D sequence (single series) or a 2-D array / mapping
    of ``{series_label: sequence}`` for grouped bars. Use ``sort_by`` (a series
    index) to rank categories, e.g. rank wells by HCPV.
    """
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if isinstance(values, Mapping):
        series_labels = list(values.keys()) if series_labels is None else list(series_labels)
        mat = np.array([np.asarray(values[k], dtype=float) for k in series_labels])
    else:
        mat = np.atleast_2d(np.asarray(values, dtype=float))
        if series_labels is None:
            series_labels = [None] if mat.shape[0] == 1 else [f"series {i}" for i in range(mat.shape[0])]

    cats = list(categories)
    order = None
    if sort_by is not None:
        order = np.argsort(mat[sort_by])
        if descending:
            order = order[::-1]
        cats = [cats[i] for i in order]
        mat = mat[:, order]

    n_series = mat.shape[0]
    n_cat = mat.shape[1]
    idx = np.arange(n_cat)
    width = 0.8 / n_series
    default_colors = colors or plt.get_cmap("tab10")(np.linspace(0, 1, max(n_series, 2)))

    # A single series with one color per category means per-bar coloring
    # (e.g. red/orange/green status bars); reorder with any sort applied.
    per_bar = (
        n_series == 1
        and not isinstance(default_colors, np.ndarray)
        and len(default_colors) == n_cat
        and n_cat > 1
    )
    if per_bar and order is not None:
        default_colors = [default_colors[i] for i in order]

    for s in range(n_series):
        offset = (s - (n_series - 1) / 2) * width
        if per_bar:
            color = list(default_colors)
        else:
            color = default_colors[s % len(default_colors)] if not isinstance(default_colors, np.ndarray) else default_colors[s]
        if horizontal:
            bars = ax.barh(idx + offset, mat[s], height=width, color=color, label=series_labels[s])
        else:
            bars = ax.bar(idx + offset, mat[s], width=width, color=color, label=series_labels[s])
        if value_labels:
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    if horizontal:
        ax.set_yticks(idx)
        ax.set_yticklabels(cats)
        _depth_axis_down(ax)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
    else:
        ax.set_xticks(idx)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_ylabel(ylabel or "")
        ax.set_xlabel(xlabel or "")

    if any(lbl is not None for lbl in series_labels):
        ax.legend(fontsize=8)
    ax.grid(True, axis=("x" if horizontal else "y"), alpha=0.3)
    if title:
        ax.set_title(title)
    return fig, ax


@_styled_plot
def create_tornado_plot(
    labels: Sequence[str],
    low: Sequence[float],
    high: Sequence[float],
    base: float,
    figsize: Tuple[float, float] = (9, 5),
    title: Optional[str] = None,
    xlabel: str = "Outcome",
    low_color: str = "#c44e52",
    high_color: str = "#4c72b0",
    ax: Optional["Axes"] = None,
) -> Tuple["Figure", "Axes"]:
    """Sensitivity tornado: horizontal bars of low/high outcome about a base case.

    Parameters are sorted by total span (widest at top). ``low``/``high`` are the
    outcome values when each parameter takes its low/high setting; ``base`` is the
    base-case outcome (drawn as a vertical reference).
    """
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    labels = list(labels)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    span = np.abs(high - low)
    order = np.argsort(span)  # smallest first -> widest ends on top

    ypos = np.arange(len(labels))
    for row, i in enumerate(order):
        lo, hi = low[i], high[i]
        if (lo - base) * (hi - base) > 0:
            # Both settings move the outcome the same way: the bars sit side
            # by side, so neither hides the other.
            low_y, high_y, height = row + 0.2, row - 0.2, 0.4
        else:
            low_y, high_y, height = row, row, 0.8
        ax.barh(low_y, lo - base, height=height, left=base, color=low_color, alpha=0.85,
                label="low" if row == 0 else None)
        ax.barh(high_y, hi - base, height=height, left=base, color=high_color, alpha=0.85,
                label="high" if row == 0 else None)

    values = np.concatenate([low, high, [base]])
    values = values[np.isfinite(values)]
    if values.size:
        pad = 0.06 * (values.max() - values.min()) or 0.05 * abs(base) or 1.0
        ax.set_xlim(values.min() - pad, values.max() + pad)
    ax.axvline(base, color="k", lw=1.2, ls="--", zorder=3, label=f"base {base:.3g}")
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, axis="x", alpha=0.3)
    if title:
        ax.set_title(title)
    return fig, ax


def _curve_by_well(wells: Any, curve: str) -> Tuple[Dict[str, np.ndarray], str]:
    """``{well name: values}`` of ``curve`` in each well or result, and its unit.

    A well name that repeats in a list (two runs of one well) is numbered:
    ``"30/6-5"``, ``"30/6-5 (2)"``.
    """
    from petrophys.visualization.log_viewer import _resolve_curve_name_for_well

    items = list(wells.items()) if isinstance(wells, Mapping) else [(None, well) for well in wells]
    data: Dict[str, np.ndarray] = {}
    units = set()
    for label, item in items:
        well = getattr(item, "well", item)
        name = _resolve_curve_name_for_well(well, curve)
        if name is None:
            raise KeyError(f"{well.name} has no {curve} curve")
        values = well.get(name)
        key = base = str(label if label is not None else well.name)
        count = 1
        while key in data:
            count += 1
            key = f"{base} ({count})"
        data[key] = np.asarray(values.data, dtype=float)
        units.add(str(values.unit or "").strip())
    units.discard("")
    return data, (units.pop() if len(units) == 1 else "")


@_styled_plot
def create_histogram_comparison(
    datasets: Any,
    curve: Optional[str] = None,
    bins: int = 40,
    density: bool = False,
    figsize: Tuple[float, float] = (9, 5),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    alpha: float = 0.5,
    ax: Optional["Axes"] = None,
    value_range: Optional[Tuple[float, float]] = None,
    histtype: str = "bar",
) -> Tuple["Figure", "Axes"]:
    """Overlay histograms of several datasets (e.g. GR before/after normalization).

    ``datasets`` maps a label to a 1-D array; NaNs are dropped. Or it is
    several wells or interpretation results (a list, the wells
    :func:`petrophys.read` returns, or a mapping of label to well) and
    ``curve`` names the curve to compare (``"GR"``, a standard or the well's
    own name), labelled by well name; the x-axis is labelled with its unit.
    Wells in a list that share a name (two runs of one well) are labelled
    ``"30/6-5"``, ``"30/6-5 (2)"`` in list order; pass a mapping
    (``{"Archie": run_a, "Indonesia": run_b}``) to name them. Wells or
    results, or a list, without ``curve`` raise ValueError.
    ``value_range`` fixes the bin range (spikes otherwise stretch it), and
    ``histtype="step"`` draws outlines, which stay readable for several wells.
    """
    _require_mpl()
    if curve is not None:
        datasets, unit = _curve_by_well(datasets, curve)
        xlabel = xlabel or (f"{curve} ({unit})" if unit else curve)
    elif not isinstance(datasets, Mapping) or any(
            hasattr(getattr(value, "well", value), "curves") for value in datasets.values()):
        raise ValueError(
            "Without curve=, datasets must map a label to an array. To compare wells or "
            'interpretation results, name the curve: create_histogram_comparison(wells, curve="GR")'
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    # More than ten datasets would repeat the default ten colours.
    colors = plt.get_cmap("tab20" if len(datasets) > 10 else "tab10")

    all_finite = []
    for arr in datasets.values():
        a = np.asarray(arr, dtype=float)
        all_finite.append(a[np.isfinite(a)])
    if value_range is not None:
        lo, hi = value_range
    else:
        lo = min((a.min() for a in all_finite if a.size), default=0.0)
        hi = max((a.max() for a in all_finite if a.size), default=1.0)
    edges = np.linspace(lo, hi, bins + 1)

    for index, (label, arr) in enumerate(datasets.items()):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        color = colors(index % colors.N)
        if histtype == "step":
            ax.hist(a, bins=edges, density=density, histtype="step", linewidth=1.4, label=label, color=color)
        else:
            ax.hist(a, bins=edges, density=density, alpha=alpha, histtype=histtype, label=label, color=color)

    ax.set_xlabel(xlabel or "")
    ax.set_ylabel("density" if density else "count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    return fig, ax


@_styled_plot
def create_line_comparison(
    x: Any,
    series: "Mapping[str, Any]",
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    logx: bool = False,
    logy: bool = False,
    markers: bool = False,
    invert_yaxis: bool = False,
    ax: Optional["Axes"] = None,
) -> Tuple["Figure", "Axes"]:
    """Generic x versus multiple y-series line plot.

    Covers transfer functions (e.g. Vcl vs IGR for several methods), temperature
    profiles (temperature vs depth with ``invert_yaxis``), and similar comparisons
    that are not depth-track logs. ``series`` maps a label to a y-array aligned to ``x``.
    """
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.asarray(x, dtype=float)
    style = "-o" if markers else "-"
    for label, y in series.items():
        ax.plot(x, np.asarray(y, dtype=float), style, ms=3, lw=1.2, label=label)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if invert_yaxis:
        _depth_axis_down(ax)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    return fig, ax


@_styled_plot
def create_heatmap(
    matrix: Any,
    x: Optional[Sequence[float]] = None,
    y: Optional[Sequence[float]] = None,
    figsize: Tuple[float, float] = (7, 10),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "viridis",
    invert_yaxis: bool = True,
    colorbar_label: Optional[str] = None,
    log_color: bool = False,
    ax: Optional["Axes"] = None,
) -> Tuple["Figure", "Axes"]:
    """2-D intensity map, e.g. an NMR T2-distribution vs depth or a binned density crossplot.

    Parameters
    ----------
    matrix : array-like, shape (n_y, n_x)
        Cell values; NaN cells are left blank.
    x, y : array-like, optional
        Cell coordinates along each axis: either the ``n`` cell centres (e.g.
        T2 bin centres, sample depths) or the ``n + 1`` cell edges. Each cell is
        drawn centred on its coordinate. Log-spaced centres such as T2 bins get
        log-symmetric cells, so ``ax.set_xscale("log")`` shows them correctly.
        When omitted, cells are indexed 0..n.
    figsize : (float, float)
        Figure size in inches, used when ``ax`` is not given.
    title : str, optional
        Axes title.
    xlabel, ylabel : str, optional
        Axis labels; blank by default.
    cmap : str
        Matplotlib colour map name.
    invert_yaxis : bool
        Show y increasing downwards, like a depth track.
    colorbar_label : str, optional
        Label of the colour bar.
    log_color : bool
        Log-scaled colour (LogNorm).
    ax : matplotlib Axes, optional
        Axes to draw on.

    Returns
    -------
    (Figure, Axes)
    """
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mat = np.asarray(matrix, dtype=float)
    n_y, n_x = mat.shape
    norm = None
    if log_color:
        from matplotlib.colors import LogNorm

        positive = mat[np.isfinite(mat) & (mat > 0)]
        if positive.size:
            norm = LogNorm(vmin=positive.min(), vmax=positive.max())

    x_edges = _cell_edges(x, n_x, "x") if x is not None else np.arange(n_x + 1, dtype=float)
    y_edges = _cell_edges(y, n_y, "y") if y is not None else np.arange(n_y + 1, dtype=float)

    im = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(mat), cmap=cmap, norm=norm,
                       shading="flat")
    if invert_yaxis:
        _depth_axis_down(ax)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    if colorbar_label:
        cb.set_label(colorbar_label)
    if title:
        ax.set_title(title)
    return fig, ax


__all__ = [
    "create_multiwell_overlay",
    "create_bar_comparison",
    "create_tornado_plot",
    "create_histogram_comparison",
    "create_line_comparison",
    "create_heatmap",
]
