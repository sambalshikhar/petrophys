"""Plots of several wells: ``pp.plot`` on a list of wells or results.

Without ``curves``, :class:`MultiWellPlot` draws each well's own log plot, one
figure at a time. With ``curves``, :func:`compare_wells` draws one figure with
a track per curve and every well's curve in it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from petrophys.visualization.style import _styled_plot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from petrophys.visualization.log_viewer import LogViewer


def _plotted_well(item: Any) -> Any:
    """The well of a well or an interpretation result."""
    return getattr(item, "well", item)


class MultiWellPlot:
    """One log plot per well, drawn when needed and closed once shown or saved.

    Returned by :func:`petrophys.plot` for several wells without ``curves``.
    The figures are not held together in memory: ``save(folder)`` writes one
    file per well, ``show()`` and a notebook display draw each well in turn,
    and ``plot[i]`` or ``plot["15/9-12"]`` gives that well's
    :class:`~petrophys.visualization.log_viewer.LogViewer`.
    """

    def __init__(self, items: Sequence[Any], options: Optional[Dict[str, Any]] = None,
                 labels: Optional[Sequence[str]] = None):
        self.items: List[Any] = list(items)
        self.options: Dict[str, Any] = dict(options or {})
        self.labels: Optional[List[str]] = None if labels is None else [str(label) for label in labels]
        #: Each drawn well's viewer warnings, by its label (else its well name).
        self.warnings: Dict[str, List[str]] = {}

    @property
    def names(self) -> List[str]:
        """Labels, or else well names, in plotting order."""
        if self.labels is not None:
            return list(self.labels)
        return [str(_plotted_well(item).name) for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"MultiWellPlot({len(self)} wells: {', '.join(self.names[:6])}{' ...' if len(self) > 6 else ''})"

    def _viewer(self, index: int) -> "LogViewer":
        """The log plot of the well at position ``index``; its warnings are kept under its name."""
        from petrophys.api import plot

        viewer = plot(self.items[index], **self.options)
        self.warnings[self.names[index]] = list(viewer.warnings)
        return viewer

    def __getitem__(self, key: Union[int, str]) -> "LogViewer":
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(f"No well {key!r}; the wells are {', '.join(self.names)}")
            key = self.names.index(key)
        return self._viewer(range(len(self.items))[key])

    def save(self, folder: Union[str, Path], format: str = "png", dpi: int = 150) -> List[Path]:
        """Save one plot per well to ``folder/<well name>.<format>`` (png, pdf, svg, ...).

        A repeated well name gets _2, _3, ...; each figure is closed once saved.
        Returns the written paths, in plotting order.
        """
        from petrophys.workflow.multi_well import _unique_file_names

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, name in enumerate(_unique_file_names(self.names)):
            path = folder / f"{name}.{format.lstrip('.')}"
            viewer = self._viewer(index)
            viewer.save(path, dpi=dpi)
            viewer.close()
            paths.append(path)
        return paths

    def show(self) -> None:
        """Show each well's plot in turn."""
        for index in range(len(self.items)):
            viewer = self._viewer(index)
            viewer.show()
            viewer.close()

    def close(self) -> None:
        """Nothing stays open: each figure is closed once shown or saved."""

    def _ipython_display_(self) -> None:
        for index in range(len(self.items)):
            viewer = self._viewer(index)
            viewer._ipython_display_()  # the PNG, once, under any backend
            viewer.close()


#: Standard curves drawn on a logarithmic scale in a comparison.
_LOG_CURVES = {"RDEP", "RMED", "RSHA", "RXO", "RT", "PERM", "PERM_CC", "KH_CORE", "NMR_KSDR", "NMR_KTIM"}


def _log_scale(name: str, mnemonic: str) -> bool:
    from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic

    standard = get_standard_mnemonic(mnemonic).upper()
    return name.upper() in _LOG_CURVES or standard in _LOG_CURVES or name.upper().startswith("PERM")


class WellComparisonPlot:
    """One figure comparing curves across wells: a track per curve, a line per well.

    Returned by :func:`petrophys.plot` for several wells with ``curves``.
    ``fig`` and ``axes`` are the matplotlib figure and tracks; ``warnings``
    names the curves a well does not have.
    """

    def __init__(self, fig: "Figure", axes: List["Axes"], warnings: List[str]):
        self.fig: Optional["Figure"] = fig
        self.axes = axes
        self.warnings = warnings

    def save(self, filename: Union[str, Path], dpi: int = 150) -> None:
        """Save the figure (png, pdf, svg, ... by the extension).

        Outside an interactive session the figure is released from pyplot
        after saving; ``.fig`` keeps it for further edits and saves.
        """
        from petrophys.visualization.log_viewer import _non_interactive_backend
        from petrophys.visualization.style import save_figure

        if self.fig is None:
            raise ValueError("The figure was closed; draw it again with petrophys.plot.")
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(self.fig, path, dpi=dpi)
        if _non_interactive_backend():
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def show(self) -> None:
        """Display the figure."""
        import matplotlib.pyplot as plt

        plt.show()

    def _ipython_display_(self) -> None:
        # At the end of a notebook cell the comparison shows its figure, once.
        import matplotlib.pyplot as plt
        from IPython.core.pylabtools import print_figure
        from IPython.display import Image, display

        if self.fig is not None:
            display(Image(data=print_figure(self.fig, "png")))
            plt.close(self.fig)

    def close(self) -> None:
        """Close the figure and free its memory."""
        import matplotlib.pyplot as plt

        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


@_styled_plot
def compare_wells(
    items: Sequence[Any],
    curves: Sequence[str],
    *,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    labels: Optional[Sequence[str]] = None,
) -> WellComparisonPlot:
    """Draw ``curves`` of several wells in one figure, each curve in its own track.

    Curves are found by the well's own name or a standard name (``"GR"``,
    ``"RDEP"``, ``"PHIE"``...), as in a single-well plot, and converted to
    standard units as a single-well plot draws them (percent porosity to
    v/v, kg/m3 to g/cm3, ...). Each track keeps the single-well plot's
    scale for its curve (GR 0-150 API, RHOB 1.95-2.95 g/cc, Sw 1-0, resistivity
    0.2-2000 ohm.m logarithmic, ...), so wells and runs compare on the scales a
    log plot uses; a curve the layout does not know spans the wells' 0.5-99.5
    percentiles. A curve identical in every run is drawn once, in black. Depths are in the first well's depth unit; a well
    logged in the other unit is converted. ``top`` and ``bottom`` bound the
    depth axis. ``warnings`` names the wells without a curve, and curves
    whose units still differ after conversion. ``labels`` name the lines
    (default: the well names), so runs of one well can be compared; a number
    of labels other than the number of wells raises ValueError. ``figsize``
    is the figure size in inches (default: 3.2 in per curve, at least 6 in,
    by 11 in). Returns a :class:`WellComparisonPlot`.
    """
    import matplotlib.pyplot as plt

    from petrophys.visualization.log_viewer import (
        _display_well,
        _resolve_curve_name_for_well,
        create_adaptive_log_viewer_template,
    )

    wells = [_display_well(_plotted_well(item), interpreted=hasattr(item, "steps_executed")) for item in items]
    names = [str(label) for label in labels] if labels is not None else [str(well.name) for well in wells]
    if len(names) != len(wells):
        raise ValueError(f"{len(names)} labels for {len(wells)} wells")
    # Each curve's scale, direction and track as the single-well plot draws it.
    standards: Dict[str, Tuple[Any, str]] = {}
    for name in curves:
        for well in wells:
            resolved = _resolve_curve_name_for_well(well, name)
            if resolved is None:
                continue
            for spec_track in create_adaptive_log_viewer_template(well, include_on_request=True).tracks:
                for spec in spec_track.curves:
                    found = _resolve_curve_name_for_well(well, spec.curve_name, spec.candidates)
                    if found is not None and found.upper() == resolved.upper():
                        standards[name] = (spec, str(spec_track.scale).lower())
                        break
                if name in standards:
                    break
            break
    unit = str(wells[0].depth_unit or "")
    width = max(2.2 * len(curves), 6.0)
    fig, axes_grid = plt.subplots(1, len(curves), figsize=figsize or (width, 11), sharey=True, squeeze=False)
    axes = list(axes_grid[0])
    colors = plt.get_cmap("tab20" if len(wells) > 10 else "tab10")
    notes: List[str] = []
    with_data: set = set()

    def depth_of(well: Any) -> np.ndarray:
        depth = np.asarray(well.depth, dtype=float)
        if {str(well.depth_unit or ""), unit} == {"m", "ft"}:
            depth = depth * 0.3048 if unit == "m" else depth / 0.3048
        return depth

    def in_window(depth: np.ndarray) -> np.ndarray:
        inside = np.isfinite(depth)
        if top is not None:
            inside &= depth >= top
        if bottom is not None:
            inside &= depth <= bottom
        return inside

    for track, (ax, name) in enumerate(zip(axes, curves)):
        units = set()
        series = []
        for index, well in enumerate(wells):
            resolved = _resolve_curve_name_for_well(well, name)
            if resolved is None:
                notes.append(f"{names[index]}: no {name} curve")
                continue
            curve = well.get(resolved)
            series.append((index, depth_of(well), np.asarray(curve.data, dtype=float), resolved))
            units.add(str(curve.unit or "").strip())
        spec, scale = standards.get(name, (None, "linear"))
        log_track = scale == "log" or any(_log_scale(name, resolved) for *_, resolved in series)
        # A curve the same in every run (the inputs of a model comparison) is drawn once.
        same = len(series) > 1 and all(
            d.shape == series[0][1].shape and np.allclose(d, series[0][1], equal_nan=True)
            and np.allclose(v, series[0][2], equal_nan=True) for _, d, v, _ in series[1:])
        drawn = series[:1] if same else series
        shown = []
        for index, depth, values, _ in drawn:
            inside = in_window(depth) & np.isfinite(values)
            color = "#000000" if same else colors(index % colors.N)
            label = "every run" if same else names[index]
            ax.plot(values, depth, color=color, linewidth=0.8, label=label)
            if inside.any():
                with_data.add(label)
            shown.append(values[inside])
        if log_track:
            ax.set_xscale("log")
        left = right = None
        if spec is not None and spec.left is not None and spec.right is not None:
            left, right = float(spec.left), float(spec.right)
        else:
            # A curve the layout does not know: the wells' 0.5-99.5 percentiles.
            data = np.concatenate(shown) if shown else np.empty(0)
            data = data[data > 0] if log_track else data
            if data.size:
                low, high = np.percentile(data, [0.5, 99.5])
                if high > low:
                    pad = 0.03 * (high - low)
                    left, right = (low / 1.5, high * 1.5) if log_track else (low - pad, high + pad)
        if left is not None:
            ax.set_xlim(left, right)
        units.discard("")
        if len(units) > 1:
            notes.append(f"{name}: the wells give it in different units ({', '.join(sorted(units))}); "
                         "the track compares unlike values")
        unit_text = next(iter(units)) if len(units) == 1 else ("mixed units" if units else "")
        # The header of a log-plot track: name and unit, with the scale ends.
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        if left is not None:
            ax.set_xticks([left, right])
            ax.set_xticklabels([f"{left:g}", f"{right:g}"], fontsize=8)
            ax.set_xlim(left, right)
        ax.set_xlabel(f"{name} [{unit_text}]" if unit_text else name, fontsize=9, fontweight="bold")
        from matplotlib.ticker import NullFormatter

        ax.xaxis.set_minor_formatter(NullFormatter())
        if not log_track and left is not None:
            ax.set_xticks(np.linspace(left, right, 6), minor=True)
        ax.grid(True, which="both", axis="x", alpha=0.3)
        ax.grid(True, which="major", axis="y", alpha=0.3)
        if track == 0:
            ax.set_ylabel(f"Depth [{unit}]" if unit else "Depth")
    if top is not None or bottom is not None:
        low, high = axes[0].get_ylim()
        axes[0].set_ylim(bottom if bottom is not None else max(low, high), top if top is not None else min(low, high))
    elif not axes[0].yaxis_inverted():
        axes[0].invert_yaxis()
    handles, legend_labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in legend_labels and label in with_data:
                handles.append(handle)
                legend_labels.append(label)
    well_names = list(dict.fromkeys(str(well.name) for well in wells))
    title = " | ".join(well_names) if len(well_names) <= 3 else f"{len(well_names)} wells"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                   ncol=min(len(legend_labels), 6), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return WellComparisonPlot(fig, axes, notes)
