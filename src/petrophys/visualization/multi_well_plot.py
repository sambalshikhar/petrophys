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
    convention for its curve: logarithmic resistivity and permeability,
    neutron and sonic increasing to the left. The scale spans the wells'
    0.5-99.5 percentiles. Depths are in the first well's depth unit; a well
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
    conventions: Dict[str, Tuple[bool, bool]] = {}
    for name in curves:
        for well in wells:
            resolved = _resolve_curve_name_for_well(well, name)
            if resolved is None:
                continue
            for spec_track in create_adaptive_log_viewer_template(well).tracks:
                for spec in spec_track.curves:
                    found = _resolve_curve_name_for_well(well, spec.curve_name, spec.candidates)
                    if found is not None and found.upper() == resolved.upper():
                        reversed_scale = (spec.left is not None and spec.right is not None
                                          and float(spec.left) > float(spec.right))
                        conventions[name] = (str(spec_track.scale).lower() == "log", reversed_scale)
                        break
                if name in conventions:
                    break
            break
    unit = str(wells[0].depth_unit or "")
    width = max(3.2 * len(curves), 6.0)
    fig, axes_grid = plt.subplots(1, len(curves), figsize=figsize or (width, 11), sharey=True, squeeze=False)
    axes = list(axes_grid[0])
    colors = plt.get_cmap("tab20" if len(wells) > 10 else "tab10")
    notes: List[str] = []
    for track, (ax, name) in enumerate(zip(axes, curves)):
        units = set()
        log_track = False
        shown = []
        for index, well in enumerate(wells):
            resolved = _resolve_curve_name_for_well(well, name)
            if resolved is None:
                notes.append(f"{names[index]}: no {name} curve")
                continue
            curve = well.get(resolved)
            depth = np.asarray(well.depth, dtype=float)
            well_unit = str(well.depth_unit or "")
            if {well_unit, unit} == {"m", "ft"}:
                depth = depth * 0.3048 if unit == "m" else depth / 0.3048
            values = np.asarray(curve.data, dtype=float)
            log_track = log_track or conventions.get(name, (False, False))[0] or _log_scale(name, resolved)
            ax.plot(values, depth, color=colors(index % colors.N), linewidth=0.8, label=names[index])
            units.add(str(curve.unit or "").strip())
            inside = np.isfinite(values) & np.isfinite(depth)
            if top is not None:
                inside &= depth >= top
            if bottom is not None:
                inside &= depth <= bottom
            shown.append(values[inside])
        if log_track:
            ax.set_xscale("log")
        # The scale spans the wells' 0.5-99.5 percentiles, so one spike does not flatten the rest.
        data = np.concatenate(shown) if shown else np.empty(0)
        data = data[data > 0] if log_track else data
        if data.size:
            low, high = np.percentile(data, [0.5, 99.5])
            if high > low:
                if log_track:
                    ax.set_xlim(low / 1.5, high * 1.5)
                else:
                    pad = 0.03 * (high - low)
                    ax.set_xlim(low - pad, high + pad)
        if conventions.get(name, (False, False))[1]:
            ax.invert_xaxis()
        units.discard("")
        if len(units) > 1:
            notes.append(f"{name}: the wells give it in different units ({', '.join(sorted(units))}); "
                         "the track compares unlike values")
        ax.set_title(f"{name} ({next(iter(units))})" if len(units) == 1 else
                     (f"{name} (mixed units)" if units else name), fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        if track == 0:
            ax.set_ylabel(f"Depth ({unit})" if unit else "Depth")
    if top is not None or bottom is not None:
        low, high = axes[0].get_ylim()
        axes[0].set_ylim(bottom if bottom is not None else max(low, high), top if top is not None else min(low, high))
    elif not axes[0].yaxis_inverted():
        axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        more_handles, more_labels = ax.get_legend_handles_labels()
        for handle, label in zip(more_handles, more_labels):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=min(len(legend_labels), 6),
                   frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return WellComparisonPlot(fig, axes, notes)
