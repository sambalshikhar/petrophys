"""Interpretation charts: data over the lines of a chart.

Each chart draws its points over fixed lines that carry the interpretation:
mineral lines on the lithology charts, lines of constant Sw on the Pickett and
Hingle plots, of constant bulk-volume water on the Buckles plot, of constant
FZI on the flow-unit plot, the fitted transform on the poro-perm plot. Each
takes a well or an interpretation result with its usual curves by default,
``top=`` and ``bottom=`` for the depth window, ``color=`` for the curve that
colours the points, and returns a plot that shows itself at the end of a
notebook cell and saves with ``.save(path)``.

:func:`~petrophys.visualization.crossplot` reaches the same charts by the pair
of curves named: ``crossplot(result, "PHIE", "RDEP")`` is
``pickett_plot(result)``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

#: ``color`` left at its default: each chart's own colour curve.
_DEFAULT = object()


def _window(top, bottom):
    if top is None and bottom is None:
        return None
    return (-np.inf if top is None else float(top), np.inf if bottom is None else float(bottom))


def _well(data):
    from petrophys._internal.analysis_helpers import as_well

    return as_well(data)


def _plot(figure, title=None):
    from petrophys.visualization.crossplot_viewer import CrossPlot

    plot = CrossPlot(figsize=tuple(figure.get_size_inches()))
    plot.fig = figure
    plot.ax = figure.axes[0] if figure.axes else None
    if title and plot.ax is not None:
        plot.ax.set_title(title)
    return plot


def _colour(color, default):
    return {"color_curve": default if color is _DEFAULT else color}


def _size(figsize):
    return {} if figsize is None else {"figsize": figsize}


def nd_plot(data, *, top: Optional[float] = None, bottom: Optional[float] = None,
            color: Optional[str] = _DEFAULT, title: Optional[str] = None,  # type: ignore[assignment]
            figsize: Optional[Tuple[float, float]] = None):
    """Density-neutron chart: RHOB against NPHI (limestone units), with the
    sandstone, limestone and dolomite lines ticked 0-40 % porosity.

    Points are coloured by GR by default. Gas moves points up and left of the
    lines, clay down and right.
    """
    from petrophys.visualization.lithology_crossplots import _nd_chart

    return _plot(_nd_chart(_well(data), depth_range=_window(top, bottom), **_colour(color, "GR"),
                           **_size(figsize)), title)


def pef_plot(data, *, top: Optional[float] = None, bottom: Optional[float] = None,
             color: Optional[str] = _DEFAULT, title: Optional[str] = None,  # type: ignore[assignment]
             figsize: Optional[Tuple[float, float]] = None):
    """Density-PEF chart: RHOB against PEF, with the quartz, calcite and
    dolomite lines (ticked 0-40 % porosity, mixed in U = PEF x electron
    density) and the anhydrite point. PEF beyond 8 b/e (barite) is counted.
    """
    from petrophys.visualization.lithology_crossplots import _pef_chart

    return _plot(_pef_chart(_well(data), depth_range=_window(top, bottom), **_colour(color, "GR"),
                            **_size(figsize)), title)


def mn_plot(data, *, top: Optional[float] = None, bottom: Optional[float] = None,
            color: Optional[str] = _DEFAULT, title: Optional[str] = None,  # type: ignore[assignment]
            figsize: Optional[Tuple[float, float]] = None):
    """M-N plot: M (sonic and density) against N (neutron and density), with
    the quartz-calcite-dolomite triangle; porosity cancels out."""
    from petrophys.visualization.lithology_crossplots import _mn_chart

    return _plot(_mn_chart(_well(data), depth_range=_window(top, bottom), **_colour(color, "GR"),
                           **_size(figsize)), title)


def mid_plot(data, *, porosity: Optional[str] = None, top: Optional[float] = None,
             bottom: Optional[float] = None, color: Optional[str] = _DEFAULT,  # type: ignore[assignment]
             title: Optional[str] = None, figsize: Optional[Tuple[float, float]] = None):
    """Matrix identification (MID) plot: apparent matrix density against
    apparent matrix slowness, with the sandstone, limestone and dolomite points.

    ``porosity`` names the total-porosity curve that removes the fluid (default
    PHIT, else estimated from the clean binary-mineral triangles).
    """
    from petrophys.visualization.lithology_crossplots import _mid_chart

    return _plot(_mid_chart(_well(data), depth_range=_window(top, bottom), porosity_curve=porosity,
                            **_colour(color, "GR"), **_size(figsize)), title)


def pickett_plot(data, phi: str = "PHIE", rt: Optional[str] = None, *, rw: Optional[float] = None,
                 a: float = 1.0, m: float = 2.0, n: float = 2.0, vcl_max: Optional[float] = 0.15,
                 top: Optional[float] = None, bottom: Optional[float] = None,
                 color: Optional[str] = _DEFAULT, title: Optional[str] = None,  # type: ignore[assignment]
                 figsize: Optional[Tuple[float, float]] = None, **limits: Any):
    """Pickett plot: deep resistivity against porosity on log-log axes, with
    lines of constant Sw (100, 80, 60, 40 and 20 %) of slope -m.

    ``rw`` is Rw at formation temperature; by default the median of the
    interpretation's ``RW`` curve over the plotted points. Only clean rock is
    plotted (VCL up to ``vcl_max``, 0.15 by default; ``None`` for every
    sample), the number left out noted. ``rt`` defaults to RT, else RDEP;
    ``xlim=`` and ``ylim=`` set the axes.
    """
    from petrophys.visualization.pickett import _pickett_chart

    plot = _pickett_chart(_well(data), rw=rw, phie_curve=phi, rt_curve=rt, a=a, m=m, n=n, vcl_max=vcl_max,
                          depth_range=_window(top, bottom), **_colour(color, "VCL"), **_size(figsize), **limits)
    if title and plot.ax is not None:
        plot.ax.set_title(title)
    return plot


def hingle_plot(data, phi: str = "PHIE", rt: str = "RDEP", *, rw: Optional[float] = None,
                a: float = 1.0, m: float = 2.0, n: float = 2.0, vcl_max: Optional[float] = 0.15,
                top: Optional[float] = None, bottom: Optional[float] = None,
                color: Optional[str] = _DEFAULT, title: Optional[str] = None,  # type: ignore[assignment]
                figsize: Optional[Tuple[float, float]] = None, **limits: Any):
    """Hingle plot: Rt^(-1/m) against porosity on linear axes, where lines of
    constant Sw pass through the matrix point; the water line's slope gives
    a * Rw. Rw and the clean-rock filter as :func:`pickett_plot`.
    """
    from petrophys.visualization.lithology_crossplots import _hingle_chart

    return _plot(_hingle_chart(_well(data), rw=rw, a=a, m=m, n=n, phie_curve=phi, rt_curve=rt, vcl_max=vcl_max,
                               depth_range=_window(top, bottom), **_colour(color, "VCL"), **_size(figsize),
                               **limits), title)


def buckles_plot(data, phi: str = "PHIE", sw: str = "SW", *, top: Optional[float] = None,
                 bottom: Optional[float] = None, color: Optional[str] = _DEFAULT,  # type: ignore[assignment]
                 title: Optional[str] = None, figsize: Optional[Tuple[float, float]] = None, **limits: Any):
    """Buckles plot: Sw against porosity, with hyperbolas of constant bulk-volume
    water (porosity x Sw). Points along one hyperbola can mark rock at
    irreducible water. Coloured by permeability by default.
    """
    from petrophys.visualization.lithology_crossplots import _buckles_chart

    return _plot(_buckles_chart(_well(data), phie_curve=phi, sw_curve=sw, depth_range=_window(top, bottom),
                                **_colour(color, "PERM"), **_size(figsize), **limits), title)


def flow_unit_plot(data, *, title: Optional[str] = None, figsize: Optional[Tuple[float, float]] = None,
                   **options: Any):
    """Hydraulic flow units: RQI against normalized porosity on log-log axes,
    with a line of constant FZI for each unit.

    ``data`` is a flow-unit result (``FlowUnitModule().calculate(...)``), core
    plugs (:func:`~petrophys.interpretation.core_analysis.core_data`), or a well
    or result whose core plugs are used. ``options`` go to
    :func:`~petrophys.evaluation.calculate_flow_units` (``max_units=``, ...).
    """
    from petrophys.visualization.flow_units import _flow_unit_chart

    if hasattr(data, "curves") or hasattr(data, "well"):
        from petrophys.interpretation.core_analysis import core_data

        data = core_data(data)
    figure = _flow_unit_chart(data, title=title or "Hydraulic flow units (RQI vs phi_z)",
                              figsize=figsize or (7.5, 6.5), **options)
    return _plot(figure)


_FIT = object()


def poro_perm_plot(core, transform: Any = _FIT, *, title: Optional[str] = None,
                   figsize: Optional[Tuple[float, float]] = None):
    """Core poro-perm: plug permeability (log) against plug porosity, coloured by
    depth, with a porosity-permeability transform over the plugs' range.

    ``core`` is core plugs (:func:`~petrophys.interpretation.core_analysis.core_data`),
    a well or result whose core curves are read, or a mapping of titles to
    either (or to ``(core, transform)`` pairs) for panels side by side on
    shared axes. ``transform`` defaults to the one fitted to the plugs
    (:func:`~petrophys.interpretation.core_analysis.fit_poro_perm`); ``None``
    draws none.
    """
    from collections.abc import Mapping

    from petrophys.interpretation.core_analysis import core_data, fit_poro_perm
    from petrophys.model.core import CoreData
    from petrophys.visualization.core import _poro_perm_chart

    def plugs(item):
        return item if isinstance(item, CoreData) else core_data(item)

    def fitted(item, given):
        return fit_poro_perm(item) if given is _FIT else given

    if isinstance(core, Mapping):
        panels = {}
        for name, item in core.items():
            item, given = item if isinstance(item, tuple) else (item, transform)
            item = plugs(item)
            panels[name] = (item, fitted(item, given))
        figure, _ = _poro_perm_chart(panels, figsize=figsize)
        return _plot(figure)
    core = plugs(core)
    figure, _ = _poro_perm_chart(core, transform=fitted(core, transform), title=title or "Core poro-perm",
                                 figsize=figsize)
    return _plot(figure)


def saturation_height_plot(model, j: Optional[Sequence[float]] = None, sw: Optional[Sequence[float]] = None,
                           rock_classes: Optional[Sequence[Tuple[float, float]]] = None,
                           max_height_ft: float = 300.0, figsize: Tuple[float, float] = (12.0, 5.2),
                           title: str = "Saturation-Height Model", data_label: str = "data"):
    """Saturation height: the fitted J-function over its (J, Sw) points, and Sw
    against height above the free-water level for a few rock classes.

    ``model`` is a fitted J-function
    (:func:`~petrophys.interpretation.saturation.height.fit_j_function_from_well`,
    ``fit_j_function_from_core``). ``rock_classes`` lists (permeability mD,
    porosity) pairs to draw height curves for; default a poor, moderate and
    good rock. ``data_label`` names the source of the points in the legend.
    """
    from petrophys.visualization.saturation_height import _saturation_height_chart

    return _plot(_saturation_height_chart(model, j=j, sw=sw, rock_classes=rock_classes,
                                          max_height_ft=max_height_ft, figsize=figsize, title=title,
                                          data_label=data_label))


__all__ = [
    "nd_plot",
    "pef_plot",
    "mn_plot",
    "mid_plot",
    "pickett_plot",
    "hingle_plot",
    "buckles_plot",
    "flow_unit_plot",
    "poro_perm_plot",
    "saturation_height_plot",
]
