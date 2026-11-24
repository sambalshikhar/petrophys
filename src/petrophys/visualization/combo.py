"""Triple-combo display.

The triple combo (gamma ray + caliper, resistivity, density-neutron) is the
standard field display. It is rendered by the template-driven log viewer; this
module just provides the conventional name as a shortcut.
"""

from __future__ import annotations

from petrophys.visualization.style import _styled_plot

from typing import Optional, Tuple


@_styled_plot
def triple_combo(
    well,
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Standard triple-combo display: GR/caliper, resistivity, density-neutron.

    The same as ``pp.plot(well, template="triple_combo")``.
    Returns the :class:`~petrophys.visualization.log_viewer.LogViewer` (use
    ``.fig`` / ``.save(path)``).
    """
    from petrophys.visualization.log_viewer import _log_viewer

    return _log_viewer(
        well,
        template_defaults="triple_combo",
        depth_range=depth_range,
        figsize=figsize,
    )
