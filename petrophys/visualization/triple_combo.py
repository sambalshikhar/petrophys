"""Triple-combo display.

The triple combo (gamma ray + caliper, resistivity, density-neutron) is the
standard field display. It is rendered by the template-driven log viewer; this
module just provides the conventional name as a shortcut.
"""

from __future__ import annotations

from typing import Optional, Tuple


def create_triple_combo(
    well,
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Standard triple-combo display: GR/caliper, resistivity, density-neutron.

    A shortcut for ``create_log_viewer(well, template_defaults="triple_combo")``.
    Returns the :class:`~petrophys.visualization.log_viewer.LogViewer` (use
    ``.fig`` / ``.save(path)``).
    """
    from petrophys.visualization.log_viewer import create_log_viewer

    return create_log_viewer(
        well,
        template_defaults="triple_combo",
        depth_range=depth_range,
        figsize=figsize,
    )
