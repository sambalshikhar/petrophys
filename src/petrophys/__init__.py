"""
Petrophys - A comprehensive petrophysics library for well log analysis.

This library provides tools for:
- Curve standardization and quality control
- Clay volume (GR, SP, N-D methods) and clay resistivity
- Lithology determination (multi-mineral solver, crossplots, CSV intervals)
- Porosity calculation (density, neutron, sonic)
- Water saturation modeling (Archie, Simandoux, Indonesia, etc.)
- Permeability estimation
- Net pay determination
- Visualization and reporting
"""

from petrophys.version import __version__
from petrophys.api import read, plot, interpret
from petrophys.io import read_formations
from petrophys.reporting.context import export_context
from petrophys.model.well import Well
from petrophys.datasets import sample, list_samples
from petrophys.workflow import DynamicWorkflowConfig
from petrophys.preprocessing.imputation import impute

#: Plots reachable as ``pp.<name>``; loaded with their module on first use,
#: so ``import petrophys`` does not import Matplotlib.
_PLOTS = (
    "crossplot",
    "nmr_panel", "fmi_panel", "xrf_panel", "triple_combo",
    "saturation_height_plot", "pressure_plot", "contact_plot", "pretest_plot",
    "calibration_plot", "core_on_logs", "zone_flags_plot",
    "bar_chart", "line_chart", "histograms", "heatmap", "tornado_plot", "multiwell_overlay",
)


def __getattr__(name):
    if name in _PLOTS:
        import petrophys.visualization as visualization

        return getattr(visualization, name)
    raise AttributeError(f"module 'petrophys' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_PLOTS))


__all__ = [
    "read",
    "interpret",
    "plot",
    *_PLOTS,
    "impute",
    "export_context",
    "read_formations",
    "sample",
    "list_samples",
    "Well",
    "DynamicWorkflowConfig",
    "__version__",
]
