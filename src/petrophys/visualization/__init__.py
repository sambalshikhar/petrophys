"""
Plots for wells, interpretations and specialist data.

``crossplot`` draws any two curves, with the chart their pair calls for
(density-neutron, density-PEF, M-N, MID, Pickett, Hingle, Buckles, flow units,
core poro-perm). Beside it: the log viewer and its templates, the triple combo,
specialist panels (NMR, borehole image, XRF), saturation-height and pressure
plots, zone flags, core-calibration plots and charts for multi-well and
sensitivity studies. Every plot keeps its Matplotlib figure, so it can be
styled further or saved.
"""

from petrophys.visualization.style import plot_style, use_plot_style, save_figure
from petrophys.visualization.xrf_panel import xrf_panel
from petrophys.visualization.saturation_height_plot import saturation_height_plot
from petrophys.visualization.formation_pressure_plot import contact_plot, pressure_plot
from petrophys.visualization.mdt_pretest_plot import pretest_plot
from petrophys.visualization.batch_export import (
    BatchVisualizationResult,
    export_plots,
    write_visualization_report,
)
from petrophys.visualization.log_viewer import (
    DEFAULT_LOG_VIEWER_TEMPLATES,
    GEOLINK_LITHOLOGY_CODE_LABELS,
    LITHOLOGY_CODE_LABELS,
    LogViewer,
    LogViewerTemplate,
    auto_template,
    load_template,
)
from petrophys.visualization.triple_combo import triple_combo
from petrophys.visualization.crossplot_viewer import CrossPlot, crossplot
from petrophys.visualization.pickett_plot import PickettPlot
from petrophys.visualization.zone_flags import add_zone_shading, zone_flags_plot
from petrophys.visualization.specialist_panels import fmi_panel, has_nmr_curves, nmr_panel
from petrophys.visualization.core import calibration_plot, core_on_logs
from petrophys.visualization.multi_well_plot import MultiWellPlot, WellComparisonPlot, compare_wells
from petrophys.visualization.analytics import (
    bar_chart,
    heatmap,
    histograms,
    line_chart,
    multiwell_overlay,
    tornado_plot,
)

__all__ = [
    # Any two curves
    "crossplot",
    "CrossPlot",
    "PickettPlot",
    # Log plots
    "LogViewer",
    "LogViewerTemplate",
    "auto_template",
    "load_template",
    "triple_combo",
    "DEFAULT_LOG_VIEWER_TEMPLATES",
    "GEOLINK_LITHOLOGY_CODE_LABELS",
    "LITHOLOGY_CODE_LABELS",
    # Specialist panels
    "nmr_panel",
    "fmi_panel",
    "xrf_panel",
    "has_nmr_curves",
    # Saturation height and pressures
    "saturation_height_plot",
    "pressure_plot",
    "contact_plot",
    "pretest_plot",
    # Core data
    "calibration_plot",
    "core_on_logs",
    # Zone flags
    "zone_flags_plot",
    "add_zone_shading",
    # Several wells
    "MultiWellPlot",
    "WellComparisonPlot",
    "compare_wells",
    # Charts
    "bar_chart",
    "line_chart",
    "histograms",
    "heatmap",
    "tornado_plot",
    "multiwell_overlay",
    # Style and export
    "plot_style",
    "use_plot_style",
    "save_figure",
    "BatchVisualizationResult",
    "export_plots",
    "write_visualization_report",
]
