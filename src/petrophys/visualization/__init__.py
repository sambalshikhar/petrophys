"""
Plots for wells, interpretations and specialist data.

Log viewers and templates, triple combos, crossplots (neutron-density, M-N,
MID, Pickett, Buckles, Hingle), specialist panels (NMR, borehole image, XRF),
flow-unit and saturation-height plots, zone flags, core-calibration plots and
analytics charts for multi-well and sensitivity studies. Every function returns
the Matplotlib figure (and axes) so it can be styled further or saved.
"""

from petrophys.visualization.lithology_crossplots import (
    create_nd_crossplot, create_mn_plot, create_mid_plot,
    create_buckles_plot, create_hingle_plot,
)
from petrophys.visualization.xrf_panel import create_xrf_panel
from petrophys.visualization.flow_unit_plot import create_flow_unit_plot
from petrophys.visualization.saturation_height_plot import create_saturation_height_plot
from petrophys.visualization.formation_pressure_plot import (
    create_formation_pressure_plot, create_pressure_contact_plot,
)
from petrophys.visualization.mdt_pretest_plot import create_mdt_pretest_plot


from petrophys.visualization.batch_export import (
    BatchVisualizationResult,
    export_interpretation_plots,
    write_visualization_report,
)
from petrophys.visualization.log_plot import LogPlot, create_log_plot
from petrophys.visualization.log_viewer import (
    DEFAULT_LOG_VIEWER_TEMPLATES,
    GEOLINK_LITHOLOGY_CODE_LABELS,
    LogViewer,
    LogViewerTemplate,
    create_adaptive_log_viewer_template,
    create_log_viewer,
    load_log_viewer_template,
)
from petrophys.visualization.triple_combo import create_triple_combo
from petrophys.visualization.crossplot_viewer import CrossPlot, create_crossplot
from petrophys.visualization.pickett_plot import PickettPlot, create_pickett_plot
from petrophys.visualization.zone_flags import plot_zone_flags, add_zone_shading
from petrophys.visualization.specialist_panels import (
    create_fmi_panel,
    create_nmr_panel,
    has_nmr_curves,
)
from petrophys.visualization.core import (
    plot_calibration,
    plot_core_on_logs,
    plot_poro_perm,
)
from petrophys.visualization.analytics import (
    create_multiwell_overlay,
    create_bar_comparison,
    create_tornado_plot,
    create_histogram_comparison,
    create_line_comparison,
    create_heatmap,
)

__all__ = [
    "create_buckles_plot", "create_hingle_plot", "create_xrf_panel",
    "create_flow_unit_plot", "create_saturation_height_plot",
    "create_formation_pressure_plot",
    "create_mdt_pretest_plot",
    "create_pressure_contact_plot",
    "create_nd_crossplot", "create_mn_plot", "create_mid_plot",
    # Batch export
    "BatchVisualizationResult",
    "export_interpretation_plots",
    "write_visualization_report",
    # Log plots
    "LogPlot",
    "create_log_plot",
    "DEFAULT_LOG_VIEWER_TEMPLATES",
    "GEOLINK_LITHOLOGY_CODE_LABELS",
    "LogViewer",
    "LogViewerTemplate",
    "create_adaptive_log_viewer_template",
    "create_log_viewer",
    "load_log_viewer_template",
    # Triple combo
    "create_triple_combo",
    # Crossplots
    "CrossPlot",
    "create_crossplot",
    # Pickett
    "PickettPlot",
    "create_pickett_plot",
    # Zone flags
    "plot_zone_flags",
    "add_zone_shading",
    # Specialist panels
    "create_fmi_panel",
    "create_nmr_panel",
    "has_nmr_curves",
    # Analytics charts
    "create_multiwell_overlay",
    "create_bar_comparison",
    "create_tornado_plot",
    "create_histogram_comparison",
    "create_line_comparison",
    "create_heatmap",
    # Core data
    "plot_calibration",
    "plot_poro_perm",
    "plot_core_on_logs",
]
