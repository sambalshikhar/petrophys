"""
Visualization module for petrophysics log plots.
"""

from petrophys.visualization.batch_export import (
    BatchVisualizationResult,
    export_interpretation_plots,
    write_visualization_report,
)
from petrophys.visualization.log_plot import LogPlot, create_log_plot
from petrophys.visualization.log_viewer import (
    DEFAULT_LOG_VIEWER_TEMPLATES,
    LogViewer,
    LogViewerTemplate,
    create_adaptive_log_viewer_template,
    create_log_viewer,
    load_log_viewer_template,
)
from petrophys.visualization.triple_combo import TripleComboPlot, create_triple_combo
from petrophys.visualization.interpretation_plot import (
    InterpretationPlot,
    create_interpretation_plot,
)
from petrophys.visualization.crossplot_viewer import CrossPlot, create_crossplot
from petrophys.visualization.pickett_plot import PickettPlot, create_pickett_plot
from petrophys.visualization.zone_flags import plot_zone_flags, add_zone_shading
from petrophys.visualization.specialist_panels import (
    create_fmi_panel,
    create_nmr_panel,
    has_nmr_curves,
)

__all__ = [
    # Batch export
    "BatchVisualizationResult",
    "export_interpretation_plots",
    "write_visualization_report",
    # Log plots
    "LogPlot",
    "create_log_plot",
    "DEFAULT_LOG_VIEWER_TEMPLATES",
    "LogViewer",
    "LogViewerTemplate",
    "create_adaptive_log_viewer_template",
    "create_log_viewer",
    "load_log_viewer_template",
    # Triple combo
    "TripleComboPlot",
    "create_triple_combo",
    # Interpretation
    "InterpretationPlot",
    "create_interpretation_plot",
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
]
