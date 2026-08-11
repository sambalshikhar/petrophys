Visualization and Reporting
===========================

Visualization is built around the same ``Well`` object used for
interpretation. Plotters resolve canonical curve aliases, retain depth units,
and omit unsupported tracks rather than plotting fabricated data.

Adaptive log viewer
-------------------

:func:`petrophys.visualization.create_log_viewer` is the main multi-track
interface. ``template_defaults="auto"`` selects tracks from the curves that are
actually available. Named templates provide stable layouts for repeated work:
``raw``, ``triple_combo``, ``petrophysics``, ``minerals``, ``comprehensive``,
and ``geolink_full``.

.. code-block:: python

   from petrophys.visualization import create_log_viewer

   viewer = create_log_viewer(
       result.well,
       template_defaults="petrophysics",
       depth_range=(2200, 2750),
       figsize=(15, 8),
   )
   viewer.save("interpretation.png", dpi=180)
   viewer.close()

Use :func:`petrophys.visualization.create_adaptive_log_viewer_template` to
inspect the tracks selected for an unfamiliar well before rendering it. XML
templates are also supported when a project needs fixed tracks, scales,
styles, and fills.

Conventional and specialist plots
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 45 29

   * - Plot family
     - Available displays
     - Entry points
   * - Conventional logs
     - Adaptive multi-track panels, triple combo, lithology patterns, formation
       markers, and zone flags
     - ``create_log_viewer``, ``create_triple_combo``
   * - Lithology
     - Density-neutron, M-N, MID, Buckles, and Hingle plots
     - ``create_nd_crossplot``, ``create_mn_plot`` and related functions
   * - Electrical
     - Generic crossplots and Pickett plots
     - ``create_crossplot``, ``create_pickett_plot``
   * - Specialist logs
     - NMR distributions, borehole images, and XRF/element panels
     - ``create_nmr_panel``, ``create_fmi_panel``, ``create_xrf_panel``
   * - Reservoir quality
     - Flow units and saturation-height relationships
     - ``create_flow_unit_plot``, ``create_saturation_height_plot``
   * - Multiwell analytics
     - Overlays, bar comparisons, tornado plots, histograms, lines, and heatmaps
     - Functions under ``petrophys.visualization.analytics``

Batch visualization
-------------------

:func:`petrophys.visualization.export_interpretation_plots` creates a standard
set of available plot products for a collection of wells and returns a
structured result. :func:`petrophys.visualization.write_visualization_report`
records generated and skipped figures, making missing specialist panels
explicit.

Curve export
------------

The basic I/O layer writes the current well state to LAS, CSV, or Excel.

.. code-block:: python

   from petrophys.io import export_csv, export_excel, export_las

   export_las(result.well, "interpreted.las")
   export_csv(result.well, "interpreted.csv")
   export_excel(result.well, "interpreted.xlsx")

Computed LAS curves carry units and descriptions. Export preserves the well's
depth order and writes missing values using the configured LAS null value.

Auditable work packages
-----------------------

``petrophys.reporting`` builds complete delivery packages when a plain curve
file is not enough. Available exports include:

* interpreted LAS with standardized curve families and provenance;
* method and parameter reports;
* reference JSON grouped by well, category, formation, and lithology;
* curve-name mapping files for downstream consumers;
* directory-level work packages and summaries.

.. code-block:: python

   from petrophys.reporting import export_example_format_las

   export_example_format_las(
       result.well,
       "delivery/30_6-5_interpreted.las",
       set_name="Petrophys interpretation",
   )

Workflow traces and delivery exports complement one another: the LAS carries
the curves, while the trace records why each calculation ran and which inputs,
parameters, and assumptions it used.
