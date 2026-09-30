Advanced Workflows
==================

Move from individual equations to operational interpretation workflows. These
lessons show how Petrophys selects methods, works across intervals and wells,
combines conventional and specialist measurements, and records enough context
to review the result.

**Recommended preparation:** complete the :doc:`foundations` path or be
comfortable with conventional porosity, saturation, permeability, and
formation-evaluation terminology.

Optional dependencies
---------------------

The workflow, formation, visualization, and context lessons need only the
base install. Specialist lessons identify their own inputs and
use these package extras where needed:

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Extra
     - Install command
     - Used for
   * - DLIS
     - ``pip install "petrophys[dlis]"``
     - Native DLIS frames and borehole-image channels (lesson 16)
   * - Machine learning
     - ``pip install "petrophys[ml]"``
     - K-means electrofacies (lesson 12) and pretrained curve-imputation
       models (lesson 13); requires Python 3.11 or later
   * - MCP server
     - ``pip install "petrophys[mcp]"``
     - The Petrophys MCP server for LLM agents (lesson 20); requires Python
       3.10 or later

Course outline
--------------

.. list-table::
   :header-rows: 1
   :widths: 9 32 59

   * - Lesson
     - Topic
     - Work completed
   * - 10
     - :doc:`Dynamic workflow glass box <part_b/10_dynamic_workflow_glass_box>`
     - Inspect routing, skip reasons, parameters, and warnings, then measure a
       controlled net-pay sensitivity over the reservoir interval.
   * - 11
     - :doc:`Formation and interval analysis <part_b/11_formation_and_interval_analysis>`
     - Summarize pay per formation, keep overburden flags out of reservoir
       totals, and shade formations on depth plots.
   * - 12
     - :doc:`Core calibration and digital core <part_b/12_core_calibration_and_digital_core>`
     - Tie core plugs to logs, fit porosity and poro-perm transforms, and
       cluster wireline electrofacies over the cored interval.
   * - 13
     - :doc:`ML curve imputation <part_b/13_ml_curve_imputation>`
     - Fill gaps in existing curves with pretrained models, score them against
       withheld data, and record provenance.
   * - 14
     - :doc:`NMR petrophysics <part_b/14_nmr_petrophysics>`
     - Interpret a processed CMR summary log (no T2 bins): bound and free
       fluid, Swirr, and Timur-Coates permeability checked against the vendor.
   * - 15
     - :doc:`Saturation height <part_b/15_capillary_pressure_saturation_height>`
     - Build capillary-pressure, Leverett-J, and saturation-height relationships
       on a carbonate reservoir interval.
   * - 16
     - :doc:`Borehole images and FMI <part_b/16_borehole_image_fmi>`
     - Read image arrays, orient and equalize data, assess coverage, and render images (needs the ``dlis`` extra and the Git LFS image file of a source checkout).
   * - 17
     - :doc:`Visualization masterclass <part_b/17_visualization_masterclass>`
     - Compose interpretation panels, specialist displays, crossplots, and analytics.
   * - 18
     - :doc:`Multiwell field study <part_b/18_multiwell_field_study>`
     - Run comparable interpretations across wells and screen field-scale results.
   * - 19
     - :doc:`LLM context generation <part_b/19_llm_context_generation>`
     - Build compact, auditable JSON context from interpreted wells for any
       LLM client or review workflow.
   * - 20
     - :doc:`Petrophys for LLM agents (MCP) <part_b/20_mcp_server>`
     - Let LLM agents load wells, interpret, summarize zones, plot, and export
       through the Petrophys MCP server.

.. toctree::
   :hidden:
   :maxdepth: 1

   Dynamic workflow glass box <part_b/10_dynamic_workflow_glass_box>
   Formation and interval analysis <part_b/11_formation_and_interval_analysis>
   Core calibration and digital core <part_b/12_core_calibration_and_digital_core>
   ML curve imputation <part_b/13_ml_curve_imputation>
   NMR petrophysics <part_b/14_nmr_petrophysics>
   Saturation height <part_b/15_capillary_pressure_saturation_height>
   Borehole images and FMI <part_b/16_borehole_image_fmi>
   Visualization masterclass <part_b/17_visualization_masterclass>
   Multiwell field study <part_b/18_multiwell_field_study>
   LLM context generation <part_b/19_llm_context_generation>
   Petrophys for LLM agents (MCP) <part_b/20_mcp_server>
