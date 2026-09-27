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

The workflow, formation, visualization, and context lessons run with the base
documentation environment. Specialist lessons identify their own inputs and
use these package extras where needed:

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Extra
     - Install command
     - Used for
   * - DLIS
     - ``pip install "petrophys[dlis]"``
     - Native DLIS frames, NMR arrays, and borehole-image channels
   * - Machine learning
     - ``pip install "petrophys[ml]"``
     - Classical curve-imputation models and training utilities

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
     - Inspect routing, prerequisites, parameters, warnings, and execution states.
   * - 11
     - :doc:`Formation and interval analysis <part_b/11_formation_and_interval_analysis>`
     - Apply tops and zones, then build depth-aware interval summaries and plots.
   * - 12
     - :doc:`Core calibration and digital core <part_b/12_core_calibration_and_digital_core>`
     - Compare core and logs, calibrate properties, and inspect digital facies and mechanics.
   * - 13
     - :doc:`ML curve imputation <part_b/13_ml_curve_imputation>`
     - Predict missing samples while preserving measured values and recording provenance.
   * - 14
     - :doc:`NMR petrophysics <part_b/14_nmr_petrophysics>`
     - Interpret processed T2 distributions, fluid volumes, and NMR permeability.
   * - 15
     - :doc:`Saturation height <part_b/15_capillary_pressure_saturation_height>`
     - Build capillary-pressure, Leverett-J, and saturation-height relationships.
   * - 16
     - :doc:`Borehole images and FMI <part_b/16_borehole_image_fmi>`
     - Read image arrays, orient and equalize data, assess coverage, and render images.
   * - 17
     - :doc:`Visualization masterclass <part_b/17_visualization_masterclass>`
     - Compose interpretation panels, specialist displays, crossplots, and analytics.
   * - 18
     - :doc:`Multiwell field study <part_b/18_multiwell_field_study>`
     - Run comparable interpretations across wells and screen field-scale results.
   * - 19
     - :doc:`LLM context generation <part_b/19_llm_context_generation>`
     - Export portable, structured context from interpreted LAS data.
   * - 20
     - :doc:`Petrophys for LLM agents (MCP) <part_b/20_mcp_server>`
     - Drive loading, interpretation, zone summaries and plots from an MCP client.

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
