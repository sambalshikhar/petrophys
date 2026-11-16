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
     - K-means electrofacies (lesson 12) and curve imputation (lesson 13)
   * - MCP server
     - ``pip install "petrophys[mcp]"``
     - The Petrophys MCP server for LLM agents (lesson 20); requires Python
       3.10 or later

.. note::

   The specialist lessons work on the bundled data and say what it lacks.
   Lesson 12 uses Epslog scanner data and a matching routine-core extract.
   Lesson 14 interprets the Daydream 1 CMR summary curves, which carry no T2
   distribution, and lists the bundled Phillips NMR examples. Lesson 15 fits
   the COSTA mercury-injection capillary-pressure curves and reads the
   formation-pressure stations of 35/11-13 and 35/11-10 with
   ``read_formation_pressure``; the station files carry no fluid labels, and
   ``infer_contacts`` finds an oil/water contact in the first well and reports
   the Fensfjord interval of the second as ambiguous, without a definite FWL.
   The contacts of the well reports are used only for comparison. The HW-3
   saturation-height example fits log Sw above an assumed FWL. Collection
   READMEs list the calibration measurements and metadata each collection
   does not hold.

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
       cluster wireline electrofacies over the cored interval; read spectral
       gamma (Th/U, thorium clay indicator) on 25/5-1 and test a transform
       against a second well's core.
   * - 13
     - :doc:`ML curve imputation <part_b/13_ml_curve_imputation>`
     - Rebuild missing curve samples with ``pp.impute``, from the well itself
       or from other wells, and score the result against withheld data.
   * - 14
     - :doc:`NMR petrophysics <part_b/14_nmr_petrophysics>`
     - Interpret a processed CMR summary log (no T2 bins): bound and free
       fluid, Swirr, and Timur-Coates permeability checked against the vendor.
   * - 15
     - :doc:`Saturation height <part_b/15_capillary_pressure_saturation_height>`
     - Fit laboratory capillary-pressure curves and a Leverett J-function, find
       the free-water level from logs and from formation pressures, and predict
       saturation against height on a carbonate reservoir interval.
   * - 16
     - :doc:`Borehole images and FMI <part_b/16_borehole_image_fmi>`
     - Read image arrays, orient and equalize data, assess coverage, and render images (needs the ``dlis`` extra; the 112 MB image file is downloaded on first use).
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
   * - 21
     - :doc:`Custom workflows <part_b/21_custom_workflows>`
     - Build, check, save and share your own interpretation recipe, add your own
       steps around ``pp.interpret``, and run it across the COSTA field.

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
   Custom workflows <part_b/21_custom_workflows>
