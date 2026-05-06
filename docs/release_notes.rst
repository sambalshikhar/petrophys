Release Notes
=============

Version 2.0.0
-------------

Released
~~~~~~~~

Released with the ``dev`` to ``main`` merge on 2026-05-06.

Summary
~~~~~~~

Version 2.0.0 is the agent-ready petrophysics release. It moves the project
from a collection of calculation modules into a reproducible workflow and
context-generation environment. The release adds traceable dynamic routing,
LAS parameter extraction, context JSON generation, category-level analysis,
and stronger workflow verification.

Major Changes
~~~~~~~~~~~~~

Workflow and traceability
^^^^^^^^^^^^^^^^^^^^^^^^^

* Expanded ``DynamicWorkflow`` with richer step metadata, including method
  names, input curves, output curves, parameters, decisions, warnings, errors,
  and per-step curve summaries.
* Added final curve summaries to workflow JSON so curve statistics can be
  reproduced and compared across runs.
* Added ``run_dynamic_workflow_with_zones`` for LAS files with global and zoned
  parameters.
* Added ``write_workflow_json_traces_directory`` for batch generation of
  per-well ``workflow_json`` trace files.
* Added curve audit and batch report helpers for workflow output review.

LAS parameter extraction
^^^^^^^^^^^^^^^^^^^^^^^^

* Added mapping from LAS ``~Params`` metadata into ``DynamicWorkflowConfig``.
* Recognized parameters include ``RW``, ``RWT``, ``RMF``, ``RMFT``, ``RM``,
  ``RMT``, ``CLFL``, ``WSAL``, ``SAL``, ``BHT``, ``TD``, ``MUD``, ``LITH``,
  ``MATRIX``, ``A``, ``M``, ``N``, ``RTSH``, ``CEC``, ``B_COEFF``, ``RWB``,
  ``PNSH``, ``RBFL``, ``RHMA``, and related aliases.
* Unmapped LAS parameters are now retained in JSON as audit information instead
  of being silently discarded.

Context JSON export
^^^^^^^^^^^^^^^^^^^

* Added reference JSON work package export for whole-well context generation.
* Added formation-level context JSON export.
* Added formation/lithology-group context JSON export.
* Added category-first context JSON export for agent workflows.
* Added reference curve name mapping output for generated packages.

New scripts
^^^^^^^^^^^

* ``scripts/generate_workflow_json_traces.py``
* ``scripts/generate_context_json_by_formation_lithology.py``
* ``scripts/generate_category_context_json_by_formation_lithology.py``

These scripts are wrappers around public library APIs and are intended as
reproducible command-line examples.

New analysis package
^^^^^^^^^^^^^^^^^^^^

Added ``petrophys.analysis`` with:

* ``PorosityAnalysis``
* ``SaturationAnalysis``
* ``PermeabilityAnalysis``
* ``LithologyAnalysis``
* ``HydrocarbonAnalysis``
* ``NumericalStatisticalAnalysis``
* ``WellboreQC``

These classes provide category-oriented summaries for context export, review,
and LLM-agent inspection.

Core and formation support
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Documented ``petrophys.core`` as the public module for core plug data,
  core-to-log calibration, porosity correction, poro-perm transforms, Archie
  ``m`` estimation, and calibration plots.
* Documented ``petrophys.formation`` as the public module for formation tops,
  zones, sub-zones, manual intervals, and interval-level summaries.
* Kept backward-compatible formation re-exports for clay volume and lithology
  helpers while directing whole-well interpretation through the top-level
  modules and workflow APIs.

Petrophysical calculation updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Improved temperature handling and BHT/fallback gradient routing.
* Improved RMF and RW routing with explicit measured-temperature versus assumed
  reference-temperature behavior.
* Added cleaner RWA clean-wet fallback traces for RW estimation.
* Expanded saturation model routing and per-model sample count reporting.
* Added SXO calculation details and fallback reporting.
* Added lithology-aware Archie coefficients where ``LITHOLOGY_GEOLINK`` is
  available.
* Added apparent Swirr estimation and HC-mask-based permeability routing.
* Added same-well core correction metadata for porosity and permeability.
* Added reusable elastic property calculations.
* Added lithology/VCL-aware permeability support.
* Improved net-pay, BVW, MHI, rock-quality, invasion, bad-hole, and VCL trace
  output.

Visualization
^^^^^^^^^^^^^

* Added a template-driven ``LogViewer``.
* Added reusable log viewer templates.
* Added batch interpretation plot export and visualization reporting.
* Updated interpretation plotting to support the expanded Version 2 curve set.

Tests
^^^^^

Added or expanded tests for:

* alias/unit standardization
* analysis modules
* same-well core correction
* dynamic workflow routing
* log viewer behavior
* lithology/VCL permeability
* porosity calculations
* saturation routing
* value range QC
* workflow trace export

Public APIs Added or Expanded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workflow APIs:

* ``petrophys.workflow.run_dynamic_workflow_with_zones``
* ``petrophys.workflow.write_workflow_json_traces_directory``
* ``petrophys.workflow.extract_zoned_parameters``
* ``petrophys.workflow.process_las_directory_with_workflow``
* ``petrophys.workflow.write_batch_report``
* ``petrophys.workflow.write_curve_audit_report``

I/O and context APIs:

* ``petrophys.io.export_example_format_las``
* ``petrophys.io.export_example_format_directory``
* ``petrophys.io.export_example_work_package_directory``
* ``petrophys.io.export_reference_json_work_package_directory``
* ``petrophys.io.export_reference_json_work_package_by_formation``
* ``petrophys.io.export_reference_json_work_package_by_formation_lithology``
* ``petrophys.io.export_reference_json_work_package_by_category_formation_lithology``
* ``petrophys.io.write_reference_curve_name_mapping``

Analysis APIs:

* ``petrophys.analysis.PorosityAnalysis``
* ``petrophys.analysis.SaturationAnalysis``
* ``petrophys.analysis.PermeabilityAnalysis``
* ``petrophys.analysis.LithologyAnalysis``
* ``petrophys.analysis.HydrocarbonAnalysis``
* ``petrophys.analysis.NumericalStatisticalAnalysis``
* ``petrophys.analysis.WellboreQC``

Visualization APIs:

* ``petrophys.visualization.LogViewer``
* ``petrophys.visualization.LogViewerTemplate``
* ``petrophys.visualization.create_log_viewer``
* ``petrophys.visualization.load_log_viewer_template``
* ``petrophys.visualization.export_interpretation_plots``
* ``petrophys.visualization.write_visualization_report``

Migration Notes
~~~~~~~~~~~~~~~

* Use ``write_workflow_json_traces_directory`` for reproducible batch trace
  generation instead of custom scripts that directly call workflow internals.
* Use ``run_dynamic_workflow_with_zones`` when LAS files contain global or zoned
  ``~Params`` metadata.
* For historical trace comparisons, pin the package commit or explicitly pass
  ``rw`` and ``rw_meas_temp_f`` when older RW temperature-reference behavior is
  required.
* ``LITHOLOGY_GEOLINK`` is treated as authoritative for lithology-aware routing
  when present.
* The workflow now removes or consolidates several intermediate method curves
  before final export. Use step-level curve summaries in the JSON trace to audit
  intermediate behavior.

Agent Reproducibility
~~~~~~~~~~~~~~~~~~~~~

The recommended reproducibility path for an LLM agent is:

.. code-block:: python

   from pathlib import Path
   from petrophys.workflow import write_workflow_json_traces_directory

   results = write_workflow_json_traces_directory(
       input_dir=Path("geolink_las_with_params"),
       output_dir=Path("workflow_trace_output"),
       pattern="*.las",
   )

The generated JSON records:

* source LAS file and well name
* global parameter overrides extracted from LAS metadata
* zone metadata, when present
* input, output, and computed curve names
* workflow step order and methods
* step parameters and decisions
* per-step input/output curve summaries
* final curve summaries
* warnings, errors, and unmapped parameters

Verification
~~~~~~~~~~~~

Recommended checks for this release:

.. code-block:: bash

   pytest tests/test_workflow_trace_export.py
   pytest tests/test_dynamic_workflow_routing.py
   pytest tests/test_analysis_modules.py
   pytest tests/test_saturation_routing_logic.py
   pytest tests/test_value_range_qc.py

End-to-end trace generation:

.. code-block:: bash

   python scripts/generate_workflow_json_traces.py \
     --input-dir /path/to/geolink_las_with_params \
     --output-dir /path/to/reproduction_output

Known Follow-Up
~~~~~~~~~~~~~~~

* Add package version and git commit hash to every generated workflow JSON file.
* Keep ``petrophys.workflow.workflow_api`` synchronized with the full dynamic
  workflow step list.
* Decide whether LAS parameters such as ``FMT`` and ``TGRAD`` should become
  active configuration inputs or remain audit-only metadata.
