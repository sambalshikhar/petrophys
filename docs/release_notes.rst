Release Notes
=============

Version 2.0.0
-------------

Status
~~~~~~

Pending merge from ``dev`` into ``main``.

Version 2 is the agent-ready petrophysics release. It expands the library from
a calculation toolkit into a reproducible workflow environment with traceable
dynamic routing, context JSON export, category-level analysis, and richer
verification coverage.

Highlights
~~~~~~~~~~

* Added workflow JSON trace generation for LAS batches through
  ``write_workflow_json_traces_directory`` and
  ``scripts/generate_workflow_json_traces.py``.
* Added context JSON export workflows for reference packages, formation-level
  packages, lithology-grouped packages, and category-first packages.
* Added the ``petrophys.analysis`` package for category-oriented analysis:
  porosity, saturation, permeability, lithology, hydrocarbon, numerical
  statistics, and wellbore quality checks.
* Expanded ``DynamicWorkflow`` with richer step traces, input/output curve
  summaries, method decisions, parameter records, and final curve summaries.
* Added zone-aware and formation/lithology-aware batch processing helpers for
  reproducible LAS enrichment and audit output.
* Added a template-driven well log viewer and batch visualization/export
  utilities.
* Added reusable elastic property calculations and a lithology/VCL-aware
  permeability method.
* Added broader tests for routing, trace export, value range QC, standardization,
  log viewing, porosity, saturation, permeability, core correction, and analysis
  modules.

Agent and Reproducibility Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version 2 exposes the main workflow surfaces needed by an LLM agent:

* ``DynamicWorkflow``
* ``DynamicWorkflowConfig``
* ``run_dynamic_workflow``
* ``run_dynamic_workflow_with_zones``
* ``write_workflow_json_traces_directory``
* ``extract_zoned_parameters``
* ``WorkflowStep``
* ``StepResult``

For LAS files that include ``~Params`` metadata, the zone-aware workflow maps
recognized LAS parameters into ``DynamicWorkflowConfig`` fields. Examples
include ``RW``, ``RWT``, ``RMF``, ``RMFT``, ``BHT``, ``TD``, ``CLFL``, ``WSAL``,
``RBFL``, ``MUD``, ``LITH``, ``A``, ``M``, ``N``, ``RTSH``, ``CEC``, ``RWB``,
and cutoff/model parameters. This allows an agent to reproduce calculations
from the same LAS input without hand-building every configuration object.

The recommended reproducibility entry point is:

.. code-block:: python

   from pathlib import Path
   from petrophys.workflow import write_workflow_json_traces_directory

   results = write_workflow_json_traces_directory(
       input_dir=Path("geolink_las_with_params"),
       output_dir=Path("workflow_trace_output"),
       pattern="*.las",
   )

The generated ``workflow_json`` output records:

* LAS source and well name
* extracted global parameter overrides
* zone parameter intervals, when present
* input, output, and computed curve names
* executed workflow steps
* step methods, parameters, decisions, warnings, and errors
* input and output curve summaries for each step
* final curve summaries for reproduced statistics
* lithology group summaries and derived log computation notes

New Scripts
~~~~~~~~~~~

Version 2 adds these reusable CLI scripts:

* ``scripts/generate_workflow_json_traces.py``
* ``scripts/generate_context_json_by_formation_lithology.py``
* ``scripts/generate_category_context_json_by_formation_lithology.py``

These scripts are thin wrappers around public library APIs and are intended to
be reproducible examples, not one-off notebooks.

Workflow Behavior Changes
~~~~~~~~~~~~~~~~~~~~~~~~~

* The dynamic workflow now records richer trace metadata for each calculation
  step, including curve statistics and routing decisions.
* Saturation routing is more explicit and records per-model sample counts for
  auto-routed runs.
* Rw handling is stricter and more explicit about measured temperature versus
  assumed reference temperature. Exact reproduction of older trace files may
  require the same library commit or explicit ``rw``/``rw_meas_temp_f`` inputs.
* Bad-hole, VCL, porosity, saturation, permeability, and net-pay steps now expose
  more detailed parameter and QC notes in workflow JSON.
* Input ``LITHOLOGY_GEOLINK`` values are kept authoritative for lithology-aware
  routing where available.

New and Updated Public APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workflow APIs:

* ``petrophys.workflow.write_workflow_json_traces_directory``
* ``petrophys.workflow.run_dynamic_workflow_with_zones``
* ``petrophys.workflow.extract_zoned_parameters``
* ``petrophys.workflow.process_las_directory_with_workflow``

I/O and context export APIs:

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
* ``petrophys.visualization.batch_export``

Verification Guidance
~~~~~~~~~~~~~~~~~~~~~

Before merging Version 2 into ``main``, verify at minimum:

.. code-block:: bash

   pytest tests/test_workflow_trace_export.py
   pytest tests/test_dynamic_workflow_routing.py
   pytest tests/test_analysis_modules.py
   pytest tests/test_saturation_routing_logic.py
   pytest tests/test_value_range_qc.py

For end-to-end workflow trace reproduction, run:

.. code-block:: bash

   python scripts/generate_workflow_json_traces.py \
     --input-dir /path/to/geolink_las_with_params \
     --output-dir /path/to/reproduction_output

Known Follow-Up
~~~~~~~~~~~~~~~

* Add the package version and git commit hash into every generated workflow JSON
  file for stronger long-term reproducibility.
* Keep ``petrophys.workflow.workflow_api`` aligned with the current dynamic
  workflow step list, including ``CORE_CORRECTION`` and ``ROCK_QUALITY``.
* Decide whether currently unmapped LAS parameters such as ``FMT`` and ``TGRAD``
  should be mapped into configuration fields or intentionally reported as audit
  metadata only.
