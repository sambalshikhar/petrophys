Workflows
=========

Petrophys can be used one calculation at a time, but most complete-well work
starts with :func:`petrophys.workflow.run_dynamic_workflow`. The dynamic
workflow examines the available curves and parameters before each stage,
executes applicable calculations in dependency order, and records the outcome
in a :class:`petrophys.workflow.DynamicWorkflowResult`.

Dynamic workflow
----------------

.. code-block:: python

   import petrophys
   from petrophys.io import read_las
   from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow

   well = read_las(petrophys.sample("30_6-5"))
   result = run_dynamic_workflow(
       well,
       DynamicWorkflowConfig(
           matrix_type="auto",
           lithology="auto",
           rw=0.07,
           sw_model="auto",
       ),
   )

   print(result.status)
   print([step.step for step in result.steps_executed])
   print(result.steps_skipped)

The workflow does not require every possible input. A missing prerequisite
normally causes the dependent step to be listed in ``steps_skipped``. A hard
input error or a failed executed step makes ``result.success`` false. Warnings
remain separate so a usable but reviewable interpretation can be distinguished
from a failed run.

Pipeline order
--------------

The exact executed path depends on the well. The complete pipeline is grouped
below in its current dependency order.

.. list-table::
   :header-rows: 1
   :widths: 18 27 55

   * - Stage
     - Steps
     - Purpose
   * - Preparation
     - Normalization, temperature, bad-hole QC
     - Create canonical working aliases, convert supported units, establish
       environmental context, and identify compromised borehole intervals.
   * - Specialist inputs
     - Spectral gamma, elemental spectroscopy
     - Interpret compatible specialist channels when they are present.
   * - Matrix and fluids
     - VCL, Rcl, Rmf, hydrocarbon correction
     - Establish clay, electrical, mud-filtrate, and optional hydrocarbon
       corrections required by later calculations.
   * - Pore system
     - Porosity, lithology, porosity refinement, core correction
     - Calculate available porosity families, solve mineral volumes where the
       response system is supported, refine matrix endpoints, and apply
       compatible laboratory calibration.
   * - Electrical interpretation
     - Rw, invasion, saturation
     - Resolve formation-water resistivity, diagnose invasion, and route water
       saturation through a compatible clean- or shaly-formation model.
   * - Reservoir properties
     - Permeability, rock quality, BVW, MHI
     - Estimate flow capacity and derive reservoir-quality indicators.
   * - Decisions
     - Net pay
     - Apply configurable net-rock, net-reservoir, net-pay, and quality masks.

Transparent routing
-------------------

Each executed :class:`petrophys.workflow.StepResult` records its method,
input and output curves, parameters, decisions, warnings, and message. The
result can be printed, converted to a DataFrame, or serialized as a trace
dictionary.

.. code-block:: python

   result.print_trace()

   trace_table = result.to_dataframe()
   trace_json = result.to_trace_dict(
       include_curves=["VCL", "PHIE", "SW", "PERM", "NET_PAY"]
   )

``interpretation_profile="auto"`` uses a validated vendor lithology route
when an authoritative lithology-code curve is available and otherwise uses
the generic inference route. Callers can request ``"generic_inference"``
explicitly when testing the log-derived matrix workflow. Imported interpreted
curves remain available on the well; the trace identifies the curves selected
for each calculation.

Assumptions remain explicit
---------------------------

The workflow accepts environmental, matrix, saturation, permeability, and
cutoff assumptions through :class:`petrophys.workflow.DynamicWorkflowConfig`.
Formation-water resistivity is not silently defaulted unless
``allow_default_rw=True`` is deliberately set. If no measured, header-derived,
salinity-derived, or SP-derived Rw route is supported, saturation-dependent
steps are reduced or skipped.

.. code-block:: python

   config = DynamicWorkflowConfig(
       bht_f=220.0,
       td_ft=8500.0,
       rw=0.05,
       rw_meas_temp_f=180.0,
       salinity_ppm=25000.0,
       lithology="clastic",
       matrix_type="sandstone",
       a=1.0,
       m=2.0,
       n=2.0,
       vcl_cutoff=0.35,
       phie_cutoff=0.08,
       sw_cutoff=0.60,
   )

Minerals with limited logs
--------------------------

The generic solver uses density and unity plus each available neutron, PEF, or
sonic response. Clay volume is a supplied constraint. Porosity is either solved
jointly or fixed by a supplied core-related porosity curve. A solution requires
enough independent equations for the selected unknowns.

For reduced mineral selection, ``LithologyModule(lithology="auto")`` accepts a
recognized interpreted-lithology log or core-constrained matrix-density evidence
when at least 80% of its classified samples agree on the rock family. This is a
well-level modelling constraint, not a calibrated confidence probability or a
replacement for zonation. The selected source, agreement, and mineral set appear
in the workflow trace. There are no well-name or dataset-name selection rules.

For an independently established carbonate interval, the workflow can be run as:

.. code-block:: python

   config = DynamicWorkflowConfig(
       interpretation_profile="generic_inference",
       lithology="carbonate",
       matrix_type="auto",
   )
   result = run_dynamic_workflow(well, config)

With limited responses the carbonate constraint uses calcite and dolomite;
quartz and anhydrite remain in the full system when enough equations are
available. Without sufficient geological evidence, automatic mineral selection
does not assert that a sparse log suite is carbonate. Custom mineral lists are
respected. Check matrix endpoints, fluid assumptions, and applicability over
mixed-lithology intervals before accepting the result.

Rank-deficient systems and zero solutions with unallocated rock volume are
unsolved: mineral values are NaN and QC is flagged. A finite solution may still
carry a poor-closure flag. Normalized closure alone does not demonstrate that
the minerals or their calculated log responses are geologically correct. The
historical vendor-benchmark profile uses its separate solver and is not altered
by generic mineral-set selection.

Zoned and batch execution
-------------------------

Use :func:`petrophys.workflow.run_dynamic_workflow_with_zones` when LAS header
parameters or project intervals provide depth-varying overrides. For folders
of wells, :func:`petrophys.workflow.process_las_directory_with_workflow`
combines input-aware execution, LAS output, traces, and batch reports.

.. code-block:: python

   from petrophys.workflow import process_las_directory_with_workflow

   batch = process_las_directory_with_workflow(
       input_dir="incoming_las",
       output_dir="interpreted_las",
       config=config,
   )

Selective and granular use
--------------------------

For controlled studies, ``DynamicWorkflow.run_selected_steps`` executes a
specified subset in pipeline order. Individual domain functions under
``petrophys.interpretation`` and ``petrophys.evaluation`` remain available for
experiments that need direct control over inputs and equations.

The fixed-configuration :class:`petrophys.workflow.StandardWorkflow` remains
available for compatibility but is deprecated for new applications.
