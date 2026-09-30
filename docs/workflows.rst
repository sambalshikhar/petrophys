Workflows
=========

Petrophys can be used one calculation at a time, but most complete-well work
starts with :func:`~petrophys.api.interpret`. It runs the dynamic workflow,
examines available curves and parameters before each stage, and records every
executed or skipped calculation in a
:class:`petrophys.workflow.DynamicWorkflowResult`.

Dynamic workflow
----------------

.. code-block:: python

   import petrophys as pp

   well = pp.read(pp.sample("30_6-5"))
   result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230, sw_model="auto")

   print(result.status)
   print([step.step for step in result.steps_executed])
   print(result.specific_steps_skipped)

Settings are :class:`~petrophys.workflow.DynamicWorkflowConfig` fields,
passed as keywords or as one config object. Invalid values raise
``ValueError`` before calculation starts. ``pp.interpret`` works on a copy
of the well. The lower-level
:func:`petrophys.workflow.run_dynamic_workflow` adds curves to the well it is
given. A :class:`petrophys.workflow.DynamicWorkflow` can be reused across
wells; each run resolves ``auto`` settings independently.

The workflow does not require every possible input. A missing prerequisite
causes the dependent step to appear in ``steps_skipped``, with its reason in
``specific_steps_skipped``. A failed executed step makes
``result.success`` false. Warnings remain separate so a usable but reviewable
interpretation can be distinguished from a failed run.

One calculation path
--------------------

There are no dataset profiles or context replay modes. Dataset names, well
names, and the presence of a vendor curve do not switch workflow versions.
Imported curves remain on the well, and each trace step identifies the
measurements, assumptions, method, and outputs that were used.

``lithology_domain`` selects only the mineral calculation family:

* ``clastic`` selects the canonical multi-mineral calculation.
* ``carbonate`` selects the COSTA carbonate calculation.
* ``auto`` resolves a supported domain from attached or embedded geological
  evidence, then records the decision. Insufficient evidence falls back to
  clastic with a warning.

The domain is not the per-depth lithology. External intervals such as shale,
sandstone, limestone, dolomite, and marl are separate inputs and take
precedence over embedded or solver-generated categorical labels. Use the
standardized tables in :doc:`input_templates` to make project assumptions
reproducible.

Pipeline order
--------------

The executed path depends on the available inputs. The pipeline follows this
dependency order:

.. list-table::
   :header-rows: 1
   :widths: 18 27 55

   * - Stage
     - Steps
     - Purpose
   * - Preparation
     - Normalization, temperature, bad-hole QC
     - Create canonical aliases, convert supported units, establish
       environmental context, and identify compromised borehole intervals.
   * - Matrix and fluids
     - VCL, Rcl, Rmf, hydrocarbon correction
     - Establish clay, electrical, mud-filtrate, and optional hydrocarbon
       corrections required by later calculations.
   * - Pore system
     - Porosity, lithology, core correction
     - Calculate porosity, solve mineral volumes when the response system is
       supported, and apply compatible laboratory calibration.
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

Spectral-gamma and elemental-spectroscopy modules remain available through
their direct APIs, but they are not hidden dataset-specific branches in the
context workflow.

Transparent routing
-------------------

Each executed :class:`petrophys.workflow.StepResult` records its method,
input and output curves, parameters, decisions, warnings, and message.

.. code-block:: python

   result.print_trace()

   trace_table = result.to_dataframe()
   trace_json = result.to_trace_dict(
       include_curves=["VCL", "PHIE", "SW", "PERM", "NET_PAY"]
   )

Net-pay curves are cutoff-screening results, not confirmation of hydrocarbons.
Review Rw and its reference temperature, clay-volume endpoints, porosity
calibration, borehole quality, and project-specific cutoffs. Marl is a mixed
clay-carbonate rock and requires project calibration; a categorical label is
not a calibrated mineral model.

Assumptions remain explicit
---------------------------

The workflow accepts environmental, matrix, saturation, permeability, and
cutoff assumptions through
:class:`petrophys.workflow.DynamicWorkflowConfig`. Formation-water
resistivity is resolved from the supplied setting, LAS metadata, salinity, or
supported log evidence. The trace records the selected source.

.. code-block:: python

   config = pp.DynamicWorkflowConfig(
       bht_f=220.0,
       td_ft=8500.0,
       rw=0.05,
       rw_meas_temp_f=180.0,
       salinity_ppm=25000.0,
       lithology_domain="clastic",
       matrix_type="sandstone",
       a=1.0,
       m=2.0,
       n=2.0,
       gr_min=20.0,
       gr_max=100.0,
       vcl_cutoff=0.30,
       phie_cutoff=0.08,
       sw_cutoff=0.70,
   )

The well-parameter template uses the friendlier headers ``gr_clean`` and
``gr_shale``; the reader maps them to ``gr_min`` and ``gr_max``.
The shared defaults are 20 and 100 for all lithology domains.

Mineral calculations
--------------------

The clastic route requires RHOB, NPHI/PHIN, and VCL. It uses the canonical
multi-mineral equation set and closure gate; PEF or DTC can add response
information when available.

The carbonate route follows the COSTA method. It requires RHOB, GR, and
PHIT/PHIE. Bulk illite is calculated as
``-0.0424035 + 0.00562239 * GR``, followed by a bounded
calcite-dolomite density and closure solve.

.. code-block:: python

   config = pp.DynamicWorkflowConfig(
       lithology_domain="carbonate",
       matrix_type="auto",
   )
   result = pp.run_dynamic_workflow(well, config)

External lithology can supply the categorical rock log while the selected
solver still calculates mineral volumes from log responses. Without the
required responses, mineral solving is skipped and the reason is retained.
There are no well-name or dataset-name rules.

The lower-level ``LithologyModule`` remains available for experimental
mineral systems and direct equation studies. Those choices do not create a
second context-generation workflow. Rank-deficient systems and zero solutions
with unallocated rock volume are unsolved: mineral values are NaN and QC is
flagged. Closure alone does not prove that a mineral solution is geologically
correct.

Zoned and batch execution
-------------------------

Use :func:`petrophys.workflow.run_dynamic_workflow_with_zones` when LAS
header parameters or project intervals provide depth-varying overrides. For
folders of wells,
:func:`petrophys.workflow.process_las_directory_with_workflow` combines
input-aware execution, LAS output, traces, and batch reports.

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
``petrophys.interpretation`` and ``petrophys.evaluation`` remain available
for experiments that need direct control over equations.

The fixed-configuration :class:`petrophys.workflow.StandardWorkflow` remains
available for compatibility but is deprecated for new applications.
