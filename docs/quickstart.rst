Quickstart
==========

This guide runs a traceable interpretation on a bundled public LAS well,
inspects the computed curves, creates a multi-track plot, and exports the
result.

Load a well
-----------

.. code-block:: python

   import petrophys
   from petrophys.io import read_las

   path = petrophys.sample("30_6-5")
   well = read_las(path)

   print(well.name)
   print(well.top, well.bottom, well.depth_unit)
   print(well.curves[:12])

The package also includes COSTA carbonate wells and processed NMR samples:

.. code-block:: python

   print(petrophys.list_samples())
   print(petrophys.list_samples("costa"))

Inspect curves
--------------

``well.curves`` contains mnemonic strings. ``well.get`` returns the associated
:class:`petrophys.model.Curve` or ``None`` when the curve is absent.

.. code-block:: python

   gr = well.get("GR")
   if gr is not None:
       print(gr.mnemonic, gr.unit, gr.valid_count())
       print(gr.describe())

Every scalar curve in this LAS is associated with the file's supplied depth
index. See :doc:`data and depth <data_model>` for DLIS resampling behavior and
the distinction between grid alignment and physical log-depth matching.

Run the dynamic workflow
------------------------

The example supplies ``Rw`` as a study assumption. Petrophys does not silently
invent formation-water resistivity when no supported measured or calculated
route is available.

.. code-block:: python

   from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow

   config = DynamicWorkflowConfig(
       matrix_type="auto",
       lithology="auto",
       rw=0.07,
       sw_model="auto",
   )

   result = run_dynamic_workflow(well, config)

   print("status:", result.status)
   print("executed:", [step.step for step in result.steps_executed])
   print("skipped:", result.steps_skipped)
   print("warnings:", result.warnings)

Normalization is enabled by default. The workflow adds canonical working
aliases while retaining vendor mnemonics, converts supported physical units,
and screens recognized null and range problems before routing calculations.

Inspect the trace
-----------------

Each executed step records its method, inputs, outputs, parameters, and routing
decisions.

.. code-block:: python

   result.print_trace()

   trace = result.to_dataframe()
   print(trace[["Step", "Status", "Method", "Message"]])

   payload = result.to_trace_dict(
       include_curves=["VCL", "PHIE", "SW", "PERM", "NET_PAY"]
   )
   print(payload["status"])
   print(payload["curve_summaries"])

Use calculated curves
---------------------

Calculated products are attached to ``result.well`` on the same depth index.
Only products supported by the available inputs and assumptions are present.

.. code-block:: python

   interpreted = result.well

   for name in ["VCL", "PHIT", "PHIE", "SW", "PERM", "RQI", "NET_PAY"]:
       curve = interpreted.get(name)
       if curve is not None:
           print(name, curve.unit, curve.valid_count())

Create an interpretation plot
-----------------------------

The adaptive viewer can select tracks from the available curves. A named
template gives a stable layout for reports.

.. code-block:: python

   from petrophys.visualization import create_log_viewer

   viewer = create_log_viewer(
       interpreted,
       template_defaults="petrophysics",
       depth_range=(2200, 2750),
       figsize=(15, 8),
   )
   viewer.save("30_6-5_interpretation.png", dpi=180)
   viewer.close()

Export the well
---------------

.. code-block:: python

   from petrophys.io import export_csv, export_las

   export_las(interpreted, "30_6-5_interpreted.las")
   export_csv(interpreted, "30_6-5_interpreted.csv")

For delivery packages that include calculation provenance, category mappings,
and structured JSON, continue to :doc:`visualization and reporting
<visualization_reporting>`.

Use individual calculations
---------------------------

Workflow orchestration is optional. Scientific functions remain directly
callable for controlled studies.

.. code-block:: python

   from petrophys.interpretation.clay_volume import vcl_from_gr
   from petrophys.interpretation.porosity import density_porosity
   from petrophys.interpretation.saturation import archie, simandoux

   vcl_gr = vcl_from_gr(gr.data, gr_min=15.0, gr_max=110.0)
   phid = density_porosity(rhob=2.35, rhob_matrix=2.65, rhob_fluid=1.0)
   sw_clean = archie(rt=10.0, rw=0.07, phi=0.18)
   sw_shaly = simandoux(
       rt=10.0,
       rw=0.07,
       phi=0.18,
       vcl=0.25,
       rcl=2.0,
   )

Where to go next
----------------

* :doc:`Workflows <workflows>` explains routing, traces, zones, and batch runs.
* :doc:`Capability map <capabilities>` lists the current scientific modules.
* :doc:`Specialist data <specialist_data>` covers NMR, images, geochemistry,
  core, saturation height, and sonic saturation.
* :doc:`Tutorials <tutorials/index>` provides executed notebooks and case
  studies on public datasets.
