Quickstart
==========

Start by plotting existing measurements. Interpretation is a separate,
optional step. No formation or lithology file is required.

.. code-block:: python

   import petrophys as pp
   well = pp.read(pp.sample("30_6-5"))
   viewer = pp.plot(well, depth_range=(2800, 3000))
   viewer.show()

Replace the sample path with your LAS path. See :doc:`own_data` for optional
geology, format controls, and troubleshooting.

Load a well
-----------

.. code-block:: python

   import petrophys as pp

   well = pp.read(pp.sample("30_6-5"))

   print(well.name)
   print(well.top, well.bottom, well.depth_unit)
   print(well.curves[:12])

``pp.read`` handles LAS, DLIS, CSV and TSV. The package also includes COSTA
carbonate wells and processed NMR samples:

.. code-block:: python

   print(pp.list_samples())
   print(pp.list_samples("costa"))

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

Interpret the well
------------------

The example supplies ``Rw`` (formation water resistivity, ohm.m) as a study
assumption. Petrophys does not invent formation-water resistivity when no
supported measured or calculated route is available.

.. code-block:: python

   result = pp.interpret(well, rw=0.07)
   result.raise_for_errors()

   print("status:", result.status)
   print("matrix:", result.config.matrix_type, "| lithology:", result.config.lithology)
   print("executed:", [step.step for step in result.steps_executed])
   for step, reason in result.specific_steps_skipped.items():
       print("skipped", step, "->", reason)
   print("warnings:", result.warnings)

``pp.interpret`` accepts any :class:`~petrophys.workflow.DynamicWorkflowConfig`
setting as a keyword (``sw_model="indonesia"``, ``bht_f=220``, ...) or a whole
config object. Matrix and lithology default to ``"auto"`` and are resolved from
the data; ``result.config`` shows the values used.

Normalization runs first: canonical working aliases are added while vendor
mnemonics are kept, supported units are converted, and implausible values are
screened. When most of a curve is implausible, which usually means a wrong
unit label, :class:`petrophys.DataQualityWarning` is raised and the problem is
listed in ``result.warnings``. ``pp.interpret`` works on a copy unless
``inplace=True``.

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

Whole-well statistics mix reservoir and overburden. Summarize the reservoir
interval instead, here the Brent and Dunlin groups (2841-3223 m):

.. code-block:: python

   import numpy as np

   reservoir = (interpreted.depth >= 2841) & (interpreted.depth <= 3223)
   for name in ["VCL", "PHIE", "SW"]:
       values = interpreted.get(name).data[reservoir]
       print(name, round(float(np.nanmedian(values)), 3))

Create an interpretation plot
-----------------------------

The adaptive viewer can select tracks from the available curves. A named
template gives a stable layout for reports.

.. code-block:: python

   viewer = pp.plot(
       result,
       template="petrophysics",
       depth_range=(2800, 3000),
       figsize=(15, 8),
   )
   viewer.save("30_6-5_interpretation.png", dpi=180)
   viewer.close()

Export the well
---------------

.. code-block:: python

   result.to_las("30_6-5_interpreted.las")    # curves plus the settings used (PP_* parameters)
   result.to_json("30_6-5_trace.json")        # complete machine-readable trace
   interpreted.to_csv("30_6-5_interpreted.csv")

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
