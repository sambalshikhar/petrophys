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

Reading 30/6-5 prints two warnings about the core saturation curves ``SWC`` and
``SOC``: they are labelled as fractions but hold percentages, so Petrophys
converts them and says so (see :ref:`core-saturation-units`).

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

Formation water resistivity (``Rw``) is the input logs rarely provide. For
30/6-5 the study value is 0.07 ohm.m measured at 230 degF, the reservoir
temperature. Give the temperature too: Rw changes strongly with temperature,
and a value without one is taken to be at 75 degF.

.. code-block:: python

   result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)   # Rw in ohm.m at 230 degF
   result.raise_for_errors()

   print("status:", result.status)
   print("matrix:", result.config.matrix_type, "| lithology:", result.config.lithology_domain)
   print("executed:", [step.step for step in result.steps_executed])
   for step, reason in result.specific_steps_skipped.items():
       print("skipped", step, "->", reason)
   print("warnings:", result.warnings)

.. admonition:: Where Rw comes from

   Without an ``rw`` setting, the workflow uses the first source available:
   an ``RW`` value in the LAS header, a formation-water salinity
   (``salinity_ppm``), the SP or Rxo/Rt response with a known mud-filtrate
   resistivity, and finally the apparent Rw (Rwa) of clean, wet intervals.
   The 30/6-5 header prints ``RW = 0.7`` with no temperature, which is why the
   example passes its own value. See which source was used in the ``rw`` step
   of the trace:

   .. code-block:: python

      rw_step = next(step for step in result.steps_executed if step.step == "rw")
      print(rw_step.method, rw_step.decisions)

``pp.interpret`` accepts any :class:`~petrophys.workflow.DynamicWorkflowConfig`
setting as a keyword (``sw_model="indonesia"``, ``bht_f=220``, ...) or a whole
config object. Matrix and lithology default to ``"auto"`` and are resolved from
the data; ``result.config`` shows the values used.

Normalization runs first: canonical working aliases are added while vendor
mnemonics are kept, supported units are converted, and implausible values are
screened. When most of a curve is implausible, which usually means a wrong
unit label, :class:`~petrophys.exceptions.DataQualityWarning` is raised and the
problem is listed in ``result.warnings``. ``pp.interpret`` works on a copy
unless ``inplace=True``.

``result.status`` tells you how far the interpretation got:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Status
     - Meaning
   * - ``"interpreted"``
     - Steps ran and produced interpretation curves (porosity, saturation, ...).
   * - ``"no_interpretation"``
     - Nothing failed, but the inputs supported no interpretation step; read
       ``result.specific_steps_skipped`` to see what is missing.
   * - ``"failed"``
     - A step or an input check failed; ``result.errors`` says why, and
       ``result.raise_for_errors()`` raises it.

For a compact overview of an interval, use ``result.summary``:

.. code-block:: python

   overview = result.summary(top=2841, bottom=2958)   # the Brent interval, in metres
   print(overview["key_results"]["PHIE"]["median"], overview["net_thickness"])

Inspect the trace
-----------------

Each executed step records its method, inputs, outputs, parameters, and routing
decisions.

.. code-block:: python

   result.print_trace()

   trace = result.to_dataframe()
   print(trace[["Step", "Status", "Method", "Decisions"]])

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
