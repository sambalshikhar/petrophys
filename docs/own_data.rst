Your Own Data
=============

Choose the task
---------------

* View/QC: ``pp.read`` then ``pp.plot``. No interpretation required.
* Interpret: ``pp.interpret`` then inspect the result trace and calculated curves.
* Individual equations: use the modules under ``petrophys.interpretation``.
* Batch interpretation: see :doc:`workflows`.
* Computed context packages: see :doc:`context`.

Read and plot
-------------

.. code-block:: python

   import petrophys as pp
   well = pp.read("my_well.las")
   viewer = pp.plot(well)
   viewer.show()
   viewer.save("my_well.png", dpi=180)
   viewer.close()

``read`` dispatches LAS, DLIS, CSV and TSV; DLIS requires the optional ``dlis``
extra. Unsupported LAS options raise ``TypeError``.
Tabular well logs require explicit depth units:

.. code-block:: python

   well = pp.read("logs.csv", depth_column="MD", depth_unit="m")

For Excel well logs, use pandas then ``Well.from_dataframe`` with explicit
depth columns/units; Excel geology is supported directly below. For multiple
wells in one log table, use ``petrophys.io.read_csv_wells``.

Optional geology
----------------

.. code-block:: python

   well = pp.read(
       "my_well.las",
       formations="formations.csv",
       lithology="lithology.csv",
   )
   viewer = pp.plot(well, depth_range=(2800, 3000))
   print(viewer.diagnostics)

Either geology argument may be omitted. Use :doc:`input_templates` for the
downloadable schemas and industry mapping guidance. Geology can also be a
validated DataFrame. Alternate column layouts do not require editing files:

.. code-block:: python

   well = pp.read(
       "my_well.las",
       depth_reference="MD", datum="RKB",
       formations="vendor_tops.xlsx",
       formation_options={
           "sheet_name": "Tops", "well_name": "30/6-5",
           "columns": {"formation_name": "Marker", "top_depth": "Pick MD"},
           "depth_unit": "m", "depth_reference": "MD", "datum": "RKB",
       },
       lithology="rocks.txt",
       lithology_options={
           "sep": ";",
           "columns": {"top": "From MD", "bottom": "To MD", "lithology": "Rock"},
           "depth_unit": "m", "depth_reference": "MD", "datum": "RKB",
       },
   )

Mappings run from canonical name to source header. Headers otherwise match
case-insensitively with aliases such as FROM/TO/LITH and well/formation/md_top.
File well identifiers remain text, including leading zeroes. Explicit mappings
may omit vendor comment columns; omitted columns are reported.
This is not a WITSML XML, PDF, or multi-component cuttings importer.

From a repository checkout, the same plot-only journey is available as
``python examples/plot_your_well.py my_well.las --output my_well.png``.
Add ``--formations formations.csv --lithology lithology.csv`` only
when those inputs are available. Omitting the LAS path uses the bundled sample.

Interpret with visible assumptions
----------------------------------

.. code-block:: python

   result = pp.interpret(well, rw=0.07)  # illustrative Rw; use a justified value
   result.raise_for_errors()
   print(result.status)
   print(result.warnings)
   print(result.specific_steps_skipped)
   result.print_trace()
   result.plot().show()

The well is copied by default; ``inplace=True`` opts into mutation.
``DynamicWorkflowConfig`` remains supported as the second argument.
With no config, matrix and lithology resolve automatically from available
inputs. Review fluid properties, temperatures, matrix choices and cutoffs;
automation is not a substitute for petrophysical interpretation.
No default Rw is silently introduced. Missing inputs may leave steps skipped,
so check ``status`` as well as ``success``. ``raise_on_error=True`` re-raises
execution failures; ``raise_for_errors`` checks a returned result.

Lithology source choices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``lithology_source="auto"`` selects external intervals first, then a recognized
vendor log, otherwise computed lithology. Explicit choices are ``external``,
``vendor``, ``computed`` and ``config``. ``config`` requires explicit
``matrix_type`` and ``lithology``. An unavailable explicitly requested input
fails instead of silently selecting a different source.

Selection is recorded in the ``input_context`` trace step. Alternate curves
are retained. External labels remain separate from numeric codes and calculation
families. Unmapped labels and uncovered samples are reported; calculation
fallbacks are not geological picks. The mineral solver still runs without
external geology when its required logs are available. Its estimates and QC
are not equivalent to observed lithology.

Plot controls and troubleshooting
-------------------------------------

``pp.plot`` returns ``LogViewer`` with ``fig``, ``axes``, ``show``,
``save`` and ``close``. ``result.plot`` returns the same type.
Use ``depth_range=(top, bottom)`` or ``top`` with ``height`` in well depth units.
Choose ``tracks`` by display name; available names are
``[track.display_name for track in viewer.template.tracks]``.
XML templates retain detailed style and scale configuration.

``viewer.diagnostics`` reports imports, curve choices, omitted empty tracks,
and labels suppressed to avoid crowding. Coincident named tops are grouped,
not discarded. A missing saturation curve is not generated by plotting.
Measured core saturation and calculated log saturation are separate products.

Compatibility changes
---------------------

* Formation model and summary loaders no longer invent a final bottom 100 units
  after the last top. Supply ``terminal_depth`` for top-only analysis, or use
  ``pp.read_formations`` for point annotations.
* Invalid geology rows raise instead of being silently skipped.
* Summary totals count the covered depth union, not the sum of nested rows.
* Unknown reader/workflow options fail instead of being ignored.
* The old ``load_lithology_csv`` helper remains for compatibility. Prefer
  the shared validated readers for new code.
