Context Export
==============

The context generator is model-neutral. It performs local petrophysical
calculations and writes JSON; it does not call an LLM or send data over the
network. It also never reads an earlier context package, copies archived
numbers, or compares a run with reference JSON.

One well
--------

For a reproducible interpretation, provide the LAS plus the standardized
tables that are available for the well:

.. code-block:: python

   import petrophys as pp

   well = pp.read(
       "well.las",
       formations="formation_intervals.csv",  # optional
       lithology="lithology_intervals.csv",    # optional
   )
   pp.attach_core_measurements(well, "core_measurements.csv")  # optional
   settings = pp.read_well_parameters("well_parameters.csv")

   result = pp.interpret(well, **settings).raise_for_errors()
   pp.export_context(result, "well_context.json")

Only the LAS is mandatory. Blank parameter cells use a supported LAS value or
the documented workflow default. Formation, lithology, core, and codebook
tables add measured or interpreted evidence; none is fabricated when omitted.
See :doc:`input_templates` for the exact columns and unit rules.

The equivalent one-command workflow is:

.. code-block:: bash

   petrophys-context well.las well_context.json --parameters well_parameters.csv --formations formation_intervals.csv --lithology lithology_intervals.csv --core core_measurements.csv

Remove an optional flag when that input is unavailable. The command always
runs the interpretation before exporting context. Use ``--overwrite`` only to
replace an existing single-well JSON file.

``build_context(well_or_result)`` remains a low-level serializer. It summarizes
the object supplied to it and never runs hidden calculations. Pass an
interpretation result, rather than a raw well, when calculated curves and
workflow provenance are required.

One calculation engine
----------------------

There are no dataset profiles or legacy replay modes. Every run uses the same
workflow. ``lithology_domain`` only selects the appropriate mineral model:

* ``auto`` derives a supported domain from supplied geological evidence.
* ``clastic`` uses the canonical multi-mineral route.
* ``carbonate`` uses the COSTA carbonate route for illite, calcite, and
  dolomite.

This setting describes the interpretation domain; it is not a per-depth rock
label. Detailed limestone, dolomite, marl, sandstone, and shale intervals
belong in ``lithology_intervals.csv``. The standard gamma-ray endpoints are
``gr_clean=20`` and ``gr_shale=100`` for every domain unless the parameter
table explicitly changes them.

Formation and lithology packages
--------------------------------

A directory of LAS files can be interpreted in one call:

.. code-block:: python

   records = pp.export_context(
       "logs",
       "outputs/context_run_01",
       layout="category_formation_lithology",
       parameters_file="well_parameters.csv",
       tops_file="formation_intervals.csv",       # optional
       lithology_file="lithology_intervals.csv",  # optional
       core_file="core_measurements.csv",         # optional
       lithology_mapping_file="lithology_codes.csv",  # optional
       tops_options={"rank": "formation"},
   )

Available package layouts are ``formation_lithology`` and
``category_formation_lithology``. Package exports read each LAS, select its
rows from the standardized tables, run the same workflow, group the computed
results, and write direct context JSON documents. They do not wrap category
documents in a ``payload`` object.

If formations are absent, the logged interval is analyzed as ``WHOLE_WELL``.
If detailed lithology is absent, the workflow uses embedded evidence or its
mineral solver. If core is absent, core-dependent calibration is skipped and
recorded in the workflow trace. A codebook is needed only for coded lithology
whose labels are not already known.

``minimum_group_thickness`` defaults to ``2.0`` in the LAS depth unit. This is
the historical grouping rule retained for deterministic output. It therefore
means 2 m for a metric LAS and 2 ft for a foot-based LAS; change it explicitly
when a project requires one physical threshold across mixed units.

The command-line package form is:

.. code-block:: bash

   petrophys-context logs outputs/context_run_01 --layout category_formation_lithology --parameters well_parameters.csv --formations formation_intervals.csv --lithology lithology_intervals.csv --core core_measurements.csv --lithology-codes lithology_codes.csv --tops-rank formation

Use ``--well`` repeatedly to select wells, ``--recursive`` for nested LAS
directories, and ``--min-thickness`` to override the grouping threshold.
``python -m petrophys.reporting.context_cli`` is equivalent.

Input matching
--------------

CSV, TSV, Excel, and pandas DataFrames are accepted by the standardized
readers. Multiwell rows are matched to the well name read from the LAS header,
with common separator normalization (for example, ``30/6-5`` and
``30_6-5`` match). Duplicate or missing matches raise clear errors rather than
silently selecting a row.

Formation inputs accept the canonical templates and NPD/SODIR columns
``Well identifier`` / ``Surface`` / ``MD``. Use
``tops_options={"sheet_name": "Tops", "rank": "formation"}`` for a
multi-sheet or hierarchical workbook. Depth references and datums must match
the well; the importer can convert metres and feet, but it does not invent an
MD-to-TVD or datum transformation.

``lithology_mapping_file`` accepts canonical ``code,lithology`` or Kingdom
``Abbreviation,Attribute`` columns. ``Attribute`` is preferred over
``Lithology Attribute`` when both exist; disagreement generates a warning.

Schema and size contracts
-------------------------

Single-well JSON uses schema ``1.0`` and contains ``schema_version``,
``petrophys_version``, ``context_kind``, ``well``, ``interpretation``, and
``geology``. Raw depth-by-depth arrays are excluded. Missing numeric values
serialize as JSON ``null``, never NaN.

``max_bytes`` defaults to 1,000,000 UTF-8 bytes per JSON document. Packages
also enforce ``max_total_bytes=50_000_000``, including ancillary files.
Oversized outputs raise ``pp.ContextSizeError``; nothing is silently
truncated. Existing package directories are never overwritten. A package is
staged and validated before publication, so a failed run does not leave a
partial destination.
