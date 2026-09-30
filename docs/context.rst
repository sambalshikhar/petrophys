Context Export
==============

A *context* is a compact JSON summary of one interpreted well: its header,
curves and units, the settings and decisions of every workflow step, and
statistics per formation and lithology. It holds no depth-by-depth arrays,
so it is small enough to give to an LLM, a report generator or a database.

Context export runs locally. It computes every number from the files you pass
and writes JSON; it does not call an LLM or send data over the network. To let
an LLM run the interpretation itself, see :doc:`mcp`.

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

``pp.build_context(well_or_result)`` returns the same summary as a Python
dictionary without writing a file. It summarizes only what you give it and
runs no calculations, so pass an interpretation result, not a raw well, when
you want calculated curves and workflow provenance in the context.

Interpretation domain
---------------------

``lithology_domain`` selects the mineral model used by the interpretation:

* ``auto`` derives a supported domain from supplied geological evidence.
* ``clastic`` uses the canonical multi-mineral route.
* ``carbonate`` uses the COSTA carbonate route for illite, calcite, and
  dolomite.

This setting describes the interpretation domain; it is not a per-depth rock
label. Detailed limestone, dolomite, marl, sandstone, and shale intervals
belong in ``lithology_intervals.csv``. The standard gamma-ray endpoints are
``gr_clean=20`` and ``gr_shale=100`` for every domain unless the parameter
table explicitly changes them.

Many wells: context packages
----------------------------

A package interprets every LAS file in a folder and writes one context JSON
per well, formation and lithology group:

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

Groups thinner than ``minimum_group_thickness`` (default ``2.0``) are not
written. The value is in each LAS file's own depth unit, so it means 2 m for a
metric LAS and 2 ft for a foot-based LAS; set it explicitly when a project
mixes units and needs one physical threshold.

Packages interpret wells as clastic (sandstone matrix) unless the
parameters table or ``workflow_config`` says otherwise. For carbonate wells,
set ``lithology_domain`` to ``carbonate`` in the parameters table, or pass
``workflow_config=pp.DynamicWorkflowConfig(lithology_domain="carbonate")``.

.. tip::

   The COSTA sample wells are carbonates, and each LAS covers only the
   Kharaib-2 reservoir interval. Export them with
   ``tops_file=pp.sample("formations_costa")`` and the carbonate setting
   above, so their groups are labelled ``KHARAIB-2`` with limestone and
   dolomite lithology.

The command-line package form is:

.. code-block:: bash

   petrophys-context logs outputs/context_run_01 --layout category_formation_lithology --parameters well_parameters.csv --formations formation_intervals.csv --lithology lithology_intervals.csv --core core_measurements.csv --lithology-codes lithology_codes.csv --tops-rank formation

Use ``--well`` repeatedly to select wells, ``--recursive`` for nested LAS
directories, and ``--min-thickness`` to override the grouping threshold.
``python -m petrophys.reporting.context_cli`` is equivalent.

Input matching
--------------

CSV, TSV, Excel, and pandas DataFrames are accepted by the standardized
readers. In a package, each table row is matched to a LAS file by the well
name in the LAS header, ignoring separators (``30/6-5`` matches ``30_6-5``).
This gives two rules for multi-well tables:

* every table that covers more than one well needs a ``well`` column;
* the parameters table needs exactly one row for each LAS file in the
  folder. A missing or duplicate row stops the export with an error naming
  the well; use ``well_names=[...]`` (``--well`` on the command line) to export
  a subset.

A table without a ``well`` column is accepted only when the folder holds a
single LAS file.

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
