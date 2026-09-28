Context Export
==============

Start with one well
-------------------

The public context API is model-neutral. It does not call an LLM or send
data over the network.

.. code-block:: python

   import petrophys as pp

   well = pp.read(pp.sample("30_6-5"))
   context = pp.build_context(well)
   pp.export_context(well, "well_context.json")

Raw-well context contains input summaries, units, parameters and attached
geology. It does **not** run an interpretation. To include calculations:

.. code-block:: python

   result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230).raise_for_errors()
   pp.export_context(result, "interpreted_context.json")

Rw and temperature here are illustrative, not a calibration for your well.
Formations and lithology are optional; see :doc:`input_templates` and
:doc:`data_quality`.

Schema and size contracts
-------------------------

The default schema is ``"1.0"``. Single-well documents contain
``schema_version``, ``petrophys_version``, ``context_kind="well_summary"``,
``well``, ``interpretation`` and ``geology``. Raw depth-by-depth arrays are
excluded. Missing numeric values serialize as JSON null, not NaN.

``max_bytes`` defaults to 1,000,000 UTF-8 bytes per JSON document. Packages
also enforce ``max_total_bytes=50_000_000``, including ancillary files.
Oversized outputs raise ``pp.ContextSizeError``; nothing is silently
truncated. These are serialized-output budgets, not limits on computation,
memory or temporary disk space.

Existing single files are preserved unless ``overwrite=True``. Packages
require a **new directory**. They are staged and validated before publication,
so a failed or oversized run does not leave a partial package at its destination.

Formation and lithology packages
--------------------------------

.. code-block:: python

   results = pp.export_context(
       "logs", "outputs/context_run_01",
       layout="category_formation_lithology",
       tops_file="formation_intervals.csv",
       tops_options={"rank": "formation"},
       workflow_config=pp.DynamicWorkflowConfig(
           rw=0.07, rw_meas_temp_f=230,
       ),
   )

Packages run the workflow and split outputs by formation and available
lithology. Layouts are ``formation_lithology`` and
``category_formation_lithology``. The return value is a list of export
records with ``json_file``, ``warnings`` and ``success``. A failed record
prevents publication. Versioned package documents wrap the existing
analysis structure under ``payload`` and add schema/package metadata.

Unlike single-well context, these layouts need a formation table because
their purpose is formation grouping. CSV, TSV and Excel are accepted,
including canonical templates and NPD/SODIR columns
``Well identifier`` / ``Surface`` / ``MD``. Explicit bottoms and depth units
are preserved. Use ``tops_options={"sheet_name": "Tops", "rank": "formation"}``
for multi-sheet or hierarchical files. Depth references and datums must
match the well; this API does not transform MD to TVD.

Well identifiers match the LAS filename stem or well name with common
separator normalization. A multiwell package needs a well column.
A single-well table is matched only when unambiguous.

A codebook is optional; otherwise built-in vendor labels are used.
``lithology_mapping_file="codes.xlsx"`` accepts canonical
``code,lithology`` or Kingdom ``Abbreviation,Attribute``.
``Attribute`` is preferred over ``Lithology Attribute`` when both exist;
disagreement generates a warning. For deliberate grouping, select it with
``pp.read_lithology_codes(..., label_column=...)`` and prepare a canonical
codebook containing that mapping.

Command line
------------

.. code-block:: bash

   petrophys-context well.las well_context.json
   petrophys-context well.las interpreted.json --interpret --rw 0.07 --rw-temp-f 230
   petrophys-context logs outputs/run_01 --layout formation_lithology --formations tops.csv

``python -m petrophys.reporting.context_cli`` is equivalent. Use ``--help``
for schema, byte-budget and input options.

The historical ``scripts/generate_*context_json*`` entry points now call
this API. ``--input-dir`` and ``--tops-file`` are required unless supplied
through ``PETROPHYS_DATA_DIR`` and ``PETROPHYS_TOPS_FILE``. No private
workbook is assumed. ``--tops-sheet`` and ``--tops-rank`` select sheet/rank.

Legacy compatibility
--------------------

Use ``schema_version="legacy"`` (CLI: ``--schema-version legacy``) when an
existing consumer needs the unwrapped payload. Package replay retains the
old automatic workflow profile unless a configuration is supplied.
Versioned packages default to ``generic_inference``.

Interpreted legacy single-well exports require ``original_curves`` captured
before interpretation, to distinguish measured inputs from generated curves.
``source_file`` preserves source naming.

Regression tests compare bytes against the existing exporters for a single
well and both package layouts. This is fixture-based compatibility, not a
claim that every historical well was replayed. Corrected RMF note labels
and decorated Celsius-unit parsing remain corrected. Conflicting custom
codebooks also use corrected label selection; fixes are not reverted for replay.

``--legacy-benchmark-compat`` on the historical scripts additionally enables
archived benchmark conventions. It is not recommended for new interpretations.
