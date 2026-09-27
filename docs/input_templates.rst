Input Templates
===============

**Only a well log is required. Formations and lithology are optional.**
Without them, plotting shows available curves and interpretation follows its
normal input-aware workflow, including mineral inversion when supported.
When supplying external geology, use the templates below. These are input profiles,
not a claim that every industry's native export has the same columns.

Start with a plot
-----------------

LAS alone:

.. code-block:: python

   import petrophys as pp
   well = pp.read("my_well.las")
   pp.plot(well).show()

Attach either or both optional files for use by plotting and interpretation:

.. code-block:: python

   well = pp.read("my_well.las", formations="formations.csv", lithology="lithology.csv")
   pp.plot(well).show()

Attached external lithology takes precedence over embedded lithology in
``pp.interpret(well)`` as well. Source overrides and advanced imports are in
:doc:`own_data`. To override geology only for a single plot:

.. code-block:: python

   from petrophys.io import read_las
   from petrophys.visualization import create_log_viewer

   well = read_las("my_well.las")
   viewer = create_log_viewer(well, formation_tops="formations.csv",
                             lithology_intervals="lithology.csv")
   viewer.show()

The lithology file takes precedence over embedded lithology **for this plot
only**. It does not change the well, resample interval boundaries, fill gaps,
or select lithology-dependent interpretation parameters. The plot displays
available logs and core measurements; it does not calculate an absent ``SW``
curve. See :doc:`workflows` to calculate log-derived water saturation using
appropriate formation-water and rock parameters.

Download and fill the templates
-------------------------------

The rows are **synthetic examples**, not picks for your well. Replace all
example rows. Keep the header and use comma-delimited UTF-8 CSV with decimal
points, no merged cells, no extra title rows, and no units inside numeric cells.
Excel users can save as CSV UTF-8 or use ``.xlsx`` with the same columns.
Multiple worksheets require ``sheet_name``; multiwell tables require explicit
well selection or a matching Well. The minimal templates remain one-well files.

Formation tops (plot annotations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`formation_tops_template.csv <../src/petrophys/datasets/formation_tops_template.csv>`

.. literalinclude:: ../src/petrophys/datasets/formation_tops_template.csv
   :language: text

``formation_name`` is the stratigraphic unit or pick name; ``top_depth`` is its
depth on the same coordinate as the LAS. This table marks boundaries only.
It does not define formation bottoms or a lithology log. Pass the path as
``formation_tops=...``. No final bottom is inferred by the viewer.

Formation intervals (bounded analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`formation_intervals_template.csv <../src/petrophys/datasets/formation_intervals_template.csv>`

.. literalinclude:: ../src/petrophys/datasets/formation_intervals_template.csv
   :language: text

Add ``bottom_depth`` when interval thickness or summaries are needed. Supply
the actual last bottom; do not invent a fixed thickness. Use finite depths
with ``top_depth < bottom_depth``. Pick a single stratigraphic level for
nonoverlapping summaries, or analyze nested groups and formations separately.
An optional ``lithology`` column can describe the formation generally, but does
not replace a separate detailed lithology interval file.

.. code-block:: python

   from petrophys.model.formation_tops import FormationTops

   formations = FormationTops.from_csv("formation_intervals.csv")

The model, viewer, and formation summaries now share ``read_formations``.
Missing names, nonfinite depths, missing files, and reversed intervals raise
errors rather than disappearing. CSV, TSV, Excel and DataFrames are supported.
Top-only picks need an explicit ``terminal_depth`` to become analysis intervals;
the last bottom is never fabricated. Rank and parent columns are retained.
Coincident named tops are preserved. Summary totals use the union of covered
depth samples, so nested groups and members do not inflate total net or gross.
Adjacent analysis intervals use ``[top, bottom)``; the well's final sample is
included. Thickness retains the existing sample-step convention, not simply
``bottom - top``.

Lithology intervals (categorical rock log)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`lithology_intervals_template.csv <../src/petrophys/datasets/lithology_intervals_template.csv>`

.. literalinclude:: ../src/petrophys/datasets/lithology_intervals_template.csv
   :language: text

The canonical columns are ``top,bottom,lithology``. Each row is a bounded interval with a
rock name such as Sandstone, Shale, Limestone, Dolomite, Chalk, or Marl. Custom
rock names are retained, not silently converted to sandstone or clastic.
Names use the viewer's existing colors and patterns where available; otherwise
a fallback color is used. Related known names can share a legend category.

The reader accepts CSV, TSV, Excel, or a pandas DataFrame, sorts by top depth, and
rejects malformed headers, empty input, missing names, nonfinite depths,
zero/reversed intervals, and overlaps with an actionable error. Adjacent
intervals may share a boundary. Gaps remain unfilled and repeated rock names
are allowed. Sub-sample beds keep their true boundaries when plotted.

.. code-block:: python

   from petrophys.io import read_lithology_intervals

   lithology = read_lithology_intervals("lithology.csv")  # validate before plotting
   viewer = create_log_viewer(well, lithology_intervals=lithology)

This categorical profile does **not** accept multiple percentage components
at the same depth. Keep cuttings mixtures and mineral fractions as separate
source data. Do not silently discard secondary components to fit this template.
The older ``interpretation.lithology.load_lithology_csv`` maps rock names into
broad calculation families; use the reader above for label-preserving plots.

Optional codebook (already coded LAS lithology)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`lithology_codes_template.csv <../src/petrophys/datasets/lithology_codes_template.csv>`

Use this only when the LAS already contains a recognized coded lithology curve
and you need its code-to-name dictionary. Codes are vendor-specific: the sample
assignments are illustrative, not a universal standard. This is **not** an
interval file and should not be passed as ``lithology_intervals``.

.. code-block:: python

   import pandas as pd

   viewer = create_log_viewer(
       well, lithology_mapping_df=pd.read_csv("lithology_codes.csv")
   )

The files also ship with the installed package. Locate them with
``petrophys.sample("lithology_intervals_template")`` or
``petrophys.sample("formation_tops_template")``. Work on your own copies;
do not edit the installed package's examples.

Depth and source checklist
--------------------------

When asking a colleague or data provider for inputs, request:

* The well/wellbore identifier and the exact LAS used for the interpretation.
* Depth units (metres or feet), coordinate (MD, TVD, or TVDSS), reference datum
  (for example RKB, ground level, or mean sea level), and any core-to-log shift.
* Formation tops or explicit top/base intervals, the stratigraphic rank, and
  the source/version of the picks.
* Lithology intervals plus any codebook, whether they come from core, cuttings,
  or an interpretation, and whether they describe categories or mixtures.
* For core saturation, fluid type, units (percent or fraction), and the lab
  measurement basis. A unit correction does not make measured core saturation
  equivalent to in-situ log-derived water saturation.

**Verify depth alignment.** Optional ``well``, ``depth_unit``,
``depth_reference`` and ``datum`` columns are accepted by both shared readers.
Equivalent arguments can declare this metadata for a whole table. Compatible
metres/feet conversions are automatic when ``well`` is supplied; MD/TVD or
datum changes require a real transformation and are rejected. Set the well's
reference/datum explicitly before joining declared geology. Missing metadata
produces recorded assumptions in ``table.attrs["import_report"]``, not a claim
of verified alignment. Preserve original exports for traceability.

Real industry practice and mapping
----------------------------------

There is no single universal CSV layout. These design choices are based on
documented industry workflows, reviewed on 2026-09-25:

* `Geocap well-top import <https://geocap.atlassian.net/wiki/spaces/ug/pages/22479309/Well%2BTops>`_
  uses a tabular ASCII/CSV exchange with well identity, marker name and measured
  depth. Match the well explicitly, then map the marker and depth columns.
* `RockWorks lithology entry <https://help.rockware.com/rockworks17/WebHelp/data_bh_lith.htm>`_
  records measured top/base depths and a material keyword linked to a lithology
  dictionary. Its depth units and collar/ground datum are explicit project
  choices. This motivates bounded intervals and a separate optional codebook.
* `Energistics WITSML lithology usage guide <https://energistics.org/sites/default/files/2023-03/WITSML_Lithology_Object_Usage_Guide_1.1Revisions%20Version%202.pdf>`_
  shows richer XML geological intervals with depth units, stratigraphic ranks,
  cuttings versus interpreted lithology, and percentage mixtures. The examples
  include work contributed by Geologix and Petrolink. These CSV templates are
  a deliberately narrower profile, **not a WITSML importer**.
* `Norwegian Offshore Directorate: 30/6-5 <https://factpages.sodir.no/en/wellbore/PageView/Exploration/By/ByCompletionDate/118>`_
  is a real example for the bundled LAS's well. Its lithostratigraphic table
  states mMD RKB and has coincident group and formation tops, including Brent
  Group and Ness Formation at 2841 m. Preserve this hierarchy in the source;
  select the intended rank for a simple display or nonoverlapping analysis.

Our recommendation is to retain rich vendor exports as the source of record
and prepare the small, explicit plotting tables as derived inputs. For example,
a multiwell formation export can be reduced deliberately:

.. code-block:: python

   import pandas as pd

   source = pd.read_csv("vendor_tops.csv")
   tops = source.loc[source["Wellname"] == "30/6-5", ["Name", "MD"]]
   tops = tops.rename(columns={"Name": "formation_name", "MD": "top_depth"})
   if tops.empty:
       raise ValueError("No tops found for the selected well")
   viewer = create_log_viewer(well, formation_tops=tops)

For lithology, similarly select one well and rename ``FROM``/``TO``/``ROCK`` to
``top``/``bottom``/``lithology``, decode any numeric rock codes using the supplied
dictionary, and select those three columns. WITSML XML, vendor ASCII headers,
PDFs, and multi-component lithology are not directly accepted by this reader.

Core saturation units
---------------------

``SWC``, ``SOC`` and ``SGC`` and their ``SW_CORE``/``CORE_SW``-style aliases
are normalized to fractions during standardization and dynamic workflows.
The viewer does the same on a display copy without changing the source well.
An explicitly declared percent unit is divided by 100, even for values below 1.

For missing or incorrectly fractional units, percent is inferred only if at
least two plausible samples exceed 1, at least 80% of positive samples in
0-100 exceed 1, and their median is at least 2. This is a **heuristic**, not
proof of units. It corrects the bundled 30/6-5 ``SWC`` range from 13.7-84.6 to
0.137-0.846 despite the ``frac`` header, and handles the percent-like ``SOC``
distribution including values below 1 percent.

A single outlier or a strongly mixed scale does not rescale the whole curve.
Ambiguous out-of-range values are masked with a warning; they are not clipped
into plausible saturation. NaNs remain missing. Unlabelled percentages entirely
below 1 cannot be distinguished from fractions numerically. Confirm the units
and override on the original, unnormalized curve when necessary:

.. code-block:: python

   from petrophys.preprocessing import standardize_core_saturation

   standardize_core_saturation(well["SWC"], source_unit="%", inplace=True)
   # Use source_unit="v/v" to explicitly confirm fractions instead.

The curve records its original units, factor, reason, masked count and warnings
in ``curve.metadata.custom["core_saturation_normalization"]``. Workflow
normalization also records this evidence in its step parameters. Plot-only
diagnostics are in ``viewer.warnings``. Repeated normalization does not divide
again. ``standardize_curves(..., convert_units=False)`` opts out of the core
unit correction during preprocessing.

Reproduce the bundled well plot
-------------------------------

.. code-block:: python

   from petrophys import sample
   from petrophys.io import read_las
   from petrophys.visualization import create_log_viewer

   well = read_las(sample("30_6-5"))
   viewer = create_log_viewer(
       well, formation_tops=sample("formations_30_6-5"),
       lithology_intervals=sample("lithology_30_6-5"),
       depth_range=(2800, 3000), figsize=(18, 9),
   )
   viewer.show()

This uses only library APIs. The bundled formation/lithology files are example
interpretations, not a substitute for checking the original geological sources.
External lithology is used explicitly; core saturation points are shown in
fractions. For a continuous computed saturation curve, run an interpretation
workflow before passing its ``result.well`` to the same plot call.
