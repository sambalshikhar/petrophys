Architecture
============

Petrophys is organized by processing lifecycle at the top level and by
scientific domain below that. A feature has one implementation owner even when
it is re-exported from a convenience API.

Package responsibilities
------------------------

``petrophys.model``
   Dependency-light data structures such as wells, curves, core data,
   intervals, units, and metadata.

``petrophys.io``
   Reading and writing interchange formats. This layer does not perform an
   interpretation or run a workflow.

``petrophys.preprocessing``
   Standardization, normalization, input quality checks, and curve imputation.

``petrophys.interpretation``
   Scientific calculations that turn measurements into petrophysical
   properties. Domains include clay volume, lithology, porosity, saturation,
   permeability, electrical properties, NMR, borehole images, core analysis,
   geochemistry, temperature, and rock physics.

``petrophys.evaluation``
   Reservoir decisions and aggregations built from interpreted properties,
   including net pay, rock quality, flow units, fluid volumes, volumetrics,
   interval summaries, and diagnostics.

``petrophys.workflow``
   Orchestration of preprocessing, interpretation, and evaluation steps.

``petrophys.reporting``
   Delivery packages, interpretation reports, reference JSON, and audit
   outputs.

``petrophys.visualization``
   Log panels, crossplots, specialist displays, and other presentation logic.

``petrophys.datasets``
   Small datasets bundled for tests, examples, and tutorials.

Dependency direction
--------------------

Dependencies flow from foundational packages toward orchestration and
presentation::

   model
     ^
   io / preprocessing / interpretation
     ^
   evaluation
     ^
   workflow
     ^
   reporting / visualization

Foundational and scientific modules must not import workflows, reports, or
plots. Shared private numerical helpers live in ``petrophys._internal`` and are
not part of the public API.

Placement rules
---------------

* Put a new curve calculation under its scientific domain in
  ``interpretation``.
* Put a reservoir cutoff, aggregation, or decision product in ``evaluation``.
* Put file parsing and basic serialization in ``io``; put complete delivery
  products in ``reporting``.
* Put data cleaning before interpretation in ``preprocessing``.
* Put sequencing and automatic method selection in ``workflow``.
* Keep plotting out of scientific calculation modules.
* Prefer descriptive filenames such as ``calibration.py``, ``routing.py``, or
  ``calculator.py`` over new ``*_module.py`` filenames.

Compatibility
-------------

Imports from the pre-0.2 package layout remain available through thin
compatibility namespaces, including ``petrophys.base``,
``petrophys.petrophysics``, and ``petrophys.reservoir``. New code and
documentation must use the canonical paths described above.
