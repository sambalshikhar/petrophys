Architecture
============

Petrophys is organized by processing lifecycle at the top level and by
scientific domain within interpretation and evaluation. A feature has one
implementation owner even when it is re-exported through a convenience API.

Package map
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 53 25

   * - Package
     - Responsibility
     - Typical objects
   * - ``petrophys.model``
     - Dependency-light data structures and shared domain metadata
     - ``Well``, ``Curve``, ``CoreData``, ``FormationTops``
   * - ``petrophys.io``
     - Reading and writing interchange formats without running an
       interpretation
     - LAS, DLIS, CSV, ASCII, Excel
   * - ``petrophys.preprocessing``
     - Input preparation before scientific interpretation
     - Standardization, normalization, QC, imputation
   * - ``petrophys.interpretation``
     - Deterministic calculations that transform measurements into
       petrophysical properties
     - VCL, lithology, porosity, Sw, permeability, NMR
   * - ``petrophys.evaluation``
     - Reservoir decisions, classifications, diagnostics, and aggregations
     - Net pay, RQI/FZI, flow units, summaries
   * - ``petrophys.workflow``
     - Dependency-aware sequencing and method routing
     - Dynamic, zoned, batch, quick-look workflows
   * - ``petrophys.reporting``
     - Complete delivery products and audit artifacts
     - LAS/JSON work packages, reports, mappings
   * - ``petrophys.visualization``
     - Presentation of wells, interpretations, and specialist data
     - Log panels, crossplots, image and NMR displays
   * - ``petrophys.datasets``
     - Small public samples and optional repository integration fixtures
     - ``sample()``, ``list_samples()``

Dependency direction
--------------------

Foundational packages do not depend on orchestration or plotting. Scientific
modules operate on arrays or model objects; workflows compose them; reporting
and visualization consume the resulting well and trace.

.. code-block:: text

   model
     |
     +-- io
     +-- preprocessing
     +-- interpretation
            |
            +-- evaluation
                    |
                    +-- workflow
                           |
                           +-- reporting
                           +-- visualization

Visualization can also consume a raw or partially interpreted ``Well``
directly. Shared private numerical helpers live in ``petrophys._internal`` and
are not part of the supported public API.

Interpretation domains
----------------------

``petrophys.interpretation`` is divided by physical problem rather than by
file format:

.. code-block:: text

   interpretation/
   |-- borehole_imaging/
   |-- clay_volume/
   |-- core_analysis/
   |-- electrical/
   |   |-- cementation/
   |   |-- invasion/
   |   |-- rmf/
   |   `-- rw/
   |-- geochemistry/
   |-- lithology/
   |-- nmr/
   |-- permeability/
   |-- porosity/
   |   `-- hydrocarbon_correction/
   |-- rock_physics/
   |-- saturation/
   |   |-- height/
   |   `-- sonic/
   `-- temperature/

Evaluation domains
------------------

``petrophys.evaluation`` owns products that depend on interpreted properties:

.. code-block:: text

   evaluation/
   |-- bvw/
   |-- diagnostics/
   |-- flow_units/
   |-- intervals/
   |-- mhi/
   |-- net_pay/
   |-- rock_quality/
   |-- summaries/
   `-- volumetrics/

Core execution model
--------------------

1. A reader creates a ``Well`` and source ``Curve`` objects.
2. Preprocessing establishes canonical working aliases, supported units, and
   quality evidence.
3. Interpretation functions calculate physical properties.
4. Evaluation functions derive reservoir indicators and interval products.
5. A workflow records the dependency-aware execution path.
6. Plotters and exporters consume the final ``Well`` and workflow trace.

The dynamic workflow mutates the supplied ``Well`` by attaching calculated
curves, then returns that same object in ``DynamicWorkflowResult.well``. This
keeps granular and orchestrated use consistent: downstream APIs operate on one
well model rather than a separate opaque workflow format.

Contributing
------------

The rules for where new code belongs, and how to test and document it, are in
`CONTRIBUTING.md <https://github.com/sambalshikhar/petrophys/blob/main/CONTRIBUTING.md>`_.
