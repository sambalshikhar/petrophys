``import petrophys as pp``
==========================

The top-level ``petrophys`` package holds the functions that do a whole job
in one call, the sample data, the well and the interpretation settings.
Everything else is imported from its package. Each name links to its full
documentation.

Read, interpret, plot
---------------------

.. list-table::
   :widths: 38 62

   * - :func:`pp.read <petrophys.api.read>`
     - Read a LAS, DLIS, CSV or TSV well and optionally attach formations and
       lithology
   * - :func:`pp.interpret <petrophys.api.interpret>`
     - Run the full interpretation; returns a
       :class:`~petrophys.workflow.DynamicWorkflowResult`
   * - :func:`pp.plot <petrophys.api.plot>`
     - Multi-track log plot of a well, a result or a well file
   * - :class:`pp.DynamicWorkflowConfig <petrophys.workflow.DynamicWorkflowConfig>`
     - All interpretation settings; see :doc:`settings`

Inputs
------

.. list-table::
   :widths: 38 62

   * - :func:`pp.sample <petrophys.datasets.sample>`,
       :func:`pp.list_samples <petrophys.datasets.list_samples>`
     - Paths of the bundled sample wells and input templates
   * - :func:`pp.read_formations <petrophys.io.read_formations>`
     - Formation tops or intervals from CSV, Excel or a DataFrame
   * - :class:`pp.Well <petrophys.model.Well>`
     - A well: curves on a shared depth index, header and parameters

Context for LLMs and other tools
--------------------------------

.. list-table::
   :widths: 38 62

   * - :func:`pp.export_context <petrophys.reporting.context.export_context>`
     - Compact JSON-ready summary of a well or result, returned as a dict or
       written to a file, or a multi-well context package written to disk

Curve imputation
----------------

Requires the ``ml`` extra.

.. list-table::
   :widths: 38 62

   * - :func:`pp.impute <petrophys.preprocessing.imputation.impute>`
     - Reconstruct a curve's missing samples with a model trained on the
       well and any wells you pass; adds ``<CURVE>_IMPUTED``. Takes a well or
       an interpretation result and returns what it was given

From their packages
-------------------

.. list-table::
   :widths: 38 62

   * - ``from petrophys.workflow import`` :class:`~petrophys.workflow.DynamicWorkflowResult`
     - Interpreted well, trace, warnings, summaries and exports
   * - ``from petrophys.workflow import`` :func:`~petrophys.workflow.run_dynamic_workflow`
     - Like ``pp.interpret`` but adds curves to the well it is given
   * - ``from petrophys.workflow import`` :class:`~petrophys.workflow.custom.Workflow`
     - Your own reusable sequence of steps: check it against a well, run it
       on one well or many, save it and share it; see :doc:`../custom_workflows`
   * - ``from petrophys.workflow import`` :func:`~petrophys.workflow.custom.workflow_step`
     - Turn your own function into a workflow step
   * - :mod:`petrophys.workflow.steps`
     - Built-in steps: clay volume, porosity, Rw, saturation, permeability,
       net pay, spectral gamma, the whole of ``pp.interpret`` and more
   * - ``from petrophys.io import`` :func:`~petrophys.io.read_lithology_intervals`
     - Per-depth lithology intervals
   * - ``from petrophys.io import`` :func:`~petrophys.io.read_lithology_codes`
     - Lithology codebooks (canonical or Kingdom layout)
   * - ``from petrophys.reporting import`` :class:`~petrophys.reporting.context.ContextSizeError`
     - Raised when a context would exceed its size limit
   * - ``from petrophys.reporting import`` :func:`~petrophys.reporting.context_profiles.list_context_profiles`,
       :func:`~petrophys.reporting.context_profiles.describe_context_profile`
     - The dataset context profiles; see :doc:`../benchmark_context`
   * - ``from petrophys.model import`` :class:`~petrophys.model.Curve`
     - One log curve with mnemonic, unit and values
   * - ``from petrophys.model import`` :class:`~petrophys.model.DataType`
     - Curve categories
   * - ``from petrophys.exceptions import`` :class:`~petrophys.exceptions.DataQualityWarning`
     - Warning for unit, range and metadata problems found in the input

Classification and diagnostics
------------------------------

Each property is graded into classes by one canonical table.
:data:`CLASS_TABLES <petrophys.model.CLASS_TABLES>`, in ``petrophys.model``,
holds the tables of clay volume, porosity, permeability,
water saturation, BVW, RQI, FZI, MHI, Rxo/Rt and reservoir quality, each with
the unit of its values, the label of a sample without a value (such as
``NO_VCL_DATA``) and the source of its edges. A one-curve table is a
:class:`~petrophys.model.ClassTable` built on :class:`~petrophys.model.Bands`.
Reservoir quality is a :class:`~petrophys.model.ReservoirQualityTable`, whose
grades (:class:`~petrophys.model.ReservoirGrade`) test effective porosity,
permeability and clay volume with limits per lithology family.
:class:`~petrophys.model.TwoCurveClassTable` and
:class:`~petrophys.model.TwoCurveBands` grade from two curves; no canonical
table uses them.

.. list-table::
   :widths: 38 62

   * - ``from petrophys.model import``
       :class:`~petrophys.model.VclClassification`,
       :class:`~petrophys.model.PorosityClassification`,
       :class:`~petrophys.model.HcSaturationClassification`,
       :class:`~petrophys.model.PermeabilityClassification`,
       :class:`~petrophys.model.RQIClassification`,
       :class:`~petrophys.model.MhiClassification`,
       :class:`~petrophys.model.InvasionClassification`
     - Class enumerations. Each classifies through its canonical table, as
       the analyzers do by default; a value that is not finite gets None
   * - ``from petrophys.evaluation.diagnostics import``
       :class:`~petrophys.evaluation.diagnostics.HydrocarbonAnalysis`,
       :class:`~petrophys.evaluation.diagnostics.LithologyAnalysis`,
       :class:`~petrophys.evaluation.diagnostics.PorosityAnalysis`,
       :class:`~petrophys.evaluation.diagnostics.PermeabilityAnalysis`,
       :class:`~petrophys.evaluation.diagnostics.SaturationAnalysis`
     - Category analyses of an interpreted well. ``class_tables="current"``
       (default) classifies with the canonical tables; ``"geolink"`` keeps
       the tables and texts of the GeoLink eval package. All but the
       saturation analysis grade reservoir quality with the limits of
       ``lithology_family`` (``"clastic"``, ``"carbonate"`` or a lithology,
       for every sample or one per sample); without it, every sample is graded
       as clastic rock
   * - ``from petrophys.evaluation.diagnostics import``
       :class:`~petrophys.evaluation.diagnostics.NumericalStatisticalAnalysis`
     - Numerical and statistical data-quality analysis
   * - ``from petrophys.preprocessing.quality import``
       :class:`~petrophys.preprocessing.quality.WellboreQC`
     - Bad-hole detection from caliper roughness, with DRHO as a fallback
   * - ``from petrophys.workflow import``
       :func:`~petrophys.workflow.run_comprehensive_analysis`
     - Runs all of these analyses on a well and, by default, stores the
       results on it; passes on ``class_tables`` and ``lithology_family``
