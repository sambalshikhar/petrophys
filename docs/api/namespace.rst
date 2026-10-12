``import petrophys as pp``
==========================

Everything most users need is available directly from the top-level
``petrophys`` package. Each name links to its full documentation.

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
     - Multi-track log plot of a well or result
   * - :class:`pp.DynamicWorkflowConfig <petrophys.workflow.DynamicWorkflowConfig>`
     - All interpretation settings; see :doc:`settings`
   * - :class:`pp.DynamicWorkflowResult <petrophys.workflow.DynamicWorkflowResult>`
     - Interpreted well, trace, warnings, summaries and exports
   * - :func:`pp.run_dynamic_workflow <petrophys.workflow.run_dynamic_workflow>`
     - Like ``pp.interpret`` but adds curves to the well it is given

Custom workflows
----------------

.. list-table::
   :widths: 38 62

   * - :class:`pp.Workflow <petrophys.workflow.custom.Workflow>`
     - Your own reusable sequence of steps: check it against a well, run it
       on one well or many, save it and share it; see :doc:`../custom_workflows`
   * - :func:`pp.workflow_step <petrophys.workflow.custom.workflow_step>`
     - Turn your own function into a workflow step
   * - :mod:`petrophys.workflow.steps`
     - Built-in steps: clay volume, porosity, Rw, saturation, permeability,
       net pay, spectral gamma, the whole of ``pp.interpret`` and more

Inputs
------

.. list-table::
   :widths: 38 62

   * - :func:`pp.sample <petrophys.datasets.sample>`,
       :func:`pp.list_samples <petrophys.datasets.list_samples>`
     - Paths of the bundled sample wells and input templates
   * - :func:`pp.read_formations <petrophys.io.read_formations>`
     - Formation tops or intervals from CSV, Excel or a DataFrame
   * - :func:`pp.read_lithology_intervals <petrophys.io.read_lithology_intervals>`
     - Per-depth lithology intervals
   * - :func:`pp.read_lithology_codes <petrophys.io.read_lithology_codes>`
     - Lithology codebooks (canonical or Kingdom layout)

Context for LLMs and other tools
--------------------------------

.. list-table::
   :widths: 38 62

   * - :func:`pp.build_context <petrophys.reporting.context.build_context>`
     - Compact JSON-ready summary of a well or result
   * - :func:`pp.export_context <petrophys.reporting.context.export_context>`
     - Write that summary, or a multi-well context package, to disk
   * - :class:`pp.ContextSizeError <petrophys.reporting.context.ContextSizeError>`
     - Raised when a context would exceed its size limit

Data model
----------

.. list-table::
   :widths: 38 62

   * - :class:`pp.Well <petrophys.model.Well>`
     - A well: curves on a shared depth index, header and parameters
   * - :class:`pp.Curve <petrophys.model.Curve>`
     - One log curve with mnemonic, unit and values
   * - :class:`pp.DataType <petrophys.model.DataType>`
     - Curve categories
   * - :class:`pp.DataQualityWarning <petrophys.exceptions.DataQualityWarning>`
     - Warning for unit, range and metadata problems found in the input

Curve imputation
----------------

Requires the ``ml`` extra (Python 3.11+).

.. list-table::
   :widths: 38 62

   * - :func:`pp.impute_missing_curve <petrophys.preprocessing.imputation.impute_missing_curve>`
     - Fill missing samples of one curve with a pretrained model
   * - :func:`pp.impute_las_file <petrophys.preprocessing.imputation.impute_las_file>`
     - Impute curves in a LAS file and write the result
   * - :class:`pp.PretrainedCurveImputer <petrophys.preprocessing.imputation.PretrainedCurveImputer>`,
       :class:`pp.CurveImputationResult <petrophys.preprocessing.imputation.CurveImputationResult>`,
       :class:`pp.ModelCompatibilityError <petrophys.preprocessing.imputation.ModelCompatibilityError>`
     - The imputer, its result and its dependency-version error

Classification and diagnostics
------------------------------

Each property is graded into classes by one canonical table.
:data:`CLASS_TABLES <petrophys.model.CLASS_TABLES>`, in ``petrophys.model``
rather than ``pp``, holds the tables of clay volume, porosity, permeability,
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

   * - :class:`pp.VclClassification <petrophys.model.VclClassification>`,
       :class:`pp.PorosityClassification <petrophys.model.PorosityClassification>`,
       :class:`pp.HcSaturationClassification <petrophys.model.HcSaturationClassification>`,
       :class:`pp.PermeabilityClassification <petrophys.model.PermeabilityClassification>`,
       :class:`pp.RQIClassification <petrophys.model.RQIClassification>`,
       :class:`pp.MhiClassification <petrophys.model.MhiClassification>`,
       :class:`pp.InvasionClassification <petrophys.model.InvasionClassification>`
     - Class enumerations. Each classifies through its canonical table, as
       the analyzers do by default; a value that is not finite gets None
   * - :class:`pp.HydrocarbonAnalysis <petrophys.evaluation.diagnostics.HydrocarbonAnalysis>`,
       :class:`pp.LithologyAnalysis <petrophys.evaluation.diagnostics.LithologyAnalysis>`,
       :class:`pp.PorosityAnalysis <petrophys.evaluation.diagnostics.PorosityAnalysis>`,
       :class:`pp.PermeabilityAnalysis <petrophys.evaluation.diagnostics.PermeabilityAnalysis>`,
       :class:`pp.SaturationAnalysis <petrophys.evaluation.diagnostics.SaturationAnalysis>`
     - Category analyses of an interpreted well. ``class_tables="current"``
       (default) classifies with the canonical tables; ``"benchmark"`` keeps
       the tables and texts of the GeoLink eval package. All but the
       saturation analysis grade reservoir quality with the limits of
       ``lithology_family`` (``"clastic"``, ``"carbonate"`` or a lithology,
       for every sample or one per sample); without it, every sample is graded
       as clastic rock
   * - :class:`pp.NumericalStatisticalAnalysis <petrophys.evaluation.diagnostics.NumericalStatisticalAnalysis>`
     - Numerical and statistical data-quality analysis
   * - :class:`pp.WellboreQC <petrophys.preprocessing.quality.WellboreQC>`
     - Bad-hole detection from caliper roughness, with DRHO as a fallback
   * - :func:`pp.run_comprehensive_analysis <petrophys.workflow.run_comprehensive_analysis>`
     - Runs all of these analyses on a well and, by default, stores the
       results on it; passes on ``class_tables`` and ``lithology_family``
