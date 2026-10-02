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

Interval classes and summary analyses used by reports and the
:func:`pp.run_comprehensive_analysis <petrophys.workflow.run_comprehensive_analysis>`
report:
:class:`~petrophys.model.VclClassification`,
:class:`~petrophys.model.PorosityClassification`,
:class:`~petrophys.model.HcSaturationClassification`,
:class:`~petrophys.model.PermeabilityClassification`,
:class:`~petrophys.model.RQIClassification`,
:class:`~petrophys.model.MhiClassification`,
:class:`~petrophys.model.InvasionClassification`,
:class:`~petrophys.evaluation.diagnostics.HydrocarbonAnalysis`,
:class:`~petrophys.evaluation.diagnostics.LithologyAnalysis`,
:class:`~petrophys.evaluation.diagnostics.PorosityAnalysis`,
:class:`~petrophys.evaluation.diagnostics.PermeabilityAnalysis`,
:class:`~petrophys.evaluation.diagnostics.SaturationAnalysis`,
:class:`~petrophys.evaluation.diagnostics.NumericalStatisticalAnalysis` and
:class:`~petrophys.preprocessing.quality.WellboreQC`.
