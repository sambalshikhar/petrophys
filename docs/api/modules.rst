API Reference
=============

This section contains the complete API documentation for all Petrophys modules.

Core Module
-----------

The core module contains fundamental data structures and utilities.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   petrophys.base

.. automodule:: petrophys.base
   :members:
   :undoc-members:
   :show-inheritance:

Well Class
~~~~~~~~~~

.. autoclass:: petrophys.base.Well
   :members:
   :undoc-members:
   :show-inheritance:

Curve Class
~~~~~~~~~~~

.. autoclass:: petrophys.base.Curve
   :members:
   :undoc-members:
   :show-inheritance:

I/O Module
----------

File reading and writing functionality.

.. automodule:: petrophys.io
   :members:
   :undoc-members:
   :show-inheritance:

QAQC Module
-----------

Quality assurance and quality control tools.

Bad Hole Detection
~~~~~~~~~~~~~~~~~~

.. automodule:: petrophys.qaqc.bad_hole
   :members:
   :undoc-members:
   :show-inheritance:

Normalization
~~~~~~~~~~~~~

.. automodule:: petrophys.qaqc.normalization
   :members:
   :undoc-members:
   :show-inheritance:

Standardization
~~~~~~~~~~~~~~~

.. automodule:: petrophys.qaqc.standardization
   :members:
   :undoc-members:
   :show-inheritance:

Environmental Module
--------------------

Environmental corrections and temperature calculations.

Temperature
~~~~~~~~~~~

.. automodule:: petrophys.environmental.temperature
   :members:
   :undoc-members:
   :show-inheritance:

Clay Volume Module
------------------

Methods: GR (linear, Larionov, Steiber, Clavier), SP, Neutron-Density.
Final VCL is the minimum of all valid methods at each depth.

.. automodule:: petrophys.clay_volume
   :members:
   :undoc-members:
   :show-inheritance:

Clay Resistivity (Rcl)
~~~~~~~~~~~~~~~~~~~~~~

Calculates clay resistivity by averaging deep resistivity in pure-shale zones (VCL > 0.9).

.. automodule:: petrophys.clay_volume.rcl
   :members:
   :undoc-members:
   :show-inheritance:

Lithology Module
----------------

Multi-mineral solver and lithology determination from well logs.

.. automodule:: petrophys.lithology
   :members:
   :undoc-members:
   :show-inheritance:

Lithology Log (Per-Interval CSV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load per-interval lithology from a CSV file and build a per-depth lithology
array for use in saturation model routing and net pay cutoffs.

.. automodule:: petrophys.lithology.lithology_log
   :members:
   :undoc-members:
   :show-inheritance:

Resistivity Module
------------------

Resistivity calculations and analysis.

Rmf (Mud Filtrate Resistivity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: petrophys.resistivity.rmf
   :members:
   :undoc-members:
   :show-inheritance:

Rw (Formation Water Resistivity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: petrophys.resistivity.rw
   :members:
   :undoc-members:
   :show-inheritance:

Invasion
~~~~~~~~

.. automodule:: petrophys.resistivity.invasion
   :members:
   :undoc-members:
   :show-inheritance:

Petrophysics Module
-------------------

Core petrophysical calculations.

Porosity
~~~~~~~~

Calculates density, sonic, total (weighted N-D), effective (clay-corrected),
and secondary porosity (fractures/vugs from sonic comparison).

.. automodule:: petrophys.petrophysics.porosity
   :members:
   :undoc-members:
   :show-inheritance:

Saturation
~~~~~~~~~~

Five water saturation models with automatic routing:

- **Archie**: Clean formations (VCL < cutoff)
- **Simandoux**: Shaly clastics, salinity >= 20 kppm
- **Indonesia**: Shaly clastics, salinity < 20 kppm
- **Dual Water**: Shaly carbonates
- **Waxman-Smits**: Shaly clastics with CEC data

.. automodule:: petrophys.petrophysics.saturation
   :members:
   :undoc-members:
   :show-inheritance:

Dual Water Model
^^^^^^^^^^^^^^^^

.. automodule:: petrophys.petrophysics.saturation.dual_water
   :members:
   :undoc-members:
   :show-inheritance:

Waxman-Smits Model
^^^^^^^^^^^^^^^^^^

.. automodule:: petrophys.petrophysics.saturation.waxman_smits
   :members:
   :undoc-members:
   :show-inheritance:

Permeability
~~~~~~~~~~~~

Timur, Coates-Denoo, and Wyllie-Rose equations with impermeability detection.

.. automodule:: petrophys.petrophysics.permeability
   :members:
   :undoc-members:
   :show-inheritance:

Impermeability Detection
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: petrophys.petrophysics.permeability.impermeable_flag
   :members:
   :undoc-members:
   :show-inheritance:

HC Correction
~~~~~~~~~~~~~

Hydrocarbon corrections using preliminary porosity approach with VCL support.

.. automodule:: petrophys.petrophysics.hc_correction
   :members:
   :undoc-members:
   :show-inheritance:

Cementation
~~~~~~~~~~~

.. automodule:: petrophys.petrophysics.cementation
   :members:
   :undoc-members:
   :show-inheritance:

Reservoir Module
----------------

Reservoir evaluation tools.

BVW (Bulk Volume Water)
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: petrophys.reservoir.bvw
   :members:
   :undoc-members:
   :show-inheritance:

MHI (Moveable Hydrocarbon Index)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MHI = Sw / Sxo

.. automodule:: petrophys.reservoir.mhi
   :members:
   :undoc-members:
   :show-inheritance:

Net Pay
~~~~~~~

Three-stage filtering with lithology-aware cutoffs:
Net Rock (VCL <= 0.3) -> Net Reservoir (PHIE >= cutoff) -> Net Pay (Sw <= 0.7).
Carbonate PHIE cutoff: 0.045. Clastic PHIE cutoff: 0.08.

.. automodule:: petrophys.reservoir.net_pay
   :members:
   :undoc-members:
   :show-inheritance:

Rock Quality (RQI, FZI)
~~~~~~~~~~~~~~~~~~~~~~~~

Reservoir Quality Index, Flow Zone Indicator, and normalized porosity:

- RQI = 0.0314 * sqrt(k / phi_e)
- phi_z = phi_e / (1 - phi_e)
- FZI = RQI / phi_z

.. automodule:: petrophys.reservoir.rock_quality
   :members:
   :undoc-members:
   :show-inheritance:

Visualization Module
--------------------

Comprehensive plotting and visualization tools for well log analysis.

Log Plot
~~~~~~~~

Configurable multi-track log display.

.. autoclass:: petrophys.visualization.LogPlot
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: petrophys.visualization.create_log_plot

Triple Combo Plot
~~~~~~~~~~~~~~~~~

Industry-standard 3-track display: GR+Caliper | Resistivity | Density+Neutron.

.. autoclass:: petrophys.visualization.TripleComboPlot
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: petrophys.visualization.create_triple_combo

Interpretation Plot
~~~~~~~~~~~~~~~~~~~

8-track full interpretation display: GR | Resistivity | Density-Neutron |
VCL | Porosity | Sw | Permeability | Net Pay Flags.

.. autoclass:: petrophys.visualization.InterpretationPlot
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: petrophys.visualization.create_interpretation_plot

Crossplot
~~~~~~~~~

General-purpose crossplot (NPHI vs RHOB, PHIE vs Sw, etc.) with optional
mineral end-points and trend lines.

.. autoclass:: petrophys.visualization.CrossPlot
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: petrophys.visualization.create_crossplot

Pickett Plot
~~~~~~~~~~~~

Log(Rt) vs Log(PHIE) with Sw iso-saturation lines for Archie parameter
estimation and verification.

.. autoclass:: petrophys.visualization.PickettPlot
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: petrophys.visualization.create_pickett_plot

Zone Flags & Formation Shading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: petrophys.visualization.plot_zone_flags

.. autofunction:: petrophys.visualization.add_zone_shading

.. automodule:: petrophys.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Core Data Module
----------------

Load core plug measurements (porosity, permeability, grain density, Sw)
and calibrate log curves against laboratory data.

.. automodule:: petrophys.core
   :members:
   :undoc-members:
   :show-inheritance:

CoreData Container
~~~~~~~~~~~~~~~~~~

.. autoclass:: petrophys.core.CoreData
   :members:
   :undoc-members:
   :show-inheritance:

Calibration Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: petrophys.core.calibrate_porosity

.. autofunction:: petrophys.core.calibrate_permeability

Poro-Perm Transform
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: petrophys.core.fit_poro_perm_transform

.. autofunction:: petrophys.core.apply_poro_perm_transform

Grain Density Correction
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: petrophys.core.correct_porosity_with_grain_density

Archie m Estimation
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: petrophys.core.estimate_archie_m

Core Visualization
~~~~~~~~~~~~~~~~~~

.. autofunction:: petrophys.core.plot_calibration

.. autofunction:: petrophys.core.plot_poro_perm

.. autofunction:: petrophys.core.plot_core_on_logs

Formation Module
----------------

Interval-based analysis: formation tops, zones, sub-zones, and manual intervals
with per-interval petrophysical summaries.

Interval Types
~~~~~~~~~~~~~~

Four interval types sharing a common ``Interval`` base class:

- **Formation** — geological formation (with optional ``age``)
- **Zone** — petrophysical zone (with optional ``parent_formation``)
- **SubZone** — sub-zone within a zone (with optional ``parent_zone``)
- **ManualInterval** — user-defined arbitrary depth interval

.. automodule:: petrophys.formation.intervals
   :members:
   :undoc-members:
   :show-inheritance:

FormationTops Manager
~~~~~~~~~~~~~~~~~~~~~

Load, validate, query, and export collections of formation intervals.
Supports CSV (simple, NPD, Danish formats), dictionaries, and DataFrames.

.. autoclass:: petrophys.formation.tops.FormationTops
   :members:
   :undoc-members:
   :show-inheritance:

IntervalAnalyzer
~~~~~~~~~~~~~~~~

Per-interval petrophysical analysis: net pay, NTG, curve averages, HCPV,
dominant lithology, and data coverage.

.. automodule:: petrophys.formation.analysis
   :members:
   :undoc-members:
   :show-inheritance:

Summation Module
----------------

Summary calculations and reporting.

.. automodule:: petrophys.summation
   :members:
   :undoc-members:
   :show-inheritance:

Formation Analysis (Legacy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-formation analysis with CSV loading, zone summaries, and report export.
For new code, prefer ``petrophys.formation`` which provides a richer interval
type hierarchy and the ``IntervalAnalyzer`` class.

.. automodule:: petrophys.summation.formation_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Workflow Module
---------------

Workflow orchestration and batch processing. Both workflows provide full
step-by-step trace metadata for transparency and auditability.

Standard Workflow
~~~~~~~~~~~~~~~~~

Fixed-configuration pipeline for straightforward interpretation.
Returns ``WorkflowResult`` with ``print_trace()`` for full transparency.

.. automodule:: petrophys.workflow.standard_workflow
   :members:
   :undoc-members:
   :show-inheritance:

StepTrace (Standard Workflow)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each step in the standard workflow returns a ``StepTrace`` with:

- **step**: Step name (temperature, vclay, rw, porosity, saturation, permeability, net_pay)
- **method**: Algorithm used (e.g., ``linear_gradient_from_BHT``, ``archie``)
- **parameters**: All numeric parameters (Rw, matrix density, cutoffs, etc.)
- **input_curves**: Log curves consumed
- **output_curves**: Curves added to the well
- **decisions**: Human-readable trace of logic branches and values

.. autoclass:: petrophys.workflow.standard_workflow.StepTrace
   :members:
   :undoc-members:
   :show-inheritance:

Dynamic Workflow
~~~~~~~~~~~~~~~~

Input-aware pipeline that auto-detects available curves and runs only
applicable modules in the correct pipeline order. Returns
``DynamicWorkflowResult`` with ``print_trace()`` and ``to_dataframe()``
for comprehensive trace export.

.. automodule:: petrophys.workflow.dynamic_workflow
   :members:
   :undoc-members:
   :show-inheritance:

StepResult (Dynamic Workflow)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each step returns a ``StepResult`` with comprehensive metadata:

- **step**: Step name (temperature, bad_hole, vclay, rcl, lithology, rmf, rw, invasion, hc_correction, porosity, saturation, permeability, rock_quality, bvw, mhi, net_pay)
- **method**: Specific algorithm or equation used
- **parameters**: Every numeric parameter, threshold, and constant with source (user-provided, auto-detected, default)
- **input_curves**: All input log curves consumed
- **output_curves**: All curves added/modified on the well
- **decisions**: Ordered trace of every decision (auto-detection logic, fallback paths, parameter sources, output statistics)

.. autoclass:: petrophys.workflow.dynamic_workflow.StepResult
   :members:
   :undoc-members:
   :show-inheritance:

Workflow Trace Output
^^^^^^^^^^^^^^^^^^^^^

Both ``WorkflowResult.print_trace()`` and ``DynamicWorkflowResult.print_trace()``
produce formatted output like:

.. code-block:: text

   ==========================================================================================
   WORKFLOW TRACE — 15/9-15
   ==========================================================================================
   Overall: SUCCESS | Executed: 12 | Skipped: 2

   [ OK ] temperature
          Method: linear_gradient_from_BHT
          Inputs: DEPT
          Outputs: TEMP
          surface_temp_f: 68.0
          bht_f: 220.0
          -> Gradient = 0.0144 F/ft
          -> Temp range: 75.0 - 219.5 F

   [ OK ] vclay
          Method: larionov_tertiary
          Inputs: GR, SP, NPHI, RHOB
          Outputs: VCL, VCL_GR, VCL_SP, VCL_ND
          gr_min: 22.3 (auto, P5 of GR)
          gr_max: 98.7 (auto, P95 of GR)
          -> SP available and WBM without clay/KCl -> VCL_SP computed
          -> NPHI + RHOB available -> VCL_ND computed
          -> Final VCL = min(available methods)

   [ OK ] saturation
          Method: simandoux
          Inputs: RDEP, PHIE, VCL
          Outputs: SW
          a: 1.0, m: 2.0, n: 2.0
          rw: 0.04
          -> Model selected: simandoux (auto-routed: clastic + salinity >= 20k)
          -> Sw range: 0.051 - 1.000

``DynamicWorkflowResult.to_dataframe()`` exports all trace data as a pandas
DataFrame for inclusion in QC reports.

.. automodule:: petrophys.workflow
   :members:
   :undoc-members:
   :show-inheritance:

Utilities Module
----------------

Utility functions and helpers.

.. automodule:: petrophys.utils
   :members:
   :undoc-members:
   :show-inheritance:
