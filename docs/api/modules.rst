API Reference
=============

This section contains the complete API documentation for all Petrophys modules.

Core Module
-----------

The core module contains fundamental data structures and utilities.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   petrophys.core

.. automodule:: petrophys.core
   :members:
   :undoc-members:
   :show-inheritance:

Well Class
~~~~~~~~~~

.. autoclass:: petrophys.core.Well
   :members:
   :undoc-members:
   :show-inheritance:

Curve Class
~~~~~~~~~~~

.. autoclass:: petrophys.core.Curve
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

MHI = Sxo / Sw

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

Plotting and visualization tools.

.. automodule:: petrophys.visualization
   :members:
   :undoc-members:
   :show-inheritance:

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

Workflow orchestration and batch processing.

Standard Workflow
~~~~~~~~~~~~~~~~~

Fixed-configuration pipeline for straightforward interpretation.

.. automodule:: petrophys.workflow.standard_workflow
   :members:
   :undoc-members:
   :show-inheritance:

Dynamic Workflow
~~~~~~~~~~~~~~~~

Input-aware pipeline that auto-detects available curves and runs only
applicable modules in the correct pipeline order.

.. automodule:: petrophys.workflow.dynamic_workflow
   :members:
   :undoc-members:
   :show-inheritance:

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
