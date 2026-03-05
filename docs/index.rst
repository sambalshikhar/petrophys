Petrophys Documentation
=======================

**Petrophys** is a comprehensive Python library for petrophysical well log analysis.
It provides tools for loading, processing, and interpreting wireline and LWD log data.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT

Features
--------

* **Data I/O**: Read LAS, DLIS, and ASCII log files
* **Quality Control**: Bad hole detection, normalization, curve standardization
* **Clay Volume**: Multiple VCL calculation methods (GR, SP, N-D) with clay resistivity (Rcl)
* **Porosity**: Density, neutron, sonic, total (weighted N-D), effective, and secondary porosity
* **Hydrocarbon Correction**: Preliminary porosity-based NPHI and RHOB corrections
* **Water Saturation**: 5 models — Archie, Simandoux, Indonesia, Waxman-Smits, Dual Water — with automatic routing based on VCL, lithology, CEC, and salinity
* **Permeability**: Timur, Coates-Denoo, Wyllie-Rose equations with impermeability detection
* **Rock Quality**: RQI, FZI, and normalized porosity (phi_z)
* **Net Pay**: Lithology-aware cutoff-based determination (clastic/carbonate)
* **Formation Analysis**: Interval-based summaries for formations, zones, sub-zones, and manual intervals with CSV loading (simple, NPD, Danish)
* **Dynamic Workflow**: Auto-detects available curves and runs only applicable modules
* **Visualization**: Log plots, crossplots, Pickett plots

Quick Example
-------------

**Standard Workflow:**

.. code-block:: python

   from petrophys import Well
   from petrophys.io import read_las
   from petrophys.workflow import run_standard_workflow, WorkflowConfig

   well = read_las("well_data.las")

   config = WorkflowConfig(
       rw=0.05,
       matrix_density=2.65,
       saturation_model="archie"
   )

   result = run_standard_workflow(well, config)
   print(f"Net pay: {result.well.get('NET_PAY')}")

**Dynamic Workflow (auto-detects inputs):**

.. code-block:: python

   from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig

   well = read_las("well_data.las")

   config = DynamicWorkflowConfig(
       bht_f=220, td_ft=8500,
       rw=0.05, salinity_ppm=25000,
       lithology="clastic",
   )

   result = run_dynamic_workflow(well, config)

   # Only applicable modules run automatically
   print(f"Steps executed: {[s.step for s in result.steps_executed]}")
   print(f"Steps skipped: {result.steps_skipped}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/index

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/modules


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
