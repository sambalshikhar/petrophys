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
* **Clay Volume**: Multiple VCL calculation methods (GR, SP, N-D)
* **Porosity**: Density, neutron, sonic, and combination methods
* **Water Saturation**: Archie, Simandoux, Indonesia, Dual Water models
* **Permeability**: Timur, Coates-Denoo, Wyllie-Rose equations
* **Net Pay**: Automated cutoff-based determination
* **Visualization**: Log plots, crossplots, Pickett plots

Quick Example
-------------

.. code-block:: python

   from petrophys import Well
   from petrophys.io import read_las
   from petrophys.workflow import run_standard_workflow, WorkflowConfig

   # Load well data
   well = read_las("well_data.las")

   # Configure workflow
   config = WorkflowConfig(
       rw=0.05,
       matrix_density=2.65,
       saturation_model="archie"
   )

   # Run interpretation
   result = run_standard_workflow(well, config)

   # Access results
   print(f"Net pay: {result.well.get('NET_PAY')}")

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
