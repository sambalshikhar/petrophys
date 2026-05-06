Petrophys Documentation
=======================

.. raw:: html

   <div style="display: flex; align-items: center; gap: 30px; margin-bottom: 20px; flex-wrap: wrap;">
     <a href="https://mbzuai.ac.ae" target="_blank">
       <img src="_static/mbzuai-logo.svg" alt="MBZUAI" style="height: 60px;">
     </a>
     <a href="https://www.inceptionai.ai" target="_blank">
       <img src="_static/inception-logo.png" alt="Inception AI" style="height: 50px; background: #222; padding: 8px 12px; border-radius: 6px;">
     </a>
   </div>

*Developed by* `Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) <https://mbzuai.ac.ae>`_
*in collaboration with* `Inception AI <https://www.inceptionai.ai>`_.

**Petrophys** is a comprehensive Python library for petrophysical well log analysis.
It provides tools for loading, processing, and interpreting wireline and LWD log data.

This library is designed to serve as a **tool-calling environment for Agentic AI workflows**
and **Agentic Reinforcement Learning (RL)** research. Each module exposes deterministic,
composable functions that an LLM agent can invoke as tools — enabling autonomous
petrophysical interpretation pipelines where the agent selects methods, tunes parameters,
and iterates on results. The modular architecture, full execution traces, and
structured inputs/outputs make it ideal for:

* **Agentic tool-calling** — LLM agents can call individual modules (VCL, porosity, Sw, net pay, etc.) as discrete tools with typed parameters and structured returns.
* **Agentic RL environments** — The workflow engine provides a step-by-step MDP-like interface where actions are petrophysical computation steps, observations are well log curves, and rewards can be derived from interpretation quality metrics (NTG, HCPV, calibration error).
* **Human-in-the-loop and fully autonomous modes** — The standard workflow supports fixed configurations while the dynamic workflow auto-detects inputs and routes models, enabling both supervised and autonomous agent operation.

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
* **Core Data**: Load core plug measurements, calibrate log porosity/permeability, fit poro-perm transforms, estimate Archie m
* **Dynamic Workflow**: Auto-detects available curves and runs only applicable modules with full step-by-step trace
* **Visualization**: Log plots, triple combos, interpretation plots, crossplots, Pickett plots, zone flags, formation shading

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

   # Full transparency: print every step's method, parameters, and decisions
   result.print_trace()

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

   # Full trace: every step's method, inputs, outputs, parameters, decisions
   result.print_trace()

   # Or export the trace as a DataFrame for reporting
   trace_df = result.to_dataframe()

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
   :maxdepth: 2
   :caption: Releases

   release_notes

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/modules


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
