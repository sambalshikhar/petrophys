.. title:: Petrophys documentation

.. raw:: html

   <header class="ph-intro">
     <p class="ph-kicker">Open-source formation evaluation</p>
     <h1>Petrophys</h1>
     <p class="ph-lead">
       A modular Python library for turning well logs, core measurements,
       and specialist data into traceable petrophysical interpretations.
     </p>
     <div class="ph-actions">
       <a href="quickstart.html">Run the quickstart</a>
       <a href="api/index.html">Browse the API</a>
     </div>
   </header>

From measurements to decisions
==============================

Petrophys keeps the scientific calculations accessible as independent Python
functions while also providing an input-aware workflow for complete wells.
Recognized curve aliases and units are standardized before interpretation;
every executed, skipped, or warning-producing step is recorded in the result.

.. figure:: _static/petrophys-heimdal-formation.png
   :alt: Multi-track Petrophys interpretation of the 15/9-17 Heimdal Formation
   :width: 100%
   :class: ph-log-figure

   Formation-scale interpretation of 15/9-17 in the Heimdal Formation,
   showing lithology, conventional logs, porosity, saturation, permeability,
   mineral volumes, core points, and reservoir flags.

What is in the library
======================

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Real-world inputs
      :class-card: ph-card-teal

      Read LAS, tabular logs, and optional DLIS data. Work with core
      measurements, formation intervals, NMR, borehole images, and
      geochemical channels through domain-specific APIs.

      +++
      :doc:`Data and depth <data_model>`

   .. grid-item-card:: Preparation and QC
      :class-card: ph-card-blue

      Resolve recognized mnemonic aliases, convert supported units, screen
      implausible values, identify washout and bad-hole intervals, normalize
      curves, and impute only missing samples when a compatible model exists.

      +++
      :doc:`Capability map <capabilities>`

   .. grid-item-card:: Formation evaluation
      :class-card: ph-card-amber

      Calculate clay volume, matrix and mineral volumes, porosity, electrical
      properties, saturation, permeability, temperature, and invasion using
      explicit methods or automatic routing.

      +++
      :doc:`Interpretation workflows <workflows>`

   .. grid-item-card:: Specialist interpretation
      :class-card: ph-card-coral

      Analyze processed NMR distributions, borehole-image arrays, spectral
      gamma and elemental spectroscopy, core measurements, saturation height,
      sonic saturation, and elastic properties.

      +++
      :doc:`Specialist data <specialist_data>`

   .. grid-item-card:: Reservoir evaluation
      :class-card: ph-card-green

      Derive BVW, moveable-hydrocarbon indicators, RQI and FZI, hydraulic flow
      units, net rock, net reservoir, net pay, volumetrics, diagnostics, and
      interval summaries.

      +++
      :doc:`Capability map <capabilities>`

   .. grid-item-card:: Plots and delivery
      :class-card: ph-card-blue

      Build adaptive log panels, triple combos, crossplots, Pickett plots,
      specialist displays, and multiwell charts. Export curves and auditable
      LAS, CSV, Excel, JSON, and report packages.

      +++
      :doc:`Visualization and reporting <visualization_reporting>`

How data moves through Petrophys
================================

.. raw:: html

   <div class="ph-flow" role="list" aria-label="Petrophys processing flow">
     <div class="ph-flow-node" role="listitem">
       <span class="ph-flow-step">01 / Inputs</span>
       <strong>Read</strong>
       <small>LAS, DLIS, CSV, ASCII, core, tops and specialist channels</small>
     </div>
     <div class="ph-flow-node" role="listitem">
       <span class="ph-flow-step">02 / Preparation</span>
       <strong>Standardize</strong>
       <small>Aliases, units, nulls, ranges, wellbore quality and provenance</small>
     </div>
     <div class="ph-flow-node" role="listitem">
       <span class="ph-flow-step">03 / Science</span>
       <strong>Interpret</strong>
       <small>VCL, lithology, porosity, electrical properties, Sw and perm</small>
     </div>
     <div class="ph-flow-node" role="listitem">
       <span class="ph-flow-step">04 / Decisions</span>
       <strong>Evaluate</strong>
       <small>Rock quality, flow units, flags, intervals and volumetrics</small>
     </div>
     <div class="ph-flow-node" role="listitem">
       <span class="ph-flow-step">05 / Delivery</span>
       <strong>Explain</strong>
       <small>Computed curves, traces, plots, summaries and export packages</small>
     </div>
   </div>

Input-aware, not all-or-nothing
===============================

Incomplete log suites produce a reduced interpretation with an explicit trace.
Prerequisites are evaluated before each workflow step, so the result explains
what ran, what could not run, and where assumptions or data-quality warnings
need review.

.. raw:: html

   <div class="ph-trace" aria-label="Example workflow trace states">
     <div>
       <strong>Execution record</strong>
       <span>Method, inputs, outputs, parameters and decisions</span>
     </div>
     <div>
       <span class="ph-status ph-status-run">Executed</span>
       <strong>Porosity</strong>
       <span>RHOB and NPHI available</span>
     </div>
     <div>
       <span class="ph-status ph-status-skip">Skipped</span>
       <strong>NMR</strong>
       <span>No compatible NMR channels</span>
     </div>
     <div>
       <span class="ph-status ph-status-warn">Review</span>
       <strong>Water saturation</strong>
       <span>Inspect assumptions and quality flags</span>
     </div>
   </div>

Start with a bundled well
=========================

The package includes small public datasets for reproducible examples. The
following run reads a LAS file, applies recognized input standardization, and
returns the same in-memory well with computed curves attached.

.. code-block:: python

   import petrophys
   from petrophys.io import read_las
   from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow

   well = read_las(petrophys.sample("30_6-5"))
   config = DynamicWorkflowConfig(
       matrix_type="auto",
       lithology="auto",
       rw=0.07,  # example study assumption, ohm.m
   )

   result = run_dynamic_workflow(well, config)

   print(result.status)
   print([step.step for step in result.steps_executed])
   print(result.steps_skipped)
   print(result.well.get("PHIE"))

Continue with the :doc:`installation guide <installation>`, the
:doc:`quickstart <quickstart>`, or the browser-rendered :doc:`tutorials
<tutorials/index>`.

.. raw:: html

   <div class="ph-institution-row">
     <a href="https://mbzuai.ac.ae" target="_blank" rel="noreferrer">
       <img src="_static/mbzuai-logo.svg" alt="MBZUAI">
     </a>
     <a href="https://www.inceptionai.ai" target="_blank" rel="noreferrer">
       <img class="ph-inception" src="_static/inception-logo.png" alt="Inception AI">
     </a>
     <p>Developed by MBZUAI in collaboration with Inception AI.</p>
   </div>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   input_templates
   own_data

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Fundamentals

   data_model
   architecture

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   workflows
   capabilities
   specialist_data
   visualization_reporting
   mcp

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Learn

   tutorials/index
   examples

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Project

   changelog
