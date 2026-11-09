Tutorials
=========

Learn Petrophys through complete, browser-based lessons. Every tutorial is
rendered directly into this documentation with its explanations, equations,
Python code, tables, and saved figures. Jupyter is only needed when you want to
rerun or modify a lesson locally.

The curriculum uses public well data bundled with the package: North Sea
GeoLink wells (chiefly 30/6-5, with 15/9-14, 16/7-1 and 25/5-1), 17 wells from
the COSTA carbonate benchmark (each file covers only the Kharaib-2 reservoir
interval), and Daydream 1 NMR logs. The borehole-image lesson uses a larger
Daydream 1 DLIS file that is available in source checkouts through Git LFS.

Choose a path
-------------

New to Petrophys? The :doc:`../quickstart` and :doc:`../own_data` pages run one
well, then your own files; the lessons below go deeper.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Complete tour
      :class-card: ph-card-teal
      :link: complete_tour
      :link-type: doc

      Every part of the library in one notebook: loading and QC, each
      interpretation step, the full workflow, four-well comparison, core
      calibration, sensitivity analysis and a gallery of every plot type.

      +++
      One long notebook on four GeoLink wells

   .. grid-item-card:: Foundations
      :class-card: ph-card-amber
      :link: foundations
      :link-type: doc

      Build the conventional formation-evaluation chain one physical quantity
      at a time, from input inspection and QC through net pay.

      +++
      10 sequential lessons (00-09)

   .. grid-item-card:: Advanced workflows
      :class-card: ph-card-blue
      :link: advanced
      :link-type: doc

      Work with workflow orchestration, formation intervals, core calibration,
      imputation, NMR, saturation height, borehole images, visualization,
      multiwell analysis, LLM context export, LLM agents over MCP, and your own
      reusable custom workflows.

      +++
      12 focused lessons (10-21)

   .. grid-item-card:: Case studies
      :class-card: ph-card-green
      :link: case_studies
      :link-type: doc

      Answer realistic asset questions using end-to-end interpretations,
      public ground truth, incomplete inputs, and sensitivity analysis.

      +++
      5 project-style studies

How the curriculum works
------------------------

.. raw:: html

   <div class="ph-course-map" role="list" aria-label="Tutorial learning path">
     <div class="ph-course-stage" role="listitem">
       <span>01</span>
       <strong>Observe</strong>
       <small>Inspect curves, units, metadata, depth coverage, and quality.</small>
     </div>
     <div class="ph-course-stage" role="listitem">
       <span>02</span>
       <strong>Calculate</strong>
       <small>Apply explicit petrophysical methods and inspect intermediate curves.</small>
     </div>
     <div class="ph-course-stage" role="listitem">
       <span>03</span>
       <strong>Route</strong>
       <small>Use available inputs to build a traceable interpretation path.</small>
     </div>
     <div class="ph-course-stage" role="listitem">
       <span>04</span>
       <strong>Evaluate</strong>
       <small>Interpret intervals, compare evidence, and deliver reproducible results.</small>
     </div>
   </div>

Each page can be read without running code. To experiment, install Petrophys
with the relevant optional extras and open the source lesson from the page
toolbar. Saved outputs let the online lesson retain its plots and numerical
results without executing code during the documentation build.

Most lessons need only the base install and the bundled samples. Lesson 13
(curve imputation) needs the ``ml`` extra, lesson 16 (borehole images) needs
the ``dlis`` extra and the Git LFS files of a source checkout, and lesson 20
needs the ``mcp`` extra; see :doc:`../installation`.

.. note::

   Tutorial parameters are study assumptions for the supplied public data.
   They are visible in the lesson and should be reviewed before applying the
   workflow to another field or formation.

.. toctree::
   :hidden:
   :maxdepth: 2

   Complete tour <complete_tour>
   Foundations <foundations>
   Advanced workflows <advanced>
   Case studies <case_studies>
