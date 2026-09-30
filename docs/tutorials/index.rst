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

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Quickstart
      :class-card: ph-card-teal
      :link: quickstart
      :link-type: doc

      Plot a bundled LAS first, optionally attach geology, then interpret on a
      copy, read the decision trace, and summarize the reservoir interval.

      +++
      One complete well in a single lesson

   .. grid-item-card:: Your own data
      :class-card: ph-card-teal
      :link: own_data
      :link-type: doc

      Map CSV/TSV/Excel inputs, select wells and sheets, check depth metadata,
      diagnose optional geology and core saturation units, summarize results
      per formation, and export LAS and JSON.

      +++
      Beginner examples followed by advanced import controls

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
      multiwell analysis, LLM context export, and LLM agents over MCP.

      +++
      11 focused lessons (10-20)

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

   Quickstart notebook <quickstart>
   Your own data notebook <own_data>
   Foundations <foundations>
   Advanced workflows <advanced>
   Case studies <case_studies>
