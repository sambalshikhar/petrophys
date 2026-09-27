Examples
========

The :doc:`tutorial curriculum <tutorials/index>` is the main example
collection. Every lesson is readable as a normal documentation page with its
code and saved outputs in place. The same material remains runnable from a
source checkout and uses public datasets exposed by :func:`petrophys.datasets.sample`.

Choose a learning path
----------------------

.. grid:: 1 2 2 3
   :gutter: 2

   .. grid-item-card:: Start with one well
      :class-card: ph-card-teal
      :link: tutorials/quickstart
      :link-type: doc

      Read a bundled LAS file, run the dynamic workflow, inspect its trace,
      plot the interpretation, and export calculated curves.

   .. grid-item-card:: Learn the calculations
      :class-card: ph-card-amber
      :link: tutorials/part_a/00_orientation_and_setup
      :link-type: doc

      Work through QC, clay volume, porosity, lithology, formation water,
      saturation, permeability, rock quality, and net pay.

   .. grid-item-card:: Apply advanced workflows
      :class-card: ph-card-blue
      :link: tutorials/part_b/10_dynamic_workflow_glass_box
      :link-type: doc

      Explore execution traces, intervals, core calibration, imputation, NMR,
      saturation height, borehole images, visualization, and multiwell work.

   .. grid-item-card:: Interpret a clastic well
      :class-card: ph-card-green
      :link: tutorials/case_studies/cs1_full_formation_evaluation_30_6-5
      :link-type: doc

      Follow an end-to-end formation evaluation on public North Sea data.

   .. grid-item-card:: Characterize carbonates
      :class-card: ph-card-coral
      :link: tutorials/case_studies/cs2_carbonate_rock_typing_costa
      :link-type: doc

      Combine COSTA core and log data for carbonate characterization and rock
      typing.

   .. grid-item-card:: Test imperfect inputs
      :class-card: ph-card-blue
      :link: tutorials/case_studies/cs4_imperfect_well_recovery_16_7-1
      :link-type: doc

      Observe how the input-aware workflow reduces its route when a well has an
      incomplete log suite.

Foundation tutorials
--------------------

.. list-table::
   :header-rows: 1
   :widths: 10 38 52

   * - No.
     - Tutorial
     - Focus
   * - 00
     - :doc:`Orientation and setup <tutorials/part_a/00_orientation_and_setup>`
     - Package layout, samples, and interpretation assumptions
   * - 01
     - :doc:`Loading and inspection <tutorials/part_a/01_loading_and_inspecting_data>`
     - Wells, curves, headers, depth, and units
   * - 02
     - :doc:`QC and standardization <tutorials/part_a/02_qaqc_and_standardization>`
     - Aliases, units, value ranges, and bad-hole evidence
   * - 03
     - :doc:`Clay volume <tutorials/part_a/03_clay_volume>`
     - GR, SP, neutron-density VCL and clay resistivity
   * - 04
     - :doc:`Porosity <tutorials/part_a/04_porosity>`
     - Density, neutron, sonic, total, and effective porosity
   * - 05
     - :doc:`Lithology and minerals <tutorials/part_a/05_lithology_and_mineral_solver>`
     - Crossplots, response equations, mineral volumes, and classification
   * - 06
     - :doc:`Formation water and environment <tutorials/part_a/06_formation_water_and_environment>`
     - Rw, Rmf, temperature, and invasion
   * - 07
     - :doc:`Water saturation <tutorials/part_a/07_water_saturation>`
     - Clean and shaly-formation models and routing
   * - 08
     - :doc:`Permeability <tutorials/part_a/08_permeability>`
     - Empirical, NMR, and core-calibrated routes
   * - 09
     - :doc:`Reservoir quality and net pay <tutorials/part_a/09_reservoir_quality_and_net_pay>`
     - RQI, FZI, flow units, cutoffs, and thickness summaries

Advanced tutorials
------------------

.. list-table::
   :header-rows: 1
   :widths: 10 38 52

   * - No.
     - Tutorial
     - Focus
   * - 10
     - :doc:`Dynamic workflow glass box <tutorials/part_b/10_dynamic_workflow_glass_box>`
     - Routing, assumptions, execution states, and sensitivity
   * - 11
     - :doc:`Formation and interval analysis <tutorials/part_b/11_formation_and_interval_analysis>`
     - Tops, zones, summaries, and formation-aware plotting
   * - 12
     - :doc:`Core calibration and digital core <tutorials/part_b/12_core_calibration_and_digital_core>`
     - Core/log calibration, XRF, digital facies, and mechanics
   * - 13
     - :doc:`ML curve imputation <tutorials/part_b/13_ml_curve_imputation>`
     - Missing-sample prediction and provenance
   * - 14
     - :doc:`NMR petrophysics <tutorials/part_b/14_nmr_petrophysics>`
     - T2 distributions, fluid volumes, and permeability
   * - 15
     - :doc:`Saturation height <tutorials/part_b/15_capillary_pressure_saturation_height>`
     - Capillary pressure, Leverett J, and saturation functions
   * - 16
     - :doc:`Borehole image and FMI <tutorials/part_b/16_borehole_image_fmi>`
     - Image arrays, orientation, equalization, coverage, and display
   * - 17
     - :doc:`Visualization masterclass <tutorials/part_b/17_visualization_masterclass>`
     - Templates, interpretation panels, crossplots, and analytics
   * - 18
     - :doc:`Multiwell field study <tutorials/part_b/18_multiwell_field_study>`
     - Batch interpretation and field screening
   * - 19
     - :doc:`LLM context generation <tutorials/part_b/19_llm_context_generation>`
     - Portable, structured context export from interpreted LAS data

Data and optional dependencies
------------------------------

Most notebooks run with the package's default dependencies and bundled sample
data. DLIS/FMI examples require ``petrophys[dlis]`` and the repository's Git
LFS fixtures. ML examples require ``petrophys[ml]``. The installation guide
lists each optional extra and its scope.
