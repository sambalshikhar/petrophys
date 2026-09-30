Foundations
===========

Build a conventional formation evaluation from the measurements upward. This
path keeps every method visible, so you can inspect inputs, assumptions,
intermediate curves, and failure conditions before using automatic routing.

**Recommended preparation:** complete the :doc:`quickstart`, install the base
package, and be comfortable reading NumPy arrays and Matplotlib figures.

What you will build
-------------------

By the end of this path you will have loaded and quality-controlled a well,
estimated clay volume, porosity, lithology, formation-water properties, water
saturation, permeability, reservoir quality, and net pay. The lessons use the
same in-memory well model so calculated curves can feed later calculations.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Start lesson 00
      :class-card: ph-card-teal
      :link: part_a/00_orientation_and_setup
      :link-type: doc

      Tour the package, bundled datasets, conventions, and interpretation
      assumptions used throughout the curriculum.

   .. grid-item-card:: Prefer a complete run?
      :class-card: ph-card-blue
      :link: quickstart
      :link-type: doc

      See the dynamic workflow interpret one bundled well before examining each
      calculation separately.

Course outline
--------------

.. list-table::
   :header-rows: 1
   :widths: 9 32 59

   * - Lesson
     - Topic
     - Work completed
   * - 00
     - :doc:`Orientation and setup <part_a/00_orientation_and_setup>`
     - Set up a kernel, tour the package and bundled samples, meet the GeoLink and COSTA teaching wells, and read the curriculum roadmap.
   * - 01
     - :doc:`Loading and inspecting data <part_a/01_loading_and_inspecting_data>`
     - Read wells with ``pp.read``, inspect headers, curves, statistics and null coverage, slice by depth, and plot existing measurements.
   * - 02
     - :doc:`QC and standardization <part_a/02_qaqc_and_standardization>`
     - Resolve aliases, standardize units, check core saturation units, screen value ranges, flag bad hole, and normalize GR across wells.
   * - 03
     - :doc:`Clay volume <part_a/03_clay_volume>`
     - Compare gamma-ray, SP, and neutron-density clay-volume methods, combine them, and estimate clay resistivity.
   * - 04
     - :doc:`Porosity <part_a/04_porosity>`
     - Derive density, neutron, sonic, total, and effective porosity, and apply a hydrocarbon correction.
   * - 05
     - :doc:`Lithology and mineral solving <part_a/05_lithology_and_mineral_solver>`
     - Read neutron-density and PEF crossplots, solve mineral volumes from response equations, and infer lithology without external geology.
   * - 06
     - :doc:`Formation water and environment <part_a/06_formation_water_and_environment>`
     - Build a temperature profile, correct Rmf and Rw to formation temperature, estimate Rw from salinity and SP, and characterize invasion.
   * - 07
     - :doc:`Water saturation <part_a/07_water_saturation>`
     - Compare Archie and shaly-sand saturation models, read a Pickett plot, compute flushed-zone saturation, and treat a carbonate case.
   * - 08
     - :doc:`Permeability <part_a/08_permeability>`
     - Estimate permeability with Timur and Coates-Denoo, flag impermeable zones, and calibrate a poro-perm transform to core; NMR permeability is in lesson 14.
   * - 09
     - :doc:`Reservoir quality and net pay <part_a/09_reservoir_quality_and_net_pay>`
     - Calculate RQI, FZI and flow units, BVW and MHI, three-stage net pay, and HCPV/OOIP/OGIP volumetrics.

.. toctree::
   :hidden:
   :maxdepth: 1

   Orientation and setup <part_a/00_orientation_and_setup>
   Loading and inspecting data <part_a/01_loading_and_inspecting_data>
   QC and standardization <part_a/02_qaqc_and_standardization>
   Clay volume <part_a/03_clay_volume>
   Porosity <part_a/04_porosity>
   Lithology and mineral solving <part_a/05_lithology_and_mineral_solver>
   Formation water and environment <part_a/06_formation_water_and_environment>
   Water saturation <part_a/07_water_saturation>
   Permeability <part_a/08_permeability>
   Reservoir quality and net pay <part_a/09_reservoir_quality_and_net_pay>
