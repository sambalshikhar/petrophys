Quickstart Guide
================

This guide will help you get started with Petrophys in just a few minutes.

Loading Well Data
-----------------

Petrophys supports multiple file formats for well log data.

**LAS Files**

.. code-block:: python

   from petrophys.io import read_las

   # Load a LAS file
   well = read_las("path/to/well.las")

   # View well information
   print(f"Well name: {well.header.get('WELL')}")
   print(f"Depth range: {well.start_depth} - {well.stop_depth}")
   print(f"Number of curves: {len(well.curves)}")

   # List available curves
   for curve in well.curves:
       print(f"  {curve.name}: {curve.unit}")

**DLIS Files**

.. code-block:: python

   from petrophys.io import read_dlis

   well = read_dlis("path/to/well.dlis")

Accessing Curve Data
--------------------

.. code-block:: python

   # Get a specific curve
   gr = well.get("GR")

   if gr is not None:
       print(f"GR range: {gr.min():.2f} - {gr.max():.2f} {gr.unit}")

   # Access the numpy array directly
   gr_data = gr.data

   # Get depth array
   depth = well.depth

Granular Module Usage
---------------------

Each calculation module can be used independently for maximum flexibility.

**Clay Volume**

.. code-block:: python

   from petrophys.formation.clay_volume import calculate_vcl, vcl_from_gr, calculate_rcl

   # Multi-method VCL (minimum of GR, SP, N-D)
   result = calculate_vcl(well, gr_min=15, gr_max=110)
   well.add_curve("VCL", result.vcl, "v/v", "Clay Volume")

   # Or use individual methods directly
   import numpy as np
   vcl_gr = vcl_from_gr(gr_data, gr_min=15, gr_max=110, method="linear")

   # Calculate clay resistivity from pure-shale zones
   rcl = calculate_rcl(rdep_array, vcl_array, vcl_threshold=0.9)

**Porosity**

.. code-block:: python

   from petrophys.petrophysics.porosity import calculate_porosity

   # Calculate all porosity types (density, sonic, total, effective, secondary)
   result = calculate_porosity(well, matrix="sandstone")

   # Access the results
   print(f"Total porosity: {result.phit.mean():.3f}")
   print(f"Effective porosity: {result.phie.mean():.3f}")
   if result.phi_sec is not None:
       print(f"Secondary porosity: {result.phi_sec.mean():.3f}")

**Water Saturation**

.. code-block:: python

   from petrophys.petrophysics.saturation import (
       calculate_sw, archie, simandoux, indonesia, dual_water, waxman_smits
   )

   # Auto-routed Sw (selects model based on VCL, lithology, salinity)
   sw = calculate_sw(
       well, rw=0.05,
       lithology="clastic",
       salinity=25000  # ppm — routes to Simandoux
   )

   # Or use specific models directly
   sw = archie(rt, rw=0.05, phi=0.15, a=1.0, m=2.0, n=2.0)
   sw = dual_water(phi_t=0.2, phi_e=0.15, rt=10.0, rw=0.05, rwb=0.02,
                    vcl=0.3, phi_cl=0.15)
   sw = waxman_smits(phi_t=0.2, rt=10.0, rw=0.05, cec=10.0,
                      rho_ma=2.65, b_coeff=0.5)

**Rock Quality**

.. code-block:: python

   from petrophys.reservoir import calculate_rock_quality

   rq = calculate_rock_quality(perm_k=100.0, phi_e=0.2)
   print(f"RQI: {rq['RQI']:.4f}")
   print(f"FZI: {rq['FZI']:.4f}")
   print(f"phi_z: {rq['phi_z']:.4f}")

**Impermeability Detection**

.. code-block:: python

   from petrophys.petrophysics.permeability import flag_impermeable

   imperm = flag_impermeable(phi_s, phi_d, phi_n)
   # True where phi_S > 1.5*phi_D or phi_N > phi_D + 0.15

**Net Pay (Lithology-Aware)**

.. code-block:: python

   from petrophys.reservoir import calculate_net_pay

   # Clastics: PHIE cutoff = 0.08
   result = calculate_net_pay(well, lithology="clastic")

   # Carbonates: PHIE cutoff = 0.045
   result = calculate_net_pay(well, lithology="carbonate")

Standard Workflow
-----------------

For a complete interpretation with fixed configuration:

.. code-block:: python

   from petrophys.workflow import run_standard_workflow, WorkflowConfig

   config = WorkflowConfig(
       surface_temp_f=70, bht_f=200, td_ft=10000,
       rw=0.05,
       matrix_density=2.65,
       a=1.0, m=2.0, n=2.0,
       vcl_cutoff=0.3, phie_cutoff=0.08, sw_cutoff=0.7
   )

   result = run_standard_workflow(well, config)

   if result.warnings:
       for warn in result.warnings:
           print(f"Warning: {warn}")

Dynamic Workflow
----------------

The dynamic workflow auto-detects available curves and runs only applicable
modules in the correct pipeline order:

.. code-block:: python

   from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig

   config = DynamicWorkflowConfig(
       # Temperature (optional)
       bht_f=220, td_ft=8500,

       # Formation water
       rw=0.05,
       salinity_ppm=25000,

       # Lithology controls Sw model routing and net pay cutoffs
       lithology="clastic",
       matrix_type="sandstone",

       # HC correction type
       hc_type="oil",

       # Waxman-Smits (optional — if CEC available)
       # cec=10.0, b_coeff=0.5,

       # Dual Water (optional — if rwb available for carbonates)
       # rwb=0.02,
   )

   result = run_dynamic_workflow(well, config)

   # Inspect what happened
   print(f"Steps executed: {[s.step for s in result.steps_executed]}")
   print(f"Steps skipped:  {result.steps_skipped}")
   print(f"Rw used: {result.rw_computed}")
   print(f"Rcl: {result.rcl_computed}")

   # Skip specific steps
   config.skip_steps = ["bad_hole", "lithology"]
   result = run_dynamic_workflow(well, config)

Water Saturation Model Routing
------------------------------

The Sw module automatically selects the appropriate model:

.. code-block:: text

   VCL < 0.3 (clean)  -->  Archie

   VCL >= 0.3 (shaly):
     Carbonate           -->  Dual Water
     Clastic + CEC       -->  Waxman-Smits
     Clastic + salt>=20k -->  Simandoux
     Clastic + salt<20k  -->  Indonesia

Override with ``sw_model="archie"`` (or any model name) to force a specific model.

Visualization
-------------

**Triple Combo Plot**

.. code-block:: python

   from petrophys.visualization import create_triple_combo

   plot = create_triple_combo(well, depth_range=(8000, 8500))
   plot.show()

**Crossplot**

.. code-block:: python

   from petrophys.visualization import create_crossplot

   plot = create_crossplot(
       well,
       x_curve="NPHI",
       y_curve="RHOB",
       color_curve="GR"
   )
   plot.show()

**Pickett Plot**

.. code-block:: python

   from petrophys.visualization import create_pickett_plot

   plot = create_pickett_plot(well, rw=0.05)
   plot.show()

Zone Summaries
--------------

.. code-block:: python

   from petrophys.summation import calculate_zone_summary

   zones = [
       {"name": "Zone A", "top": 8000, "bottom": 8100},
       {"name": "Zone B", "top": 8100, "bottom": 8200},
   ]

   for zone in zones:
       summary = calculate_zone_summary(
           well,
           top_depth=zone["top"],
           bottom_depth=zone["bottom"],
           zone_name=zone["name"]
       )

       print(f"\n{summary.zone_name}:")
       print(f"  Gross: {summary.gross_thickness:.1f} ft")
       print(f"  Net Pay: {summary.net_pay:.1f} ft")
       print(f"  NTG: {summary.ntg:.1%}")
       print(f"  Avg PHIE: {summary.avg_phie:.3f}")
       print(f"  Avg Sw: {summary.avg_sw:.3f}")

Exporting Results
-----------------

.. code-block:: python

   from petrophys.io import export_las, export_csv

   # Export to LAS format
   export_las(well, "output_well.las")

   # Export to CSV
   export_csv(well, "output_well.csv")

Next Steps
----------

* See :doc:`tutorials/index` for detailed tutorials
* See :doc:`api/modules` for complete API reference
