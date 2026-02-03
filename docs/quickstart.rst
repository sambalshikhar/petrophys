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

Basic Calculations
------------------

**Clay Volume**

.. code-block:: python

   from petrophys.formation.clay_volume import calculate_vcl

   # Calculate VCL from GR
   vcl = calculate_vcl(well, method="gr_linear")

   # Add result to well
   well.add_curve("VCL", vcl, "v/v", "Clay Volume")

**Porosity**

.. code-block:: python

   from petrophys.petrophysics.porosity import calculate_porosity

   # Calculate density porosity
   result = calculate_porosity(well, method="density", rho_ma=2.65)

   # Access the results
   phit = result.phi_density
   print(f"Average porosity: {phit.mean():.3f}")

**Water Saturation**

.. code-block:: python

   from petrophys.petrophysics.saturation import calculate_sw

   # Calculate Sw using Archie equation
   sw = calculate_sw(
       well,
       model="archie",
       rw=0.05,
       a=1.0,
       m=2.0,
       n=2.0
   )

Full Interpretation Workflow
----------------------------

For a complete interpretation, use the standard workflow:

.. code-block:: python

   from petrophys.workflow import run_standard_workflow, WorkflowConfig

   # Configure the workflow
   config = WorkflowConfig(
       # Temperature
       surface_temp_f=70,
       bht_f=200,
       td_ft=10000,

       # Formation water
       rw=0.05,

       # Matrix properties
       matrix_density=2.65,

       # Archie parameters
       a=1.0,
       m=2.0,
       n=2.0,

       # Cutoffs
       vcl_cutoff=0.3,
       phie_cutoff=0.08,
       sw_cutoff=0.7
   )

   # Run the workflow
   result = run_standard_workflow(well, config)

   # Check for warnings
   if result.warnings:
       for warn in result.warnings:
           print(f"Warning: {warn}")

   # Access computed curves
   print("Computed curves:")
   for name in ["VCL", "PHIE", "SW", "PERM", "NET_PAY"]:
       curve = result.well.get(name)
       if curve is not None:
           print(f"  {name}: added")

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

   # Define zones
   zones = [
       {"name": "Zone A", "top": 8000, "bottom": 8100},
       {"name": "Zone B", "top": 8100, "bottom": 8200},
   ]

   # Calculate summary for each zone
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
