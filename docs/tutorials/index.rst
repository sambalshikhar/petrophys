Tutorials
=========

These tutorials provide step-by-step guides for common petrophysical workflows.

.. contents:: Table of Contents
   :local:
   :depth: 2

Basic Well Log Analysis
-----------------------

Loading and Inspecting Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial shows how to load well log data and perform basic inspection.

.. code-block:: python

   from petrophys.io import read_las
   from petrophys import Well

   # Load a LAS file
   well = read_las("example_well.las")

   # Display well information
   print("Well Header Information:")
   print("-" * 40)
   for key, value in well.header.items():
       print(f"{key}: {value}")

   # Display curve inventory
   print("\nAvailable Curves:")
   print("-" * 40)
   for curve in well.curves:
       stats = curve.statistics()
       print(f"{curve.name:10s} [{curve.unit:6s}] "
             f"min={stats['min']:8.2f} max={stats['max']:8.2f}")

Clay Volume Calculation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrophys.clay_volume import VclModule

   # Initialize VCL calculator
   vcl_module = VclModule(
       gr_clean=20,      # GR in clean sand
       gr_shale=120,     # GR in shale
       method="larionov_tertiary"
   )

   # Calculate VCL
   result = vcl_module.calculate(well)

   # Add to well
   vcl_module.add_to_well(well, result)

   # View results
   vcl = well.get("VCL")
   print(f"VCL range: {vcl.min():.2f} - {vcl.max():.2f}")
   print(f"Mean VCL: {vcl.mean():.2f}")

Porosity Calculation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrophys.petrophysics.porosity import PorosityModule

   # Initialize porosity calculator
   por_module = PorosityModule(
       rho_ma=2.65,   # Matrix density (sandstone)
       rho_fl=1.0,    # Fluid density
       dt_ma=55.5,    # Matrix sonic
       dt_fl=189.0    # Fluid sonic
   )

   # Calculate porosity
   result = por_module.calculate(well)

   # Add results to well
   por_module.add_to_well(well, result)

   # View different porosity types
   print("Porosity Summary:")
   for name in ["PHID", "PHIN", "PHIS", "PHIT", "PHIE"]:
       curve = well.get(name)
       if curve is not None:
           print(f"  {name}: {curve.mean():.3f}")

Water Saturation Analysis
-------------------------

Archie Method
~~~~~~~~~~~~~

The Archie equation is used for clean (non-shaly) formations:

.. math::

   S_w^n = \\frac{a \\cdot R_w}{\\phi^m \\cdot R_t}

.. code-block:: python

   from petrophys.petrophysics.saturation import SwModule

   # Archie parameters
   sw_module = SwModule(
       rw=0.05,      # Formation water resistivity
       a=1.0,        # Tortuosity factor
       m=2.0,        # Cementation exponent
       n=2.0,        # Saturation exponent
       model="archie"
   )

   # Calculate Sw
   result = sw_module.calculate(well)
   sw_module.add_to_well(well, result)

Shaly Sand Methods
~~~~~~~~~~~~~~~~~~

For shaly formations, use Simandoux or Indonesia equations:

.. code-block:: python

   # Simandoux model
   sw_module = SwModule(
       rw=0.05,
       a=1.0,
       m=2.0,
       n=2.0,
       model="simandoux",
       rsh=2.0  # Shale resistivity
   )

   result = sw_module.calculate(well)

   # Indonesia model
   sw_module = SwModule(
       rw=0.05,
       a=1.0,
       m=2.0,
       n=2.0,
       model="indonesia",
       rsh=2.0
   )

   result = sw_module.calculate(well)

Permeability Estimation
-----------------------

Using Timur Equation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrophys.petrophysics.permeability import PermModule

   # Timur equation
   perm_module = PermModule(method="timur")
   result = perm_module.calculate(well)
   perm_module.add_to_well(well, result)

   # View permeability statistics
   perm = well.get("PERM")
   print(f"Permeability range: {perm.min():.2f} - {perm.max():.2f} mD")

Core Calibration
~~~~~~~~~~~~~~~~

When core data is available:

.. code-block:: python

   from petrophys.petrophysics.permeability import calibrate_to_core
   import numpy as np

   # Core data
   core_depth = np.array([8050, 8100, 8150])
   core_perm = np.array([100, 250, 50])

   # Calibrate log permeability
   calibrated_perm = calibrate_to_core(
       well,
       core_depth=core_depth,
       core_perm=core_perm
   )

Net Pay Determination
---------------------

.. code-block:: python

   from petrophys.reservoir.net_pay import NetPayModule

   # Define cutoffs
   np_module = NetPayModule(
       vcl_cutoff=0.3,   # Max VCL
       phie_cutoff=0.08, # Min porosity
       sw_cutoff=0.7     # Max water saturation
   )

   # Calculate net pay
   result = np_module.calculate(well)
   np_module.add_to_well(well, result)

   # Summary
   print(f"Gross thickness: {result.gross_thickness:.1f} ft")
   print(f"Net rock: {result.net_rock_thickness:.1f} ft")
   print(f"Net reservoir: {result.net_reservoir_thickness:.1f} ft")
   print(f"Net pay: {result.net_pay_thickness:.1f} ft")
   print(f"NTG: {result.net_pay_thickness/result.gross_thickness:.1%}")

Complete Workflow Example
-------------------------

.. code-block:: python

   from petrophys.io import read_las
   from petrophys.workflow import run_standard_workflow, WorkflowConfig
   from petrophys.summation import calculate_zone_summary
   from petrophys.visualization import create_interpretation_plot

   # Load data
   well = read_las("well_data.las")

   # Configure workflow
   config = WorkflowConfig(
       surface_temp_f=70,
       bht_f=180,
       td_ft=9000,
       rw=0.05,
       matrix_density=2.65,
       a=1.0,
       m=2.0,
       n=2.0,
       saturation_model="archie",
       vcl_cutoff=0.3,
       phie_cutoff=0.08,
       sw_cutoff=0.7
   )

   # Run interpretation
   result = run_standard_workflow(well, config)

   # Print warnings
   for warn in result.warnings:
       print(f"Warning: {warn}")

   # Zone summary
   zones = [
       {"name": "Zone A", "top": 8000, "bottom": 8200},
       {"name": "Zone B", "top": 8200, "bottom": 8400},
   ]

   print("\nZone Summaries:")
   print("=" * 60)
   for zone in zones:
       summary = calculate_zone_summary(
           result.well,
           zone["top"],
           zone["bottom"],
           zone["name"]
       )
       print(f"\n{summary.zone_name}:")
       print(f"  Net Pay: {summary.net_pay:.1f} ft")
       print(f"  Avg PHIE: {summary.avg_phie:.3f}")
       print(f"  Avg Sw: {summary.avg_sw:.3f}")

   # Create plot
   plot = create_interpretation_plot(
       result.well,
       depth_range=(8000, 8400)
   )
   plot.save("interpretation.png")
