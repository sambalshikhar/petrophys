Examples
========

Code-first recipes for common petrophysical workflows. For the full narrated
curriculum on real wells (with executed figures), see :doc:`tutorials/index`.

.. contents:: Table of Contents
   :local:
   :depth: 2

Basic Well Log Analysis
-----------------------

Loading and Inspecting Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial shows how to load well log data and perform basic inspection.

.. code-block:: python

   import numpy as np
   from petrophys.io import read_las

   # Load a LAS file
   well = read_las("example_well.las")

   # Display well information
   print("Well Header Information:")
   print("-" * 40)
   print(f"Well name : {well.header.well_name}")
   print(f"Depth     : {well.top} - {well.bottom} {well.depth_unit}")

   # Display curve inventory (well.curves is a list of mnemonic strings)
   print("\nAvailable Curves:")
   print("-" * 40)
   for name in well.curves:
       curve = well.get(name)
       print(f"{curve.mnemonic:10s} [{curve.unit:6s}] "
             f"min={curve.min():8.2f} max={curve.max():8.2f}")

Clay Volume Calculation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrophys.interpretation.clay_volume import calculate_vcl

   # Multi-method VCL (minimum of GR, SP, N-D). gr_min/gr_max are the
   # clean-sand and shale GR baselines.
   result = calculate_vcl(well, gr_min=20, gr_max=120, gr_method="larionov_tertiary")

   # Add to well
   well.add_curve("VCL", result.vcl, "v/v", "Clay Volume")

   # View results
   vcl = well.get("VCL")
   print(f"VCL range: {vcl.min():.2f} - {vcl.max():.2f}")
   print(f"Mean VCL: {vcl.mean():.2f}")

Porosity Calculation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrophys.interpretation.porosity import calculate_porosity

   # Calculate all porosity types (density, sonic, total, effective, secondary).
   # `matrix` selects the matrix endpoints (sandstone -> rho_ma 2.65, dt_ma 55.5).
   result = calculate_porosity(well, matrix="sandstone", rhob_fluid=1.0)

   # Add results to well
   well.add_curve("PHID", result.phid, "v/v", "Density Porosity")
   well.add_curve("PHIT", result.phit, "v/v", "Total Porosity")
   well.add_curve("PHIE", result.phie, "v/v", "Effective Porosity")
   if result.phis is not None:
       well.add_curve("PHIS", result.phis, "v/v", "Sonic Porosity")

   # View different porosity types
   print("Porosity Summary:")
   for name in ["PHID", "PHIS", "PHIT", "PHIE"]:
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

   from petrophys.interpretation.saturation import SaturationModule

   # Archie parameters
   sw_module = SaturationModule(
       a=1.0,        # Tortuosity factor
       m=2.0,        # Cementation exponent
       n=2.0,        # Saturation exponent
       model="archie",
   )

   # Calculate Sw (returns a SaturationResult with .sw, .model, .sxo)
   result = sw_module.calculate(well, rw=0.05)
   well.add_curve("SW", result.sw, "v/v", f"Water Saturation ({result.model})")

Shaly Sand Methods
~~~~~~~~~~~~~~~~~~

For shaly formations, use Simandoux or Indonesia equations:

.. code-block:: python

   # Simandoux model (rcl = clay/shale resistivity, passed to calculate())
   sw_module = SaturationModule(a=1.0, m=2.0, n=2.0, model="simandoux")
   result = sw_module.calculate(well, rw=0.05, rcl=2.0)

   # Indonesia model
   sw_module = SaturationModule(a=1.0, m=2.0, n=2.0, model="indonesia")
   result = sw_module.calculate(well, rw=0.05, rcl=2.0)

Permeability Estimation
-----------------------

Using Timur Equation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from petrophys.interpretation.permeability import PermeabilityModule

   # Timur equation (returns a PermeabilityResult with .perm, .method)
   perm_module = PermeabilityModule(method="timur")
   result = perm_module.calculate(well)
   well.add_curve("PERM", result.perm, "mD", f"Permeability ({result.method})")

   # View permeability statistics
   perm = well.get("PERM")
   print(f"Permeability range: {perm.min():.2f} - {perm.max():.2f} mD")

Core Calibration
~~~~~~~~~~~~~~~~

When core data is available:

.. code-block:: python

   import pandas as pd
   from petrophys.interpretation.core_analysis import (
       CoreData, fit_poro_perm_transform, apply_poro_perm_transform
   )

   # Core plug data. CoreData.from_dataframe defaults to column names
   # DEPTH/CPOR/CPERM/CGD; pass *_col to map custom names.
   core = CoreData.from_dataframe(
       pd.DataFrame({
           "depth": [8050.0, 8100.0, 8150.0],
           "porosity": [0.18, 0.24, 0.12],
           "permeability": [100.0, 250.0, 50.0],
       }),
       depth_col="depth", porosity_col="porosity", perm_col="permeability",
   )

   # Fit log(K) = A * PHI + B from core, then apply to the log porosity
   transform = fit_poro_perm_transform(core)
   calibrated_perm = apply_poro_perm_transform(well, transform)
   well.add_curve("PERM_CORE", calibrated_perm, "mD", "Core-Calibrated Perm")

Net Pay Determination
---------------------

.. code-block:: python

   from petrophys.evaluation import calculate_net_pay

   # Define cutoffs (or pass lithology="carbonate" for lithology-aware PHIE cutoffs)
   result = calculate_net_pay(
       well,
       vcl_cutoff=0.3,    # Max VCL
       phie_cutoff=0.08,  # Min porosity
       sw_cutoff=0.7,     # Max water saturation
   )

   # Add the net-pay flag to the well
   well.add_curve("NET_PAY", result.net_pay.astype(float), "", "Net Pay flag")

   # Summary
   print(f"Gross thickness: {result.gross_thickness:.1f} ft")
   print(f"Net rock: {result.net_rock_thickness:.1f} ft")
   print(f"Net reservoir: {result.net_reservoir_thickness:.1f} ft")
   print(f"Net pay: {result.net_pay_thickness:.1f} ft")
   print(f"NTG: {result.net_pay_thickness/result.gross_thickness:.1%}")

Complete Dynamic Workflow Example
---------------------------------

.. code-block:: python

   from petrophys.io import read_las
   from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig
   from petrophys.evaluation.summaries import calculate_zone_summary
   from petrophys.visualization import create_log_viewer

   # Load data
   well = read_las("well_data.las")

   # Configure workflow
   config = DynamicWorkflowConfig(
       surface_temp_f=70,
       bht_f=180,
       td_ft=9000,
       rw=0.05,
       rho_ma=2.65,
       a=1.0,
       m=2.0,
       n=2.0,
       sw_model="archie",
       vcl_cutoff=0.3,
       phie_cutoff=0.08,
       sw_cutoff=0.7
   )

   # Run interpretation
   result = run_dynamic_workflow(well, config)

   # Full trace: see exactly what happened at each step
   result.print_trace()

   # Zone summary
   zones = [
       {"name": "Zone A", "top": 8000, "bottom": 8200},
       {"name": "Zone B", "top": 8200, "bottom": 8400},
   ]
   for zone in zones:
       summary = calculate_zone_summary(
           result.well, zone["top"], zone["bottom"], zone["name"])
       print(f"{summary.zone_name}: Net Pay={summary.net_pay:.1f} ft, "
             f"PHIE={summary.avg_phie:.3f}, Sw={summary.avg_sw:.3f}")

   # Create plot
   plot = create_log_viewer(result.well, template_defaults="interpretation",
                            depth_range=(8000, 8400))
   plot.save("interpretation.png")

Workflow Trace & Transparency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dynamic workflow provides full step-by-step trace metadata showing exactly
what methods, parameters, thresholds, and decisions were used at each stage.

**Dynamic Workflow trace with DataFrame export:**

.. code-block:: python

   from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig

   config = DynamicWorkflowConfig(
       bht_f=220, td_ft=10522,
       rw=0.04, rmf_at_meas_temp=0.5,
       lithology="clastic",
       gr_method="larionov_tertiary",
       perm_method="timur",
       vcl_cutoff=0.30, phie_cutoff=0.08, sw_cutoff=0.70,
   )

   result = run_dynamic_workflow(well, config)

   # Print human-readable trace to console
   result.print_trace()

   # Export as DataFrame for QC reports
   trace_df = result.to_dataframe()
   print(trace_df[["Step", "Status", "Method", "Input Curves",
                    "Output Curves", "Decisions"]])

   # Drill into a specific step
   for step in result.steps_executed:
       if step.step == "vclay":
           print(f"VCL method: {step.method}")
           print(f"GR min: {step.parameters.get('gr_min')}")
           print(f"GR max: {step.parameters.get('gr_max')}")
           print(f"Used SP: {step.parameters.get('use_sp')}")
           print(f"Used N-D: {step.parameters.get('use_nd')}")

The trace captures:

- **Auto-detection decisions**: GR_min from P5 percentile, SP enabled/disabled and why
- **Method routing**: which Sw model was selected and why (lithology, VCL, salinity)
- **Parameter sources**: user-provided vs auto-detected vs default fallback
- **Output statistics**: VCL range, Sw range, net pay metres

Core Data Module
~~~~~~~~~~~~~~~~

The ``petrophys.interpretation.core_analysis`` module provides tools for working with core plug
measurements and calibrating log curves against laboratory data.

.. code-block:: python

   from petrophys.interpretation.core_analysis import (
       CoreData,
       calibrate_porosity, calibrate_permeability,
       fit_poro_perm_transform, apply_poro_perm_transform,
       correct_porosity_with_grain_density,
       estimate_archie_m,
   )
   from petrophys.visualization.core import (
       plot_calibration, plot_poro_perm, plot_core_on_logs,
   )
   import pandas as pd

   # Create synthetic core data (in practice, load from CSV/Excel).
   # from_dataframe defaults to DEPTH/CPOR/CPERM/CGD columns; map custom names
   # with the *_col arguments.
   core = CoreData.from_dataframe(
       pd.DataFrame({
           "depth": [2500, 2510, 2520, 2530, 2540, 2550, 2560],
           "porosity": [0.18, 0.22, 0.19, 0.25, 0.21, 0.17, 0.23],
           "permeability": [50, 150, 75, 300, 120, 30, 180],
           "grain_density": [2.65, 2.64, 2.66, 2.63, 2.65, 2.67, 2.64],
           "sw": [0.45, 0.35, 0.50, 0.30, 0.40, 0.55, 0.33],
       }),
       depth_col="depth", porosity_col="porosity",
       perm_col="permeability", gd_col="grain_density", sw_col="sw",
   )

   # Calibrate log porosity vs core (returns a CalibrationResult dataclass)
   cal = calibrate_porosity(well, core, log_curve="PHIE")
   print(f"R² = {cal.r_squared:.3f}, Bias = {cal.bias:.4f}")

   # Fit poro-perm transform: log(K) = A * PHI + B
   transform = fit_poro_perm_transform(core)
   perm_cal = apply_poro_perm_transform(well, transform)

   # Correct porosity using core grain density
   phie_corr = correct_porosity_with_grain_density(well, core)

   # Estimate cementation exponent m (uses core Sw and the well's Rt curve)
   m_est = estimate_archie_m(well, core, rw=0.04)
   print(f"Estimated m = {m_est['m_mean']:.2f}")

   # Visualization (plot_calibration takes the CalibrationResult)
   plot_calibration(cal)
   plot_poro_perm(core, transform)
   plot_core_on_logs(well, core)

Visualization Module
~~~~~~~~~~~~~~~~~~~~

The library provides multiple visualization tools for well log analysis.

**Custom multi-track log plot:**

.. code-block:: python

   from petrophys.visualization import LogPlot

   log_plot = LogPlot(well, figsize=(16, 12))
   fig, axes = log_plot.create_figure(n_tracks=4)
   log_plot.add_curve(0, "GR", color="green", xlim=(0, 150))
   log_plot.add_fill(0, "GR", fill_value=0, color="yellow", alpha=0.3)
   log_plot.add_curve(1, "RDEP", color="red", log_scale=True, xlim=(0.2, 2000))
   log_plot.add_curve(2, "PHIE", color="blue", xlim=(0, 0.4))
   log_plot.add_curve(3, "SW", color="blue", xlim=(0, 1))
   log_plot.set_depth_range(top=2200, bottom=3000)
   log_plot.show()

**Full interpretation plot (8 tracks):**

.. code-block:: python

   from petrophys.visualization import create_log_viewer

   interp = create_log_viewer(well, template_defaults="interpretation",
                              depth_range=(2200, 3200))
   interp.show()

**Density-Neutron crossplot with mineral endpoints:**

.. code-block:: python

   from petrophys.visualization import create_crossplot

   plot = create_crossplot(well, x_curve="NPHI", y_curve="RHOB",
                           color_curve="GR", xlim=(-0.05, 0.45),
                           ylim=(2.95, 1.95), depth_range=(2200, 3200))
   plot.add_matrix_points({
       "Sandstone": (-0.02, 2.65), "Limestone": (0.00, 2.71),
       "Dolomite": (0.02, 2.87),
   })
   plot.show()

**Pickett plot for Archie parameter estimation:**

.. code-block:: python

   from petrophys.visualization import create_pickett_plot

   pickett = create_pickett_plot(well, rw=0.05, phie_curve="PHIE",
                                 rt_curve="RDEP", color_curve="VCL")
   pickett.show()

**Formation shading on log plots:**

.. code-block:: python

   from petrophys.visualization import add_zone_shading
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(6, 14))
   ax.plot(well["GR"].data, well.depth, 'g', linewidth=0.5)
   ax.set_ylim(3200, 2200)

   zones = [
       {"name": "Balder Fm.", "top": 2208, "bottom": 2243, "color": "lightyellow"},
       {"name": "Tor Fm.",    "top": 2418, "bottom": 2606, "color": "lightsalmon"},
   ]
   add_zone_shading(ax, zones, alpha=0.3)
   plt.show()


Interval-Based Formation Analysis
----------------------------------

This tutorial demonstrates the full capabilities of the ``petrophys.evaluation.intervals``
module for analyzing well data by geological intervals.

Setting Up Interval Hierarchies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A typical well has a hierarchy of intervals: formations contain zones,
zones contain sub-zones. You can also define arbitrary manual intervals
for specific depth ranges of interest.

.. code-block:: python

   from petrophys.evaluation.intervals import (
       Formation, Zone, SubZone, ManualInterval, FormationTops
   )

   # Build a hierarchical set of intervals
   tops = FormationTops([
       # Formations (top-level geological units)
       Formation("Hordaland Group", 800, 1908, lithology="clastic"),
       Formation("Balder Fm.", 1908, 2023, lithology="clastic", age="Eocene"),
       Formation("Sele Fm.", 2023, 2150, lithology="clastic", age="Paleocene"),
       Formation("Brent Group", 2841, 2958, lithology="clastic", age="Middle Jurassic"),

       # Zones within Brent Group
       Zone("Tarbert Fm.", 2841, 2880, parent_formation="Brent Group"),
       Zone("Ness Fm.", 2880, 2920, parent_formation="Brent Group"),
       Zone("Etive Fm.", 2920, 2958, parent_formation="Brent Group"),

       # Sub-zones within Ness Fm.
       SubZone("Ness Upper", 2880, 2900, parent_zone="Ness Fm."),
       SubZone("Ness Lower", 2900, 2920, parent_zone="Ness Fm."),

       # Manual interval for a specific test zone
       ManualInterval("DST #1 Interval", 2850, 2950),
   ])

   # All intervals are auto-sorted by top depth
   print(f"Total intervals: {len(tops)}")
   print(f"Names: {tops.names}")

Loading Formations from Real-World CSV Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Case 1: Simple CSV with lithology**

.. code-block:: text

   formation_name,top_depth,bottom_depth,lithology
   Balder Fm.,1908.0,2023.0,Sandstone
   Sele Fm.,2023.0,2150.0,Shale
   Brent Group,2841.0,2958.0,Sandstone

.. code-block:: python

   tops = FormationTops.from_csv("formations.csv")
   print(tops)  # FormationTops(3 intervals)

**Case 2: NPD stratigraphic picks (multi-well, tops-only)**

NPD provides formation tops (single depth per formation). The loader
automatically converts these to intervals using the next formation top
as the bottom boundary.

.. code-block:: text

   Well identifier,Surface,MD,Geological age
   30/6-5,BALDER FM Top,1908.0,Eocene
   30/6-5,SELE FM Top,2023.0,Paleocene
   30/6-5,BRENT GP Top,2841.0,Middle Jurassic
   31/2-1,BALDER FM Top,1650.0,Eocene

.. code-block:: python

   # Filter to a single well
   tops = FormationTops.from_csv(
       "npd_strat_picks.csv",
       well_name="30/6-5",
       format="npd"
   )

**Case 3: Danish format (multi-well with lithology)**

.. code-block:: text

   wellName,formationName,topDepth,bottomDepth,majorityLithology
   DK-1,Chalk Group,800.0,1600.0,Limestone
   DK-1,Bunter Sandstone,1600.0,2200.0,Sandstone
   DK-2,Chalk Group,750.0,1500.0,Limestone

.. code-block:: python

   tops = FormationTops.from_csv(
       "danish_wells.csv",
       well_name="DK-1",
       format="danish"
   )

**Case 4: Loading zones (not formations)**

.. code-block:: python

   from petrophys.evaluation.intervals import IntervalKind

   # Load as Zone objects instead of Formation
   zone_tops = FormationTops.from_csv(
       "zones.csv",
       interval_kind=IntervalKind.ZONE
   )

**Case 5: From Python dictionaries**

.. code-block:: python

   tops = FormationTops.from_dict([
       {"name": "Unit A", "top_depth": 2000, "bottom_depth": 2200,
        "lithology": "clastic", "age": "Cretaceous"},
       {"name": "Unit B", "top_depth": 2200, "bottom_depth": 2500,
        "lithology": "carbonate"},
   ])

**Case 6: From a pandas DataFrame**

.. code-block:: python

   import pandas as pd

   df = pd.DataFrame({
       "name": ["Zone 1", "Zone 2", "Zone 3"],
       "top_depth": [1000, 1500, 2000],
       "bottom_depth": [1500, 2000, 2500],
       "lithology": ["clastic", "carbonate", "clastic"],
   })

   tops = FormationTops.from_dataframe(df)

Validation and Quality Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before analysis, validate that intervals don't overlap:

.. code-block:: python

   # Check for overlaps (raises ValueError if found)
   try:
       tops.validate()
       print("All intervals valid - no overlaps detected.")
   except ValueError as e:
       print(f"Validation failed: {e}")

   # Non-raising check
   if tops.has_overlaps():
       for issue in tops.find_overlaps():
           print(f"  WARNING: {issue}")

Note: Overlap checks are done **within the same interval kind**. A Formation
and a Zone can overlap (that's expected — zones are inside formations), but
two Formations should not overlap.

Querying Intervals
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # By name (case-insensitive)
   brent = tops.get_by_name("brent group")
   if brent:
       print(f"{brent.name}: {brent.top_depth} - {brent.bottom_depth}")

   # All intervals at a specific depth
   at_2900 = tops.get_at_depth(2900.0)
   for iv in at_2900:
       print(f"  {iv.kind.value}: {iv.name}")

   # Intervals in a depth range
   in_range = tops.get_in_range(2800.0, 3000.0)

   # Only formations (exclude zones, sub-zones, manual)
   fms = tops.filter_by_kind(IntervalKind.FORMATION)

   # Clip to well depth range
   clipped = tops.clip_to_well(well)
   print(f"Before: {len(tops)}, After clip: {len(clipped)}")

Running Per-Interval Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``IntervalAnalyzer`` computes comprehensive summaries for each interval.
The well must already have petrophysical curves computed (VCL, PHIE, SW,
PERM, NET_PAY, etc.) from a workflow run.

**Analyze all formations:**

.. code-block:: python

   from petrophys.evaluation.intervals import FormationTops, IntervalAnalyzer

   # Load formations and run analysis
   tops = FormationTops.from_csv("formations.csv")
   analyzer = IntervalAnalyzer(well)
   summaries = analyzer.analyze_all(tops)

   # Print formatted report
   #
   # ===============================...
   # INTERVAL ANALYSIS REPORT
   # ===============================...
   # Name                       Top      Bot   Gross  NtPay    NTG   PHIE     Sw    VCL     Perm
   # ---------------------------...
   # Balder Fm.              1908.0  2023.0   115.0   42.5  37.0% 0.185  0.352  0.118    25.30
   # Brent Group             2841.0  2958.0   117.0   78.3  66.9% 0.210  0.285  0.095   120.50
   # ---------------------------...
   # TOTAL                                    232.0  120.8
   # ===============================...
   IntervalAnalyzer.print_report(summaries)

**Export results:**

.. code-block:: python

   # To CSV
   IntervalAnalyzer.to_csv(summaries, "interval_results.csv")

   # To pandas DataFrame
   df = IntervalAnalyzer.to_dataframe(summaries)
   print(df[["Name", "Gross", "Net Pay", "NTG", "Avg PHIE", "Avg Sw"]])

**Analyze a single interval:**

.. code-block:: python

   from petrophys.evaluation.intervals import analyze_interval, ManualInterval

   summary = analyze_interval(
       well,
       ManualInterval("Test Zone", 2850, 2950),
   )

   zs = summary.zone_summary
   print(f"Net Pay: {zs.net_pay:.1f} ft, NTG: {zs.ntg:.1%}")
   print(f"Avg PHIE: {zs.avg_phie:.3f}, Avg Sw: {zs.avg_sw:.3f}")
   print(f"Dominant lithology: {summary.dominant_lithology}")

Full Analysis Examples
~~~~~~~~~~~~~~~~~~~~~~

The following examples show complete end-to-end workflows combining LAS data,
lithology CSV, and formation/zone/sub-zone hierarchies.

Example 1: Formation Analysis with Lithology CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example uses a real North Sea well (30/6-5) with per-interval lithology
driving Sw model selection and net pay cutoffs, then analyzes results by
geological formation.

**Input files:**

``lithology_30_6-5.csv`` — per-interval lithology:

.. code-block:: text

   top,bottom,lithology
   181.0,726.0,shale
   726.0,879.0,sandstone
   879.0,1908.0,shale
   1908.0,2023.0,sandstone
   2023.0,2203.0,shale
   2203.0,2408.0,chalk
   2408.0,2452.0,marl
   2452.0,2841.0,shale
   2841.0,2958.0,sandstone
   2958.0,3071.0,shale
   3071.0,3123.0,sandstone
   3123.0,3223.0,shale
   3223.0,3556.0,sandstone

``formations_30_6-5.csv`` — formation tops:

.. code-block:: text

   formation_name,top_depth,bottom_depth,lithology
   NORDLAND GP.,181.0,726.0,Shale
   Utsira Fm.,726.0,879.0,Sandstone
   HORDALAND GP.,879.0,1908.0,Shale
   ROGALAND GP.,1908.0,2203.0,Shale
   Balder Fm.,1908.0,2023.0,Sandstone
   SHETLAND GP.,2203.0,2408.0,Chalk
   CROMER KNOLL GP.,2408.0,2452.0,Marl
   VIKING GP.,2452.0,2841.0,Shale
   BAAT GP.,2841.0,2958.0,Sandstone
   DUNLIN GP.,2958.0,3223.0,Shale
   Statfjord Fm.,3223.0,3556.0,Sandstone

**Code:**

.. code-block:: python

   from petrophys.io import read_las
   from petrophys.preprocessing.standardization import standardize_curves
   from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig
   from petrophys.evaluation.intervals import FormationTops, IntervalAnalyzer

   # 1. Load and standardize
   well = read_las("30_6-5.las")
   well = standardize_curves(well, rename_curves=True, convert_units=True)

   # 2. Run workflow WITH per-interval lithology
   #    - Chalk intervals -> "carbonate" -> Dual Water Sw model, PHIE cutoff 0.045
   #    - Sandstone intervals -> "clastic" -> Simandoux Sw model, PHIE cutoff 0.08
   config = DynamicWorkflowConfig(
       bht_f=258.8, td_ft=11667.0, surface_temp_f=50.0,
       rw=0.070,   # header RW=0.70 reconciled to reservoir temperature (see tutorials)
       salinity_ppm=35000,
       lithology="clastic",             # default for uncovered depths
       lithology_csv="lithology_30_6-5.csv",  # per-interval lithology
       matrix_type="sandstone",
       hc_type="oil",
   )
   result = run_dynamic_workflow(well, config)
   print(f"Steps: {[s.step for s in result.steps_executed]}")

   # 3. Load formations and analyze
   tops = FormationTops.from_csv("formations_30_6-5.csv")
   print(f"Loaded {len(tops)} formations")

   analyzer = IntervalAnalyzer(well)
   summaries = analyzer.analyze_all(tops)

   # 4. Report
   IntervalAnalyzer.print_report(summaries)

   # 5. Export
   IntervalAnalyzer.to_csv(summaries, "formation_results.csv")
   df = IntervalAnalyzer.to_dataframe(summaries)
   print(df[["Name", "Gross", "Net Pay", "NTG", "Avg PHIE", "Avg Sw",
             "Dominant Lithology"]].to_string(index=False))

Example 2: Hierarchical Analysis (Formation + Zone + Sub-Zone)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze the same well at three levels of detail: the full formation, then
zones within it, then sub-zones. This shows how results nest at different
scales.

.. code-block:: python

   from petrophys.evaluation.intervals import (
       FormationTops, Formation, Zone, SubZone, ManualInterval,
       IntervalAnalyzer, IntervalKind,
   )

   # Define a hierarchy for the Brent Group (BAAT GP.)
   tops = FormationTops([
       # Level 1: Full formation
       Formation("BAAT GP.", 2841, 2958, lithology="clastic",
                 age="Middle Jurassic"),

       # Level 2: Zones within the formation
       Zone("Ness Fm.", 2841, 2865, parent_formation="BAAT GP."),
       Zone("Etive Fm.", 2865, 2920, parent_formation="BAAT GP."),
       Zone("Rannoch Fm.", 2920, 2958, parent_formation="BAAT GP."),

       # Level 3: Sub-zones within Etive Fm.
       SubZone("Etive Upper", 2865, 2890, parent_zone="Etive Fm."),
       SubZone("Etive Lower", 2890, 2920, parent_zone="Etive Fm."),

       # A manual interval across zone boundaries (e.g., DST test interval)
       ManualInterval("DST #1", 2855, 2930),
   ])

   # Analyze all
   analyzer = IntervalAnalyzer(well)
   summaries = analyzer.analyze_all(tops)
   IntervalAnalyzer.print_report(summaries)

   # Filter and compare by level
   df = IntervalAnalyzer.to_dataframe(summaries)

   print("\n--- Formation level ---")
   print(df[df["Kind"] == "formation"][
       ["Name", "Gross", "Net Pay", "NTG", "Avg PHIE"]
   ].to_string(index=False))

   print("\n--- Zone level ---")
   print(df[df["Kind"] == "zone"][
       ["Name", "Gross", "Net Pay", "NTG", "Avg PHIE"]
   ].to_string(index=False))

   print("\n--- Sub-zone level ---")
   print(df[df["Kind"] == "sub_zone"][
       ["Name", "Gross", "Net Pay", "NTG", "Avg PHIE"]
   ].to_string(index=False))

   print("\n--- Manual intervals ---")
   print(df[df["Kind"] == "manual"][
       ["Name", "Gross", "Net Pay", "NTG", "Avg PHIE"]
   ].to_string(index=False))

Example 3: Multi-Format CSV Loading with Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load formations from different CSV sources, merge them, validate,
and analyze.

.. code-block:: python

   from petrophys.evaluation.intervals import FormationTops, IntervalKind, IntervalAnalyzer

   # Load formations from NPD stratigraphic picks
   fm_tops = FormationTops.from_csv(
       "npd_strat_picks.csv",
       well_name="30/6-5",
       format="npd"
   )
   print(f"NPD formations: {len(fm_tops)}")

   # Load zones from a separate simple CSV
   zone_tops = FormationTops.from_csv(
       "zones.csv",
       interval_kind=IntervalKind.ZONE
   )
   print(f"Zones: {len(zone_tops)}")

   # Merge into one collection
   merged = FormationTops(fm_tops.intervals + zone_tops.intervals)

   # Validate: overlaps within the same kind are flagged,
   # but a Zone overlapping a Formation is OK
   if merged.has_overlaps():
       for issue in merged.find_overlaps():
           print(f"  WARNING: {issue}")
   else:
       print("No overlaps detected.")

   # Clip to well and analyze
   clipped = merged.clip_to_well(well)
   print(f"Intervals within well range: {len(clipped)}")

   analyzer = IntervalAnalyzer(well)
   summaries = analyzer.analyze_all(clipped, clip_to_well=False)  # already clipped
   IntervalAnalyzer.print_report(summaries)

Example 4: Comparing Global vs Per-Depth Lithology Impact
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the workflow twice — once with a single global lithology, once with
per-interval lithology from CSV — and compare how it affects net pay.

.. code-block:: python

   from petrophys.io import read_las
   from petrophys.preprocessing.standardization import standardize_curves
   from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig
   from petrophys.evaluation.intervals import FormationTops, IntervalAnalyzer, Formation
   import numpy as np

   # Load well
   well_global = read_las("30_6-5.las")
   well_global = standardize_curves(well_global, rename_curves=True, convert_units=True)

   well_perdepth = read_las("30_6-5.las")
   well_perdepth = standardize_curves(well_perdepth, rename_curves=True, convert_units=True)

   # Run A: global clastic lithology
   config_a = DynamicWorkflowConfig(
       bht_f=258.8, td_ft=11667.0, surface_temp_f=50.0,
       rw=0.070,   # header RW=0.70 reconciled to reservoir temperature (see tutorials) salinity_ppm=35000,
       lithology="clastic",          # everything treated as clastic
       matrix_type="sandstone",
   )
   run_dynamic_workflow(well_global, config_a)

   # Run B: per-interval lithology from CSV
   config_b = DynamicWorkflowConfig(
       bht_f=258.8, td_ft=11667.0, surface_temp_f=50.0,
       rw=0.070,   # header RW=0.70 reconciled to reservoir temperature (see tutorials) salinity_ppm=35000,
       lithology="clastic",
       lithology_csv="lithology_30_6-5.csv",  # chalk -> carbonate -> Dual Water + lower cutoff
       matrix_type="sandstone",
   )
   run_dynamic_workflow(well_perdepth, config_b)

   # Compare at the SHETLAND GP. (Chalk interval: 2203 - 2408 m)
   chalk_fm = Formation("SHETLAND GP.", 2203, 2408, lithology="carbonate")

   analyzer_a = IntervalAnalyzer(well_global)
   analyzer_b = IntervalAnalyzer(well_perdepth)

   summary_a = analyzer_a.analyze(chalk_fm)
   summary_b = analyzer_b.analyze(chalk_fm)

   print("SHETLAND GP. (Chalk) Comparison:")
   print(f"{'':30s} {'Global Clastic':>16s} {'Per-Depth Litho':>16s}")
   print(f"{'Net Pay (m)':30s} {summary_a.zone_summary.net_pay:>16.1f} "
         f"{summary_b.zone_summary.net_pay:>16.1f}")
   print(f"{'NTG':30s} {summary_a.zone_summary.ntg:>16.1%} "
         f"{summary_b.zone_summary.ntg:>16.1%}")
   print(f"{'Avg PHIE':30s} {summary_a.zone_summary.avg_phie:>16.3f} "
         f"{summary_b.zone_summary.avg_phie:>16.3f}")
   print(f"{'Avg Sw':30s} {summary_a.zone_summary.avg_sw:>16.3f} "
         f"{summary_b.zone_summary.avg_sw:>16.3f}")
   print(f"{'Dominant Lithology':30s} {summary_a.dominant_lithology:>16s} "
         f"{summary_b.dominant_lithology:>16s}")
   print()
   print("Note: Per-depth lithology uses Dual Water Sw model and PHIE cutoff")
   print("0.045 for the chalk interval, vs Simandoux and 0.08 for global clastic.")

Example 5: Building Intervals Programmatically from a DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When formation picks come from a database or spreadsheet, build the
hierarchy from a DataFrame and sub-zone columns.

.. code-block:: python

   import pandas as pd
   from petrophys.evaluation.intervals import (
       FormationTops, Formation, Zone, SubZone,
       IntervalKind, IntervalAnalyzer,
   )

   # Simulated database query result
   df = pd.DataFrame({
       "formation": ["Brent", "Brent", "Brent", "Brent", "Brent", "Brent"],
       "zone":      ["Upper", "Upper", "Upper", "Lower", "Lower", "Lower"],
       "subzone":   ["A",     "B",     "C",     "D",     "E",     "F"],
       "top":       [2841,    2860,    2880,    2900,    2920,    2940],
       "bottom":    [2860,    2880,    2900,    2920,    2940,    2958],
       "lithology": ["clastic"]*6,
   })

   intervals = []

   # Build Formation from the full range
   intervals.append(Formation(
       "Brent Group",
       top_depth=df["top"].min(),
       bottom_depth=df["bottom"].max(),
       lithology="clastic",
   ))

   # Build Zones from unique zone names
   for zone_name, grp in df.groupby("zone"):
       intervals.append(Zone(
           f"Brent {zone_name}",
           top_depth=grp["top"].min(),
           bottom_depth=grp["bottom"].max(),
           parent_formation="Brent Group",
       ))

   # Build SubZones from each row
   for _, row in df.iterrows():
       intervals.append(SubZone(
           f"{row['zone']}-{row['subzone']}",
           top_depth=row["top"],
           bottom_depth=row["bottom"],
           parent_zone=f"Brent {row['zone']}",
       ))

   tops = FormationTops(intervals)
   print(f"Built {len(tops)} intervals:")
   for iv in tops:
       print(f"  [{iv.kind.value:10s}] {iv.name:20s}  "
             f"{iv.top_depth:.0f} - {iv.bottom_depth:.0f}")

   # Validate
   tops.validate()

   # Analyze
   analyzer = IntervalAnalyzer(well)
   summaries = analyzer.analyze_all(tops)
   IntervalAnalyzer.print_report(summaries)

   # Export all levels to a single CSV
   IntervalAnalyzer.to_csv(summaries, "brent_multi_level_analysis.csv")

Example 6: Exporting Formation Tops and Re-Importing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Round-trip: load tops, clip to well, export, re-import.

.. code-block:: python

   from petrophys.evaluation.intervals import FormationTops

   # Load and clip
   tops = FormationTops.from_csv("formations_30_6-5.csv")
   clipped = tops.clip_to_well(well)

   # Export to CSV
   clipped.to_csv("clipped_formations.csv")

   # Export to DataFrame for further manipulation
   df = clipped.to_dataframe()
   print(df[["name", "top_depth", "bottom_depth", "thickness", "lithology"]])

   # Re-import from the exported CSV
   reimported = FormationTops.from_csv("clipped_formations.csv")
   print(f"Re-imported {len(reimported)} formations")

   # Re-import from a modified DataFrame
   df["lithology"] = "carbonate"  # change all to carbonate
   modified_tops = FormationTops.from_dataframe(
       df, name_col="name", top_col="top_depth", bottom_col="bottom_depth"
   )
