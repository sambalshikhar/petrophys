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

   from petrophys.clay_volume import calculate_vcl, vcl_from_gr, calculate_rcl

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

Per-Interval Lithology from CSV
--------------------------------

Drive lithology-dependent calculations (Sw routing, net pay cutoffs) with a
per-interval lithology CSV file:

.. code-block:: text

   top,bottom,lithology
   200.0,1500.0,sandstone
   1500.0,2200.0,chalk
   2200.0,3500.0,sandstone

**Standalone usage:**

.. code-block:: python

   from petrophys.lithology.lithology_log import (
       load_lithology_csv, build_lithology_log
   )

   # Load intervals (auto-detects columns, normalizes lithology names)
   intervals = load_lithology_csv("lithology.csv")

   # Build per-depth array
   litho_log = build_lithology_log(well.depth, intervals)

   # Use with Sw module (per-depth model routing)
   from petrophys.petrophysics.saturation import calculate_sw
   sw = calculate_sw(well, rw=0.05, lithology=litho_log, salinity=25000)

   # Use with Net Pay (per-depth PHIE cutoffs)
   from petrophys.reservoir import calculate_net_pay
   result = calculate_net_pay(well, lithology=litho_log)

**With dynamic workflow:**

.. code-block:: python

   config = DynamicWorkflowConfig(
       rw=0.05,
       salinity_ppm=25000,
       lithology="clastic",             # default for uncovered depths
       lithology_csv="lithology.csv",   # per-interval lithology
   )
   result = run_dynamic_workflow(well, config)

   # LITHO_CLASS curve is added to well automatically

Lithology names are normalized automatically:
``sandstone/sand/sst`` → ``clastic``,
``limestone/chalk/dolomite`` → ``carbonate``,
``shale/mudstone/marl`` → ``clastic``.

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

   # Full trace: every step's method, parameters, and decisions
   result.print_trace()

   # Access individual step traces
   for step in result.steps:
       print(f"[{step.step}] method={step.method}, "
             f"inputs={step.input_curves}, outputs={step.output_curves}")
       for d in step.decisions:
           print(f"    -> {d}")

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

   # Full trace: shows method, inputs, outputs, parameters, and decisions for each step
   result.print_trace()

   # Export trace as a DataFrame (useful for QC reports)
   trace_df = result.to_dataframe()
   print(trace_df[["Step", "Method", "Input Curves", "Output Curves"]])

   # Inspect a specific step's trace
   for step in result.steps_executed:
       if step.step == "saturation":
           print(f"Sw method: {step.method}")
           print(f"Parameters: {step.parameters}")
           for d in step.decisions:
               print(f"  -> {d}")

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

**Custom Log Plot (LogPlot)**

.. code-block:: python

   from petrophys.visualization import LogPlot, create_log_plot

   # Step-by-step API
   log_plot = LogPlot(well, figsize=(16, 12))
   fig, axes = log_plot.create_figure(n_tracks=4)
   log_plot.add_curve(0, "GR", color="green", xlim=(0, 150))
   log_plot.add_curve(1, "RDEP", color="red", log_scale=True, xlim=(0.2, 2000))
   log_plot.add_curve(2, "PHIE", color="blue", xlim=(0, 0.4))
   log_plot.add_curve(3, "SW", color="blue", xlim=(0, 1))
   log_plot.set_depth_range(top=2200, bottom=3000)
   log_plot.show()

   # Or one-liner with curve specs
   plot = create_log_plot(well, curves=[
       {"track": 0, "name": "GR",   "color": "green", "xlim": (0, 150)},
       {"track": 1, "name": "RDEP", "color": "red",   "log_scale": True},
       {"track": 2, "name": "PHIE", "color": "blue",  "xlim": (0, 0.4)},
       {"track": 3, "name": "SW",   "color": "blue",  "xlim": (0, 1)},
   ], depth_range=(2200, 3000))

**Triple Combo Plot**

.. code-block:: python

   from petrophys.visualization import create_triple_combo, TripleComboPlot

   plot = create_triple_combo(well, figsize=(14, 14))
   plot.show()

   # Zoomed with custom ranges
   tc = TripleComboPlot(well, figsize=(14, 10))
   tc.create(depth_range=(2200, 2850), gr_range=(0, 120),
             rt_range=(0.5, 500), rhob_range=(2.0, 2.85),
             nphi_range=(0.40, -0.10))
   tc.show()

**Interpretation Plot (8 Tracks)**

.. code-block:: python

   from petrophys.visualization import create_interpretation_plot

   # GR | Resistivity | Density-Neutron | VCL | Porosity | Sw | Perm | Net Pay
   interp = create_interpretation_plot(well, depth_range=(2200, 3200))
   interp.show()

**Crossplot**

.. code-block:: python

   from petrophys.visualization import create_crossplot

   plot = create_crossplot(
       well, x_curve="NPHI", y_curve="RHOB",
       color_curve="GR", xlim=(-0.05, 0.45), ylim=(2.95, 1.95),
       cmap="RdYlGn_r", depth_range=(2200, 3200)
   )
   plot.add_matrix_points({
       "Sandstone": (-0.02, 2.65),
       "Limestone": (0.00, 2.71),
       "Dolomite":  (0.02, 2.87),
   })
   plot.show()

**Pickett Plot**

.. code-block:: python

   from petrophys.visualization import create_pickett_plot

   plot = create_pickett_plot(well, rw=0.05, phie_curve="PHIE",
                              rt_curve="RDEP", color_curve="VCL",
                              depth_range=(2200, 3200))
   plot.show()

**Zone Flags & Formation Shading**

.. code-block:: python

   from petrophys.visualization import plot_zone_flags, add_zone_shading

   # Net pay flag display
   fig, ax = plot_zone_flags(well,
       flag_curves=["NET_ROCK", "NET_RES", "NET_PAY"],
       depth_range=(2200, 3200))

   # Add formation shading to any matplotlib axes
   zones = [
       {"name": "Balder Fm.", "top": 2208, "bottom": 2243, "color": "lightyellow"},
       {"name": "Tor Fm.",    "top": 2418, "bottom": 2606, "color": "lightsalmon"},
   ]
   add_zone_shading(ax, zones, alpha=0.3)

Core Data Module
----------------

Load core plug measurements and calibrate log curves against lab data.

**Loading Core Data**

.. code-block:: python

   from petrophys.core import CoreData
   import pandas as pd

   # From a CSV file
   core = CoreData.from_csv("core_data.csv")

   # From a DataFrame
   df = pd.DataFrame({
       "depth": [2500.0, 2510.0, 2520.0, 2530.0, 2540.0],
       "porosity": [0.18, 0.22, 0.19, 0.25, 0.21],
       "permeability": [50.0, 150.0, 75.0, 300.0, 120.0],
       "grain_density": [2.65, 2.64, 2.66, 2.63, 2.65],
   })
   core = CoreData.from_dataframe(df)

   # From a dictionary
   core = CoreData.from_dict({
       "depth": [2500.0, 2510.0, 2520.0],
       "porosity": [0.18, 0.22, 0.19],
       "permeability": [50.0, 150.0, 75.0],
   })

**Porosity & Permeability Calibration**

.. code-block:: python

   from petrophys.core import calibrate_porosity, calibrate_permeability

   # Compare core vs log porosity (R², bias, RMSE)
   cal_phi = calibrate_porosity(well, core, log_curve="PHIE")
   print(f"R²={cal_phi['r_squared']:.3f}, Bias={cal_phi['bias']:.4f}")

   # Compare core vs log permeability
   cal_k = calibrate_permeability(well, core, log_curve="PERM")

**Poro-Perm Transform**

.. code-block:: python

   from petrophys.core import fit_poro_perm_transform, apply_poro_perm_transform

   # Fit log(K) = A * PHI + B from core data
   transform = fit_poro_perm_transform(core)
   print(f"A={transform['A']:.2f}, B={transform['B']:.2f}, R²={transform['r_squared']:.3f}")

   # Apply to log porosity to get core-calibrated permeability
   perm_calibrated = apply_poro_perm_transform(well, transform)
   well.add_curve("PERM_CORE", perm_calibrated, "mD", "Core-Calibrated Perm")

**Grain Density Correction**

.. code-block:: python

   from petrophys.core import correct_porosity_with_grain_density

   # Use core grain density instead of assumed matrix density
   phie_corrected = correct_porosity_with_grain_density(well, core, rho_fl=1.0)
   well.add_curve("PHIE_CORE", phie_corrected, "v/v", "Core-Corrected Porosity")

**Archie m Estimation**

.. code-block:: python

   from petrophys.core import estimate_archie_m

   # Estimate cementation exponent from core Sw and Rt
   m_result = estimate_archie_m(core, rw=0.04)
   print(f"Estimated m = {m_result['m']:.2f}")

**Visualization**

.. code-block:: python

   from petrophys.core import plot_calibration, plot_poro_perm, plot_core_on_logs

   # Core vs log comparison scatter plots
   plot_calibration(well, core, curve="porosity")
   plot_poro_perm(core, transform)
   plot_core_on_logs(well, core)

Interval-Based Analysis (Formation Module)
-------------------------------------------

The formation module provides a unified framework for analyzing well data by
**formations**, **zones**, **sub-zones**, or **manual depth intervals**.

Interval Types
~~~~~~~~~~~~~~

Four interval types share a common base class with ``name``, ``top_depth``,
``bottom_depth``, ``lithology``, and computed ``thickness``:

.. code-block:: python

   from petrophys.formation import (
       Formation, Zone, SubZone, ManualInterval, IntervalKind
   )

   # Formation — geological unit (with optional age)
   fm = Formation("Brent Group", top_depth=2841.0, bottom_depth=2958.0,
                  lithology="clastic", age="Middle Jurassic")

   # Zone — petrophysical zone within a formation
   z = Zone("Brent Upper", top_depth=2841.0, bottom_depth=2900.0,
            parent_formation="Brent Group")

   # Sub-zone — finer subdivision
   sz = SubZone("Brent Upper A", top_depth=2841.0, bottom_depth=2870.0,
                parent_zone="Brent Upper")

   # Manual — arbitrary user-defined interval
   mi = ManualInterval("Test Interval", top_depth=2500.0, bottom_depth=2600.0)

   # Common interface
   print(fm.thickness)        # 117.0
   print(fm.contains(2900))   # True
   print(fm.overlaps(z))      # True
   print(fm.kind)             # IntervalKind.FORMATION

Loading Formation Tops from CSV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``FormationTops`` class loads, validates, and manages collections of
intervals from multiple CSV formats.

**Simple CSV** (``formations.csv``):

.. code-block:: text

   formation_name,top_depth,bottom_depth,lithology
   Balder Fm.,1908.0,2023.0,Sandstone
   Draupne Fm.,2452.0,2593.0,Shale
   Brent Group,2841.0,2958.0,Sandstone

**NPD stratigraphic picks** (``npd_picks.csv``):

.. code-block:: text

   Well identifier,Surface,MD,Geological age
   30/6-5,BALDER FM Top,1908.0,Eocene
   30/6-5,DRAUPNE FM Top,2452.0,Late Jurassic

**Danish format** (``danish.csv``):

.. code-block:: text

   wellName,formationName,topDepth,bottomDepth,majorityLithology
   Well-1,Chalk,1200.0,1800.0,Limestone
   Well-1,Bunter,1800.0,2400.0,Sandstone

.. code-block:: python

   from petrophys.formation import FormationTops

   # Auto-detect format from column headers
   tops = FormationTops.from_csv("formations.csv")

   # NPD format — filter by well name
   tops = FormationTops.from_csv("npd_picks.csv", well_name="30/6-5", format="npd")

   # Danish format — filter by well name
   tops = FormationTops.from_csv("danish.csv", well_name="Well-1", format="danish")

   # Load as Zone intervals instead of Formation
   from petrophys.formation import IntervalKind
   zone_tops = FormationTops.from_csv(
       "zones.csv", interval_kind=IntervalKind.ZONE
   )

Loading from Dictionaries and DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # From a list of dicts
   tops = FormationTops.from_dict([
       {"name": "Balder", "top_depth": 1908, "bottom_depth": 2023, "lithology": "clastic"},
       {"name": "Brent",  "top_depth": 2841, "bottom_depth": 2958, "lithology": "clastic"},
   ])

   # From a pandas DataFrame
   import pandas as pd
   df = pd.DataFrame({
       "name": ["Upper", "Middle", "Lower"],
       "top_depth": [2000, 2200, 2400],
       "bottom_depth": [2200, 2400, 2600],
       "lithology": ["clastic", "carbonate", "clastic"],
   })
   tops = FormationTops.from_dataframe(df)

Validation and Querying
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Validate — raises ValueError if overlapping intervals exist
   tops.validate()

   # Or check without raising
   if tops.has_overlaps():
       issues = tops.find_overlaps()
       for issue in issues:
           print(issue)

   # Query by name (case-insensitive)
   fm = tops.get_by_name("Brent Group")

   # Query by depth — returns all intervals containing this depth
   at_depth = tops.get_at_depth(2900.0)

   # Query by range — returns intervals overlapping [top, bottom]
   in_range = tops.get_in_range(2800.0, 3000.0)

   # Filter by kind
   formations_only = tops.filter_by_kind(IntervalKind.FORMATION)

   # Clip to well depth range (removes/trims intervals outside well)
   clipped = tops.clip_to_well(well)

   # Collection interface
   print(len(tops))        # number of intervals
   print(tops.names)       # list of interval names
   print(tops[0])          # first interval (by index)
   print(tops["Brent"])    # by name

   # Add / remove
   tops.add(Formation("New Fm", 3000, 3200))
   tops.remove("New Fm")

   # Export
   tops.to_csv("exported_tops.csv")
   df = tops.to_dataframe()

Per-Interval Analysis
~~~~~~~~~~~~~~~~~~~~~

The ``IntervalAnalyzer`` computes per-interval summaries including net pay,
NTG, pay-zone averages, whole-interval curve averages, dominant lithology,
and data coverage.

**Single interval:**

.. code-block:: python

   from petrophys.formation import IntervalAnalyzer, Formation

   # Analyze a single formation (well must have computed curves)
   analyzer = IntervalAnalyzer(well)
   summary = analyzer.analyze(Formation("Brent", 2841, 2958))

   # Access zone summary (net pay, NTG, pay averages)
   zs = summary.zone_summary
   print(f"Gross:    {zs.gross_thickness:.1f} ft")
   print(f"Net Rock: {zs.net_rock:.1f} ft")
   print(f"Net Res:  {zs.net_reservoir:.1f} ft")
   print(f"Net Pay:  {zs.net_pay:.1f} ft")
   print(f"NTG:      {zs.ntg:.1%}")
   print(f"Avg PHIE: {zs.avg_phie:.3f}")
   print(f"Avg Sw:   {zs.avg_sw:.3f}")
   print(f"HCPV:     {zs.hcpv:.2f}")

   # Whole-interval averages
   print(f"Avg GR:   {summary.avg_gr:.1f}")
   print(f"Lithology: {summary.dominant_lithology}")
   print(f"Coverage: {summary.data_coverage_pct:.1f}%")

**Convenience function for one-shot analysis:**

.. code-block:: python

   from petrophys.formation import analyze_interval, ManualInterval

   summary = analyze_interval(
       well,
       ManualInterval("Test Zone", 2500, 2700),
   )

**Analyze all formations from CSV:**

.. code-block:: python

   from petrophys.formation import FormationTops, IntervalAnalyzer

   tops = FormationTops.from_csv("formations.csv")
   analyzer = IntervalAnalyzer(well)

   # Analyze all (auto-clips to well depth range)
   summaries = analyzer.analyze_all(tops)

   # Print formatted report
   IntervalAnalyzer.print_report(summaries)

   # Export to CSV
   IntervalAnalyzer.to_csv(summaries, "interval_results.csv")

   # As pandas DataFrame
   df = IntervalAnalyzer.to_dataframe(summaries)

**Mixed interval types (formations + zones + manual):**

.. code-block:: python

   from petrophys.formation import (
       FormationTops, Formation, Zone, SubZone, ManualInterval,
       IntervalAnalyzer,
   )

   tops = FormationTops([
       Formation("Brent Group", 2841, 2958, lithology="clastic"),
       Zone("Brent Upper", 2841, 2900, parent_formation="Brent Group"),
       Zone("Brent Lower", 2900, 2958, parent_formation="Brent Group"),
       SubZone("Brent Upper A", 2841, 2870, parent_zone="Brent Upper"),
       ManualInterval("Pay Zone", 2860, 2930),
   ])

   analyzer = IntervalAnalyzer(well)
   summaries = analyzer.analyze_all(tops)
   IntervalAnalyzer.print_report(summaries)

**Custom net-pay curve names:**

.. code-block:: python

   # If your workflow uses different curve names
   analyzer = IntervalAnalyzer(
       well,
       net_pay_curve="MY_NET_PAY",
       net_reservoir_curve="MY_NET_RES",
       net_rock_curve="MY_NET_ROCK",
   )

Full Workflow: LAS + Lithology CSV + Formation Analysis
-------------------------------------------------------

Complete end-to-end example combining all inputs: LAS file, per-interval
lithology CSV, and formation/zone/sub-zone hierarchy.

.. code-block:: python

   from petrophys.io import read_las
   from petrophys.qaqc.standardization import standardize_curves
   from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig
   from petrophys.formation import (
       FormationTops, Formation, Zone, SubZone, ManualInterval,
       IntervalAnalyzer, IntervalKind,
   )

   # ── 1. Load and standardize ──
   well = read_las("30_6-5.las")
   well = standardize_curves(well, rename_curves=True, convert_units=True)

   # ── 2. Run workflow with per-interval lithology ──
   #    Lithology CSV drives:
   #    - Sw model routing (chalk -> Dual Water, sandstone -> Simandoux)
   #    - Net pay PHIE cutoffs (carbonate: 0.045, clastic: 0.08)
   config = DynamicWorkflowConfig(
       bht_f=250.0, td_ft=11667.0, surface_temp_f=40.0,
       rw=0.03, salinity_ppm=35000,
       lithology="clastic",
       lithology_csv="lithology_30_6-5.csv",
       matrix_type="sandstone",
       hc_type="oil",
   )
   result = run_dynamic_workflow(well, config)

   # ── 3. Build interval hierarchy ──
   tops = FormationTops([
       # Formations
       Formation("ROGALAND GP.", 1908, 2203, lithology="clastic"),
       Formation("SHETLAND GP.", 2203, 2408, lithology="carbonate"),
       Formation("VIKING GP.", 2452, 2841, lithology="clastic"),
       Formation("BAAT GP.", 2841, 2958, lithology="clastic"),
       Formation("DUNLIN GP.", 2958, 3223, lithology="clastic"),

       # Zones within BAAT GP.
       Zone("Ness Fm.", 2841, 2865, parent_formation="BAAT GP."),
       Zone("Etive Fm.", 2865, 2920, parent_formation="BAAT GP."),
       Zone("Rannoch Fm.", 2920, 2958, parent_formation="BAAT GP."),

       # Sub-zones within Etive Fm.
       SubZone("Etive Upper", 2865, 2890, parent_zone="Etive Fm."),
       SubZone("Etive Lower", 2890, 2920, parent_zone="Etive Fm."),

       # Manual: DST test interval spanning zone boundaries
       ManualInterval("DST #1", 2855, 2930),
   ])

   # ── 4. Analyze all intervals ──
   analyzer = IntervalAnalyzer(well)
   summaries = analyzer.analyze_all(tops)

   # ── 5. Report ──
   IntervalAnalyzer.print_report(summaries)

   # ── 6. Per-level summaries ──
   df = IntervalAnalyzer.to_dataframe(summaries)

   for level in ["formation", "zone", "sub_zone", "manual"]:
       level_df = df[df["Kind"] == level]
       if not level_df.empty:
           print(f"\n--- {level.upper()} level ---")
           print(level_df[["Name", "Gross", "Net Pay", "NTG",
                           "Avg PHIE", "Avg Sw", "Dominant Lithology"
                           ]].to_string(index=False))

   # ── 7. Export ──
   IntervalAnalyzer.to_csv(summaries, "full_analysis_results.csv")

Or load formations from a CSV instead of building manually:

.. code-block:: python

   # Load from CSV + add zones programmatically
   fm_tops = FormationTops.from_csv("formations_30_6-5.csv")

   zones = FormationTops.from_dict([
       {"name": "Ness Fm.", "top_depth": 2841, "bottom_depth": 2865},
       {"name": "Etive Fm.", "top_depth": 2865, "bottom_depth": 2920},
       {"name": "Rannoch Fm.", "top_depth": 2920, "bottom_depth": 2958},
   ], interval_kind=IntervalKind.ZONE)

   # Merge
   all_intervals = FormationTops(fm_tops.intervals + zones.intervals)
   summaries = IntervalAnalyzer(well).analyze_all(all_intervals)

Zone Summaries (Low-Level)
--------------------------

For quick zone summaries without the formation module, use the summation module:

.. code-block:: python

   from petrophys.summation import calculate_zone_summary

   summary = calculate_zone_summary(
       well, top_depth=8000, bottom_depth=8100, zone_name="Zone A"
   )

   print(f"Net Pay: {summary.net_pay:.1f} ft")
   print(f"NTG: {summary.ntg:.1%}")

Legacy Formation Analysis
-------------------------

The ``petrophys.summation`` module still provides the original formation
analysis functions for backward compatibility. For new code, prefer the
``petrophys.formation`` module above.

.. code-block:: python

   from petrophys.summation import run_formation_analysis

   result = run_formation_analysis(
       las_path="well.las",
       formations_csv_path="formations.csv",
       rw=0.03,
       salinity_ppm=35000,
       lithology="clastic",
   )

   result.print_report()
   result.to_csv("formation_results.csv")

**Supported CSV formats:**

- ``simple``: formation_name, top_depth, bottom_depth [, lithology]
- ``npd``: NPD stratigraphic picks (auto-converts tops to intervals)
- ``danish``: wellName, formationName, topDepth, bottomDepth [, majorityLithology]

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
