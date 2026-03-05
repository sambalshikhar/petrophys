"""
Demo 04: Formation-Level Analysis
===================================

This demo shows how to:
- Load a LAS file and a formation tops CSV
- Run the full dynamic workflow
- Analyze results by geological formation
- Generate per-formation summaries (NTG, PHIE, Sw, Perm, etc.)
- Export results to CSV
- Use the one-call convenience function (run_formation_analysis)

Well: 30/6-5 (Norwegian Continental Shelf)
Formations: from NPD stratigraphic picks (Nordland GP through Statfjord Fm)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from petrophys.io import read_las
from petrophys.qaqc.standardization import standardize_curves
from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig
from petrophys.summation import (
    load_formations_csv,
    analyze_formations,
    run_formation_analysis,
    Formation,
)

LAS_PATH = os.path.expanduser(
    "~/energy_gpt/EvalGen2/geolink_las/30_6-5.las"
)
FORMATIONS_CSV = os.path.join(os.path.dirname(__file__), "formations_30_6-5.csv")

# ─────────────────────────────────────────────
# APPROACH 1: Step-by-step
# ─────────────────────────────────────────────
print("=" * 90)
print("DEMO 04: Formation-Level Analysis - Well 30/6-5")
print("=" * 90)

# 1. Load and standardize
print("\n--- Step 1: Load LAS and standardize ---")
well = read_las(LAS_PATH)
well = standardize_curves(well, rename_curves=True, convert_units=True)
print(f"Well: {well.header.well_name}")
print(f"Depth: {well._depth.min():.1f} - {well._depth.max():.1f} {well._depth_unit}")
print(f"Curves: {well.curves}")

# 2. Run dynamic workflow
print("\n--- Step 2: Run dynamic workflow ---")
config = DynamicWorkflowConfig(
    bht_f=250.0,
    td_ft=11667.0,
    surface_temp_f=40.0,
    rw=0.03,
    salinity_ppm=35000,
    lithology="clastic",
    matrix_type="sandstone",
    hc_type="oil",
    sw_model="auto",
)

wf_result = run_dynamic_workflow(well, config)
print(f"Steps executed: {[s.step for s in wf_result.steps_executed if s.success]}")
if wf_result.warnings:
    for w in wf_result.warnings:
        print(f"  Warning: {w}")

# 3. Load formations from CSV
print("\n--- Step 3: Load formations ---")
formations = load_formations_csv(FORMATIONS_CSV)
print(f"Loaded {len(formations)} formations:")
for fm in formations:
    print(f"  {fm.name:<25s}  {fm.top_depth:>8.1f} - {fm.bottom_depth:>8.1f} {well._depth_unit}  ({fm.lithology})")

# 4. Analyze by formation
print("\n--- Step 4: Formation analysis ---")
result = analyze_formations(well, formations)

# 5. Print report
result.print_report()

# 6. Export to CSV
output_csv = os.path.join(os.path.dirname(__file__), "formation_results_30_6-5.csv")
result.to_csv(output_csv)
print(f"\nResults exported to: {output_csv}")

# 7. Also show as DataFrame
print("\n--- DataFrame view (first 10 columns) ---")
try:
    df = result.to_dataframe()
    cols_to_show = list(df.columns[:10])
    print(df[cols_to_show].to_string(index=False))
except ImportError:
    print("  (pandas not available for DataFrame display)")

# ─────────────────────────────────────────────
# APPROACH 2: One-call convenience function
# ─────────────────────────────────────────────
print("\n" + "=" * 90)
print("APPROACH 2: One-call run_formation_analysis()")
print("=" * 90)

result2 = run_formation_analysis(
    las_path=LAS_PATH,
    formations_csv_path=FORMATIONS_CSV,
    rw=0.03,
    salinity_ppm=35000,
    lithology="clastic",
    matrix_type="sandstone",
)

result2.print_report()

# ─────────────────────────────────────────────
# APPROACH 3: Using NPD stratigraphic picks directly
# ─────────────────────────────────────────────
NPD_CSV = os.path.expanduser(
    "~/energy_gpt/EvalGen2/meta_files/NPD stratigraphic picks north sea.xlsx - Sheet1.csv"
)

if os.path.exists(NPD_CSV):
    print("\n" + "=" * 90)
    print("APPROACH 3: NPD Stratigraphic Picks (auto-detected)")
    print("=" * 90)

    npd_formations = load_formations_csv(
        NPD_CSV,
        well_name="30/6-5",
        format="npd"
    )
    print(f"Loaded {len(npd_formations)} formation tops from NPD for 30/6-5:")
    for fm in npd_formations:
        print(f"  {fm.name:<25s}  {fm.top_depth:>8.1f} - {fm.bottom_depth:>8.1f} m")

    npd_result = analyze_formations(well, npd_formations)
    npd_result.print_report()

# ─────────────────────────────────────────────
# APPROACH 4: Manual formation definitions
# ─────────────────────────────────────────────
print("\n" + "=" * 90)
print("APPROACH 4: Manual formation definitions")
print("=" * 90)

manual_formations = [
    Formation("Upper Section", 200.0, 1000.0, lithology="Shale"),
    Formation("Middle Section", 1000.0, 2500.0, lithology="Mixed"),
    Formation("Brent Group", 2841.0, 2958.0, lithology="Sandstone"),
    Formation("Lower Section", 2958.0, 3500.0, lithology="Mixed"),
]

manual_result = analyze_formations(well, manual_formations)
manual_result.print_report()

print("\nDone!")
