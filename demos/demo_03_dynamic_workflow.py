"""
Demo 03: Dynamic Workflow on Real LAS Data
============================================

This demo shows the full dynamic workflow that:
- Auto-detects available curves in the well
- Runs only applicable modules in pipeline order
- Reports which steps executed vs were skipped
- Generates all computed curves (VCL, PHIE, SW, PERM, RQI, FZI, NET_PAY, etc.)

Well: 30/6-5 (Norwegian Continental Shelf)
Assumptions: Clastic lithology, North Sea salinity ~35k ppm
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from petrophys.io import read_las
from petrophys.qaqc.standardization import standardize_curves
from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig

# ─────────────────────────────────────────────
# 1. Load and standardize
# ─────────────────────────────────────────────
import petrophys
LAS_PATH = petrophys.sample("30_6-5")

print("=" * 60)
print("DEMO 03: Dynamic Workflow - Well 30/6-5")
print("=" * 60)

well = read_las(LAS_PATH)
well = standardize_curves(well, rename_curves=True, convert_units=True)

print(f"Well: {well.header.well_name}")
print(f"Depth: {well._depth.min():.1f} - {well._depth.max():.1f} {well._depth_unit}")
print(f"Input curves: {well.curves}")

# ─────────────────────────────────────────────
# 2. Configure the dynamic workflow
# ─────────────────────────────────────────────
# This well is from the Norwegian Continental Shelf
# Typical North Sea clastic parameters:
config = DynamicWorkflowConfig(
    # Temperature: ~4C/100m gradient, BHT estimated from TD
    # TD ~ 3556m ~ 11667 ft, estimated BHT ~ 250F
    bht_f=250.0,
    td_ft=11667.0,
    surface_temp_f=40.0,  # ~4C at seabed

    # Formation water (North Sea typical)
    rw=0.03,
    salinity_ppm=35000,

    # Clastic (sandstone) lithology
    lithology="clastic",
    matrix_type="sandstone",

    # HC correction
    hc_type="oil",

    # Archie parameters
    a=1.0,
    m=2.0,
    n=2.0,

    # Sw model: auto-routing (VCL-based, clastic + 35k ppm -> Simandoux for shaly)
    sw_model="auto",

    # Cutoffs
    vcl_cutoff=0.3,
    sw_cutoff=0.7,
    # phie_cutoff auto-selected: 0.08 for clastic
)

# ─────────────────────────────────────────────
# 3. Run the workflow
# ─────────────────────────────────────────────
print("\n--- Running Dynamic Workflow ---")
result = run_dynamic_workflow(well, config)

# ─────────────────────────────────────────────
# 4. Inspect results
# ─────────────────────────────────────────────
print("\n--- Workflow Results ---")
print(f"Success: {result.success}")

print(f"\nSteps executed ({len(result.steps_executed)}):")
for step in result.steps_executed:
    status = "OK" if step.success else f"FAILED: {step.message}"
    print(f"  [{status}] {step.step}")

print(f"\nSteps skipped ({len(result.steps_skipped)}):")
for name in result.steps_skipped:
    print(f"  - {name}")

if result.warnings:
    print(f"\nWarnings ({len(result.warnings)}):")
    for w in result.warnings:
        print(f"  ! {w}")

print(f"\nComputed parameters:")
print(f"  Rw:  {result.rw_computed}")
print(f"  Rcl: {result.rcl_computed}")
print(f"  Rmf: {result.rmf_computed}")
print(f"  Temperature gradient: {result.temperature_gradient}")

# ─────────────────────────────────────────────
# 5. Show all curves now on the well
# ─────────────────────────────────────────────
print(f"\n--- All Curves After Workflow ({len(well.curves)}) ---")
for name in sorted(well.curves):
    c = well.get(name)
    valid = np.sum(~np.isnan(c.data))
    print(f"  {name:20s}  [{c.unit:6s}]  valid={valid}")

# ─────────────────────────────────────────────
# 6. Key computed curve statistics
# ─────────────────────────────────────────────
print("\n--- Computed Curve Statistics ---")
computed = ["VCL", "PHIT", "PHIE", "SW", "PERM", "BVW", "NET_PAY"]
for name in computed:
    c = well.get(name)
    if c is not None:
        valid_data = c.data[~np.isnan(c.data)]
        if len(valid_data) > 0:
            print(f"  {name:10s}: mean={np.mean(valid_data):.4f}, "
                  f"P10={np.percentile(valid_data, 10):.4f}, "
                  f"P90={np.percentile(valid_data, 90):.4f}")
    else:
        print(f"  {name:10s}: not computed")

# ─────────────────────────────────────────────
# 7. Zone summary for a specific interval
# ─────────────────────────────────────────────
print("\n--- Zone Summary (example interval) ---")
from petrophys.summation import calculate_zone_summary

depth = well._depth

# Pick a depth interval where data exists (mid-well)
mid_depth = (depth.min() + depth.max()) / 2
zone_top = mid_depth - 50
zone_bot = mid_depth + 50

try:
    summary = calculate_zone_summary(
        well,
        top_depth=zone_top,
        bottom_depth=zone_bot,
        zone_name=f"Mid-well ({zone_top:.0f}-{zone_bot:.0f} {well._depth_unit})"
    )
    print(f"Zone: {summary.zone_name}")
    print(f"  Gross thickness: {summary.gross_thickness:.1f}")
    print(f"  Net Pay:         {summary.net_pay:.1f}")
    print(f"  NTG:             {summary.ntg:.1%}")
    print(f"  Avg PHIE:        {summary.avg_phie:.3f}")
    print(f"  Avg Sw:          {summary.avg_sw:.3f}")
except Exception as e:
    print(f"  Zone summary error: {e}")

# ─────────────────────────────────────────────
# 8. Re-run with skipping steps
# ─────────────────────────────────────────────
print("\n--- Re-run with skipped steps ---")
well2 = read_las(LAS_PATH)
well2 = standardize_curves(well2, rename_curves=True, convert_units=True)

config2 = DynamicWorkflowConfig(
    rw=0.03,
    salinity_ppm=35000,
    lithology="clastic",
    matrix_type="sandstone",
    skip_steps=["temperature", "bad_hole", "lithology", "invasion"],
)

result2 = run_dynamic_workflow(well2, config2)
print(f"Steps executed: {[s.step for s in result2.steps_executed]}")
print(f"Steps skipped:  {result2.steps_skipped}")

print("\nDone!")
