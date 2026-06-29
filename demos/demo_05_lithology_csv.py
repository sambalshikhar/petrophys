"""
Demo 05: Lithology CSV Integration
====================================

This demo shows how to use a per-interval lithology CSV to drive
lithology-dependent calculations throughout the workflow:

- Sw model routing: Carbonate intervals use Dual Water,
  Clastic intervals route by salinity/CEC (Simandoux, Indonesia, etc.)
- Net pay PHIE cutoffs: 0.08 for clastics, 0.045 for carbonates

The lithology CSV has columns: top, bottom, lithology
Lithology names are automatically normalized:
  sandstone/sand/sst       -> "clastic"
  limestone/chalk/dolomite -> "carbonate"
  shale/mudstone/marl      -> "clastic"

Well: 30/6-5 (Norwegian Continental Shelf)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from petrophys.io import read_las
from petrophys.qaqc.standardization import standardize_curves
from petrophys.lithology.lithology_log import (
    load_lithology_csv,
    build_lithology_log,
)
from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
import petrophys
LAS_PATH = petrophys.sample("30_6-5")
LITHOLOGY_CSV = os.path.join(os.path.dirname(__file__), "lithology_30_6-5.csv")

print("=" * 60)
print("DEMO 05: Lithology CSV Integration - Well 30/6-5")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. Load lithology CSV (standalone usage)
# ─────────────────────────────────────────────
print("\n--- 1. Load lithology intervals from CSV ---")
intervals = load_lithology_csv(LITHOLOGY_CSV)

print(f"Loaded {len(intervals)} lithology intervals:")
for iv in intervals:
    print(f"  {iv.top:8.1f} - {iv.bottom:8.1f}  {iv.lithology}")

# ─────────────────────────────────────────────
# 2. Build per-depth lithology log
# ─────────────────────────────────────────────
print("\n--- 2. Build per-depth lithology log ---")
well = read_las(LAS_PATH)
well = standardize_curves(well, rename_curves=True, convert_units=True)

depth = well.depth
litho_log = build_lithology_log(depth, intervals, default_lithology="clastic")

unique, counts = np.unique(litho_log, return_counts=True)
for lith, count in zip(unique, counts):
    pct = 100.0 * count / len(litho_log)
    print(f"  {lith:12s}: {count:6d} samples ({pct:.1f}%)")

# ─────────────────────────────────────────────
# 3. Use per-depth lithology with SaturationModule
# ─────────────────────────────────────────────
print("\n--- 3. Sw calculation with per-depth lithology ---")
from petrophys.petrophysics.saturation.sw_module import SaturationModule

sw_module = SaturationModule(a=1.0, m=2.0, n=2.0, model="auto")

try:
    result = sw_module.calculate(
        well,
        rw=0.03,
        salinity=35000,
        lithology=litho_log,
    )
    valid_sw = result.sw[~np.isnan(result.sw)]
    print(f"  Model(s) used: {result.model}")
    print(f"  Sw: mean={np.mean(valid_sw):.3f}, "
          f"P10={np.percentile(valid_sw, 10):.3f}, "
          f"P90={np.percentile(valid_sw, 90):.3f}")
except Exception as e:
    print(f"  Sw calculation requires PHIE — run porosity first: {e}")

# ─────────────────────────────────────────────
# 4. Full workflow with lithology_csv parameter
# ─────────────────────────────────────────────
print("\n--- 4. Dynamic workflow with lithology CSV ---")
well2 = read_las(LAS_PATH)
well2 = standardize_curves(well2, rename_curves=True, convert_units=True)

config = DynamicWorkflowConfig(
    # Temperature
    bht_f=250.0,
    td_ft=11667.0,
    surface_temp_f=40.0,

    # Formation water
    rw=0.03,
    salinity_ppm=35000,

    # Lithology: per-interval from CSV
    lithology="clastic",          # default for uncovered depths
    lithology_csv=LITHOLOGY_CSV,  # per-interval lithology

    # Matrix
    matrix_type="sandstone",
    hc_type="oil",

    # Archie
    a=1.0, m=2.0, n=2.0,
    sw_model="auto",
)

result = run_dynamic_workflow(well2, config)

print(f"Success: {result.success}")
print(f"Steps executed: {[s.step for s in result.steps_executed]}")

if result.warnings:
    print(f"Warnings:")
    for w in result.warnings:
        print(f"  ! {w}")

# Check the LITHO_CLASS curve was added
litho = well2.get("LITHO_CLASS")
if litho is not None:
    unique, counts = np.unique(litho.data, return_counts=True)
    print(f"\nLITHO_CLASS curve added to well:")
    for lith, count in zip(unique, counts):
        print(f"  {lith:12s}: {count:6d} samples")

# ─────────────────────────────────────────────
# 5. Compare: global lithology vs per-depth lithology
# ─────────────────────────────────────────────
print("\n--- 5. Impact comparison ---")

# Global clastic workflow (no lithology CSV)
well3 = read_las(LAS_PATH)
well3 = standardize_curves(well3, rename_curves=True, convert_units=True)

config_global = DynamicWorkflowConfig(
    bht_f=250.0, td_ft=11667.0, surface_temp_f=40.0,
    rw=0.03, salinity_ppm=35000,
    lithology="clastic",
    matrix_type="sandstone",
)

result_global = run_dynamic_workflow(well3, config_global)

# Compare net pay
np_perdepth = well2.get("NET_PAY")
np_global = well3.get("NET_PAY")

if np_perdepth is not None and np_global is not None:
    step = well2.step or 0.5
    pay_perdepth = float(np.nansum(np_perdepth.data) * step)
    pay_global = float(np.nansum(np_global.data) * step)
    diff = pay_perdepth - pay_global
    print(f"  Net pay (per-depth lithology): {pay_perdepth:.1f} {well2._depth_unit}")
    print(f"  Net pay (global clastic):      {pay_global:.1f} {well3._depth_unit}")
    print(f"  Difference:                    {diff:+.1f} {well2._depth_unit}")
    print(f"  (Difference from carbonate intervals using lower PHIE cutoff)")

print("\nDone!")
