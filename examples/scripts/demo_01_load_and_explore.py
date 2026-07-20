"""
Demo 01: Loading and Exploring a Real LAS File
================================================

This demo shows how to:
- Load a LAS file using petrophys
- Inspect well header and available curves
- Standardize curve mnemonics and units
- View basic curve statistics and data quality
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add petrophys to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from petrophys.io import read_las
from petrophys.preprocessing.standardization import standardize_curves, CurveStandardizer

# ─────────────────────────────────────────────
# 1. Load the LAS file
# ─────────────────────────────────────────────
import petrophys
LAS_PATH = petrophys.sample("30_6-5")

print("=" * 60)
print("DEMO 01: Loading and Exploring Well 30/6-5")
print("=" * 60)

well = read_las(LAS_PATH)

# ─────────────────────────────────────────────
# 2. Inspect header
# ─────────────────────────────────────────────
print(f"\nWell name : {well.header.well_name}")
print(f"UWI       : {well.header.uwi}")
print(f"Depth unit: {well._depth_unit}")
print(f"Depth range: {well._depth.min():.1f} - {well._depth.max():.1f} {well._depth_unit}")
print(f"Samples   : {len(well._depth)}")

# ─────────────────────────────────────────────
# 3. List raw curves (before standardization)
# ─────────────────────────────────────────────
print(f"\n--- Raw curves ({len(well.curves)}) ---")
for name in well.curves:
    curve = well.get(name)
    valid = np.sum(~np.isnan(curve.data))
    total = len(curve.data)
    pct = 100.0 * valid / total
    print(f"  {name:25s}  unit={curve.unit:10s}  valid={valid}/{total} ({pct:.1f}%)")

# ─────────────────────────────────────────────
# 4. Generate standardization report
# ─────────────────────────────────────────────
standardizer = CurveStandardizer()
report = standardizer.report(well)

print("\n--- Standardization Report ---")
print(f"Total curves: {report['total_curves']}")

if report["recognized"]:
    print("Recognized aliases:")
    for item in report["recognized"]:
        if item["original"] != item["standard"]:
            print(f"  {item['original']} -> {item['standard']}")
        else:
            print(f"  {item['original']} (already standard)")

if report["unrecognized"]:
    print(f"Unrecognized: {report['unrecognized']}")

if report["unit_conversions"]:
    print("Unit conversions needed:")
    for item in report["unit_conversions"]:
        print(f"  {item['curve']}: {item['from_unit']} -> {item['to_unit']}")

# ─────────────────────────────────────────────
# 5. Standardize curves
# ─────────────────────────────────────────────
print("\n--- Standardizing curves ---")
well = standardize_curves(well, rename_curves=True, convert_units=True)

print(f"Curves after standardization: {well.curves}")

# ─────────────────────────────────────────────
# 6. Curve statistics for key logs
# ─────────────────────────────────────────────
print("\n--- Key Curve Statistics ---")
key_curves = ["GR", "NPHI", "RHOB", "RDEP", "RSHA", "RMED", "DTC", "DTS", "PEF", "SP", "CALI"]
for name in key_curves:
    c = well.get(name)
    if c is not None:
        valid_data = c.data[~np.isnan(c.data)]
        if len(valid_data) > 0:
            print(f"  {name:6s} [{c.unit:6s}]  "
                  f"min={np.min(valid_data):10.3f}  "
                  f"mean={np.mean(valid_data):10.3f}  "
                  f"max={np.max(valid_data):10.3f}  "
                  f"valid={len(valid_data)}")
        else:
            print(f"  {name:6s} [{c.unit:6s}]  ALL NaN")
    else:
        print(f"  {name:6s}  NOT FOUND")

# ─────────────────────────────────────────────
# 7. Find data-rich depth interval
# ─────────────────────────────────────────────
print("\n--- Data Coverage Analysis ---")
depth = well._depth
rhob = well.get("RHOB")
nphi = well.get("NPHI")
gr = well.get("GR")
rdep = well.get("RDEP")

if rhob is not None and nphi is not None:
    # Find first/last depth where RHOB and NPHI are both valid
    both_valid = ~np.isnan(rhob.data) & ~np.isnan(nphi.data)
    valid_indices = np.where(both_valid)[0]
    if len(valid_indices) > 0:
        first_idx = valid_indices[0]
        last_idx = valid_indices[-1]
        print(f"RHOB+NPHI coverage: {depth[first_idx]:.1f} - {depth[last_idx]:.1f} {well._depth_unit}")
        print(f"  ({len(valid_indices)} samples)")

if rdep is not None:
    valid_rdep = ~np.isnan(rdep.data)
    valid_idx = np.where(valid_rdep)[0]
    if len(valid_idx) > 0:
        print(f"RDEP coverage:      {depth[valid_idx[0]]:.1f} - {depth[valid_idx[-1]]:.1f} {well._depth_unit}")

print("\nDone!")
