"""
Demo 06: Interval-Based Formation Analysis
============================================

This demo shows the full capabilities of the petrophys.formation module:

1. Load LAS + per-interval lithology CSV -> run dynamic workflow
2. Build a Formation/Zone/SubZone/ManualInterval hierarchy
3. Analyze all intervals with IntervalAnalyzer
4. Print per-level reports (formation, zone, sub-zone, manual)
5. Compare global vs per-depth lithology impact on a carbonate formation
6. Load formations from CSV, merge with programmatic zones, analyze
7. Export and round-trip

Well: 30/6-5 (Norwegian Continental Shelf)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from petrophys.io import read_las
from petrophys.qaqc.standardization import standardize_curves
from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig
from petrophys.formation import (
    FormationTops,
    Formation,
    Zone,
    SubZone,
    ManualInterval,
    IntervalAnalyzer,
    IntervalKind,
    analyze_interval,
)

LAS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "geolink_data", "30_6-5.las"
)
LITHOLOGY_CSV = os.path.join(os.path.dirname(__file__), "lithology_30_6-5.csv")
FORMATIONS_CSV = os.path.join(os.path.dirname(__file__), "formations_30_6-5.csv")

# ═══════════════════════════════════════════════
# PART 1: Load, standardize, run workflow
# ═══════════════════════════════════════════════
print("=" * 90)
print("DEMO 06: Interval-Based Formation Analysis - Well 30/6-5")
print("=" * 90)

print("\n--- 1. Load LAS and run workflow with per-interval lithology ---")
well = read_las(LAS_PATH)
well = standardize_curves(well, rename_curves=True, convert_units=True)
print(f"Well: {well.header.well_name}")
print(f"Depth: {well.depth.min():.1f} - {well.depth.max():.1f} {well._depth_unit}")

config = DynamicWorkflowConfig(
    bht_f=250.0, td_ft=11667.0, surface_temp_f=40.0,
    rw=0.03, salinity_ppm=35000,
    lithology="clastic",
    lithology_csv=LITHOLOGY_CSV,
    matrix_type="sandstone",
    hc_type="oil",
)
result = run_dynamic_workflow(well, config)
print(f"Steps: {[s.step for s in result.steps_executed if s.success]}")

# ═══════════════════════════════════════════════
# PART 2: Build hierarchical intervals manually
# ═══════════════════════════════════════════════
print("\n--- 2. Build Formation/Zone/SubZone/Manual hierarchy ---")

tops = FormationTops([
    # Formations
    Formation("ROGALAND GP.", 1908, 2203, lithology="clastic"),
    Formation("SHETLAND GP.", 2203, 2408, lithology="carbonate"),
    Formation("VIKING GP.", 2452, 2841, lithology="clastic"),
    Formation("BAAT GP.", 2841, 2958, lithology="clastic", age="Middle Jurassic"),
    Formation("DUNLIN GP.", 2958, 3223, lithology="clastic"),
    Formation("Statfjord Fm.", 3223, 3556, lithology="clastic"),

    # Zones within BAAT GP.
    Zone("Ness Fm.", 2841, 2865, parent_formation="BAAT GP."),
    Zone("Etive Fm.", 2865, 2920, parent_formation="BAAT GP."),
    Zone("Rannoch Fm.", 2920, 2958, parent_formation="BAAT GP."),

    # Sub-zones within Etive Fm.
    SubZone("Etive Upper", 2865, 2890, parent_zone="Etive Fm."),
    SubZone("Etive Lower", 2890, 2920, parent_zone="Etive Fm."),

    # Manual: DST test interval
    ManualInterval("DST #1", 2855, 2930),
])

print(f"Total intervals: {len(tops)}")
for iv in tops:
    print(f"  [{iv.kind.value:10s}] {iv.name:20s}  {iv.top_depth:.0f} - {iv.bottom_depth:.0f}")

# ═══════════════════════════════════════════════
# PART 3: Validate
# ═══════════════════════════════════════════════
print("\n--- 3. Validate (overlaps within same kind) ---")
if tops.has_overlaps():
    for issue in tops.find_overlaps():
        print(f"  WARNING: {issue}")
else:
    print("  No overlaps detected within same interval kind.")

# ═══════════════════════════════════════════════
# PART 4: Analyze all intervals
# ═══════════════════════════════════════════════
print("\n--- 4. Analyze all intervals ---")
analyzer = IntervalAnalyzer(well)
summaries = analyzer.analyze_all(tops)
IntervalAnalyzer.print_report(summaries)

# ═══════════════════════════════════════════════
# PART 5: Per-level breakdown
# ═══════════════════════════════════════════════
print("\n--- 5. Per-level breakdown ---")
try:
    df = IntervalAnalyzer.to_dataframe(summaries)
    cols = ["Name", "Gross", "Net Pay", "NTG", "Avg PHIE", "Avg Sw", "Dominant Lithology"]

    for level in ["formation", "zone", "sub_zone", "manual"]:
        level_df = df[df["Kind"] == level]
        if not level_df.empty:
            print(f"\n  {level.upper()} level ({len(level_df)} intervals):")
            print(level_df[cols].to_string(index=False))
except ImportError:
    print("  (pandas not available)")

# ═══════════════════════════════════════════════
# PART 6: Query intervals
# ═══════════════════════════════════════════════
print("\n--- 6. Query intervals ---")

baat = tops.get_by_name("BAAT GP.")
print(f"  By name 'BAAT GP.': {baat}")

at_2900 = tops.get_at_depth(2900.0)
print(f"  At depth 2900.0 ({len(at_2900)} matches):")
for iv in at_2900:
    print(f"    {iv.kind.value}: {iv.name}")

in_range = tops.get_in_range(2840, 2960)
print(f"  In range 2840-2960 ({len(in_range)} matches):")
for iv in in_range:
    print(f"    {iv.kind.value}: {iv.name}")

fms_only = tops.filter_by_kind(IntervalKind.FORMATION)
print(f"  Formations only: {len(fms_only)}")

# ═══════════════════════════════════════════════
# PART 7: Single interval convenience function
# ═══════════════════════════════════════════════
print("\n--- 7. Single interval analysis (convenience function) ---")
summary = analyze_interval(well, ManualInterval("Quick Check", 2850, 2950))
zs = summary.zone_summary
print(f"  Quick Check (2850-2950):")
print(f"    Net Pay: {zs.net_pay:.1f}, NTG: {zs.ntg:.1%}")
print(f"    Avg PHIE: {zs.avg_phie:.3f}, Avg Sw: {zs.avg_sw:.3f}")
print(f"    Dominant lithology: {summary.dominant_lithology}")

# ═══════════════════════════════════════════════
# PART 8: Compare global vs per-depth lithology
# ═══════════════════════════════════════════════
print("\n--- 8. Compare: Global clastic vs per-depth lithology ---")
print("         (Impact on SHETLAND GP. = Chalk interval)")

# Run a second workflow with global clastic (no lithology CSV)
well_global = read_las(LAS_PATH)
well_global = standardize_curves(well_global, rename_curves=True, convert_units=True)

config_global = DynamicWorkflowConfig(
    bht_f=250.0, td_ft=11667.0, surface_temp_f=40.0,
    rw=0.03, salinity_ppm=35000,
    lithology="clastic",
    matrix_type="sandstone",
)
run_dynamic_workflow(well_global, config_global)

chalk_fm = Formation("SHETLAND GP.", 2203, 2408, lithology="carbonate")

s_global = IntervalAnalyzer(well_global).analyze(chalk_fm)
s_perdepth = IntervalAnalyzer(well).analyze(chalk_fm)

print(f"  {'':25s} {'Global Clastic':>16s} {'Per-Depth Litho':>16s}")
print(f"  {'Net Pay':25s} {s_global.zone_summary.net_pay:>16.1f} "
      f"{s_perdepth.zone_summary.net_pay:>16.1f}")
print(f"  {'NTG':25s} {s_global.zone_summary.ntg:>16.1%} "
      f"{s_perdepth.zone_summary.ntg:>16.1%}")
print(f"  {'Avg PHIE':25s} {s_global.zone_summary.avg_phie:>16.3f} "
      f"{s_perdepth.zone_summary.avg_phie:>16.3f}")
print(f"  {'Avg Sw':25s} {s_global.zone_summary.avg_sw:>16.3f} "
      f"{s_perdepth.zone_summary.avg_sw:>16.3f}")

# ═══════════════════════════════════════════════
# PART 9: Load from CSV + merge with programmatic zones
# ═══════════════════════════════════════════════
print("\n--- 9. Load CSV formations + merge with programmatic zones ---")

csv_tops = FormationTops.from_csv(FORMATIONS_CSV)
print(f"  Loaded {len(csv_tops)} formations from CSV")

zone_tops = FormationTops.from_dict([
    {"name": "Ness Fm.", "top_depth": 2841, "bottom_depth": 2865},
    {"name": "Etive Fm.", "top_depth": 2865, "bottom_depth": 2920},
    {"name": "Rannoch Fm.", "top_depth": 2920, "bottom_depth": 2958},
], interval_kind=IntervalKind.ZONE)
print(f"  Created {len(zone_tops)} zones programmatically")

merged = FormationTops(csv_tops.intervals + zone_tops.intervals)
print(f"  Merged: {len(merged)} total intervals")

clipped = merged.clip_to_well(well)
print(f"  After clip to well: {len(clipped)} intervals")

summaries_merged = IntervalAnalyzer(well).analyze_all(clipped, clip_to_well=False)
IntervalAnalyzer.print_report(summaries_merged)

# ═══════════════════════════════════════════════
# PART 10: Export and round-trip
# ═══════════════════════════════════════════════
print("\n--- 10. Export and round-trip ---")

output_dir = os.path.dirname(__file__)

# Export analysis results
results_csv = os.path.join(output_dir, "interval_results_30_6-5.csv")
IntervalAnalyzer.to_csv(summaries, results_csv)
print(f"  Results exported to: {results_csv}")

# Export tops
tops_csv = os.path.join(output_dir, "exported_tops_30_6-5.csv")
tops.to_csv(tops_csv)
print(f"  Tops exported to: {tops_csv}")

# Re-import
reimported = FormationTops.from_csv(tops_csv)
print(f"  Re-imported {len(reimported)} intervals from exported CSV")

print("\nDone!")
