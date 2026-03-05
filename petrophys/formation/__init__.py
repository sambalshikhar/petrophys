"""
Formation Module - Interval-based analysis.

This module provides tools for formation-level (interval-based) analysis:
- Interval types: Formation, Zone, SubZone, ManualInterval
- Tops management: load/validate/query formation tops from CSV
- Per-interval analysis: summaries, net pay, averages, HCPV

Note: Whole-well calculations (clay volume, lithology) are top-level modules:
  - petrophys.clay_volume
  - petrophys.lithology
"""

# Interval types
from petrophys.formation.intervals import (
    IntervalKind,
    Interval,
    Formation,
    Zone,
    SubZone,
    ManualInterval,
)

# Tops management
from petrophys.formation.tops import FormationTops

# Per-interval analysis
from petrophys.formation.analysis import (
    IntervalSummary,
    IntervalAnalyzer,
    analyze_interval,
)

# Backward-compatible re-exports (deprecated paths)
from petrophys.clay_volume import (
    calculate_vcl,
    vcl_from_gr,
    vcl_from_sp,
    vcl_from_nd,
    VclModule,
)
from petrophys.lithology import (
    MultiMineralSolver,
    calculate_lithology,
)

__all__ = [
    # Interval types
    "IntervalKind",
    "Interval",
    "Formation",
    "Zone",
    "SubZone",
    "ManualInterval",
    # Tops management
    "FormationTops",
    # Analysis
    "IntervalSummary",
    "IntervalAnalyzer",
    "analyze_interval",
    # Backward-compatible re-exports
    "calculate_vcl",
    "vcl_from_gr",
    "vcl_from_sp",
    "vcl_from_nd",
    "VclModule",
    "MultiMineralSolver",
    "calculate_lithology",
]
