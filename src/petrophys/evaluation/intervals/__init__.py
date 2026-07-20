"""
Formation-, zone-, and interval-based evaluation.

This module provides tools for formation-level (interval-based) analysis:
- Interval types: Formation, Zone, SubZone, ManualInterval
- Tops management: load/validate/query formation tops from CSV
- Per-interval analysis: summaries, net pay, averages, HCPV

"""

# Interval types
from petrophys.model.intervals import (
    IntervalKind,
    Interval,
    Formation,
    Zone,
    SubZone,
    ManualInterval,
)

# Tops management
from petrophys.model.formation_tops import FormationTops

# Per-interval analysis
from petrophys.evaluation.intervals.analyzer import (
    IntervalSummary,
    IntervalAnalyzer,
    analyze_interval,
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
]
