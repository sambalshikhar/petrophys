"""
Formation-, zone-, and interval-based evaluation.

This module provides tools for formation-level (interval-based) analysis:
- Interval types: Formation, Zone, SubZone, ManualInterval
- Tops management: load/validate/query formation tops from CSV
- Per-interval analysis: summaries, net pay, averages, HCPV
- Depth labels (``label_depths``) and flag-to-interval tables (``flag_intervals``)

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
from petrophys.model.formation_tops import FormationTops, label_depths

# Flags to intervals
from petrophys.evaluation.intervals.flag_intervals import flag_intervals

# Vertical stacking patterns of zones
from petrophys.evaluation.intervals.sequence import MIN_STEP, stacking_pattern

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
    "label_depths",
    # Flags to intervals
    "flag_intervals",
    # Stacking patterns
    "MIN_STEP",
    "stacking_pattern",
    # Analysis
    "IntervalSummary",
    "IntervalAnalyzer",
    "analyze_interval",
]
