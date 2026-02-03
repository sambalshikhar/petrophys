"""
Cutoff definitions for net pay determination.
"""

from dataclasses import dataclass

from petrophys.core.constants import DefaultParameters


@dataclass
class Cutoffs:
    """Net pay cutoff values."""
    vcl: float = DefaultParameters.VCL_CUTOFF
    phie_clastic: float = DefaultParameters.PHIE_CUTOFF_CLASTIC
    phie_carbonate: float = DefaultParameters.PHIE_CUTOFF_CARBONATE
    sw: float = DefaultParameters.SW_CUTOFF


def default_cutoffs(formation_type: str = "clastic") -> Cutoffs:
    """Get default cutoffs for formation type."""
    cutoffs = Cutoffs()
    if formation_type.lower() == "carbonate":
        cutoffs.phie_clastic = DefaultParameters.PHIE_CUTOFF_CARBONATE
    return cutoffs
