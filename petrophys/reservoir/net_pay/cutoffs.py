"""
Cutoff definitions for net pay determination.
"""

from dataclasses import dataclass

from petrophys.base.constants import DefaultParameters


@dataclass
class Cutoffs:
    """Net pay cutoff values."""
    vcl: float = DefaultParameters.VCL_CUTOFF
    phie_clastic: float = DefaultParameters.PHIE_CUTOFF_CLASTIC
    phie_carbonate: float = DefaultParameters.PHIE_CUTOFF_CARBONATE
    sw: float = DefaultParameters.SW_CUTOFF
    formation_type: str = "clastic"

    @property
    def active_phie(self) -> float:
        """PHIE cutoff for the selected formation type."""
        if self.formation_type.lower() == "carbonate":
            return self.phie_carbonate
        return self.phie_clastic


def default_cutoffs(formation_type: str = "clastic") -> Cutoffs:
    """Get default cutoffs for formation type.

    The ``phie_clastic`` / ``phie_carbonate`` fields already hold the per-lithology
    defaults; ``active_phie`` returns the one matching ``formation_type``.
    """
    cutoffs = Cutoffs()
    cutoffs.formation_type = formation_type.lower()
    return cutoffs
