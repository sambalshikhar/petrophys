"""
Cutoff definitions for net pay determination.

``STANDARD_CUTOFFS`` is the curriculum-wide single source of truth for net-pay
cutoffs, keyed by lithology family. Values live in
:class:`petrophys.model.constants.DefaultParameters`.
"""

from dataclasses import dataclass

from petrophys.model.constants import DefaultParameters

#: Curriculum-standard net-pay cutoffs by lithology family. Pass a set into a
#: workflow config (``**STANDARD_CUTOFFS["clastic"]``) or leave the config
#: cutoffs unset to have them resolved from the well lithology automatically.
STANDARD_CUTOFFS = {
    "clastic": {
        "vcl_cutoff": DefaultParameters.VCL_CUTOFF_CLASTIC,
        "phie_cutoff": DefaultParameters.PHIE_CUTOFF_CLASTIC,
        "sw_cutoff": DefaultParameters.SW_CUTOFF_CLASTIC,
    },
    "carbonate": {
        "vcl_cutoff": DefaultParameters.VCL_CUTOFF_CARBONATE,
        "phie_cutoff": DefaultParameters.PHIE_CUTOFF_CARBONATE,
        "sw_cutoff": DefaultParameters.SW_CUTOFF_CARBONATE,
    },
}


def standard_cutoffs(lithology: str = "clastic") -> dict:
    """Return a copy of the standard cutoff set for a lithology family."""
    family = "carbonate" if str(lithology).lower().startswith("carb") else "clastic"
    return dict(STANDARD_CUTOFFS[family])


@dataclass
class Cutoffs:
    """Net pay cutoff values (per-lithology where the convention differs)."""
    vcl_clastic: float = DefaultParameters.VCL_CUTOFF_CLASTIC
    vcl_carbonate: float = DefaultParameters.VCL_CUTOFF_CARBONATE
    phie_clastic: float = DefaultParameters.PHIE_CUTOFF_CLASTIC
    phie_carbonate: float = DefaultParameters.PHIE_CUTOFF_CARBONATE
    sw_clastic: float = DefaultParameters.SW_CUTOFF_CLASTIC
    sw_carbonate: float = DefaultParameters.SW_CUTOFF_CARBONATE
    formation_type: str = "clastic"

    @property
    def active_phie(self) -> float:
        """PHIE cutoff for the selected formation type."""
        if self.formation_type.lower() == "carbonate":
            return self.phie_carbonate
        return self.phie_clastic

    @property
    def active_vcl(self) -> float:
        """VCL cutoff for the selected formation type."""
        if self.formation_type.lower() == "carbonate":
            return self.vcl_carbonate
        return self.vcl_clastic

    @property
    def active_sw(self) -> float:
        """Sw cutoff for the selected formation type."""
        if self.formation_type.lower() == "carbonate":
            return self.sw_carbonate
        return self.sw_clastic

    @property
    def vcl(self) -> float:
        """Backward-compatible alias for :attr:`active_vcl`."""
        return self.active_vcl

    @property
    def sw(self) -> float:
        """Backward-compatible alias for :attr:`active_sw`."""
        return self.active_sw


def default_cutoffs(formation_type: str = "clastic") -> Cutoffs:
    """Get default cutoffs for formation type.

    The per-lithology fields hold both conventions; ``active_phie`` /
    ``active_vcl`` return the ones matching ``formation_type``.
    """
    cutoffs = Cutoffs()
    cutoffs.formation_type = formation_type.lower()
    return cutoffs
