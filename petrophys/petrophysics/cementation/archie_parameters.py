"""
Archie parameter definitions and defaults.
"""

from dataclasses import dataclass


@dataclass
class ArchieParameters:
    """Archie parameters container."""
    a: float = 1.0
    m: float = 2.0
    n: float = 2.0

    def __repr__(self):
        return f"ArchieParameters(a={self.a}, m={self.m}, n={self.n})"


def default_archie_params(formation_type: str = "sandstone") -> ArchieParameters:
    """Get default Archie parameters for formation type."""
    defaults = {
        "sandstone": ArchieParameters(a=0.81, m=2.0, n=2.0),  # Humble
        "limestone": ArchieParameters(a=1.0, m=2.0, n=2.0),
        "dolomite": ArchieParameters(a=1.0, m=2.0, n=2.0),
        "unconsolidated": ArchieParameters(a=0.62, m=2.15, n=2.0),  # Humble
    }
    return defaults.get(formation_type.lower(), ArchieParameters())
