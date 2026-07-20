"""
Archie parameter definitions and defaults.
"""

from dataclasses import dataclass

from petrophys.model.constants import ArchieParameterDefaults


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
    coeffs = ArchieParameterDefaults.get(formation_type)
    return ArchieParameters(a=coeffs.a, m=coeffs.m, n=coeffs.n)
