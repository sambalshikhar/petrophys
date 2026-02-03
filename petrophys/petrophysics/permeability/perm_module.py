"""
Main permeability estimation module.
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np

from petrophys.petrophysics.permeability.timur import timur
from petrophys.petrophysics.permeability.coates_denoo import coates_denoo, wyllie_rose


@dataclass
class PermeabilityResult:
    """Result container for permeability."""
    perm: np.ndarray
    method: str


class PermeabilityModule:
    """Permeability estimation module."""

    def __init__(self, method: str = "timur"):
        self.method = method.lower()

    def calculate(
        self,
        well: "Well",
        swi: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None
    ) -> PermeabilityResult:
        """Calculate permeability."""
        if phi is None:
            phi_curve = well.get("PHIE")
            if phi_curve is None:
                raise ValueError("PHIE required")
            phi = phi_curve.data

        if swi is None:
            sw_curve = well.get("SW")
            swi = sw_curve.data if sw_curve is not None else 0.3

        if self.method == "timur":
            perm = timur(phi, swi)
        elif self.method == "coates":
            perm = coates_denoo(phi, swi)
        elif self.method == "wyllie_rose":
            perm = wyllie_rose(phi, swi)
        else:
            perm = timur(phi, swi)

        return PermeabilityResult(perm=perm, method=self.method)


def calculate_permeability(
    well: "Well",
    method: str = "timur",
    **kwargs
) -> np.ndarray:
    """Calculate permeability."""
    module = PermeabilityModule(method=method)
    result = module.calculate(well, **kwargs)
    return result.perm
