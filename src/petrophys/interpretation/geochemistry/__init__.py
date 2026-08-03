"""Core and borehole geochemical interpretation."""

from petrophys.interpretation.geochemistry.elemental_spectroscopy import (
    ELEMENT_NAMES,
    ElementalSpectroscopyResult,
    OxideClosureResult,
    has_elemental_spectroscopy_inputs,
    interpret_elemental_spectroscopy,
    oxide_closure,
)
from petrophys.interpretation.geochemistry.spectral_gamma import (
    DEFAULT_API_COEFFICIENTS,
    SpectralGammaResult,
    has_spectral_gamma_inputs,
    interpret_spectral_gamma,
)
from petrophys.interpretation.geochemistry.xrf import XRFData

__all__ = [
    "DEFAULT_API_COEFFICIENTS",
    "ELEMENT_NAMES",
    "ElementalSpectroscopyResult",
    "OxideClosureResult",
    "SpectralGammaResult",
    "XRFData",
    "has_elemental_spectroscopy_inputs",
    "has_spectral_gamma_inputs",
    "interpret_elemental_spectroscopy",
    "interpret_spectral_gamma",
    "oxide_closure",
]
