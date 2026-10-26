"""Core and borehole geochemical interpretation."""

from petrophys.interpretation.geochemistry.clay_typing import (
    TH_K_CLAY_TYPES,
    ClayTypingResult,
    classify_clay_th_k,
    clay_typing_from_well,
)
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
from petrophys.interpretation.geochemistry.thorium_uranium import (
    TH_U_ENVIRONMENTS,
    TH_U_ENVIRONMENT_THRESHOLDS,
    ThoriumClayIndex,
    ThUEnvironment,
    classify_th_u_environment,
    thorium_clay_index,
    uranium_stripped_gamma_ray,
)
from petrophys.interpretation.geochemistry.xrf import XRFData

__all__ = [
    "TH_K_CLAY_TYPES",
    "ClayTypingResult",
    "classify_clay_th_k",
    "clay_typing_from_well",
    "DEFAULT_API_COEFFICIENTS",
    "ELEMENT_NAMES",
    "ElementalSpectroscopyResult",
    "OxideClosureResult",
    "SpectralGammaResult",
    "TH_U_ENVIRONMENTS",
    "TH_U_ENVIRONMENT_THRESHOLDS",
    "ThUEnvironment",
    "ThoriumClayIndex",
    "XRFData",
    "classify_th_u_environment",
    "has_elemental_spectroscopy_inputs",
    "has_spectral_gamma_inputs",
    "interpret_elemental_spectroscopy",
    "interpret_spectral_gamma",
    "oxide_closure",
    "thorium_clay_index",
    "uranium_stripped_gamma_ray",
]
