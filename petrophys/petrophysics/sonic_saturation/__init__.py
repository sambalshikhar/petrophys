"""
Sonic (acoustic) water saturation via Gassmann fluid substitution.

A resistivity-independent Sw estimate (compressional/shear sonic + density),
useful in low-resistivity-low-contrast pay, shaly sands and fresh-water zones.
Clean-room implementation from public-domain rock physics (Gassmann, Wood,
Krief, Greenberg-Castagna).
"""

from petrophys.petrophysics.sonic_saturation.rock_physics import (
    MINERAL_ELASTIC, FLUID_ELASTIC,
    slowness_to_velocity, vs_from_vp_greenberg_castagna, velocity_to_moduli,
    dry_rock_modulus, gassmann_ksat, gassmann_kfl,
    wood_fluid_modulus, sw_from_fluid_modulus,
)
from petrophys.petrophysics.sonic_saturation.sonic_saturation_module import (
    SonicSaturationModule, SonicSaturationResult, sonic_water_saturation,
)

__all__ = [
    "MINERAL_ELASTIC", "FLUID_ELASTIC",
    "slowness_to_velocity", "vs_from_vp_greenberg_castagna", "velocity_to_moduli",
    "dry_rock_modulus", "gassmann_ksat", "gassmann_kfl",
    "wood_fluid_modulus", "sw_from_fluid_modulus",
    "SonicSaturationModule", "SonicSaturationResult", "sonic_water_saturation",
]
