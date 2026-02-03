"""
Temperature Profile Module.

This module provides tools for calculating formation temperature
profiles and temperature-related corrections.
"""

from petrophys.environmental.temperature.temp_profile import (
    TemperatureProfile,
    calculate_temperature_profile,
)
from petrophys.environmental.temperature.temp_gradient import (
    calculate_temperature_gradient,
    estimate_gradient_from_bht,
)
from petrophys.environmental.temperature.temp_corrections import (
    correct_resistivity_temperature,
    arps_equation,
)

__all__ = [
    "TemperatureProfile",
    "calculate_temperature_profile",
    "calculate_temperature_gradient",
    "estimate_gradient_from_bht",
    "correct_resistivity_temperature",
    "arps_equation",
]
