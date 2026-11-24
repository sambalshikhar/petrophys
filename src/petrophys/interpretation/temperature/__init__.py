"""
Formation temperature profiles, geothermal gradients and temperature corrections.

This module provides tools for calculating formation temperature
profiles and temperature-related corrections.
"""

from petrophys.interpretation.temperature.temp_profile import (
    TemperatureProfile,
    temperature_profile,
)
from petrophys.interpretation.temperature.temp_gradient import (
    calculate_temperature_gradient,
    estimate_gradient_from_bht,
    gradient_c_per_km_to_fahrenheit_per_100ft,
    gradient_to_celsius_per_km,
)
from petrophys.interpretation.temperature.temp_corrections import (
    correct_resistivity_temperature,
    arps_equation,
)

__all__ = [
    "TemperatureProfile",
    "temperature_profile",
    "calculate_temperature_gradient",
    "estimate_gradient_from_bht",
    "gradient_c_per_km_to_fahrenheit_per_100ft",
    "gradient_to_celsius_per_km",
    "correct_resistivity_temperature",
    "arps_equation",
]
