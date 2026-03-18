"""
Temperature profile calculation module.

This module provides the TemperatureProfile class for computing
formation temperature at any depth.
"""

from typing import Optional, Union
from dataclasses import dataclass
import numpy as np

from petrophys.base.constants import DefaultParameters
from petrophys.base.units import UnitConverter
from petrophys.environmental.temperature.temp_gradient import (
    calculate_temperature_gradient,
    estimate_gradient_from_bht,
)


@dataclass
class TemperatureResult:
    """
    Result container for temperature profile calculation.

    Attributes
    ----------
    temperature : np.ndarray
        Formation temperature at each depth (°F)
    gradient : float
        Temperature gradient (°F/100ft)
    surface_temp : float
        Surface temperature used (°F)
    """

    temperature: np.ndarray
    gradient: float
    surface_temp: float


class TemperatureProfile:
    """
    Formation temperature profile calculator.

    Calculates formation temperature at depth using either:
    - Linear interpolation from BHT and surface temperature
    - Specified gradient
    - Two reference temperature points

    Parameters
    ----------
    gradient : float, optional
        Temperature gradient in °F/100ft
    surface_temp : float, optional
        Surface temperature (default: 68°F)
    temp_unit : str, optional
        Temperature unit ("F" or "C")

    Examples
    --------
    >>> profile = TemperatureProfile(gradient=1.5, surface_temp=68)
    >>> temp_at_depth = profile.get_temperature(5000)
    >>> print(f"Temperature at 5000 ft: {temp_at_depth:.1f} °F")
    143.0

    >>> # From BHT
    >>> profile = TemperatureProfile.from_bht(bht=200, td=10000)
    >>> temps = profile.calculate(depth_array)
    """

    def __init__(
        self,
        gradient: Optional[float] = None,
        surface_temp: float = DefaultParameters.SURFACE_TEMPERATURE_F,
        temp_unit: str = "F"
    ):
        self.surface_temp = surface_temp
        self.temp_unit = temp_unit.upper()

        if self.temp_unit == "C":
            self.surface_temp = UnitConverter.celsius_to_fahrenheit(surface_temp)

        self._gradient = gradient  # °F/100ft

    @property
    def gradient(self) -> Optional[float]:
        """Return temperature gradient in °F/100ft."""
        return self._gradient

    @gradient.setter
    def gradient(self, value: float):
        """Set temperature gradient."""
        self._gradient = value

    @classmethod
    def from_bht(
        cls,
        bht: float,
        td: float,
        surface_temp: float = DefaultParameters.SURFACE_TEMPERATURE_F,
        bht_unit: str = "F",
        depth_unit: str = "ft"
    ) -> "TemperatureProfile":
        """
        Create temperature profile from BHT and TD.

        Parameters
        ----------
        bht : float
            Bottom hole temperature
        td : float
            Total depth
        surface_temp : float, optional
            Surface temperature
        bht_unit : str, optional
            BHT unit
        depth_unit : str, optional
            Depth unit

        Returns
        -------
        TemperatureProfile
            Configured temperature profile
        """
        # Convert to standard units
        if bht_unit.upper() == "C":
            bht = UnitConverter.celsius_to_fahrenheit(bht)
            surface_temp = UnitConverter.celsius_to_fahrenheit(surface_temp)

        if depth_unit.lower() == "m":
            td = UnitConverter.meters_to_feet(td)

        gradient = estimate_gradient_from_bht(bht, td, surface_temp)

        profile = cls(gradient=gradient, surface_temp=surface_temp)
        return profile

    @classmethod
    def from_two_points(
        cls,
        temp1: float,
        depth1: float,
        temp2: float,
        depth2: float,
        temp_unit: str = "F",
        depth_unit: str = "ft"
    ) -> "TemperatureProfile":
        """
        Create temperature profile from two reference points.

        Any two temperature values at known depths can be used
        to calculate the gradient.

        Parameters
        ----------
        temp1 : float
            First temperature
        depth1 : float
            First depth
        temp2 : float
            Second temperature
        depth2 : float
            Second depth
        temp_unit : str, optional
            Temperature unit
        depth_unit : str, optional
            Depth unit

        Returns
        -------
        TemperatureProfile
            Configured temperature profile
        """
        gradient = calculate_temperature_gradient(
            temp1, depth1, temp2, depth2, temp_unit, depth_unit
        )

        # Calculate surface temp by extrapolation
        if temp_unit.upper() == "C":
            temp1 = UnitConverter.celsius_to_fahrenheit(temp1)
        if depth_unit.lower() == "m":
            depth1 = UnitConverter.meters_to_feet(depth1)

        surface_temp = temp1 - (gradient * depth1 / 100)

        return cls(gradient=gradient, surface_temp=surface_temp)

    def get_temperature(
        self,
        depth: Union[float, np.ndarray],
        depth_unit: str = "ft"
    ) -> Union[float, np.ndarray]:
        """
        Get formation temperature at specified depth(s).

        Parameters
        ----------
        depth : float or np.ndarray
            Depth value(s)
        depth_unit : str, optional
            Depth unit

        Returns
        -------
        float or np.ndarray
            Temperature in °F
        """
        if self._gradient is None:
            raise ValueError("Temperature gradient not set")

        # Convert depth to feet
        if depth_unit.lower() == "m":
            depth = UnitConverter.meters_to_feet(depth)

        # Calculate temperature
        temperature = self.surface_temp + (self._gradient * depth / 100)
        return temperature

    def calculate(
        self,
        depth: np.ndarray,
        depth_unit: str = "ft"
    ) -> TemperatureResult:
        """
        Calculate temperature profile for depth array.

        Parameters
        ----------
        depth : np.ndarray
            Depth values
        depth_unit : str, optional
            Depth unit

        Returns
        -------
        TemperatureResult
            Temperature profile results
        """
        temperature = self.get_temperature(depth, depth_unit)

        return TemperatureResult(
            temperature=temperature,
            gradient=self._gradient,
            surface_temp=self.surface_temp
        )

    def add_to_well(
        self,
        well: "Well",
        curve_name: str = "TEMP"
    ) -> "Well":
        """
        Add temperature curve to well.

        Parameters
        ----------
        well : Well
            Well object
        curve_name : str, optional
            Name for temperature curve

        Returns
        -------
        Well
            Well with temperature curve added
        """
        if well.depth is None:
            raise ValueError("Well has no depth data")

        result = self.calculate(well.depth, well.depth_unit)

        well.add_curve(
            curve_name,
            result.temperature,
            unit="F",
            description="Formation Temperature"
        )

        # Store gradient
        well.set_parameter("temperature_gradient", result.gradient)
        well.set_parameter("surface_temperature", result.surface_temp)

        return well


def calculate_temperature_profile(
    well: "Well",
    bht: Optional[float] = None,
    surface_temp: float = DefaultParameters.SURFACE_TEMPERATURE_F,
    gradient: Optional[float] = None,
    bht_unit: str = "F"
) -> TemperatureResult:
    """
    Calculate temperature profile for a well.

    Parameters
    ----------
    well : Well
        Well object with depth data
    bht : float, optional
        Bottom hole temperature (uses header if not provided)
    surface_temp : float, optional
        Surface temperature
    gradient : float, optional
        Temperature gradient (calculated from BHT if not provided)
    bht_unit : str, optional
        BHT unit

    Returns
    -------
    TemperatureResult
        Temperature profile results

    Examples
    --------
    >>> from petrophys.environmental.temperature import calculate_temperature_profile
    >>> result = calculate_temperature_profile(well, bht=200)
    >>> well["TEMP"] = result.temperature
    """
    if well.depth is None:
        raise ValueError("Well has no depth data")

    # Get BHT from header if not provided
    if bht is None:
        bht = well.header.bht
        if well.header.bht_unit:
            bht_unit = well.header.bht_unit

    if bht is None and gradient is None:
        raise ValueError("Either BHT or gradient must be provided")

    # Create profile
    if gradient is not None:
        profile = TemperatureProfile(gradient=gradient, surface_temp=surface_temp)
    else:
        td = well.bottom
        if td is None:
            raise ValueError("Cannot determine TD for gradient calculation")
        profile = TemperatureProfile.from_bht(
            bht, td, surface_temp, bht_unit, well.depth_unit
        )

    return profile.calculate(well.depth, well.depth_unit)
