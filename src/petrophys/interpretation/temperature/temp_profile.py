"""
Temperature profile calculation module.

This module provides the TemperatureProfile class for computing
formation temperature at any depth.
"""

from typing import Any, Optional, Sequence, Union
from dataclasses import dataclass
import numpy as np

from petrophys.model.well import Well
from petrophys.model.constants import DefaultParameters
from petrophys.model.units import UnitConverter, temperature_to_fahrenheit, temperature_unit
from petrophys.interpretation.temperature.temp_gradient import (
    calculate_temperature_gradient,
    estimate_gradient_from_bht,
    gradient_c_per_km_to_fahrenheit_per_100ft,
)


def store_temperature_parameters(well: "Well", result: Any, *, with_metadata: bool = True) -> None:
    """Record a temperature profile's parameters on ``well``.

    ``with_metadata=False`` stores the values alone, without the ``__UNIT`` and
    ``__DESCR`` entries that a LAS export writes beside them.
    """
    values = (
        ("temperature_gradient", result.gradient, "degF/100ft", "Temperature gradient"),
        ("surface_temperature", result.surface_temp, "degF", "Surface temperature"),
        ("temperature_profile_source", result.source, "", "Temperature profile method"),
        ("temperature_reference_temperature", result.reference_temperature, "degF",
         "Reference (e.g. bottom-hole) temperature"),
        ("temperature_reference_depth", result.reference_depth, "ft", "Depth of the reference temperature"),
    )
    for name, value, unit, description in values:
        if value is None:
            continue
        well.set_parameter(name, value)
        if not with_metadata:
            continue
        if unit:
            well.set_parameter(f"{name}__UNIT", unit)
        well.set_parameter(f"{name}__DESCR", description)


def _parameter_float(well: "Well", name: str) -> Optional[float]:
    """Fetch a numeric well parameter if present."""
    value = well.get_parameter(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parameter_text(well: "Well", name: str) -> str:
    """Fetch a text well parameter if present."""
    value = well.get_parameter(name)
    return str(value or "").strip()


def _constant_zoned_parameter(well: "Well", prefix: str) -> Optional[float]:
    """Return a unique PARAM_Zn value when all zones share the same value."""
    values = set()
    prefix = prefix.upper()
    for key, value in getattr(well, "_parameters", {}).items():
        if not isinstance(key, str) or "__" in key:
            continue
        key_upper = key.upper()
        if not key_upper.startswith(f"{prefix}_Z"):
            continue
        try:
            values.add(float(value))
        except (TypeError, ValueError):
            continue
    if len(values) == 1:
        return next(iter(values))
    return None


def _metadata_temperature_f(well: "Well", name: str) -> Optional[float]:
    """Read scalar or single-valued zoned temperature metadata in Fahrenheit."""
    value = _parameter_float(well, name)
    if value is not None:
        return temperature_to_fahrenheit(value, _parameter_text(well, f"{name}__UNIT"))

    value = _constant_zoned_parameter(well, name)
    if value is not None:
        return temperature_to_fahrenheit(value, _parameter_text(well, f"{name}_Z1__UNIT"))

    return None


def temperature_curve_f(
    well: "Well", names: Sequence[str] = ("TEMP", "TMPC")
) -> Optional[np.ndarray]:
    """Return formation temperature in degF from the well's temperature curves.

    At each depth the first of ``names`` with a finite value is used. Each
    curve is converted from its own unit label (``degC``, ``K``, ...); a curve
    with no recognized temperature label is read as degF, the unit the
    workflow writes ``TEMP`` in.

    Returns
    -------
    numpy.ndarray or None
        Temperature (degF) per sample, or None when none of the curves has data.
    """
    result = np.full(well.n_samples, np.nan, dtype=float)
    for name in names:
        curve = well.get(name)
        if curve is None:
            continue
        data = np.asarray(curve.data, dtype=float)
        if data.shape != result.shape:
            continue
        scale = temperature_unit(curve.unit)
        if scale in ("C", "K"):
            data = UnitConverter.temperature(data, scale, "F")
        fill = ~np.isfinite(result) & np.isfinite(data)
        result[fill] = data[fill]
    return result if np.isfinite(result).any() else None


def _formation_temperature_f(well: "Well", names: Sequence[str]) -> Union[float, np.ndarray]:
    """Formation temperature (degF) from temperature curves, else a BHT profile."""
    temperature = temperature_curve_f(well, names)
    if temperature is not None:
        return temperature
    try:
        return calculate_temperature_profile(well).temperature
    except Exception:
        # No depth for a profile: the header BHT itself, or a nominal 150 degF.
        if well.header.bht is None:
            return 150.0
        return temperature_to_fahrenheit(well.header.bht, well.header.bht_unit)


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
    source: str = "unknown"
    reference_temperature: Optional[float] = None
    reference_depth: Optional[float] = None


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
        Unit of ``surface_temp``: "F", "C", "K" or a label such as "degC"
        (see :func:`petrophys.model.units.temperature_unit`); unknown labels
        raise ``ValueError``

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
        temp_unit: str = "F",
        source: str = "specified_gradient",
        reference_temperature: Optional[float] = None,
        reference_depth: Optional[float] = None,
    ):
        self.temp_unit = temperature_unit(temp_unit, strict=True)
        self.surface_temp = temperature_to_fahrenheit(surface_temp, temp_unit)

        self._gradient = gradient  # °F/100ft
        self.source = source
        self.reference_temperature = reference_temperature
        self.reference_depth = reference_depth

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
        depth_unit: str = "ft",
        surface_temp_unit: Optional[str] = None,
    ) -> "TemperatureProfile":
        """
        Create temperature profile from BHT and TD.

        The profile is linear from ``surface_temp`` at depth 0 to ``bht`` at
        ``td``: the BHT is taken as the static formation temperature (no
        correction for mud circulation, such as a Horner plot) and the surface
        temperature applies at depth 0, not at the seabed offshore.

        Parameters
        ----------
        bht : float
            Bottom hole temperature
        td : float
            Total depth
        surface_temp : float, optional
            Surface temperature
        bht_unit : str, optional
            BHT unit: "F", "C", "K" or a label such as "degC"; unknown labels
            raise ``ValueError``
        depth_unit : str, optional
            Depth unit

        Returns
        -------
        TemperatureProfile
            Configured temperature profile
        """
        # Convert to standard units ("C", "degC", "deg C" ... are all Celsius)
        bht = temperature_to_fahrenheit(bht, temperature_unit(bht_unit, strict=True))

        if surface_temp_unit is None:
            surface_temp_unit = (
                "F"
                if surface_temp == DefaultParameters.SURFACE_TEMPERATURE_F
                else bht_unit
            )
        surface_temp = temperature_to_fahrenheit(
            surface_temp, temperature_unit(surface_temp_unit, strict=True)
        )

        if depth_unit.lower() == "m":
            td = UnitConverter.meters_to_feet(td)

        gradient = estimate_gradient_from_bht(bht, td, surface_temp)

        profile = cls(
            gradient=gradient,
            surface_temp=surface_temp,
            source="linear_gradient_from_BHT",
            reference_temperature=float(bht),
            reference_depth=float(td),
        )
        return profile

    @classmethod
    def from_gradient_c_per_km(
        cls,
        gradient_c_per_km: float = DefaultParameters.GEOTHERMAL_GRADIENT_C_PER_KM,
        surface_temp: float = DefaultParameters.SURFACE_TEMPERATURE_C,
        surface_temp_unit: str = "C",
    ) -> "TemperatureProfile":
        """Create a fallback geothermal profile from a C/km gradient."""
        gradient = gradient_c_per_km_to_fahrenheit_per_100ft(gradient_c_per_km)
        return cls(
            gradient=gradient,
            surface_temp=temperature_to_fahrenheit(
                surface_temp, temperature_unit(surface_temp_unit, strict=True)
            ),
            source="fallback_geothermal_gradient",
        )

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
            Temperature unit: "F", "C", "K" or a label such as "degC";
            unknown labels raise ``ValueError``
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

        ref_temp = temp2
        ref_depth = depth2

        # Calculate surface temp by extrapolation
        if temperature_unit(temp_unit, strict=True) != "F":
            temp1 = UnitConverter.temperature(temp1, temp_unit, "F")
            ref_temp = UnitConverter.temperature(ref_temp, temp_unit, "F")
        if depth_unit.lower() == "m":
            depth1 = UnitConverter.meters_to_feet(depth1)
            ref_depth = UnitConverter.meters_to_feet(ref_depth)

        surface_temp = temp1 - (gradient * depth1 / 100)

        return cls(
            gradient=gradient,
            surface_temp=surface_temp,
            source="linear_gradient_from_two_points",
            reference_temperature=float(ref_temp),
            reference_depth=float(ref_depth),
        )

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
            surface_temp=self.surface_temp,
            source=self.source,
            reference_temperature=self.reference_temperature,
            reference_depth=self.reference_depth,
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

        store_temperature_parameters(well, result)

        return well


def calculate_temperature_profile(
    well: "Well",
    bht: Optional[float] = None,
    surface_temp: float = DefaultParameters.SURFACE_TEMPERATURE_F,
    gradient: Optional[float] = None,
    bht_unit: str = "F",
    surface_temp_unit: str = "F",
    td: Optional[float] = None,
    td_unit: Optional[str] = None,
    depth_unit: Optional[str] = None,
    fallback_gradient_c_per_km: float = DefaultParameters.GEOTHERMAL_GRADIENT_C_PER_KM,
) -> TemperatureResult:
    """
    Calculate temperature profile for a well.

    Parameters
    ----------
    well : Well
        Well object with depth data
    bht : float, optional
        Bottom hole temperature (uses header if not provided, with an
        unlabelled header unit read as degF)
    surface_temp : float, optional
        Surface temperature
    gradient : float, optional
        Temperature gradient (calculated from BHT if not provided)
    bht_unit : str, optional
        Unit of ``bht``: "F", "C", "K" or a label such as "degC"; unknown
        labels raise ``ValueError``

    Returns
    -------
    TemperatureResult
        Temperature profile results

    Examples
    --------
    >>> from petrophys.interpretation.temperature import calculate_temperature_profile
    >>> result = calculate_temperature_profile(well, bht=200)
    >>> well["TEMP"] = result.temperature
    """
    if well.depth is None:
        raise ValueError("Well has no depth data")

    depth_unit = depth_unit or well.depth_unit

    # Get BHT from header if not provided. Header temperatures follow the LAS
    # convention: an unlabelled or unrecognized unit is read as degF.
    if bht is None and well.header.bht is not None:
        unit = _parameter_text(well, "BHT__UNIT") or well.header.bht_unit
        bht, bht_unit = temperature_to_fahrenheit(well.header.bht, unit), "F"

    metadata_bht = _metadata_temperature_f(well, "BHT")
    if bht is None and metadata_bht is not None:
        bht = metadata_bht
        bht_unit = "F"

    if td is None:
        td = well.header.td
    if td is None:
        td = well.bottom
    td_depth_unit = td_unit or depth_unit

    # Create profile
    if gradient is not None:
        profile = TemperatureProfile(
            gradient=gradient,
            surface_temp=surface_temp,
            temp_unit=surface_temp_unit,
            source="specified_gradient",
        )
    elif bht is not None and td is not None:
        profile = TemperatureProfile.from_bht(
            bht=bht,
            td=td,
            surface_temp=surface_temp,
            bht_unit=bht_unit,
            depth_unit=td_depth_unit,
            surface_temp_unit=surface_temp_unit,
        )
    else:
        profile = TemperatureProfile.from_gradient_c_per_km(
            fallback_gradient_c_per_km,
            surface_temp=surface_temp,
            surface_temp_unit=surface_temp_unit,
        )

    return profile.calculate(well.depth, depth_unit)
