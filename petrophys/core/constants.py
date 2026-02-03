"""
Physical constants and matrix properties for petrophysical calculations.

This module provides reference values for:
- Matrix properties (density, slowness, neutron response, PEF)
- Fluid properties (water, oil, gas)
- Standard default values used throughout the library
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass(frozen=True)
class MineralProperties:
    """
    Properties of a mineral for petrophysical calculations.

    Attributes
    ----------
    name : str
        Mineral name
    density : float
        Matrix density in g/cm³
    slowness : float
        Matrix slowness in μs/ft
    nphi : float
        Apparent neutron porosity in v/v
    pef : float
        Photoelectric factor in b/e
    """

    name: str
    density: float  # g/cm³
    slowness: float  # μs/ft
    nphi: float  # v/v
    pef: float  # b/e


class MatrixProperties:
    """
    Reference matrix properties for common minerals.

    This class provides a lookup table for mineral properties
    used in lithology determination and porosity calculations.

    Examples
    --------
    >>> props = MatrixProperties.get("quartz")
    >>> print(props.density)
    2.65
    >>> print(props.slowness)
    55.5
    """

    # Standard mineral properties table
    _MINERALS: Dict[str, MineralProperties] = {
        "quartz": MineralProperties(
            name="Quartz",
            density=2.65,
            slowness=55.5,
            nphi=-0.02,
            pef=1.8
        ),
        "calcite": MineralProperties(
            name="Calcite",
            density=2.71,
            slowness=47.6,
            nphi=0.00,
            pef=5.1
        ),
        "dolomite": MineralProperties(
            name="Dolomite",
            density=2.87,
            slowness=43.5,
            nphi=0.02,
            pef=3.1
        ),
        "anhydrite": MineralProperties(
            name="Anhydrite",
            density=2.96,
            slowness=50.0,
            nphi=-0.02,
            pef=5.0
        ),
        "gypsum": MineralProperties(
            name="Gypsum",
            density=2.31,
            slowness=83.0,
            nphi=0.49,
            pef=3.99
        ),
        "halite": MineralProperties(
            name="Halite (Salt)",
            density=2.17,
            slowness=67.0,
            nphi=-0.03,
            pef=4.65
        ),
        "shale": MineralProperties(
            name="Shale",
            density=2.58,
            slowness=100.0,
            nphi=0.30,
            pef=3.5
        ),
        "coal": MineralProperties(
            name="Coal",
            density=1.35,
            slowness=250.0,
            nphi=0.50,
            pef=0.2
        ),
        "sandstone": MineralProperties(
            name="Sandstone",
            density=2.65,
            slowness=55.5,
            nphi=-0.02,
            pef=1.8
        ),
        "limestone": MineralProperties(
            name="Limestone",
            density=2.71,
            slowness=47.6,
            nphi=0.00,
            pef=5.1
        ),
    }

    # Aliases for common names
    _ALIASES: Dict[str, str] = {
        "sand": "sandstone",
        "ss": "sandstone",
        "lime": "limestone",
        "ls": "limestone",
        "dolo": "dolomite",
        "salt": "halite",
        "calcium_carbonate": "calcite",
        "sio2": "quartz",
        "silica": "quartz",
    }

    @classmethod
    def get(cls, mineral: str) -> MineralProperties:
        """
        Get properties for a mineral.

        Parameters
        ----------
        mineral : str
            Mineral name or alias

        Returns
        -------
        MineralProperties
            Properties for the specified mineral

        Raises
        ------
        KeyError
            If mineral is not found
        """
        key = mineral.lower()
        if key in cls._ALIASES:
            key = cls._ALIASES[key]

        if key not in cls._MINERALS:
            available = list(cls._MINERALS.keys())
            raise KeyError(
                f"Unknown mineral: {mineral}. Available: {available}"
            )

        return cls._MINERALS[key]

    @classmethod
    def list_minerals(cls) -> list:
        """Return list of available minerals."""
        return list(cls._MINERALS.keys())

    @classmethod
    def get_density(cls, mineral: str) -> float:
        """Get matrix density for a mineral in g/cm³."""
        return cls.get(mineral).density

    @classmethod
    def get_slowness(cls, mineral: str) -> float:
        """Get matrix slowness for a mineral in μs/ft."""
        return cls.get(mineral).slowness

    @classmethod
    def get_nphi(cls, mineral: str) -> float:
        """Get apparent neutron porosity for a mineral in v/v."""
        return cls.get(mineral).nphi

    @classmethod
    def get_pef(cls, mineral: str) -> float:
        """Get photoelectric factor for a mineral in b/e."""
        return cls.get(mineral).pef


@dataclass(frozen=True)
class FluidPropertySet:
    """
    Properties of a fluid for petrophysical calculations.

    Attributes
    ----------
    name : str
        Fluid name
    density : float
        Fluid density in g/cm³
    slowness : float
        Fluid slowness in μs/ft
    hydrogen_index : float
        Hydrogen index relative to fresh water (dimensionless)
    """

    name: str
    density: float  # g/cm³
    slowness: float  # μs/ft
    hydrogen_index: float  # dimensionless


class FluidProperties:
    """
    Reference fluid properties for petrophysical calculations.

    Examples
    --------
    >>> props = FluidProperties.get("water")
    >>> print(props.density)
    1.0
    """

    _FLUIDS: Dict[str, FluidPropertySet] = {
        "water": FluidPropertySet(
            name="Fresh Water",
            density=1.0,
            slowness=189.0,
            hydrogen_index=1.0
        ),
        "saltwater": FluidPropertySet(
            name="Salt Water",
            density=1.1,
            slowness=185.0,
            hydrogen_index=1.0
        ),
        "oil": FluidPropertySet(
            name="Oil",
            density=0.8,
            slowness=238.0,
            hydrogen_index=0.8
        ),
        "light_oil": FluidPropertySet(
            name="Light Oil",
            density=0.7,
            slowness=250.0,
            hydrogen_index=0.9
        ),
        "heavy_oil": FluidPropertySet(
            name="Heavy Oil",
            density=0.95,
            slowness=220.0,
            hydrogen_index=0.7
        ),
        "gas": FluidPropertySet(
            name="Gas",
            density=0.2,
            slowness=600.0,
            hydrogen_index=0.3
        ),
        "mud_filtrate": FluidPropertySet(
            name="Mud Filtrate",
            density=1.0,
            slowness=189.0,
            hydrogen_index=1.0
        ),
    }

    @classmethod
    def get(cls, fluid: str) -> FluidPropertySet:
        """
        Get properties for a fluid.

        Parameters
        ----------
        fluid : str
            Fluid name

        Returns
        -------
        FluidPropertySet
            Properties for the specified fluid
        """
        key = fluid.lower().replace(" ", "_")
        if key not in cls._FLUIDS:
            available = list(cls._FLUIDS.keys())
            raise KeyError(
                f"Unknown fluid: {fluid}. Available: {available}"
            )
        return cls._FLUIDS[key]

    @classmethod
    def list_fluids(cls) -> list:
        """Return list of available fluids."""
        return list(cls._FLUIDS.keys())


class DefaultParameters:
    """
    Default parameter values for petrophysical calculations.
    """

    # Temperature defaults
    SURFACE_TEMPERATURE_F = 68.0  # °F
    SURFACE_TEMPERATURE_C = 20.0  # °C

    # Bad hole thresholds
    CALI_BS_THRESHOLD = 1.5  # inches
    DRHO_THRESHOLD = 0.15  # g/cm³
    RHOB_MIN = 2.1  # g/cm³
    PEF_MAX = 7.0  # b/e
    PEF_MIN = 0.5  # b/e
    LARGE_HOLE_BS = 17.5  # inches

    # Clay volume defaults
    GR_MAX = 100.0  # API
    GR_MIN = 20.0  # API

    # Porosity defaults
    FLUID_DENSITY = 1.0  # g/cm³
    CLAY_POROSITY = 0.15  # v/v
    FLUID_SLOWNESS = 189.0  # μs/ft

    # Archie parameters
    ARCHIE_A = 1.0
    ARCHIE_M = 2.0
    ARCHIE_N = 2.0

    # Net pay cutoffs
    VCL_CUTOFF = 0.30
    PHIE_CUTOFF_CLASTIC = 0.08
    PHIE_CUTOFF_CARBONATE = 0.045
    SW_CUTOFF = 0.70

    # OBM default
    RXO_OBM_DEFAULT = 200.0  # ohm.m

    # HC correction
    OIL_HYDROGEN_INDEX = 0.8
    GAS_HYDROGEN_INDEX = 0.3
    OIL_DENSITY = 0.8  # g/cm³
    GAS_DENSITY = 0.2  # g/cm³
