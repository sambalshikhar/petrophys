"""
Physical constants and matrix properties for petrophysical calculations.

This module provides reference values for:
- Matrix properties (density, slowness, neutron response, PEF)
- Fluid properties (water, oil, gas)
- Standard default values used throughout the library
"""

from dataclasses import dataclass
from typing import Dict, Optional
import unicodedata
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
    log_density : float, optional
        Apparent density in g/cm³ that a density log reads in the pure
        mineral, where it differs materially from the grain ``density``. The
        tool reads rho_log = 1.0704 * rho_e - 0.1883, with electron density
        rho_e = density * 2Z/A (Schlumberger, 1989), so hydrogen- and
        chlorine-rich evaporites read well away from their grain density.
        It is the matrix endpoint for density porosity; the mineral solvers
        use the grain density.
    """

    name: str
    density: float  # g/cm³
    slowness: float  # μs/ft
    nphi: float  # v/v
    pef: float  # b/e
    log_density: Optional[float] = None  # g/cm³


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
        # Evaporites: log_density is the apparent density-log reading, and the
        # potash salts' slowness the sonic reading, of Schlumberger (1972),
        # Table 18-1, which has no kainite slowness. The potash salts' neutron
        # (CNL; carnallite and kainite read 0.60 or more) and PEF values are
        # from the Schlumberger Log Interpretation Charts (2009), and their
        # grain densities follow from the log densities.
        "anhydrite": MineralProperties(
            name="Anhydrite",
            density=2.96,
            slowness=50.0,
            nphi=-0.02,
            pef=5.0,
            log_density=2.977
        ),
        "gypsum": MineralProperties(
            name="Gypsum",
            density=2.31,
            slowness=52.5,
            nphi=0.49,
            pef=3.99,
            log_density=2.351
        ),
        "halite": MineralProperties(
            name="Halite (Salt)",
            density=2.17,
            slowness=67.0,
            nphi=-0.03,
            pef=4.65,
            log_density=2.032
        ),
        "sylvite": MineralProperties(
            name="Sylvite",
            density=1.984,
            slowness=74.0,
            nphi=-0.03,
            pef=8.51,
            log_density=1.863
        ),
        "carnallite": MineralProperties(
            name="Carnallite",
            density=1.61,
            slowness=78.0,
            nphi=0.60,
            pef=4.09,
            log_density=1.570
        ),
        "polyhalite": MineralProperties(
            name="Polyhalite",
            density=2.78,
            slowness=57.5,
            nphi=0.25,
            pef=4.32,
            log_density=2.790
        ),
        "kainite": MineralProperties(
            name="Kainite",
            density=2.13,
            slowness=float("nan"),
            nphi=0.60,
            pef=3.50,
            log_density=2.120
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
            # Bituminous coal reads 110-140 us/ft (Schlumberger, 1972,
            # Table 18-1), faster than the pore fluid.
            slowness=120.0,
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
    def names(cls) -> list:
        """Return every accepted mineral name and alias, sorted."""
        return sorted(set(cls._MINERALS) | set(cls._ALIASES))

    @classmethod
    def get_density(cls, mineral: str) -> float:
        """Get matrix density for a mineral in g/cm³."""
        return cls.get(mineral).density

    @classmethod
    def get_log_density(cls, mineral: str) -> float:
        """Get the density-porosity matrix endpoint for a mineral in g/cm³.

        This is the apparent log density where one is tabulated (evaporites)
        and the grain density otherwise.
        """
        props = cls.get(mineral)
        return props.log_density if props.log_density is not None else props.density

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
class ArchieCoefficients:
    """
    Default Archie coefficients for a lithology.

    Attributes
    ----------
    lithology : str
        Canonical lithology key.
    a : float
        Tortuosity coefficient.
    m : float
        Cementation exponent.
    n : float
        Saturation exponent.
    closest_lithology : str
        Closest rock type used when the input lithology is not a clean
        Archie rock type.
    note : str
        Short applicability note.
    """

    lithology: str
    a: float
    m: float
    n: float
    closest_lithology: str
    note: str = ""


class ArchieParameterDefaults:
    """
    Lithology-aware default Archie coefficients.

    These values are starting points only. Prefer Pickett plots or SCAL/core
    electrical measurements when available, especially for shaly, fractured,
    vuggy, evaporitic, organic-rich, or igneous intervals.
    """

    _DEFAULT = ArchieCoefficients(
        lithology="generic_clean",
        a=1.0,
        m=2.0,
        n=2.0,
        closest_lithology="generic_clean",
        note="Standard clean-rock Archie fallback.",
    )

    _LITHOLOGIES: Dict[str, ArchieCoefficients] = {
        "sandstone": ArchieCoefficients(
            lithology="sandstone",
            a=0.81,
            m=2.0,
            n=2.0,
            closest_lithology="granular_sandstone",
            note="Humble near-equivalent granular-rock form.",
        ),
        "unconsolidated_sand": ArchieCoefficients(
            lithology="unconsolidated_sand",
            a=0.62,
            m=2.15,
            n=2.0,
            closest_lithology="high_porosity_granular_sand",
            note="Original Humble granular-rock form.",
        ),
        "shaly_sandstone": ArchieCoefficients(
            lithology="shaly_sandstone",
            a=1.0,
            m=1.8,
            n=1.8,
            closest_lithology="shaly_sandstone",
            note="Approximate shaly-sand fallback; prefer Simandoux/Indonesia/Dual Water.",
        ),
        "silt": ArchieCoefficients(
            lithology="silt",
            a=1.0,
            m=2.0,
            n=2.0,
            closest_lithology="fine_clastic",
            note="Closest clean fine-clastic fallback.",
        ),
        "clastic_coarse": ArchieCoefficients(
            lithology="clastic_coarse",
            a=0.81,
            m=2.0,
            n=2.0,
            closest_lithology="granular_sandstone",
            note="Closest granular clastic fallback for conglomerate/breccia/gravel.",
        ),
        "limestone": ArchieCoefficients(
            lithology="limestone",
            a=1.0,
            m=2.0,
            n=2.0,
            closest_lithology="clean_carbonate",
            note="Clean interparticle/intercrystalline carbonate fallback.",
        ),
        "dolomite": ArchieCoefficients(
            lithology="dolomite",
            a=1.0,
            m=2.0,
            n=2.0,
            closest_lithology="clean_carbonate",
            note="Clean intercrystalline carbonate fallback.",
        ),
        "shale": ArchieCoefficients(
            lithology="shale",
            a=1.0,
            m=2.0,
            n=2.0,
            closest_lithology="shaly_clastic",
            note="Archie is not recommended for conductive shale; use shaly-sand models.",
        ),
        "calcareous_shale": ArchieCoefficients(
            lithology="calcareous_shale",
            a=1.0,
            m=2.0,
            n=2.0,
            closest_lithology="shaly_carbonate",
            note="Mixed shale-carbonate fallback; use with caution.",
        ),
        "chert": ArchieCoefficients(
            lithology="chert",
            a=1.0,
            m=2.0,
            n=2.0,
            closest_lithology="siliceous_clean_rock",
            note="Generic clean-rock fallback.",
        ),
        "coal": ArchieCoefficients(
            lithology="coal",
            a=1.0,
            m=2.0,
            n=2.0,
            closest_lithology="non_archie_special_lithology",
            note="Organic-rich coal is not a typical Archie reservoir rock.",
        ),
        "evaporite": ArchieCoefficients(
            lithology="evaporite",
            a=1.0,
            m=2.0,
            n=2.0,
            closest_lithology="non_archie_seal",
            note="Evaporites are usually treated as seals/nonreservoir intervals.",
        ),
        "igneous": ArchieCoefficients(
            lithology="igneous",
            a=1.0,
            m=2.0,
            n=2.0,
            closest_lithology="non_archie_special_lithology",
            note="Generic fallback; calibrate for fractures/vugs if reservoir quality exists.",
        ),
        "generic_clean": _DEFAULT,
    }

    _ALIASES: Dict[str, str] = {
        "sand": "sandstone",
        "silty_sand": "sandstone",
        "unconsolidated": "unconsolidated_sand",
        "crossbedded_sand": "sandstone",
        "cross_bedded_sand": "sandstone",
        "cross_bedded_sst": "sandstone",
        "aeolian_sandstone": "sandstone",
        "calcareous_sandstone": "sandstone",
        "porous_limestone": "limestone",
        "sst": "sandstone",
        "shaly_sand": "shaly_sandstone",
        "shaly_sst": "shaly_sandstone",
        "sandy_silt": "shaly_sandstone",
        "sandy_shale": "shale",
        "biogenic_ooze": "shale",
        "silty_shale": "shale",
        "shaly_silt": "silt",
        "siltstone": "silt",
        "conglomerate": "clastic_coarse",
        "breccia": "clastic_coarse",
        "gravel": "clastic_coarse",
        "calcareous_debris_flow": "clastic_coarse",
        "arkose": "clastic_coarse",
        "carbonate": "limestone",
        "calcite": "limestone",
        "chalk": "limestone",
        "marlstone": "limestone",
        "sandy_limestone": "limestone",
        "argillaceous_limestone": "limestone",
        "calcareous_cement": "limestone",
        "dolomite2": "dolomite",
        "anhydrite": "evaporite",
        "gypsum": "evaporite",
        "halite": "evaporite",
        "salt": "evaporite",
        "sylvinite": "evaporite",
        "sylvite": "evaporite",
        "sylvine": "evaporite",
        "potash": "evaporite",
        "potash_salt": "evaporite",
        "polyhalite": "evaporite",
        "carnallite": "evaporite",
        "kainite": "evaporite",
        "gap": "generic_clean",
        "cinerite": "igneous",
        "extrusive": "igneous",
        "intrusive": "igneous",
        "basement": "igneous",
        "plutonic_rock": "igneous",
        "volcanic_rock": "igneous",
        "volcanic_tuff": "igneous",
        "metamorphic_rock": "shaly_sandstone",
        "spiculite": "chert",
        "no_lithology_data": "generic_clean",
        "special_lithology": "generic_clean",
        "mixed": "generic_clean",
        "clastic": "sandstone",
    }

    @classmethod
    def names(cls) -> list:
        """Return every lithology name and alias with tabulated coefficients."""
        return sorted(set(cls._LITHOLOGIES) | set(cls._ALIASES))

    @classmethod
    def normalize(cls, lithology: Optional[str]) -> str:
        """Normalize lithology labels from CSV/LITHOLOGY_GEOLINK nomenclature."""
        if lithology is None:
            return "generic_clean"
        key = unicodedata.normalize("NFKD", str(lithology).strip().lower())
        key = "".join(ch for ch in key if not unicodedata.combining(ch))
        for old, new in (("&", "_"), ("+", "_"), ("/", "_"), ("-", "_"), (".", ""), (",", "")):
            key = key.replace(old, new)
        key = "_".join(part for part in key.split() if part)
        while "__" in key:
            key = key.replace("__", "_")
        return key or "generic_clean"

    @classmethod
    def get(cls, lithology: Optional[str]) -> ArchieCoefficients:
        """Return Archie coefficients for a lithology or its closest alias."""
        key = cls.normalize(lithology)
        key = cls._ALIASES.get(key, key)
        return cls._LITHOLOGIES.get(key, cls._DEFAULT)

    @classmethod
    def get_tuple(cls, lithology: Optional[str]) -> tuple:
        """Return ``(a, m, n)`` for a lithology."""
        coeffs = cls.get(lithology)
        return coeffs.a, coeffs.m, coeffs.n

    @classmethod
    def as_dict(cls) -> Dict[str, Dict[str, float]]:
        """Return a plain dictionary of canonical coefficients."""
        return {
            key: {"a": value.a, "m": value.m, "n": value.n}
            for key, value in cls._LITHOLOGIES.items()
        }

    @classmethod
    def list_lithologies(cls) -> list:
        """Return canonical lithology keys with explicit coefficients."""
        return sorted(cls._LITHOLOGIES.keys())


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

    # Liquid hydrocarbons CH_n have hydrogen index 9 n rho / (12 + n)
    # (Schlumberger, 1989): about 1.0 for oil (n = 2, 0.8 g/cm³) and for heavy
    # oil (n = 1.6, 0.95 g/cm³), 0.9 for light oil (0.7 g/cm³).
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
            hydrogen_index=1.0
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
            hydrogen_index=1.0
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
    GEOTHERMAL_GRADIENT_C_PER_KM = 30.0
    GEOTHERMAL_GRADIENT_F_PER_100FT = 1.64592

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
    ARCHIE_BY_LITHOLOGY = ArchieParameterDefaults.as_dict()

    # Net pay cutoffs - generic lithology standards, one source of truth
    # (exposed as STANDARD_CUTOFFS in petrophys.evaluation.net_pay.cutoffs):
    #   clastic   : VCL < 0.40, PHIE > 0.08,  Sw < 0.60
    #   carbonate : VCL < 0.30, PHIE > 0.045, Sw < 0.70
    # (carbonates produce from lower porosity and at higher Sw than clastics;
    #  projects may override, e.g. the Costa case studies screen the high-phi
    #  Kharaib with a tighter 0.10 / 0.60 set.)
    VCL_CUTOFF_CLASTIC = 0.40
    VCL_CUTOFF_CARBONATE = 0.30
    PHIE_CUTOFF_CLASTIC = 0.08
    PHIE_CUTOFF_CARBONATE = 0.045
    SW_CUTOFF_CLASTIC = 0.60
    SW_CUTOFF_CARBONATE = 0.70
    SW_CUTOFF = 0.60  # alias of SW_CUTOFF_CLASTIC

    # OBM default
    RXO_OBM_DEFAULT = 200.0  # ohm.m

    # HC correction
    OIL_HYDROGEN_INDEX = 1.0  # CH2 liquid at OIL_DENSITY (see FluidProperties)
    GAS_HYDROGEN_INDEX = 0.3
    OIL_DENSITY = 0.8  # g/cm³
    GAS_DENSITY = 0.2  # g/cm³
