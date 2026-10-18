"""
Matrix properties for lithology determination.

This module provides reference mineral properties for
multi-mineral solver and crossplot analysis.
"""

from typing import Dict, Optional, NamedTuple
from petrophys.model.constants import MatrixProperties, MineralProperties


# Mineral properties for multi-mineral analysis. "u" is PEF x density, the
# convention of the solvers' log term (PEF x RHOB); chartbook U values use the
# electron density instead (fresh water 0.398), as the RHOmaa-Umaa transform
# in lithology_module does.
MINERAL_PROPERTIES: Dict[str, Dict[str, float]] = {
    "quartz": {
        "density": 2.65,  # g/cm³
        "slowness": 55.5,  # μs/ft
        "nphi": -0.02,  # v/v
        "pef": 1.8,  # b/e
        "u": 4.8,  # barns/cc (PEF × density)
    },
    "calcite": {
        "density": 2.71,
        "slowness": 47.6,
        "nphi": 0.00,
        "pef": 5.1,
        "u": 13.8,
    },
    "dolomite": {
        "density": 2.87,
        "slowness": 43.5,
        "nphi": 0.02,
        "pef": 3.1,
        "u": 9.0,
    },
    "anhydrite": {
        "density": 2.96,
        "slowness": 50.0,
        "nphi": -0.02,
        "pef": 5.0,
        "u": 14.9,
    },
    "gypsum": {
        "density": 2.31,
        "slowness": 52.5,
        "nphi": 0.49,
        "pef": 3.99,
        "u": 9.2,
    },
    "halite": {
        "density": 2.17,
        "slowness": 67.0,
        "nphi": -0.03,
        "pef": 4.65,
        "u": 10.1,
    },
    "shale": {
        "density": 2.58,
        "slowness": 100.0,
        "nphi": 0.30,
        "pef": 3.5,
        "u": 9.0,
    },
    "coal": {
        "density": 1.35,
        "slowness": 120.0,  # bituminous coal (Schlumberger, 1972, Table 18-1)
        "nphi": 0.50,
        "pef": 0.2,
        "u": 0.27,
    },
    "water": {
        "density": 1.0,
        "slowness": 189.0,
        "nphi": 1.0,
        "pef": 0.36,
        "u": 0.36,
    },
    "oil": {
        "density": 0.8,
        "slowness": 238.0,
        "nphi": 1.0,  # hydrogen index of a CH2 liquid at 0.8 g/cm³
        "pef": 0.12,
        "u": 0.10,
    },
    "gas": {
        "density": 0.2,
        "slowness": 600.0,
        "nphi": 0.3,
        "pef": 0.09,
        "u": 0.02,
    },
    "clay": {
        "density": 2.65,   # g/cm³  — generic dry clay (illite/chlorite PEF; kaolinite reads ~1.8)
        "slowness": 80.0,  # μs/ft
        "nphi": 0.35,      # v/v    — high due to clay-bound water
        "pef": 3.7,        # b/e
        "u": 9.81,         # b/cm³  (3.7 × 2.65)
    },
}


def get_mineral_properties(mineral: str, property_name: Optional[str] = None):
    """
    Get properties for a mineral.

    Parameters
    ----------
    mineral : str
        Mineral name
    property_name : str, optional
        Specific property to return

    Returns
    -------
    dict or float
        Properties dict or specific property value
    """
    mineral = mineral.lower()
    if mineral not in MINERAL_PROPERTIES:
        # Try to get from core constants
        try:
            matrix = MatrixProperties.get(mineral)
        except KeyError:
            raise KeyError(f"Unknown mineral: {mineral}")
        props = {
            "density": matrix.density,
            "slowness": matrix.slowness,
            "nphi": matrix.nphi,
            "pef": matrix.pef,
            # volumetric photoelectric factor U = PEF x density (consumers
            # such as the multi-mineral solver read "u" unconditionally).
            "u": matrix.pef * matrix.density,
        }
    else:
        props = MINERAL_PROPERTIES[mineral]
    if property_name:
        if property_name not in props:
            raise KeyError(
                f"Unknown property {property_name!r} for mineral {mineral!r}; "
                f"available: {sorted(props)}"
            )
        return props[property_name]
    return props


def get_matrix_density(mineral: str) -> float:
    """Get matrix density for a mineral in g/cm³."""
    return get_mineral_properties(mineral, "density")


def get_log_density(mineral: str) -> float:
    """Density-log response of a mineral in g/cm³.

    The apparent log density where one is tabulated (the evaporites of
    :class:`~petrophys.model.constants.MatrixProperties`, e.g. halite 2.032),
    otherwise the grain density. This is the RHOB endpoint of the N-D
    crossplot and the multi-mineral solver; the benchmark packages used the
    grain density, kept inside ``historical_benchmark()``.
    """
    from petrophys._internal.benchmark import reproducing_benchmark

    density = get_mineral_properties(mineral, "density")
    if reproducing_benchmark():
        return density
    try:
        props = MatrixProperties.get(mineral)
    except KeyError:
        return density
    return props.log_density if props.log_density is not None else density


def get_matrix_slowness(mineral: str) -> float:
    """Get matrix slowness for a mineral in μs/ft."""
    return get_mineral_properties(mineral, "slowness")


def get_matrix_nphi(mineral: str) -> float:
    """Get matrix neutron response for a mineral in v/v."""
    return get_mineral_properties(mineral, "nphi")


def get_matrix_pef(mineral: str) -> float:
    """Get matrix PEF for a mineral in b/e."""
    return get_mineral_properties(mineral, "pef")


class MineralEndpoints:
    """
    Container for mineral endpoints used in multi-mineral analysis.
    """

    def __init__(
        self,
        minerals: list = None,
        fluid: str = "water"
    ):
        """
        Initialize mineral endpoints.

        Parameters
        ----------
        minerals : list, optional
            List of mineral names (default: ["quartz", "calcite", "dolomite"])
        fluid : str, optional
            Fluid type (default: "water")
        """
        if minerals is None:
            minerals = ["quartz", "calcite", "dolomite"]

        self.minerals = minerals
        self.fluid = fluid

        # Load properties
        self.endpoints = {}
        for mineral in minerals:
            self.endpoints[mineral] = get_mineral_properties(mineral)
        self.endpoints["fluid"] = get_mineral_properties(fluid)

    def get_density(self, mineral: str) -> float:
        """Get density for a component."""
        return self.endpoints[mineral]["density"]

    def get_nphi(self, mineral: str) -> float:
        """Get NPHI for a component."""
        return self.endpoints[mineral]["nphi"]

    def get_pef(self, mineral: str) -> float:
        """Get PEF for a component."""
        return self.endpoints[mineral]["pef"]

    def get_slowness(self, mineral: str) -> float:
        """Get slowness for a component."""
        return self.endpoints[mineral]["slowness"]
