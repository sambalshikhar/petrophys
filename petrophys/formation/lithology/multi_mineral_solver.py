"""
Multi-mineral solver for lithology determination.

This module provides solvers for determining mineral fractions
from multiple log measurements using simultaneous equations.
"""

from typing import Optional, Tuple, Dict, Union
import numpy as np
from scipy.optimize import nnls

from petrophys.formation.lithology.matrix_properties import (
    get_mineral_properties,
    MINERAL_PROPERTIES,
)


def solve_two_mineral(
    log_value: float,
    mineral1_value: float,
    mineral2_value: float,
    fluid_value: float,
    porosity: float
) -> Tuple[float, float]:
    """
    Solve for two mineral fractions given a log response.

    Uses the equation:
    Log = φ * Fluid + (1-φ) * (V1 * M1 + V2 * M2)
    where V1 + V2 = 1

    Parameters
    ----------
    log_value : float
        Measured log value
    mineral1_value : float
        Response of mineral 1
    mineral2_value : float
        Response of mineral 2
    fluid_value : float
        Response of fluid
    porosity : float
        Porosity (0-1)

    Returns
    -------
    tuple
        (V1, V2) mineral fractions
    """
    # Matrix response
    matrix_value = (log_value - porosity * fluid_value) / (1 - porosity)

    # Solve for V1
    if mineral1_value == mineral2_value:
        v1 = 0.5
    else:
        v1 = (matrix_value - mineral2_value) / (mineral1_value - mineral2_value)

    v1 = np.clip(v1, 0, 1)
    v2 = 1 - v1

    return v1, v2


def solve_three_mineral(
    rhob: float,
    nphi: float,
    pef: float,
    minerals: Tuple[str, str, str] = ("quartz", "calcite", "dolomite"),
    fluid: str = "water",
    porosity: Optional[float] = None
) -> Tuple[float, float, float, float]:
    """
    Solve for three mineral fractions using RHOB, NPHI, and PEF.

    Uses simultaneous equations to solve for:
    - V1, V2, V3: Mineral volume fractions
    - φ: Porosity (if not provided)

    Parameters
    ----------
    rhob : float
        Bulk density (g/cm³)
    nphi : float
        Neutron porosity (v/v)
    pef : float
        Photoelectric factor (b/e)
    minerals : tuple, optional
        Three mineral types to solve for
    fluid : str, optional
        Fluid type
    porosity : float, optional
        Porosity if known

    Returns
    -------
    tuple
        (V1, V2, V3, porosity) mineral fractions and porosity
    """
    # Get mineral properties
    m1 = get_mineral_properties(minerals[0])
    m2 = get_mineral_properties(minerals[1])
    m3 = get_mineral_properties(minerals[2])
    fl = get_mineral_properties(fluid)

    # Build matrix equation: A * x = b
    # where x = [V1, V2, V3, φ]

    # Using RHOB, NPHI, PEF, and closure (V1 + V2 + V3 + φ = 1)
    A = np.array([
        [m1["density"], m2["density"], m3["density"], fl["density"]],
        [m1["nphi"], m2["nphi"], m3["nphi"], fl["nphi"]],
        [m1["pef"], m2["pef"], m3["pef"], fl["pef"]],
        [1, 1, 1, 1]  # Closure
    ])

    b = np.array([rhob, nphi, pef, 1.0])

    # Add porosity constraint if known
    if porosity is not None:
        # Fix porosity, solve for 3 minerals
        A_reduced = np.array([
            [m1["density"], m2["density"], m3["density"]],
            [m1["nphi"], m2["nphi"], m3["nphi"]],
            [1, 1, 1]
        ])
        b_reduced = np.array([
            rhob - porosity * fl["density"],
            nphi - porosity * fl["nphi"],
            1 - porosity
        ])

        try:
            x, _ = nnls(A_reduced, b_reduced)
            return x[0], x[1], x[2], porosity
        except:
            return 0.0, 0.0, 0.0, porosity

    # Solve using non-negative least squares
    try:
        x, _ = nnls(A, b)
        # Normalize to ensure closure
        total = np.sum(x)
        if total > 0:
            x = x / total
        return x[0], x[1], x[2], x[3]
    except:
        return 0.0, 0.0, 0.0, 0.0


class MultiMineralSolver:
    """
    Multi-mineral solver for complex lithology.

    This class solves for mineral fractions using multiple
    log measurements simultaneously.

    Parameters
    ----------
    minerals : list, optional
        List of mineral types to solve for
    fluid : str, optional
        Fluid type
    use_pef : bool, optional
        Include PEF in solution (default: True)

    Examples
    --------
    >>> solver = MultiMineralSolver(minerals=["quartz", "calcite", "dolomite"])
    >>> result = solver.solve(rhob, nphi, pef)
    """

    def __init__(
        self,
        minerals: list = None,
        fluid: str = "water",
        use_pef: bool = True
    ):
        if minerals is None:
            minerals = ["quartz", "calcite", "dolomite"]

        self.minerals = minerals
        self.fluid = fluid
        self.use_pef = use_pef

        # Load properties
        self._load_properties()

    def _load_properties(self):
        """Load mineral and fluid properties."""
        self.mineral_props = {}
        for mineral in self.minerals:
            self.mineral_props[mineral] = get_mineral_properties(mineral)
        self.fluid_props = get_mineral_properties(self.fluid)

    def solve(
        self,
        rhob: Union[float, np.ndarray],
        nphi: Union[float, np.ndarray],
        pef: Optional[Union[float, np.ndarray]] = None,
        porosity: Optional[Union[float, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Solve for mineral fractions.

        Parameters
        ----------
        rhob : float or np.ndarray
            Bulk density
        nphi : float or np.ndarray
            Neutron porosity
        pef : float or np.ndarray, optional
            Photoelectric factor
        porosity : float or np.ndarray, optional
            Known porosity

        Returns
        -------
        dict
            Dictionary with mineral fractions and porosity
        """
        # Convert to arrays
        rhob = np.atleast_1d(rhob)
        nphi = np.atleast_1d(nphi)
        n = len(rhob)

        if pef is not None:
            pef = np.atleast_1d(pef)
        if porosity is not None:
            porosity = np.atleast_1d(porosity)

        # Initialize result arrays
        result = {mineral: np.zeros(n) for mineral in self.minerals}
        result["porosity"] = np.zeros(n)

        # Solve point by point
        for i in range(n):
            if np.isnan(rhob[i]) or np.isnan(nphi[i]):
                for mineral in self.minerals:
                    result[mineral][i] = np.nan
                result["porosity"][i] = np.nan
                continue

            pef_i = pef[i] if pef is not None else None
            por_i = porosity[i] if porosity is not None else None

            if len(self.minerals) == 3 and pef_i is not None and self.use_pef:
                v1, v2, v3, phi = solve_three_mineral(
                    rhob[i], nphi[i], pef_i,
                    tuple(self.minerals), self.fluid, por_i
                )
                result[self.minerals[0]][i] = v1
                result[self.minerals[1]][i] = v2
                result[self.minerals[2]][i] = v3
                result["porosity"][i] = phi
            elif len(self.minerals) == 2:
                # Two mineral solution using density
                v1, v2 = solve_two_mineral(
                    rhob[i],
                    self.mineral_props[self.minerals[0]]["density"],
                    self.mineral_props[self.minerals[1]]["density"],
                    self.fluid_props["density"],
                    por_i if por_i is not None else 0.15
                )
                result[self.minerals[0]][i] = v1
                result[self.minerals[1]][i] = v2
                result["porosity"][i] = por_i if por_i is not None else 0.15
            else:
                # Simplified solution
                for mineral in self.minerals:
                    result[mineral][i] = np.nan
                result["porosity"][i] = np.nan

        return result

    def add_to_well(self, well: "Well", **kwargs) -> "Well":
        """
        Add lithology curves to well.

        Parameters
        ----------
        well : Well
            Well object

        Returns
        -------
        Well
            Well with lithology curves
        """
        rhob = well.get("RHOB")
        nphi = well.get("NPHI")
        pef = well.get("PEF")

        if rhob is None or nphi is None:
            raise ValueError("RHOB and NPHI required for lithology")

        result = self.solve(
            rhob.data, nphi.data,
            pef.data if pef is not None else None,
            **kwargs
        )

        for mineral, values in result.items():
            if mineral == "porosity":
                continue
            curve_name = f"V{mineral.upper()[:4]}"
            well.add_curve(
                curve_name, values, unit="v/v",
                description=f"Volume {mineral}"
            )

        return well
