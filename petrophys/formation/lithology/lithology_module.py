"""
Main lithology determination module.

This module provides the LithologyModule class for comprehensive
lithology analysis from well logs.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass
import numpy as np

from petrophys.formation.lithology.multi_mineral_solver import (
    MultiMineralSolver,
    solve_three_mineral,
)
from petrophys.formation.lithology.crossplots import identify_lithology_zone


@dataclass
class LithologyResult:
    """
    Result container for lithology analysis.

    Attributes
    ----------
    v_quartz : np.ndarray
        Quartz volume fraction
    v_calcite : np.ndarray
        Calcite volume fraction
    v_dolomite : np.ndarray
        Dolomite volume fraction
    v_clay : np.ndarray
        Clay volume fraction (from VCL)
    lithology_code : np.ndarray
        Numeric lithology codes
    lithology_name : np.ndarray
        Lithology names
    """

    v_quartz: np.ndarray
    v_calcite: np.ndarray
    v_dolomite: np.ndarray
    v_clay: np.ndarray
    lithology_code: np.ndarray
    lithology_name: np.ndarray


class LithologyModule:
    """
    Lithology determination from well logs.

    This module combines multi-mineral analysis with clay volume
    to determine complete lithology fractions.

    Parameters
    ----------
    minerals : list, optional
        Primary minerals to solve for
    include_clay : bool, optional
        Include clay from VCL (default: True)

    Examples
    --------
    >>> litho = LithologyModule()
    >>> result = litho.calculate(well)
    >>> well["VQUAR"] = result.v_quartz
    """

    # Lithology codes
    LITHOLOGY_CODES = {
        "sandstone": 1,
        "limestone": 2,
        "dolomite": 3,
        "shale": 4,
        "anhydrite": 5,
        "salt": 6,
        "coal": 7,
        "mixed": 8,
        "unknown": 0,
    }

    def __init__(
        self,
        minerals: List[str] = None,
        include_clay: bool = True
    ):
        if minerals is None:
            minerals = ["quartz", "calcite", "dolomite"]

        self.minerals = minerals
        self.include_clay = include_clay
        self.solver = MultiMineralSolver(minerals=minerals)

    def calculate(
        self,
        well: "Well",
        vcl: Optional[np.ndarray] = None
    ) -> LithologyResult:
        """
        Calculate lithology fractions.

        Parameters
        ----------
        well : Well
            Well with required curves (RHOB, NPHI, PEF, VCL)
        vcl : np.ndarray, optional
            Clay volume (uses VCL curve if not provided)

        Returns
        -------
        LithologyResult
            Lithology calculation results
        """
        n_samples = well.n_samples

        # Get required curves
        rhob = well.get("RHOB")
        nphi = well.get("NPHI")
        pef = well.get("PEF")

        if rhob is None or nphi is None:
            raise ValueError("RHOB and NPHI required for lithology")

        # Get VCL
        if vcl is None and self.include_clay:
            vcl_curve = well.get("VCL")
            if vcl_curve is not None:
                vcl = vcl_curve.data
            else:
                vcl = np.zeros(n_samples)
        elif vcl is None:
            vcl = np.zeros(n_samples)

        # Solve for mineral fractions (excluding clay)
        # Adjust inputs for clay content
        rhob_clean = rhob.data.copy()
        nphi_clean = nphi.data.copy()
        pef_clean = pef.data.copy() if pef is not None else None

        # Get clay properties
        clay_rhob = 2.45
        clay_nphi = 0.30
        clay_pef = 3.5

        # Remove clay effect
        matrix_fraction = 1 - vcl
        matrix_fraction = np.where(matrix_fraction < 0.01, 0.01, matrix_fraction)

        rhob_clean = (rhob_clean - vcl * clay_rhob) / matrix_fraction
        nphi_clean = (nphi_clean - vcl * clay_nphi) / matrix_fraction
        if pef_clean is not None:
            pef_clean = (pef_clean - vcl * clay_pef) / matrix_fraction

        # Solve for minerals
        result = self.solver.solve(
            rhob_clean, nphi_clean,
            pef_clean
        )

        # Scale by matrix fraction
        v_quartz = result.get("quartz", np.zeros(n_samples)) * matrix_fraction
        v_calcite = result.get("calcite", np.zeros(n_samples)) * matrix_fraction
        v_dolomite = result.get("dolomite", np.zeros(n_samples)) * matrix_fraction

        # Determine dominant lithology
        lithology_code = np.zeros(n_samples, dtype=int)
        lithology_name = np.empty(n_samples, dtype=object)

        for i in range(n_samples):
            # Find dominant component
            volumes = {
                "sandstone": v_quartz[i],
                "limestone": v_calcite[i],
                "dolomite": v_dolomite[i],
                "shale": vcl[i],
            }

            dominant = max(volumes, key=volumes.get)

            if vcl[i] > 0.5:
                lithology_name[i] = "shale"
                lithology_code[i] = self.LITHOLOGY_CODES["shale"]
            elif volumes[dominant] > 0.5:
                lithology_name[i] = dominant
                lithology_code[i] = self.LITHOLOGY_CODES[dominant]
            else:
                lithology_name[i] = "mixed"
                lithology_code[i] = self.LITHOLOGY_CODES["mixed"]

        return LithologyResult(
            v_quartz=v_quartz,
            v_calcite=v_calcite,
            v_dolomite=v_dolomite,
            v_clay=vcl,
            lithology_code=lithology_code,
            lithology_name=lithology_name
        )

    def add_to_well(
        self,
        well: "Well",
        result: Optional[LithologyResult] = None
    ) -> "Well":
        """
        Add lithology curves to well.

        Parameters
        ----------
        well : Well
            Well object
        result : LithologyResult, optional
            Pre-computed result

        Returns
        -------
        Well
            Well with lithology curves
        """
        if result is None:
            result = self.calculate(well)

        well.add_curve("VQUAR", result.v_quartz, "v/v", "Quartz Volume")
        well.add_curve("VCALC", result.v_calcite, "v/v", "Calcite Volume")
        well.add_curve("VDOLO", result.v_dolomite, "v/v", "Dolomite Volume")
        well.add_curve("LITHO", result.lithology_code.astype(float), "", "Lithology Code")

        return well


def calculate_lithology(
    well: "Well",
    vcl: Optional[np.ndarray] = None,
    **kwargs
) -> LithologyResult:
    """
    Calculate lithology for a well.

    Convenience function for LithologyModule.

    Parameters
    ----------
    well : Well
        Well object
    vcl : np.ndarray, optional
        Clay volume
    **kwargs
        Additional arguments for LithologyModule

    Returns
    -------
    LithologyResult
        Lithology results

    Examples
    --------
    >>> from petrophys.formation.lithology import calculate_lithology
    >>> result = calculate_lithology(well)
    """
    module = LithologyModule(**kwargs)
    return module.calculate(well, vcl)
