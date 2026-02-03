"""
Main porosity calculation module.

This module provides the PorosityModule class for comprehensive
porosity analysis.
"""

from typing import Optional, Union
from dataclasses import dataclass
import numpy as np

from petrophys.core.constants import DefaultParameters, MatrixProperties
from petrophys.petrophysics.porosity.density_porosity import density_porosity
from petrophys.petrophysics.porosity.sonic_porosity import sonic_porosity
from petrophys.petrophysics.porosity.total_porosity import total_porosity
from petrophys.petrophysics.porosity.effective_porosity import effective_porosity


@dataclass
class PorosityResult:
    """
    Result container for porosity calculation.

    Attributes
    ----------
    phid : np.ndarray
        Density porosity
    phis : np.ndarray
        Sonic porosity (if available)
    phit : np.ndarray
        Total porosity
    phie : np.ndarray
        Effective porosity
    """

    phid: np.ndarray
    phis: Optional[np.ndarray]
    phit: np.ndarray
    phie: np.ndarray


class PorosityModule:
    """
    Comprehensive porosity calculator.

    Parameters
    ----------
    matrix : str, optional
        Matrix type (default: "sandstone")
    rhob_fluid : float, optional
        Fluid density (default: 1.0)
    phi_clay : float, optional
        Clay porosity (default: 0.15)

    Examples
    --------
    >>> poro = PorosityModule(matrix="sandstone")
    >>> result = poro.calculate(well)
    >>> well["PHIE"] = result.phie
    """

    def __init__(
        self,
        matrix: str = "sandstone",
        rhob_fluid: float = DefaultParameters.FLUID_DENSITY,
        phi_clay: float = DefaultParameters.CLAY_POROSITY
    ):
        self.matrix = matrix
        self.rhob_fluid = rhob_fluid
        self.phi_clay = phi_clay

        # Get matrix properties
        try:
            props = MatrixProperties.get(matrix)
            self.rhob_matrix = props.density
            self.dtc_matrix = props.slowness
        except KeyError:
            self.rhob_matrix = 2.65
            self.dtc_matrix = 55.5

    def calculate(
        self,
        well: "Well",
        vcl: Optional[np.ndarray] = None
    ) -> PorosityResult:
        """
        Calculate all porosity types.

        Parameters
        ----------
        well : Well
            Well with required curves
        vcl : np.ndarray, optional
            Clay volume (uses VCL curve if not provided)

        Returns
        -------
        PorosityResult
            Porosity calculation results
        """
        # Get required curves
        rhob = well.get("RHOB")
        nphi = well.get("NPHI")
        dtc = well.get("DTC")

        if rhob is None:
            raise ValueError("RHOB required for porosity calculation")

        # Get VCL
        if vcl is None:
            vcl_curve = well.get("VCL")
            vcl = vcl_curve.data if vcl_curve is not None else np.zeros(well.n_samples)

        # Density porosity
        phid = density_porosity(rhob.data, self.rhob_matrix, self.rhob_fluid)

        # Sonic porosity
        phis = None
        if dtc is not None:
            phis = sonic_porosity(
                dtc.data, self.matrix,
                dtc_fluid=DefaultParameters.FLUID_SLOWNESS
            )

        # Total porosity (N-D)
        if nphi is not None:
            phit = total_porosity(nphi.data, phid)
        else:
            phit = phid

        # Effective porosity
        phie = effective_porosity(phit, vcl, self.phi_clay)

        return PorosityResult(
            phid=phid,
            phis=phis,
            phit=phit,
            phie=phie
        )

    def add_to_well(
        self,
        well: "Well",
        result: Optional[PorosityResult] = None
    ) -> "Well":
        """Add porosity curves to well."""
        if result is None:
            result = self.calculate(well)

        well.add_curve("PHID", result.phid, "v/v", "Density Porosity")
        well.add_curve("PHIT", result.phit, "v/v", "Total Porosity")
        well.add_curve("PHIE", result.phie, "v/v", "Effective Porosity")

        if result.phis is not None:
            well.add_curve("PHIS", result.phis, "v/v", "Sonic Porosity")

        return well


def calculate_porosity(
    well: "Well",
    matrix: str = "sandstone",
    rhob_fluid: float = 1.0,
    phi_clay: float = 0.15,
    vcl: Optional[np.ndarray] = None,
    **kwargs
) -> PorosityResult:
    """
    Calculate porosity for a well.

    Convenience function for PorosityModule.

    Parameters
    ----------
    well : Well
        Well object
    matrix : str, optional
        Matrix type
    rhob_fluid : float, optional
        Fluid density
    phi_clay : float, optional
        Clay porosity
    vcl : np.ndarray, optional
        Clay volume
    **kwargs
        Additional arguments

    Returns
    -------
    PorosityResult
        Porosity results
    """
    module = PorosityModule(
        matrix=matrix,
        rhob_fluid=rhob_fluid,
        phi_clay=phi_clay
    )
    return module.calculate(well, vcl)
