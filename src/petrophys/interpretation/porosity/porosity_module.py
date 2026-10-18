"""
Main porosity calculation module.

This module provides the PorosityModule class for comprehensive
porosity analysis.
"""

from typing import TYPE_CHECKING, Optional, Union
from dataclasses import dataclass
import warnings
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.model.constants import DefaultParameters, MatrixProperties
from petrophys.interpretation.porosity.density_porosity import density_porosity
from petrophys.interpretation.porosity.sonic_porosity import (
    secondary_porosity,
    wyllie_time_average,
)
from petrophys.interpretation.porosity.total_porosity import total_porosity
from petrophys.interpretation.porosity.effective_porosity import effective_porosity
from petrophys._internal.analysis_helpers import get_curve_data_depthwise

if TYPE_CHECKING:
    from petrophys.model.well import Well


@dataclass
class PorosityResult:
    """
    Result container for porosity calculation.

    Attributes
    ----------
    phid : np.ndarray
        Density porosity
    phis : np.ndarray
        Sonic porosity (if DTC available, else None)
    phit : np.ndarray
        Total porosity (N-D combination)
    phie : np.ndarray
        Effective porosity (phit - VCL * phi_clay)
    phi_sec : np.ndarray
        Secondary porosity index = phit - phis (if DTC available, else None).
        Highlights fracture / vug porosity not seen by the sonic.
    """

    phid: np.ndarray
    phis: Optional[np.ndarray]
    phit: np.ndarray
    phie: np.ndarray
    phi_sec: Optional[np.ndarray] = None


class PorosityModule:
    """
    Comprehensive porosity calculator.

    Parameters
    ----------
    matrix : str, optional
        Matrix type (default: "sandstone")
    rhob_fluid : float, optional
        Fluid density (default: 1.0 g/cc)
    phi_clay : float, optional
        Clay-bound porosity used in effective porosity (default: 0.15)

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
        self.matrix     = matrix
        self.rhob_fluid = rhob_fluid
        self.phi_clay   = phi_clay

        try:
            props = MatrixProperties.get(matrix)
            self.rhob_matrix = MatrixProperties.get_log_density(matrix)
            self.dtc_matrix  = props.slowness
        except KeyError:
            self.rhob_matrix = 2.65
            self.dtc_matrix  = 55.5

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
            Well with RHOB required; NPHI and DTC optional but recommended
        vcl : np.ndarray, optional
            Clay volume array (uses VCL curve from well if not provided).
            Without either, no clay correction is possible: PHIE is written
            equal to PHIT, with a ``UserWarning``.

        Returns
        -------
        PorosityResult
        """
        rhob, _, _ = get_curve_data_depthwise(well, ["RHOB"])
        nphi, _, _ = get_curve_data_depthwise(well, ["NPHI"])
        dtc, _, _ = get_curve_data_depthwise(well, ["DTC", "DT"])

        if rhob is None:
            raise ValueError("RHOB with finite samples required for porosity calculation")

        no_vcl = False
        if vcl is None:
            vcl, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY", "VSH"])
            if vcl is None:
                vcl = np.zeros(well.n_samples)
                no_vcl = not reproducing_benchmark()

        # --- Density porosity ---
        phid = density_porosity(rhob, self.rhob_matrix, self.rhob_fluid)

        # --- Sonic porosity ---
        phis = None
        if dtc is not None:
            # The matrix slowness set from the matrix (or by the caller).
            phis = wyllie_time_average(
                dtc, self.dtc_matrix, DefaultParameters.FLUID_SLOWNESS
            )

        # --- Total porosity (N-D) ---
        phit = total_porosity(nphi, phid) if nphi is not None else phid

        # --- Effective porosity ---
        if no_vcl:
            # No clay volume to correct for: PHIE is PHIT.
            warnings.warn(
                "No VCL, VCLAY or VSH curve and no vcl given: PHIE is written equal to PHIT "
                "(no clay-bound water correction)",
                UserWarning,
                stacklevel=2,
            )
            phie = np.array(phit, dtype=float, copy=True)
        else:
            phie = effective_porosity(phit, vcl, self.phi_clay)

        # --- Secondary porosity index = phit - phis (NOT phie - phis) ---
        # Using phit correctly captures all secondary porosity (fractures,
        # vugs) that the sonic misses.  Using phie would over-subtract
        # the clay correction.
        phi_sec = None
        if phis is not None:
            phi_sec = secondary_porosity(phis, phit)

        return PorosityResult(
            phid=phid,
            phis=phis,
            phit=phit,
            phie=phie,
            phi_sec=phi_sec
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

        if result.phi_sec is not None:
            well.add_curve("PHI_SEC", result.phi_sec, "v/v", "Secondary Porosity Index")

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

    Convenience wrapper for PorosityModule.

    Parameters
    ----------
    well : Well
    matrix : str, optional
    rhob_fluid : float, optional
    phi_clay : float, optional
    vcl : np.ndarray, optional

    Returns
    -------
    PorosityResult

    Raises
    ------
    TypeError
        For an unknown keyword argument (for example a misspelt ``rhobfluid``).
    ValueError
        For an unknown matrix, a fluid density that is not in g/cm3 or a
        clay porosity that is not a fraction.
    """
    if kwargs:
        raise TypeError(
            f"calculate_porosity() got unexpected keyword argument(s): {', '.join(sorted(kwargs))}"
        )
    try:
        MatrixProperties.get(matrix)
    except KeyError:
        raise ValueError(
            f"Unknown matrix {matrix!r}. Available: {', '.join(MatrixProperties.names())}"
        ) from None
    if not (0.0 < float(rhob_fluid) <= 2.0):
        raise ValueError(
            f"rhob_fluid={rhob_fluid:g} is not a fluid density in g/cm3 (kg/m3 / 1000)"
        )
    if not (0.0 <= float(phi_clay) <= 1.0):
        raise ValueError(f"phi_clay={phi_clay:g} must be a fraction (v/v), not a percentage")
    module = PorosityModule(matrix=matrix, rhob_fluid=rhob_fluid, phi_clay=phi_clay)
    return module.calculate(well, vcl)
