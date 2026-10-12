"""
Mud filtrate resistivity (Rmf) calculation module.

This module provides calculation of Rmf at formation temperature
from header values or mud resistivity.
"""

from typing import TYPE_CHECKING, Optional, Union
from dataclasses import dataclass
import numpy as np

from petrophys.model.constants import DefaultParameters
from petrophys.interpretation.temperature.temp_corrections import (
    correct_resistivity_temperature,
    arps_equation,
)

if TYPE_CHECKING:
    from petrophys.model.well import Well


@dataclass
class RmfResult:
    """
    Result container for Rmf calculation.

    Attributes
    ----------
    rmf : float or np.ndarray
        Rmf at formation temperature (ohm.m)
    rmf_surface : float
        Rmf at surface/measured temperature
    temperature_measured : float
        Temperature at which Rmf was measured
    temperature_formation : float or np.ndarray
        Formation temperature
    rm : float or np.ndarray, optional
        Rm corrected to formation temperature when Rm was the input
    """

    rmf: Union[float, np.ndarray]
    rmf_surface: float
    temperature_measured: float
    temperature_formation: Union[float, np.ndarray]
    rm: Optional[Union[float, np.ndarray]] = None


class RmfModule:
    """
    Mud filtrate resistivity calculator.

    Calculates Rmf at formation temperature from:
    - Given Rmf at measured temperature (corrected using Arps)
    - Rm × conversion factor

    Parameters
    ----------
    rmf_conversion_factor : float, optional
        Factor to convert Rm to Rmf (default: 0.75 for fresh water mud)

    Notes
    -----
    For OBM (oil-based mud), Rmf is not applicable. Use Rxo = 200 ohm.m default.

    Examples
    --------
    >>> rmf_calc = RmfModule()
    >>> result = rmf_calc.calculate(
    ...     rmf_surface=0.85,
    ...     temp_measured=75,
    ...     temp_formation=150
    ... )
    """

    def __init__(self, rmf_conversion_factor: float = 0.75):
        self.rmf_conversion_factor = rmf_conversion_factor

    def calculate(
        self,
        rmf_surface: Optional[float] = None,
        rm_surface: Optional[float] = None,
        temp_measured: float = 75.0,
        temp_formation: Union[float, np.ndarray] = 150.0,
        temp_unit: str = "F"
    ) -> RmfResult:
        """
        Calculate Rmf at formation temperature.

        Parameters
        ----------
        rmf_surface : float, optional
            Rmf at surface/measured temperature
        rm_surface : float, optional
            Rm at surface (Rmf = Rm × factor)
        temp_measured : float
            Temperature at which resistivity was measured
        temp_formation : float or np.ndarray
            Formation temperature
        temp_unit : str, optional
            Temperature unit

        Returns
        -------
        RmfResult
            Rmf calculation results
        """
        rm_at_formation = None

        # Get Rmf at surface
        if rmf_surface is not None:
            rmf_at_surface = rmf_surface
        elif rm_surface is not None:
            rmf_at_surface = rm_surface * self.rmf_conversion_factor
            rm_at_formation = correct_resistivity_temperature(
                rm_surface,
                temp_measured,
                temp_formation,
                temp_unit,
            )
        else:
            raise ValueError("Either rmf_surface or rm_surface must be provided")

        # Correct to formation temperature
        rmf_at_formation = correct_resistivity_temperature(
            rmf_at_surface,
            temp_measured,
            temp_formation,
            temp_unit
        )

        return RmfResult(
            rmf=rmf_at_formation,
            rmf_surface=rmf_at_surface,
            temperature_measured=temp_measured,
            temperature_formation=temp_formation,
            rm=rm_at_formation,
        )

    def calculate_from_well(
        self,
        well: "Well",
        temperature_curve: str = "TEMP"
    ) -> RmfResult:
        """
        Calculate Rmf from well header and temperature curve.

        Parameters
        ----------
        well : Well
            Well object with header info and temperature curve
        temperature_curve : str
            Name of temperature curve; a degC or K unit label is converted
            to degF

        Returns
        -------
        RmfResult
            Rmf at formation temperature
        """
        # Get header values
        rmf_surface = well.header.rmf
        rm_surface = well.header.rm
        rm_temp = well.header.rm_temp or 75.0

        if rmf_surface is None and rm_surface is None:
            raise ValueError("No Rmf or Rm in well header")

        # Get formation temperature (degF), from the curve's own unit label
        from petrophys.interpretation.temperature.temp_profile import _formation_temperature_f

        temp_formation = _formation_temperature_f(well, [temperature_curve, "TEMP", "TMPC"])

        return self.calculate(
            rmf_surface=rmf_surface,
            rm_surface=rm_surface,
            temp_measured=rm_temp,
            temp_formation=temp_formation
        )


def calculate_rmf(
    rmf_surface: Optional[float] = None,
    rm_surface: Optional[float] = None,
    temp_measured: float = 75.0,
    temp_formation: Union[float, np.ndarray] = 150.0,
    temp_unit: str = "F",
    **kwargs
) -> Union[float, np.ndarray]:
    """
    Calculate Rmf at formation temperature.

    Convenience function for RmfModule.

    Parameters
    ----------
    rmf_surface : float, optional
        Rmf at measured temperature
    rm_surface : float, optional
        Rm at measured temperature
    temp_measured : float
        Temperature at which resistivity was measured
    temp_formation : float or np.ndarray
        Formation temperature
    temp_unit : str, optional
        Temperature unit

    Returns
    -------
    float or np.ndarray
        Rmf at formation temperature

    Examples
    --------
    >>> rmf = calculate_rmf(rmf_surface=0.85, temp_measured=75, temp_formation=150)
    """
    module = RmfModule(**kwargs)
    result = module.calculate(
        rmf_surface, rm_surface, temp_measured, temp_formation, temp_unit
    )
    return result.rmf
