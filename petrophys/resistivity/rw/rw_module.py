"""
Main Rw calculation module.

This module provides the RwModule class for calculating formation
water resistivity using multiple methods.
"""

from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from petrophys.resistivity.rw.sp_rw import rw_from_sp, rw_from_rxo_rt
from petrophys.resistivity.rw.salinity_rw import rw_from_salinity
from petrophys.resistivity.rw.temp_correction import correct_rw_temperature


class RwMethod(Enum):
    """Rw calculation methods."""
    GIVEN = "given"
    SP = "sp"
    SALINITY = "salinity"
    RXO_RT = "rxo_rt"


@dataclass
class RwResult:
    """
    Result container for Rw calculation.

    Attributes
    ----------
    rw : float or np.ndarray
        Rw at formation temperature
    method : str
        Method used for calculation
    temperature : float or np.ndarray
        Formation temperature
    """

    rw: Union[float, np.ndarray]
    method: str
    temperature: Union[float, np.ndarray]


class RwModule:
    """
    Formation water resistivity calculator.

    Calculates Rw using multiple methods:
    1. Given (from header/samples)
    2. From salinity equation
    3. From SP: SSP = -K × log10(Rmf/Rw)
    4. From Rxo/Rt = Rmf/Rw (clean water zone)

    Parameters
    ----------
    method : str, optional
        Preferred method (default: "auto")

    Examples
    --------
    >>> rw_calc = RwModule()
    >>> result = rw_calc.calculate(well, rmf=0.5)
    """

    def __init__(self, method: str = "auto"):
        self.method = method.lower()

    def calculate(
        self,
        well: "Well",
        rmf: Optional[float] = None,
        rw_given: Optional[float] = None,
        salinity: Optional[float] = None,
        sp_shale: Optional[float] = None,
        temperature: Optional[Union[float, np.ndarray]] = None
    ) -> RwResult:
        """
        Calculate Rw using best available method.

        Parameters
        ----------
        well : Well
            Well object
        rmf : float, optional
            Mud filtrate resistivity at formation temperature
        rw_given : float, optional
            Given Rw value
        salinity : float, optional
            Formation water salinity (ppm)
        sp_shale : float, optional
            Shale baseline SP
        temperature : float or np.ndarray, optional
            Formation temperature

        Returns
        -------
        RwResult
            Rw calculation results
        """
        # Get formation temperature
        if temperature is None:
            temp_curve = well.get("TEMP")
            if temp_curve is not None:
                temperature = temp_curve.data
            else:
                temperature = well.header.bht or 150.0

        # Method selection
        method = self.method

        if method == "given" or (method == "auto" and rw_given is not None):
            return RwResult(
                rw=rw_given,
                method="given",
                temperature=temperature
            )

        if method == "salinity" or (method == "auto" and salinity is not None):
            rw = rw_from_salinity(salinity, temperature)
            return RwResult(rw=rw, method="salinity", temperature=temperature)

        if method == "sp" or (method == "auto" and "SP" in well and rmf is not None):
            sp_curve = well.get("SP")
            if sp_curve is not None and sp_shale is not None:
                rw = rw_from_sp(
                    sp_curve.data,
                    sp_shale,
                    rmf,
                    temperature
                )
                return RwResult(rw=rw, method="sp", temperature=temperature)

        if method == "rxo_rt" or method == "auto":
            rxo = well.get("RSHA") or well.get("RXO")
            rt = well.get("RDEP") or well.get("RT")
            if rxo is not None and rt is not None and rmf is not None:
                rw = rw_from_rxo_rt(rxo.data, rt.data, rmf)
                return RwResult(rw=rw, method="rxo_rt", temperature=temperature)

        raise ValueError("Could not calculate Rw - insufficient data")


def calculate_rw(
    well: "Well" = None,
    method: str = "auto",
    rw_given: Optional[float] = None,
    salinity: Optional[float] = None,
    rmf: Optional[float] = None,
    temperature: Optional[Union[float, np.ndarray]] = None,
    **kwargs
) -> Union[float, np.ndarray]:
    """
    Calculate formation water resistivity.

    Convenience function for RwModule.

    Parameters
    ----------
    well : Well, optional
        Well object
    method : str, optional
        Calculation method
    rw_given : float, optional
        Given Rw value
    salinity : float, optional
        Formation water salinity (ppm)
    rmf : float, optional
        Mud filtrate resistivity
    temperature : float or np.ndarray, optional
        Formation temperature
    **kwargs
        Additional arguments

    Returns
    -------
    float or np.ndarray
        Formation water resistivity

    Examples
    --------
    >>> rw = calculate_rw(salinity=50000, temperature=150)
    """
    if method == "salinity" and salinity is not None:
        return rw_from_salinity(salinity, temperature or 150.0)

    if method == "given" and rw_given is not None:
        return rw_given

    if well is not None:
        module = RwModule(method=method)
        result = module.calculate(
            well, rmf=rmf, rw_given=rw_given,
            salinity=salinity, temperature=temperature, **kwargs
        )
        return result.rw

    raise ValueError("Insufficient parameters to calculate Rw")
