"""
Main water saturation module.

This module provides the SaturationModule class for automatic
model selection based on formation properties.
"""

from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from petrophys.core.constants import DefaultParameters
from petrophys.petrophysics.saturation.archie import archie
from petrophys.petrophysics.saturation.simandoux import simandoux
from petrophys.petrophysics.saturation.indonesia import indonesia
from petrophys.petrophysics.saturation.sxo import calculate_sxo


class SwModel(Enum):
    """Water saturation model selection."""
    ARCHIE = "archie"
    SIMANDOUX = "simandoux"
    INDONESIA = "indonesia"
    DUAL_WATER = "dual_water"
    WAXMAN_SMITS = "waxman_smits"


@dataclass
class SaturationResult:
    """
    Result container for saturation calculation.

    Attributes
    ----------
    sw : np.ndarray
        Water saturation (uninvaded zone)
    sxo : np.ndarray
        Water saturation (flushed zone)
    model : str
        Model used for calculation
    """

    sw: np.ndarray
    sxo: Optional[np.ndarray]
    model: str


class SaturationModule:
    """
    Water saturation calculator with automatic model selection.

    Model selection criteria:
    - VCL < 0.2: Archie (use φe for Archie, φt for shale-compensated)
    - VCL >= 0.2:
      - Clastic, salinity >= 20 kppm: Simandoux
      - Clastic, salinity < 20 kppm: Indonesia
      - Carbonates: Dual Water

    Parameters
    ----------
    a : float, optional
        Tortuosity factor (default: 1.0)
    m : float, optional
        Cementation exponent (default: 2.0)
    n : float, optional
        Saturation exponent (default: 2.0)
    model : str, optional
        Force specific model (default: "auto")

    Examples
    --------
    >>> sw_calc = SaturationModule()
    >>> result = sw_calc.calculate(well, rw=0.05)
    """

    def __init__(
        self,
        a: float = DefaultParameters.ARCHIE_A,
        m: float = DefaultParameters.ARCHIE_M,
        n: float = DefaultParameters.ARCHIE_N,
        model: str = "auto"
    ):
        self.a = a
        self.m = m
        self.n = n
        self.model = model.lower()

    def calculate(
        self,
        well: "Well",
        rw: float,
        rcl: Optional[float] = None,
        salinity: Optional[float] = None,
        rmf: Optional[float] = None,
        vcl: Optional[np.ndarray] = None,
        phie: Optional[np.ndarray] = None
    ) -> SaturationResult:
        """
        Calculate water saturation.

        Parameters
        ----------
        well : Well
            Well object
        rw : float
            Formation water resistivity
        rcl : float, optional
            Clay resistivity (auto-estimated if not provided)
        salinity : float, optional
            Formation water salinity (ppm)
        rmf : float, optional
            Mud filtrate resistivity (for Sxo)
        vcl : np.ndarray, optional
            Clay volume
        phie : np.ndarray, optional
            Effective porosity

        Returns
        -------
        SaturationResult
            Saturation calculation results
        """
        # Get required curves
        rt = well.get("RDEP") or well.get("RT")
        rxo = well.get("RSHA") or well.get("RXO")

        if rt is None:
            raise ValueError("Deep resistivity (RDEP) required")

        # Get VCL and PHIE
        if vcl is None:
            vcl_curve = well.get("VCL")
            vcl = vcl_curve.data if vcl_curve is not None else np.zeros(well.n_samples)

        if phie is None:
            phie_curve = well.get("PHIE")
            if phie_curve is None:
                raise ValueError("Effective porosity (PHIE) required")
            phie = phie_curve.data

        # Estimate Rcl if not provided
        if rcl is None:
            rcl = self._estimate_rcl(rt.data, vcl)

        # Select model
        model = self._select_model(vcl, salinity)

        # Calculate Sw
        if model == SwModel.ARCHIE:
            sw = archie(rt.data, rw, phie, self.a, self.m, self.n)
        elif model == SwModel.SIMANDOUX:
            sw = simandoux(rt.data, rw, phie, vcl, rcl, self.a, self.m, self.n)
        elif model == SwModel.INDONESIA:
            sw = indonesia(rt.data, rw, phie, vcl, rcl, self.a, self.m, self.n)
        else:
            sw = archie(rt.data, rw, phie, self.a, self.m, self.n)

        # Calculate Sxo if Rmf available
        sxo = None
        if rxo is not None and rmf is not None:
            sxo = calculate_sxo(rxo.data, rmf, phie, self.a, self.m, self.n)

        return SaturationResult(sw=sw, sxo=sxo, model=model.value)

    def _select_model(
        self,
        vcl: np.ndarray,
        salinity: Optional[float]
    ) -> SwModel:
        """Select saturation model based on formation properties."""
        if self.model != "auto":
            return SwModel(self.model)

        mean_vcl = np.nanmean(vcl)

        if mean_vcl < 0.2:
            return SwModel.ARCHIE
        else:
            if salinity is not None and salinity >= 20000:
                return SwModel.SIMANDOUX
            else:
                return SwModel.INDONESIA

    def _estimate_rcl(self, rt: np.ndarray, vcl: np.ndarray) -> float:
        """Estimate clay resistivity from shale intervals."""
        # Find high VCL zones
        shale_mask = vcl > 0.9
        if np.any(shale_mask):
            return float(np.nanmean(rt[shale_mask]))
        return 5.0  # Default


def calculate_sw(
    well: "Well",
    rw: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    model: str = "auto",
    **kwargs
) -> np.ndarray:
    """
    Calculate water saturation.

    Convenience function for SaturationModule.

    Parameters
    ----------
    well : Well
        Well object
    rw : float
        Formation water resistivity
    a, m, n : float
        Archie parameters
    model : str
        Saturation model
    **kwargs
        Additional arguments

    Returns
    -------
    np.ndarray
        Water saturation
    """
    module = SaturationModule(a=a, m=m, n=n, model=model)
    result = module.calculate(well, rw, **kwargs)
    return result.sw
