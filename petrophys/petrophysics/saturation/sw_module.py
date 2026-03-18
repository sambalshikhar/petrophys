"""
Main water saturation module.

This module provides the SaturationModule class for automatic
model selection based on formation properties: VCL, lithology,
CEC availability, and salinity.

Decision flowchart:
    VCL < cutoff -> CLEAN -> Archie
    VCL >= cutoff -> SHALY:
        Carbonate -> Dual Water
        Clastic + CEC available -> Waxman-Smits
        Clastic + salinity >= 20 kppm -> Simandoux
        Clastic + salinity < 20 kppm -> Indonesia

Lithology can be a single string (applied globally) or a per-depth
numpy array for interval-based lithology control.
"""

from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from petrophys.base.constants import DefaultParameters
from petrophys.petrophysics.saturation.archie import archie
from petrophys.petrophysics.saturation.simandoux import simandoux
from petrophys.petrophysics.saturation.indonesia import indonesia
from petrophys.petrophysics.saturation.dual_water import dual_water
from petrophys.petrophysics.saturation.waxman_smits import waxman_smits
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
        Model used for calculation (or "mixed" if per-depth lithology)
    """

    sw: np.ndarray
    sxo: Optional[np.ndarray]
    model: str


class SaturationModule:
    """
    Water saturation calculator with automatic model selection.

    Model selection criteria (from design doc flowchart):
    - VCL < vcl_cutoff -> CLEAN -> Archie
    - VCL >= vcl_cutoff -> SHALY:
      - Carbonate -> Dual Water
      - Clastic + CEC available -> Waxman-Smits
      - Clastic + salinity >= 20 kppm -> Simandoux
      - Clastic + salinity < 20 kppm -> Indonesia

    Lithology may be a single string or a per-depth numpy array.
    When per-depth, the routing logic is applied per-interval.

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
    vcl_cutoff : float, optional
        VCL threshold for clean vs shaly (default: 0.3)

    Examples
    --------
    >>> sw_calc = SaturationModule()
    >>> result = sw_calc.calculate(well, rw=0.05, lithology='carbonate')
    """

    def __init__(
        self,
        a: float = DefaultParameters.ARCHIE_A,
        m: float = DefaultParameters.ARCHIE_M,
        n: float = DefaultParameters.ARCHIE_N,
        model: str = "auto",
        vcl_cutoff: float = 0.3
    ):
        self.a = a
        self.m = m
        self.n = n
        self.model = model.lower()
        self.vcl_cutoff = vcl_cutoff

    def calculate(
        self,
        well: "Well",
        rw: float,
        rcl: Optional[float] = None,
        salinity: Optional[float] = None,
        rmf: Optional[float] = None,
        vcl: Optional[np.ndarray] = None,
        phie: Optional[np.ndarray] = None,
        phit: Optional[np.ndarray] = None,
        lithology: Union[str, np.ndarray] = "clastic",
        cec: Optional[float] = None,
        b_coeff: Optional[float] = None,
        rho_ma: float = 2.65,
        m_star: float = 2.0,
        rwb: Optional[float] = None,
        phi_cl: float = DefaultParameters.CLAY_POROSITY
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
        phit : np.ndarray, optional
            Total porosity (required for Dual Water / Waxman-Smits)
        lithology : str or np.ndarray, optional
            'clastic' or 'carbonate' globally, or per-depth array
        cec : float, optional
            Cation exchange capacity (for Waxman-Smits, meq/100g)
        b_coeff : float, optional
            Counter-ion conductance (for Waxman-Smits)
        rho_ma : float, optional
            Matrix density (for Waxman-Smits, default: 2.65)
        m_star : float, optional
            Intrinsic cementation exponent (for Waxman-Smits, default: 2.0)
        rwb : float, optional
            Bound water resistivity (for Dual Water)
        phi_cl : float, optional
            Clay porosity (for Dual Water, default from constants)

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

        if phit is None:
            phit_curve = well.get("PHIT")
            phit = phit_curve.data if phit_curve is not None else phie

        # Estimate Rcl if not provided
        if rcl is None:
            rcl = self._estimate_rcl(rt.data, vcl)

        # Check if per-depth lithology routing is needed
        if isinstance(lithology, np.ndarray) and self.model == "auto":
            sw, model_name = self._calculate_per_depth(
                rt.data, rw, rcl, salinity, vcl, phie, phit,
                lithology, cec, b_coeff, rho_ma, m_star, rwb, phi_cl
            )
        else:
            # Single model for entire well
            model = self._select_model(vcl, salinity,
                                       lithology if isinstance(lithology, str) else "clastic",
                                       cec)
            sw = self._run_model(model, rt.data, rw, rcl, vcl, phie, phit,
                                 cec, b_coeff, rho_ma, m_star, rwb, phi_cl)
            model_name = model.value

        # Calculate Sxo if Rmf available
        sxo = None
        if rxo is not None and rmf is not None:
            sxo = calculate_sxo(rxo.data, rmf, phie, self.a, self.m, self.n)

        return SaturationResult(sw=sw, sxo=sxo, model=model_name)

    def _calculate_per_depth(
        self,
        rt: np.ndarray, rw: float, rcl: float,
        salinity: Optional[float],
        vcl: np.ndarray, phie: np.ndarray, phit: np.ndarray,
        lithology: np.ndarray,
        cec, b_coeff, rho_ma, m_star, rwb, phi_cl
    ):
        """Calculate Sw with per-depth lithology routing (vectorized by interval)."""
        sw = np.full_like(rt, np.nan)
        models_used = set()

        # Process each unique lithology
        unique_liths = np.unique(lithology)
        for lith in unique_liths:
            lith_str = str(lith).lower()
            mask = np.array([str(v).lower() == lith_str for v in lithology])

            if not np.any(mask):
                continue

            model = self._select_model(vcl[mask], salinity, lith_str, cec)
            models_used.add(model.value)

            sw_segment = self._run_model(
                model, rt[mask], rw, rcl,
                vcl[mask], phie[mask], phit[mask],
                cec, b_coeff, rho_ma, m_star, rwb, phi_cl
            )
            sw[mask] = sw_segment

        model_name = "+".join(sorted(models_used)) if len(models_used) > 1 else (
            models_used.pop() if models_used else "archie"
        )
        return sw, model_name

    def _run_model(
        self, model: SwModel,
        rt, rw, rcl, vcl, phie, phit,
        cec, b_coeff, rho_ma, m_star, rwb, phi_cl
    ) -> np.ndarray:
        """Run a specific Sw model."""
        if model == SwModel.ARCHIE:
            return archie(rt, rw, phie, self.a, self.m, self.n)

        elif model == SwModel.SIMANDOUX:
            return simandoux(rt, rw, phie, vcl, rcl, self.a, self.m, self.n)

        elif model == SwModel.INDONESIA:
            return indonesia(rt, rw, phie, vcl, rcl, self.a, self.m, self.n)

        elif model == SwModel.DUAL_WATER:
            if rwb is None:
                rwb = rw * 0.3
            return dual_water(
                phit, phie, rt, rw, rwb, vcl, phi_cl,
                self.a, self.m, self.n
            )

        elif model == SwModel.WAXMAN_SMITS:
            return waxman_smits(
                phit, rt, rw, cec, rho_ma, b_coeff,
                m_star, self.n
            )

        else:
            return archie(rt, rw, phie, self.a, self.m, self.n)

    def _select_model(
        self,
        vcl: np.ndarray,
        salinity: Optional[float],
        lithology: str = "clastic",
        cec: Optional[float] = None
    ) -> SwModel:
        """
        Select saturation model based on formation properties.

        Decision logic (from design doc flowchart):
        - VCL < cutoff -> Archie (clean)
        - VCL >= cutoff -> Shaly:
          - Carbonate -> Dual Water
          - Clastic + CEC -> Waxman-Smits
          - Clastic + salinity >= 20 kppm -> Simandoux
          - Clastic + salinity < 20 kppm -> Indonesia
        """
        if self.model != "auto":
            return SwModel(self.model)

        mean_vcl = float(np.nanmean(vcl))

        # Clean formation
        if mean_vcl < self.vcl_cutoff:
            return SwModel.ARCHIE

        # Shaly formation
        lithology = lithology.lower()

        if lithology == "carbonate":
            return SwModel.DUAL_WATER

        # Clastic
        if cec is not None:
            return SwModel.WAXMAN_SMITS

        if salinity is not None and salinity >= 20000:
            return SwModel.SIMANDOUX
        else:
            return SwModel.INDONESIA

    def _estimate_rcl(self, rt: np.ndarray, vcl: np.ndarray) -> float:
        """Estimate clay resistivity from shale intervals."""
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
        Additional arguments passed to SaturationModule.calculate()

    Returns
    -------
    np.ndarray
        Water saturation
    """
    module = SaturationModule(a=a, m=m, n=n, model=model)
    result = module.calculate(well, rw, **kwargs)
    return result.sw
