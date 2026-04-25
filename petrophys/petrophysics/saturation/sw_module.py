"""
Main water saturation module.

This module provides the SaturationModule class for automatic
model selection based on formation properties: VCL, lithology,
CEC availability, and salinity.

Decision flowchart:
    VCL < cutoff -> CLEAN -> Archie
    VCL >= cutoff -> SHALY:
        Carbonate -> Dual Water
        Clastic + CEC and B_COEFF available -> Waxman-Smits
        Clastic + salinity >= 20 kppm -> Simandoux
        Clastic + salinity < 20 kppm -> Indonesia

Lithology can be a single string (applied globally) or a per-depth
numpy array for interval-based lithology control.
"""

from typing import Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from petrophys.base.constants import DefaultParameters
from petrophys.petrophysics.saturation.archie import archie
from petrophys.petrophysics.saturation.north_sea import north_sea_shaly_sand
from petrophys.petrophysics.saturation.nigeria_schlumberger import nigeria_schlumberger
from petrophys.petrophysics.saturation.simandoux import simandoux
from petrophys.petrophysics.saturation.indonesia import indonesia
from petrophys.petrophysics.saturation.dual_water import dual_water
from petrophys.petrophysics.saturation.waxman_smits import waxman_smits
from petrophys.petrophysics.saturation.sxo import calculate_sxo


class SwModel(Enum):
    """Water saturation model selection."""
    ARCHIE = "archie"
    NORTH_SEA = "north_sea"
    NIGERIA_SCHLUMBERGER = "nigeria_schlumberger"
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
    sxo_method: Optional[str] = None
    sxo_reason: Optional[str] = None
    model_counts: Optional[Dict[str, int]] = None


class SaturationModule:
    """
    Water saturation calculator with automatic model selection.

    Model selection criteria (from design doc flowchart):
    - VCL < vcl_cutoff -> CLEAN -> Archie
    - VCL >= vcl_cutoff -> SHALY:
      - Carbonate -> Dual Water
      - Clastic + CEC and B_COEFF available -> Waxman-Smits
      - Clastic + salinity >= 20 kppm -> Simandoux
      - Clastic + salinity < 20 kppm -> Indonesia

    Lithology may be a single string or a per-depth numpy array.
    When auto-routing is active, the clean/shaly decision is applied
    sample-by-sample so a single well can use multiple Sw models across depth.

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
        rw: Union[float, np.ndarray],
        rcl: Optional[float] = None,
        salinity: Optional[float] = None,
        rmf: Optional[Union[float, np.ndarray]] = None,
        vcl: Optional[np.ndarray] = None,
        phie: Optional[np.ndarray] = None,
        phit: Optional[np.ndarray] = None,
        lithology: Union[str, np.ndarray] = "clastic",
        cec: Optional[float] = None,
        b_coeff: Optional[float] = None,
        rho_ma: float = 2.65,
        m_star: float = 2.0,
        rwb: Optional[float] = None,
        phi_cl: float = DefaultParameters.CLAY_POROSITY,
        rtsh: Optional[float] = None,
        vsh_exp: float = 1.6,
        shale_exp: float = 1.6,
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
            vcl_curve = well.get("VCL") or well.get("VCLAY")
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

        # Auto-routing is applied sample-by-sample so clean and shaly
        # intervals can coexist within the same well/zone.
        if self.model == "auto":
            if isinstance(lithology, np.ndarray):
                lithology_array = np.asarray(lithology, dtype=object)
            else:
                lithology_array = np.full(rt.data.shape, lithology, dtype=object)
            sw, model_name, model_counts = self._calculate_depth_routed(
                rt.data, rw, rcl, salinity, vcl, phie, phit,
                lithology_array, cec, b_coeff, rho_ma, m_star, rwb, phi_cl, rtsh, vsh_exp, shale_exp
            )
        else:
            # Single model for entire well
            model = self._select_model(vcl, salinity,
                                       lithology if isinstance(lithology, str) else "clastic",
                                       cec, b_coeff)
            sw = self._run_model(model, rt.data, rw, rcl, vcl, phie, phit,
                                 cec, b_coeff, rho_ma, m_star, rwb, phi_cl, rtsh, vsh_exp, shale_exp)
            model_name = model.value
            model_counts = {model_name: int(rt.data.shape[0])}

        # Calculate Sxo from invaded-zone inputs when available; otherwise
        # fall back to a quick-look empirical relation requested by the user.
        sxo = None
        sxo_method = None
        sxo_reason = None
        if rxo is not None and rmf is not None:
            sxo = calculate_sxo(rxo.data, rmf, phie, self.a, self.m, self.n)
            sxo_method = "archie_rxo_rmf"
            sxo_reason = "RSHA/RXO and RMF available"
        elif sw is not None:
            sw_clipped = np.clip(np.asarray(sw, dtype=float), 0.0, 1.0)
            sxo = np.power(sw_clipped, 0.2)
            sxo_method = "fifth_root_quicklook"
            if rxo is None and rmf is None:
                sxo_reason = "RSHA/RXO and RMF unavailable"
            elif rxo is None:
                sxo_reason = "RSHA/RXO unavailable"
            else:
                sxo_reason = "RMF unavailable"

        return SaturationResult(
            sw=sw,
            sxo=sxo,
            model=model_name,
            sxo_method=sxo_method,
            sxo_reason=sxo_reason,
            model_counts=model_counts,
        )

    @staticmethod
    def _segment_value(value, mask: np.ndarray):
        """Return a masked view for per-depth arrays, else pass scalars through."""
        if isinstance(value, np.ndarray):
            if value.shape == mask.shape:
                return value[mask]
            return value
        return value

    def _calculate_depth_routed(
        self,
        rt: np.ndarray, rw: Union[float, np.ndarray], rcl: float,
        salinity: Optional[float],
        vcl: np.ndarray, phie: np.ndarray, phit: np.ndarray,
        lithology: np.ndarray,
        cec, b_coeff, rho_ma, m_star, rwb, phi_cl, rtsh, vsh_exp, shale_exp
    ):
        """Calculate Sw with sample-by-sample routing across depth."""
        sw = np.full_like(rt, np.nan)
        model_per_sample = np.empty(rt.shape, dtype=object)

        for i in range(rt.shape[0]):
            lith_i = str(lithology[i]).lower() if lithology is not None else "clastic"
            model_per_sample[i] = self._select_model_scalar(
                float(vcl[i]) if np.isfinite(vcl[i]) else np.nan,
                salinity,
                lith_i,
                cec,
                b_coeff,
            ).value

        ordered_models = []
        for model_name in model_per_sample:
            if model_name not in ordered_models:
                ordered_models.append(model_name)

        for model_name in ordered_models:
            mask = model_per_sample == model_name
            if not np.any(mask):
                continue
            sw_segment = self._run_model(
                SwModel(model_name),
                rt[mask], self._segment_value(rw, mask), rcl,
                vcl[mask], phie[mask], phit[mask],
                self._segment_value(cec, mask),
                self._segment_value(b_coeff, mask),
                self._segment_value(rho_ma, mask),
                self._segment_value(m_star, mask),
                self._segment_value(rwb, mask),
                self._segment_value(phi_cl, mask),
                self._segment_value(rtsh, mask),
                self._segment_value(vsh_exp, mask),
                self._segment_value(shale_exp, mask),
            )
            sw[mask] = sw_segment

        model_name = "+".join(ordered_models) if len(ordered_models) > 1 else (
            ordered_models[0] if ordered_models else "archie"
        )
        model_counts = {
            model_name: int(np.count_nonzero(model_per_sample == model_name))
            for model_name in ordered_models
        }
        return sw, model_name, model_counts

    def _select_model_scalar(
        self,
        vcl_value: float,
        salinity: Optional[float],
        lithology: str = "clastic",
        cec: Optional[float] = None,
        b_coeff: Optional[float] = None,
    ) -> SwModel:
        """Select saturation model for a single depth sample."""
        if self.model != "auto":
            return SwModel(self.model)

        if not np.isfinite(vcl_value):
            vcl_value = 0.0

        if vcl_value < self.vcl_cutoff:
            return SwModel.ARCHIE

        lithology = lithology.lower()

        if lithology in {"carbonate", "limestone", "dolomite", "calcite"}:
            return SwModel.DUAL_WATER

        if cec is not None and b_coeff is not None:
            return SwModel.WAXMAN_SMITS

        if salinity is not None and salinity >= 20000:
            return SwModel.SIMANDOUX
        return SwModel.INDONESIA

    def _run_model(
        self, model: SwModel,
        rt, rw, rcl, vcl, phie, phit,
        cec, b_coeff, rho_ma, m_star, rwb, phi_cl, rtsh, vsh_exp, shale_exp
    ) -> np.ndarray:
        """Run a specific Sw model."""
        if model == SwModel.ARCHIE:
            return archie(rt, rw, phie, self.a, self.m, self.n)

        elif model == SwModel.NORTH_SEA:
            return north_sea_shaly_sand(
                rt, rw, phie, vcl, rtsh if rtsh is not None else rcl,
                self.a, self.m, self.n, vsh_exp
            )

        elif model == SwModel.NIGERIA_SCHLUMBERGER:
            return nigeria_schlumberger(
                rt, rw, phie, vcl, rtsh if rtsh is not None else rcl,
                self.a, self.m, self.n, shale_exp
            )

        elif model == SwModel.SIMANDOUX:
            return simandoux(rt, rw, phie, vcl, rcl, self.a, self.m, self.n)

        elif model == SwModel.INDONESIA:
            return indonesia(rt, rw, phie, vcl, rcl, self.a, self.m, self.n)

        elif model == SwModel.DUAL_WATER:
            if rwb is None:
                rwb = rw * 0.3
            return dual_water(
                phit, rt, rw, rwb, vcl, phi_cl,
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
        cec: Optional[float] = None,
        b_coeff: Optional[float] = None,
    ) -> SwModel:
        """
        Select saturation model based on formation properties.

        Decision logic (from design doc flowchart):
        - VCL < cutoff -> Archie (clean)
        - VCL >= cutoff -> Shaly:
          - Carbonate -> Dual Water
          - Clastic + CEC and B_COEFF -> Waxman-Smits
          - Clastic + salinity >= 20 kppm -> Simandoux
          - Clastic + salinity < 20 kppm -> Indonesia
        """
        if self.model != "auto":
            return SwModel(self.model)

        mean_vcl = float(np.nanmean(vcl))
        return self._select_model_scalar(mean_vcl, salinity, lithology, cec, b_coeff)

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
