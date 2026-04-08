"""
Main Rw calculation module.

This module provides the RwModule class for calculating formation
water resistivity using multiple methods.
"""

from typing import Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from petrophys.resistivity.rw.sp_rw import estimate_sp_lines, rw_from_sp, rw_from_rxo_rt
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
    n_points: int = 0
    metadata: dict = field(default_factory=dict)


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

    def _build_clean_water_mask(self, well: "Well") -> Optional[np.ndarray]:
        """Build a pragmatic clean-water candidate mask for Rw estimation."""
        n_samples = well.n_samples
        mask = np.ones(n_samples, dtype=bool)
        used_any = False

        vcl_curve = well.get("VCL")
        if vcl_curve is not None:
            vcl = np.asarray(vcl_curve.data, dtype=float)
            valid = np.isfinite(vcl)
            if np.any(valid):
                mask &= valid & (vcl <= 0.35)
                used_any = True

        phie_curve = well.get("PHIE")
        if phie_curve is not None:
            phie = np.asarray(phie_curve.data, dtype=float)
            valid = np.isfinite(phie)
            if np.any(valid):
                mask &= valid & (phie >= 0.10)
                used_any = True

        rt_curve = well.get("RDEP") or well.get("RT")
        if rt_curve is not None:
            rt = np.asarray(rt_curve.data, dtype=float)
            valid = np.isfinite(rt) & (rt > 0)
            candidate = mask & valid if used_any else valid
            if np.any(candidate):
                threshold = float(np.nanpercentile(rt[candidate], 35))
                mask = candidate & (rt <= threshold)
                used_any = True

        if not used_any or np.count_nonzero(mask) < 10:
            return None

        return mask

    def _build_shale_mask(self, well: "Well") -> Optional[np.ndarray]:
        """Build a shale candidate mask for SP shale-line estimation."""
        vcl_curve = well.get("VCL")
        if vcl_curve is not None:
            vcl = np.asarray(vcl_curve.data, dtype=float)
            mask = np.isfinite(vcl) & (vcl >= 0.7)
            if np.count_nonzero(mask) >= 10:
                return mask

        gr_curve = well.get("GR")
        if gr_curve is not None:
            gr = np.asarray(gr_curve.data, dtype=float)
            valid = np.isfinite(gr)
            if np.any(valid):
                threshold = float(np.nanpercentile(gr[valid], 75))
                mask = valid & (gr >= threshold)
                if np.count_nonzero(mask) >= 10:
                    return mask

        return None

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
            if rmf is None:
                raise ValueError("Rmf is required for SP-based Rw")
            sp_curve = well.get("SP")
            if sp_curve is not None:
                sp_data = np.asarray(sp_curve.data, dtype=float)
                clean_mask = self._build_clean_water_mask(well)
                shale_mask = self._build_shale_mask(well)

                clean_sp = sp_data[clean_mask] if clean_mask is not None else sp_data
                clean_sp = clean_sp[np.isfinite(clean_sp)]
                if clean_sp.size > 0:
                    sp_clean, _ = estimate_sp_lines(clean_sp)
                else:
                    sp_clean, _ = estimate_sp_lines(sp_data)

                if sp_shale is None:
                    if shale_mask is not None:
                        shale_sp = sp_data[shale_mask]
                        shale_sp = shale_sp[np.isfinite(shale_sp)]
                        if shale_sp.size > 0:
                            sp_shale = float(np.nanmedian(shale_sp))
                    if sp_shale is None:
                        _, sp_shale = estimate_sp_lines(sp_data)

                ssp = sp_clean - sp_shale
                rw = rw_from_sp(sp_clean, sp_shale, rmf, temperature)
                return RwResult(
                    rw=float(np.nanmedian(np.asarray(rw, dtype=float))),
                    method="sp_clean_water",
                    temperature=temperature,
                    n_points=int(clean_sp.size),
                    metadata={
                        "sp_clean_mv": float(sp_clean),
                        "sp_shale_mv": float(sp_shale),
                        "ssp_mv": float(ssp),
                    },
                )

        if method == "rxo_rt" or method == "auto":
            rxo = well.get("RSHA") or well.get("RXO")
            rt = well.get("RDEP") or well.get("RT")
            if method == "rxo_rt" and rmf is None:
                raise ValueError("Rmf is required for Rxo/Rt-based Rw")
            if rxo is not None and rt is not None and rmf is not None:
                rxo_data = np.asarray(rxo.data, dtype=float)
                rt_data = np.asarray(rt.data, dtype=float)
                clean_mask = self._build_clean_water_mask(well)
                if clean_mask is not None:
                    valid = clean_mask & np.isfinite(rxo_data) & np.isfinite(rt_data)
                else:
                    valid = np.isfinite(rxo_data) & np.isfinite(rt_data)
                if np.any(valid):
                    rw_vals = rw_from_rxo_rt(rxo_data[valid], rt_data[valid], rmf)
                    return RwResult(
                        rw=float(np.nanmedian(np.asarray(rw_vals, dtype=float))),
                        method="rxo_rt_clean_water",
                        temperature=temperature,
                        n_points=int(np.count_nonzero(valid)),
                    )

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
