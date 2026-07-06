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
from petrophys.utils.analysis_helpers import get_curve_data_depthwise


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

        vcl, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY"])
        if vcl is not None:
            valid = np.isfinite(vcl)
            if np.any(valid):
                mask &= valid & (vcl <= 0.35)
                used_any = True

        phie, _, _ = get_curve_data_depthwise(well, ["PHIE", "PHIT"])
        if phie is not None:
            valid = np.isfinite(phie)
            if np.any(valid):
                mask &= valid & (phie >= 0.10)
                used_any = True

        rt, _, _ = get_curve_data_depthwise(well, ["RDEP", "RT"])
        if rt is not None:
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
        vcl, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY"])
        if vcl is not None:
            mask = np.isfinite(vcl) & (vcl >= 0.7)
            if np.count_nonzero(mask) >= 10:
                return mask

        gr, _, _ = get_curve_data_depthwise(well, ["GR"])
        if gr is not None:
            gr = np.asarray(gr, dtype=float)
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
            temperature, _, _ = get_curve_data_depthwise(well, ["TEMP", "TMPC"])
            if temperature is not None:
                pass
            else:
                try:
                    from petrophys.environmental.temperature import calculate_temperature_profile

                    temperature = calculate_temperature_profile(well).temperature
                except Exception:
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

        sp_data, _, sp_meta = get_curve_data_depthwise(well, ["SP"])
        sp_present = bool(sp_meta.get("existing_curves"))
        if method == "sp" or (method == "auto" and sp_present and sp_data is not None and rmf is not None):
            if rmf is None:
                raise ValueError("Rmf is required for SP-based Rw")
            if sp_data is not None:
                sp_data = np.asarray(sp_data, dtype=float)
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
                # The median of per-sample Rw corresponds to the MEDIAN profile
                # temperature, not the reference/BHT temperature; record that
                # basis so downstream labels can state it (and Arps-shift it).
                temp_arr = np.asarray(temperature, dtype=float)
                eff_temp = (
                    float(np.nanmedian(temp_arr)) if temp_arr.ndim else float(temp_arr)
                )
                return RwResult(
                    rw=float(np.nanmedian(np.asarray(rw, dtype=float))),
                    method="sp_clean_water",
                    temperature=temperature,
                    n_points=int(clean_sp.size),
                    metadata={
                        "sp_clean_mv": float(sp_clean),
                        "sp_shale_mv": float(sp_shale),
                        "ssp_mv": float(ssp),
                        "rw_temperature_f": eff_temp,
                    },
                )

        if method == "rxo_rt" or method == "auto":
            rxo_data, _, _ = get_curve_data_depthwise(well, ["RSHA", "RXO", "RESS"])
            rt_data, _, _ = get_curve_data_depthwise(well, ["RDEP", "RT"])
            if method == "rxo_rt" and rmf is None:
                raise ValueError("Rmf is required for Rxo/Rt-based Rw")
            if rxo_data is not None and rt_data is not None and rmf is not None:
                rxo_data = np.asarray(rxo_data, dtype=float)
                rt_data = np.asarray(rt_data, dtype=float)
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
