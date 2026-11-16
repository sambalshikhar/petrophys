"""
Main Rw calculation module.

This module provides the RwModule class for calculating formation
water resistivity using multiple methods.
"""

from typing import TYPE_CHECKING, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from petrophys.interpretation.electrical.rw.sp_rw import estimate_sp_lines, rw_from_sp, rw_from_rxo_rt
from petrophys.interpretation.electrical.rw.salinity_rw import rw_from_salinity
from petrophys.interpretation.electrical.rw.temp_correction import correct_rw_temperature
from petrophys._internal.analysis_helpers import get_curve_data_depthwise
from petrophys._internal.benchmark import reproducing_benchmark

if TYPE_CHECKING:
    from petrophys.model.well import Well


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
        Preferred method (default: "auto"): ``"given"`` (needs ``rw_given``),
        ``"salinity"`` (needs ``salinity``), ``"sp"``, ``"rxo_rt"``, or
        ``"auto"``, which takes the first of a given Rw, a given salinity,
        the header RW, the SP and Rxo/Rt.

    Notes
    -----
    An SP-derived Rw below the resistivity of NaCl-saturated brine
    (260,000 ppm) at the same temperature is not a water resistivity. It is
    not returned: a :class:`~petrophys.exceptions.DataQualityWarning` gives
    the value, ``"auto"`` goes on to Rxo/Rt, and ``"sp"`` returns NaN with
    the value in ``metadata["rw_rejected"]``.

    Examples
    --------
    >>> rw_calc = RwModule()
    >>> result = rw_calc.calculate(well, rmf=0.5)
    """

    def __init__(self, method: str = "auto"):
        self.method = str(method).strip().lower()
        allowed = ("auto", *(item.value for item in RwMethod))
        if self.method not in allowed:
            raise ValueError(f"method must be one of {list(allowed)}; got {method!r}")

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
            Formation temperature, degF. Defaults to the TEMP (or TMPC)
            curve, converted from its unit label, then to a profile from the
            header BHT, or without a BHT the default 30 degC/km gradient.

        Returns
        -------
        RwResult
            Rw calculation results
        """
        # Get formation temperature
        if temperature is None:
            from petrophys.interpretation.temperature.temp_profile import _formation_temperature_f

            temperature = _formation_temperature_f(well, ["TEMP", "TMPC"])

        # Method selection
        method = self.method
        header_note = {}
        if method == "auto" and rw_given is None and salinity is None:
            # "auto" takes the Rw the well's header states before estimating one.
            header_rw = _header_resistivity(well, "RW", "RWT", temperature)
            if header_rw is not None:
                rw_given, header_note = header_rw
        if rmf is None and method in ("auto", "sp", "rxo_rt"):
            header_rmf = _header_resistivity(well, "RMF", "RMFT", temperature, scalar=True)
            if header_rmf is not None:
                rmf, _ = header_rmf

        if method == "given" and rw_given is None:
            raise ValueError(
                "RwModule(method='given') needs the water resistivity: calculate(well, rw_given=...), "
                "in ohm.m at formation temperature"
            )
        if method == "salinity" and salinity is None:
            raise ValueError(
                "RwModule(method='salinity') needs the water salinity: calculate(well, salinity=...), "
                "in ppm NaCl equivalent"
            )
        if method == "given" or (method == "auto" and rw_given is not None):
            if header_note:
                return RwResult(rw=rw_given, method="header", temperature=temperature, metadata=header_note)
            return RwResult(
                rw=rw_given,
                method="given",
                temperature=temperature
            )

        if method == "salinity" or (method == "auto" and salinity is not None):
            rw = rw_from_salinity(salinity, temperature)
            return RwResult(rw=rw, method="salinity", temperature=temperature)

        sp_rejected: Optional[str] = None
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
                rw_sp = float(np.nanmedian(np.asarray(rw, dtype=float)))
                metadata = {
                    "sp_clean_mv": float(sp_clean),
                    "sp_shale_mv": float(sp_shale),
                    "ssp_mv": float(ssp),
                    "rw_temperature_f": eff_temp,
                }
                sp_rejected = _below_saturated_brine(rw_sp, eff_temp, float(ssp))
                if sp_rejected is None:
                    return RwResult(
                        rw=rw_sp,
                        method="sp_clean_water",
                        temperature=temperature,
                        n_points=int(clean_sp.size),
                        metadata=metadata,
                    )
                if method == "sp":
                    # The value is reported, not returned as a water resistivity.
                    return RwResult(
                        rw=float("nan"),
                        method="sp_clean_water",
                        temperature=temperature,
                        n_points=int(clean_sp.size),
                        metadata={**metadata, "rw_rejected": rw_sp, "rejected": sp_rejected},
                    )

        if method == "rxo_rt" or method == "auto":
            # The flushed-zone RXO first; the benchmark packages read the
            # shallow RSHA first.
            rxo_curves = ["RSHA", "RXO", "RESS"] if reproducing_benchmark() else ["RXO", "RSHA", "RESS"]
            rxo_data, _, _ = get_curve_data_depthwise(well, rxo_curves)
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

        if sp_rejected is not None:
            raise ValueError(
                f"Could not calculate Rw: {sp_rejected}, and the Rxo/Rt method found no clean water "
                "zone with RXO and RT. Pass rw_given= or salinity=."
            )
        raise ValueError(
            "Could not calculate Rw: the header has no RW, and the SP and Rxo/Rt methods need "
            "an Rmf (pass rmf=, or a header RMF). Pass rw_given= or salinity= otherwise; "
            "petrophys.interpret estimates Rmf from the mud data."
        )


#: NaCl brine saturates near 260,000 ppm; a water cannot be more conductive
#: than that brine at the same temperature.
SATURATED_NACL_PPM = 260_000.0


def _below_saturated_brine(rw: float, temperature_f: float, ssp: float) -> Optional[str]:
    """Why an SP-derived Rw is not a water resistivity, or None when it can be one.

    An Rw below that of NaCl-saturated brine at the same temperature (about
    0.017 ohm.m at 168 degF) comes from SP baselines or an Rmf that do not
    describe the water, not from a saltier water. The reason is also issued as
    a :class:`~petrophys.exceptions.DataQualityWarning`. The benchmark
    packages returned every SP value.
    """
    if reproducing_benchmark() or not (np.isfinite(rw) and np.isfinite(temperature_f)):
        return None
    saturated = float(rw_from_salinity(SATURATED_NACL_PPM, temperature_f))
    if rw >= saturated:
        return None
    import warnings

    from petrophys.exceptions import DataQualityWarning

    reason = (
        f"the SP gives Rw = {rw:.3g} ohm.m at {temperature_f:.0f} degF (SSP {ssp:.0f} mV), below the "
        f"{saturated:.3g} ohm.m of NaCl-saturated brine at that temperature, which no formation "
        "water reaches"
    )
    warnings.warn(
        f"Rw from the SP is not used: {reason}. Check the SP baselines and Rmf, or give the water "
        "resistivity or salinity.",
        DataQualityWarning,
        stacklevel=3,
    )
    return reason


def _header_resistivity(
    well: "Well", name: str, temp_name: str, temperature, scalar: bool = False
) -> Optional[tuple]:
    """A header resistivity (``RW``/``RMF``) at formation temperature, with its source.

    Its measurement temperature (``RWT``/``RMFT``) moves it to ``temperature``
    by Arps; without one it is taken as stated. ``scalar`` gives the value at
    the median formation temperature.
    """
    from petrophys.interpretation.temperature.temp_corrections import arps_equation
    from petrophys.io.survey_reader import _item
    from petrophys.model.units import temperature_to_fahrenheit

    value, _ = _item(well, name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    temp_value, temp_unit = _item(well, temp_name)
    try:
        measured_f = temperature_to_fahrenheit(float(temp_value), temp_unit or "degF")
    except (TypeError, ValueError):
        measured_f = None
    note = {"source": f"header {name}", f"{name.lower()}_header": value}
    if measured_f is None or temperature is None:
        note["measurement_temperature"] = "not stated: taken as at formation temperature"
        return value, note
    target = np.asarray(temperature, dtype=float)
    if scalar and target.ndim:
        target = np.asarray(np.nanmedian(target))
    converted = np.asarray(arps_equation(value, measured_f, target), dtype=float)
    note[f"{name.lower()}_measured_temp_f"] = measured_f
    return (float(converted) if converted.ndim == 0 else converted), note


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
        Formation temperature, degF. Without it the well's temperature is
        used (TEMP curve, else a profile from the header BHT); without a
        well either, ``method="salinity"`` computes Rw at an assumed
        150 degF, with a ``DataQualityWarning``.
    **kwargs
        Additional arguments

    Returns
    -------
    float or np.ndarray
        Formation water resistivity

    Examples
    --------
    >>> rw = calculate_rw(method="salinity", salinity=50000, temperature=150)
    """
    if method == "salinity" and salinity is not None:
        if temperature is None and well is not None and not reproducing_benchmark():
            # The well's temperature (TEMP curve, else a BHT profile).
            return RwModule(method="salinity").calculate(well, salinity=salinity).rw
        if temperature is None and not reproducing_benchmark():
            import warnings

            from petrophys.exceptions import DataQualityWarning

            warnings.warn(
                "calculate_rw(method='salinity') was given no temperature: Rw is computed at an "
                "assumed 150 degF. Pass temperature= (degF), or a well.",
                DataQualityWarning,
                stacklevel=2,
            )
        return rw_from_salinity(salinity, 150.0 if temperature is None else temperature)

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
