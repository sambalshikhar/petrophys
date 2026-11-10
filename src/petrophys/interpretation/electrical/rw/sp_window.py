"""SP-derived Rw over a chosen depth window, quoted at a stated temperature.

The static SP relation ``SSP = -K log10(Rmf / Rw)`` with
``K = 61 + 0.133 T`` (T in degF) gives Rw from the SP deflection between a
clean water-bearing sand and the shale next to it. Two practical traps make a
window function worth having:

- **Baseline drift.** The SP shale baseline wanders by tens of mV down a
  well. Baselines picked over the whole log mostly measure that drift, not
  the deflection. Pick them in a local window: a sand and the shale beside it.
- **Temperature bookkeeping.** Rmf must be moved to formation temperature
  before it is used, and the Rw that comes back is quoted at the median
  temperature of the samples used. Comparing it with another Rw quoted at a
  different temperature is a classic error, so the result is Arps-shifted to a
  reference temperature you choose.

References
----------
Arps, J.J. (1953). The effect of temperature on the density and electrical
resistivity of sodium chloride solutions. *Petroleum Transactions, AIME* 198,
Technical Note 195.

Asquith, G. and Krygowski, D. (2004). *Basic Well Log Analysis*, 2nd ed. AAPG
Methods in Exploration 16.

Schlumberger (1989). *Log Interpretation Principles/Applications*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from petrophys.interpretation.electrical.rw.rw_module import RwModule
from petrophys.interpretation.temperature.temp_corrections import arps_equation


@dataclass(repr=False)
class SpRwEstimate:
    """Rw from the SP in one depth window.

    Attributes
    ----------
    rw : float
        Rw (ohm.m) at ``reference_temp_f``.
    reference_temp_f : float
        Temperature (degF) at which ``rw`` is quoted.
    rw_window : float
        Rw (ohm.m) at the window's median temperature, as the SP method
        returns it.
    window_temp_f : float
        Median formation temperature of the window (degF).
    rmf_window : float
        Mud-filtrate resistivity (ohm.m) moved to the window temperature.
    sp_clean_mv, sp_shale_mv, ssp_mv : float
        Clean-sand line, shale line and static SP (clean - shale), in mV.
    top, bottom : float
        Depth window used (well depth units).
    n_samples : int
        Samples in the window.
    n_clean_samples : int
        Clean water-bearing candidate samples behind the clean line.
    depth_unit : str
        Depth unit of the well.
    """

    rw: float
    reference_temp_f: float
    rw_window: float
    window_temp_f: float
    rmf_window: float
    sp_clean_mv: float
    sp_shale_mv: float
    ssp_mv: float
    top: float
    bottom: float
    n_samples: int
    n_clean_samples: int
    depth_unit: str = ""

    def __repr__(self) -> str:
        unit = f" {self.depth_unit}" if self.depth_unit else ""
        return (
            f"SpRwEstimate(Rw {self.rw:.4f} ohm.m at {self.reference_temp_f:.0f} degF; "
            f"SSP {self.ssp_mv:.1f} mV (clean {self.sp_clean_mv:.1f}, shale "
            f"{self.sp_shale_mv:.1f} mV) over {self.top:.1f}-{self.bottom:.1f}{unit})"
        )


def sp_rw_in_window(
    well,
    *,
    rmf: float,
    rmf_temp_f: float,
    temperature_f: Union[float, np.ndarray],
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    reference_temp_f: Optional[float] = None,
    sp_shale_mv: Optional[float] = None,
) -> SpRwEstimate:
    """Estimate Rw from the SP in a depth window, at a chosen temperature.

    The window is cut from the well, Rmf is moved from its measurement
    temperature to the formation temperature of each sample (Arps), and
    :class:`~petrophys.interpretation.electrical.rw.RwModule` (``method="sp"``)
    picks the clean-sand and shale SP lines inside the window and inverts the
    static SP for Rw. The result is then Arps-shifted to
    ``reference_temp_f``.

    Parameters
    ----------
    well : Well
        Well with an ``SP`` curve (mV). Clean water-bearing candidates are
        the samples with VCL <= 0.35 and PHIE >= 0.10 where those curves
        exist, and among them the lowest 35 % of deep resistivity; the
        shale line from high VCL or high GR.
    rmf : float
        Mud-filtrate resistivity (ohm.m) measured at ``rmf_temp_f``.
    rmf_temp_f : float
        Temperature (degF) at which ``rmf`` was measured.
    temperature_f : float or array-like
        Formation temperature in degF: one value, or one per sample of the
        whole well (for example ``calculate_temperature_profile(...).temperature``).
    top, bottom : float, optional
        Depth window (inclusive), in well depth units. Defaults to the whole
        log, which is only safe when the SP baseline does not drift.
    reference_temp_f : float, optional
        Temperature (degF) to quote Rw at. Defaults to the window's median
        temperature.
    sp_shale_mv : float, optional
        Shale-line SP (mV) to use instead of picking it from the window.

    Returns
    -------
    SpRwEstimate
        Rw at the reference temperature plus the SP lines, Rmf and
        temperatures behind it.

    Raises
    ------
    ValueError
        If the well has no SP, the window is empty, the temperature does not
        match the well, or a resistivity or temperature is not physical.

    Notes
    -----
    A small SSP (a few mV) means formation water about as saline as the
    filtrate; baseline noise is then a large relative error, so treat the
    answer as a consistency check. Hydrocarbons and clay suppress the SP
    deflection. See Schlumberger, *Log Interpretation Principles/Applications*
    (1989), chapter 7.

    Examples
    --------
    >>> import petrophys as pp
    >>> from petrophys.interpretation.temperature import calculate_temperature_profile
    >>> well = pp.read(pp.sample("30_6-5"))
    >>> temp_f = calculate_temperature_profile(
    ...     well, bht=258.8, bht_unit="F", surface_temp=68.0, td=well.bottom).temperature
    >>> local = sp_rw_in_window(well, rmf=0.186, rmf_temp_f=75.0, temperature_f=temp_f,
    ...                         top=2800.0, bottom=2958.0, reference_temp_f=224.0)
    >>> round(local.ssp_mv, 1), round(local.rw, 3)
    (-5.2, 0.058)
    """
    if well.depth is None:
        raise ValueError("Well has no depth data")
    if well.get("SP") is None:
        raise ValueError("Well has no SP curve")
    for name, value in (("rmf", rmf), ("rmf_temp_f", rmf_temp_f)):
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(rmf) <= 0:
        raise ValueError("rmf must be positive (ohm.m)")
    if reference_temp_f is not None and not np.isfinite(float(reference_temp_f)):
        raise ValueError("reference_temp_f must be a finite number (degF)")

    depth = np.asarray(well.depth, dtype=float)
    temperature = np.asarray(temperature_f, dtype=float)
    if temperature.ndim == 0:
        temperature = np.full(depth.size, float(temperature))
    temperature = temperature.reshape(-1)
    if temperature.size != depth.size:
        raise ValueError(
            f"temperature_f has {temperature.size} values but the well has {depth.size} samples"
        )
    if top is not None and bottom is not None and float(top) > float(bottom):
        raise ValueError(f"top ({top}) must not be below bottom ({bottom})")

    window = np.isfinite(depth)
    if top is not None:
        window &= depth >= float(top)
    if bottom is not None:
        window &= depth <= float(bottom)
    if not np.any(window):
        raise ValueError("No samples in the requested depth window")
    window_temp = temperature[window]
    if not np.any(np.isfinite(window_temp)):
        raise ValueError("temperature_f has no finite values in the window")

    local = well.depth_slice(top, bottom) if (top is not None or bottom is not None) else well
    rmf_at_formation = arps_equation(float(rmf), float(rmf_temp_f), window_temp)
    rmf_window = float(np.nanmedian(rmf_at_formation))
    result = RwModule(method="sp").calculate(
        local, rmf=rmf_window, temperature=window_temp, sp_shale=sp_shale_mv
    )
    if not result.metadata or "rw_temperature_f" not in result.metadata:
        raise ValueError("The SP method returned no Rw; check the SP curve in the window")

    window_temp_f = float(result.metadata["rw_temperature_f"])
    reference = window_temp_f if reference_temp_f is None else float(reference_temp_f)
    rw_window = float(result.rw)
    rw_reference = float(arps_equation(rw_window, window_temp_f, reference))
    window_depth = depth[window]
    return SpRwEstimate(
        rw=rw_reference,
        reference_temp_f=reference,
        rw_window=rw_window,
        window_temp_f=window_temp_f,
        rmf_window=rmf_window,
        sp_clean_mv=float(result.metadata["sp_clean_mv"]),
        sp_shale_mv=float(result.metadata["sp_shale_mv"]),
        ssp_mv=float(result.metadata["ssp_mv"]),
        top=float(np.nanmin(window_depth)),
        bottom=float(np.nanmax(window_depth)),
        n_samples=int(np.count_nonzero(window)),
        n_clean_samples=int(result.n_points),
        depth_unit=str(getattr(well, "depth_unit", "") or ""),
    )


__all__ = ["SpRwEstimate", "sp_rw_in_window"]
