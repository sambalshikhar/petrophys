"""
Zone summary calculations.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Union
from dataclasses import dataclass
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np

from petrophys._internal.analysis_helpers import (
    cell_thickness,
    get_curve_data_depthwise,
    well_depth_array,
)
from petrophys._internal.benchmark import reproducing_benchmark

if TYPE_CHECKING:
    from petrophys.model.well import Well


@dataclass
class ZoneSummary:
    """Summary results for a zone or interval."""

    zone_name: str
    top_depth: float
    bottom_depth: float
    gross_thickness: float
    net_rock: float
    net_reservoir: float
    net_pay: float
    ntg: float  # Pay/gross: net pay thickness / gross thickness

    # Average properties (over net pay): porosity and clay volume weighted
    # by sample thickness, water saturation by pore volume (porosity times
    # thickness), permeability a thickness-weighted geometric mean. The
    # benchmark reproduction keeps plain sample means.
    avg_phie: float = np.nan
    avg_sw: float = np.nan
    avg_vcl: float = np.nan
    avg_perm: float = np.nan

    # Hydrocarbon properties
    # NOTE: hcpv here is a hydrocarbon pore THICKNESS, the sum of
    # PHIE*(1-SW)*thickness over the net-pay samples (units of depth/length,
    # per unit area), NOT a volume. For volumetric OOIP/OGIP in acre-ft/bbl/m3
    # use volumetrics.hydrocarbon_pore_volume.calculate_hcpv. It equals
    # avg_phie*(1-avg_sw)*net_pay where porosity and SW are defined at every
    # pay sample.
    hcpv: float = np.nan  # Hydrocarbon pore thickness (length, per unit area)
    # Bulk volume water (fraction): thickness-weighted mean of PHIE*SW over
    # net pay (avg_phie*avg_sw in the benchmark reproduction)
    bvw: float = np.nan

    # Statistics
    n_samples: int = 0
    n_pay_samples: int = 0

    #: Curves the porosity (avg_phie, hcpv, bvw) and permeability averages
    #: are taken from (see pay_average_data); None when the well has neither.
    phie_curve: Optional[str] = None
    perm_curve: Optional[str] = None
    #: Net-pay samples whose porosity and permeability were read from the
    #: plain curve (PHIE, PERM) because the core-corrected one is undefined
    #: there; 0 when the average rests on one curve.
    phie_plain_samples: int = 0
    perm_plain_samples: int = 0

    @property
    def reservoir_ntg(self) -> float:
        """Net reservoir thickness / gross thickness (the net-to-gross of the net reservoir)."""
        if not self.gross_thickness or not np.isfinite(self.net_reservoir):
            return np.nan
        return self.net_reservoir / self.gross_thickness


#: Undefined pay-average samples collected over the zones of one summary.
_PAY_GAPS: "ContextVar[Optional[list]]" = ContextVar("_pay_gaps", default=None)


@contextmanager
def _one_pay_gap_warning():
    """Warn once, for all the zones of a summary, where a pay-average curve is undefined."""
    if _PAY_GAPS.get() is not None:
        yield
        return
    gaps: list = []
    token = _PAY_GAPS.set(gaps)
    try:
        yield
    finally:
        _PAY_GAPS.reset(token)
        by_curve: Dict[str, List[str]] = {}
        for name, zone, missing, total in gaps:
            by_curve.setdefault(name, []).append(f"{zone} {missing} of {total}")
        for name, zones in by_curve.items():
            warnings.warn(
                f"{name} is undefined at some net-pay samples ({'; '.join(zones)}); "
                "its average covers the other pay samples.",
                UserWarning, stacklevel=4,
            )


def _written_by_workflow_step(curve) -> bool:
    """Whether a custom :class:`~petrophys.workflow.Workflow` step wrote ``curve``."""
    return "workflow_step" in (getattr(curve.metadata, "custom", None) or {})


def pay_average_curves(well: "Well") -> Dict[str, Optional[str]]:
    """The porosity and the permeability curve the pay averages are taken from.

    The core-corrected curve (``PHIE_CC``, ``PERM_CC``) is used when the
    interpretation produced or used it, else the plain one (``PHIE``,
    ``PERM``). A core-corrected curve covers the depths its core model
    applies to; :func:`pay_average_data` gives the values per sample.

    - :func:`petrophys.interpret` reads ``PHIE_CC``/``PERM_CC`` wherever they
      are on the well (its own core model, or input curves used as they
      stand), so they are used;
    - a custom :class:`~petrophys.workflow.Workflow` run reads the curves its
      steps name; a core-corrected curve that no step wrote is an input the
      run did not compute, so the run's plain curve is used (none when the
      run has no such step).

    Returns
    -------
    dict
        ``{"PHIE": name or None, "PERM": name or None}``.
    """
    custom_run = any(
        well.get(name) is not None and _written_by_workflow_step(well.get(name))
        for name in ("SW", "NET_PAY")
    )
    chosen: Dict[str, Optional[str]] = {}
    for plain, corrected in (("PHIE", "PHIE_CC"), ("PERM", "PERM_CC")):
        name = None
        cc = well.get(corrected)
        if cc is not None and np.any(np.isfinite(np.asarray(cc.data, dtype=float))):
            if not custom_run or _written_by_workflow_step(cc):
                name = corrected
        if name is None and well.get(plain) is not None:
            name = plain
        chosen[plain] = name
    return chosen


_PLAIN_CURVE = {"PHIE_CC": "PHIE", "PERM_CC": "PERM"}


def pay_average_data(
    well: "Well", phie_curve: Optional[str] = None, perm_curve: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Porosity and permeability of the pay averages at every sample.

    The curves of :func:`pay_average_curves` unless named. Where the curve is
    the core-corrected one (``PHIE_CC``, ``PERM_CC``) and it is undefined at
    a sample, the plain curve (``PHIE``, ``PERM``) is read there: the same
    per-sample choice the saturation, net-pay and flow-unit calculations of
    :func:`petrophys.interpret` make, so every net-pay sample carries the
    porosity its water saturation was computed with.

    Returns
    -------
    dict
        ``{"PHIE": array, "PERM": array}``, v/v and mD, NaN where no curve
        has a value.
    """
    if phie_curve is None or perm_curve is None:
        chosen = pay_average_curves(well)
        phie_curve = phie_curve or chosen["PHIE"]
        perm_curve = perm_curve or chosen["PERM"]
    data = {}
    for key, name in (("PHIE", phie_curve), ("PERM", perm_curve)):
        names = [name, _PLAIN_CURVE[name]] if name in _PLAIN_CURVE else [name] if name else []
        data[key] = _pay_data(well, names)
    return data


def _weighted_mean(data: np.ndarray, weight: np.ndarray) -> float:
    """Mean of ``data`` weighted by ``weight`` over the samples where both are finite."""
    valid = np.isfinite(data) & np.isfinite(weight) & (weight > 0)
    total = float(np.sum(weight[valid]))
    return float(np.sum(data[valid] * weight[valid]) / total) if total > 0 else np.nan


def calculate_zone_summary(
    well: "Well",
    top_depth: float,
    bottom_depth: float,
    zone_name: str = "Zone",
    net_pay_curve: str = "NET_PAY",
    net_reservoir_curve: str = "NET_RES",
    net_rock_curve: str = "NET_ROCK",
    include_bottom: bool = True,
    *,
    phie_curve: Optional[str] = None,
    perm_curve: Optional[str] = None,
) -> ZoneSummary:
    """
    Calculate summary for a depth interval.

    Parameters
    ----------
    well : Well
        Well object with computed curves
    top_depth : float
        Top depth of zone
    bottom_depth : float
        Bottom depth of zone
    zone_name : str
        Name of the zone
    net_pay_curve : str
        Name of net pay flag curve
    net_reservoir_curve : str
        Name of net reservoir flag curve
    net_rock_curve : str
        Name of net rock flag curve
    include_bottom : bool
        Count a sample at ``bottom_depth``
    phie_curve, perm_curve : str, optional
        Porosity and permeability curves of the pay averages; by default
        those of :func:`pay_average_curves`, read per sample as
        :func:`pay_average_data` describes.

    Returns
    -------
    ZoneSummary
        Zone summary results, thicknesses in the well's depth unit; a net
        thickness (and pay/gross ``ntg``) whose flag curve is not on the
        well is NaN. Over the net-pay samples: ``avg_phie`` and ``avg_vcl``
        are weighted by sample thickness, ``avg_sw`` by pore volume
        (porosity times thickness), and ``avg_perm`` is the
        thickness-weighted geometric mean of the positive permeability
        samples. ``hcpv`` is the hydrocarbon pore thickness, the sum of
        porosity * (1 - SW) * thickness over net pay, in the well's depth
        unit; ``bvw`` is the thickness-weighted mean of porosity * SW.
        A window without samples gives NaN averages and zero thicknesses.
    """
    depth = well_depth_array(well)
    # Get depth mask
    mask = (depth >= top_depth) & (depth <= bottom_depth)
    if not include_bottom:
        mask &= depth < bottom_depth
    n_samples = np.sum(mask)

    if n_samples == 0:
        return ZoneSummary(
            zone_name=zone_name,
            top_depth=top_depth,
            bottom_depth=bottom_depth,
            gross_thickness=0,
            net_rock=0,
            net_reservoir=0,
            net_pay=0,
            ntg=0 if reproducing_benchmark() else np.nan,
            n_samples=0,
            n_pay_samples=0
        )

    sample_thickness = cell_thickness(depth)
    gross_thickness = float(np.sum(sample_thickness[mask]))
    if gross_thickness <= 0:
        gross_thickness = abs(bottom_depth - top_depth)

    # Get flags
    net_pay_c = well.get(net_pay_curve)
    net_res_c = well.get(net_reservoir_curve)
    net_rock_c = well.get(net_rock_curve)

    net_pay_flag = net_pay_c.data[mask] if net_pay_c is not None else np.zeros(n_samples)
    net_res_flag = net_res_c.data[mask] if net_res_c is not None else np.zeros(n_samples)
    net_rock_flag = net_rock_c.data[mask] if net_rock_c is not None else np.zeros(n_samples)

    zone_thickness = sample_thickness[mask]
    net_rock_thick = float(np.sum(zone_thickness[net_rock_flag > 0.5]))
    net_res_thick = float(np.sum(zone_thickness[net_res_flag > 0.5]))
    net_pay_thick = float(np.sum(zone_thickness[net_pay_flag > 0.5]))

    if not reproducing_benchmark():
        # A missing flag curve is unknown net thickness, not zero.
        net_rock_thick = net_rock_thick if net_rock_c is not None else np.nan
        net_res_thick = net_res_thick if net_res_c is not None else np.nan
        net_pay_thick = net_pay_thick if net_pay_c is not None else np.nan

    ntg = net_pay_thick / gross_thickness if gross_thickness > 0 else 0

    # Calculate averages over net pay
    pay_mask = mask & (net_pay_c.data > 0.5) if net_pay_c is not None else np.zeros_like(mask, dtype=bool)
    n_pay_samples = int(np.sum(pay_mask))

    if reproducing_benchmark():
        avg_phie = _get_average(well, "PHIE", pay_mask)
        avg_sw = _get_average(well, "SW", pay_mask)
        avg_vcl = _get_average(well, "VCL", pay_mask)
        avg_perm = _get_geometric_average(well, "PERM", pay_mask)

        # Calculate HCPV and BVW
        hcpv = avg_phie * (1 - avg_sw) * net_pay_thick if not np.isnan(avg_phie) else np.nan
        bvw = avg_phie * avg_sw if not np.isnan(avg_phie) else np.nan
    else:
        # The core-corrected porosity and permeability where the
        # interpretation produced or used them, the plain curve at the
        # samples they do not cover (pay_average_data).
        if phie_curve is None or perm_curve is None:
            chosen = pay_average_curves(well)
            phie_curve = phie_curve or chosen["PHIE"]
            perm_curve = perm_curve or chosen["PERM"]
        pay_values = pay_average_data(well, phie_curve, perm_curve)
        phie, perm = pay_values["PHIE"], pay_values["PERM"]
        plain_samples = {}
        for key, name in (("PHIE", phie_curve), ("PERM", perm_curve)):
            corrected = _pay_data(well, [name]) if name in _PLAIN_CURVE else pay_values[key]
            plain_samples[key] = int(np.count_nonzero(
                pay_mask & np.isfinite(pay_values[key]) & ~np.isfinite(corrected)))
        sw = _pay_data(well, ["SW"])
        h = np.where(pay_mask, sample_thickness, np.nan)
        # Porosity and clay volume by thickness, Sw by pore volume, so that
        # avg_phie * (1 - avg_sw) * net pay is the hydrocarbon pore thickness.
        avg_phie = _weighted_mean(phie, h)
        avg_sw = _weighted_mean(sw, phie * h)
        if np.isnan(avg_sw):
            avg_sw = _weighted_mean(sw, h)
        avg_vcl = _weighted_mean(_pay_data(well, ["VCL"]), h)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_perm = np.where(perm > 0, np.log(perm), np.nan)
        avg_perm = float(np.exp(_weighted_mean(log_perm, h)))
        for name, data in ((phie_curve, phie), (perm_curve, perm)):
            name = f"{name}/{_PLAIN_CURVE[name]}" if name in _PLAIN_CURVE else name
            missing = int(np.count_nonzero(pay_mask & ~np.isfinite(data)))
            if name and n_pay_samples and missing and _PAY_GAPS.get() is not None:
                _PAY_GAPS.get().append((name, zone_name, missing, n_pay_samples))
            elif name and n_pay_samples and missing:
                warnings.warn(
                    f"{zone_name}: {name} is undefined at {missing} of {n_pay_samples} net-pay "
                    f"samples; its average covers the other {n_pay_samples - missing}.",
                    UserWarning, stacklevel=2,
                )

        # HCPV sums PHIE*(1-SW)*thickness over pay; BVW is the thickness-weighted
        # mean of PHIE*SW over pay, so hcpv = (avg_phie - bvw)*net_pay.
        hc_valid = pay_mask & np.isfinite(phie) & np.isfinite(sw)
        if np.any(hc_valid):
            h_valid = sample_thickness[hc_valid]
            hcpv = float(np.sum(phie[hc_valid] * (1.0 - sw[hc_valid]) * h_valid))
            bvw = (float(np.sum(phie[hc_valid] * sw[hc_valid] * h_valid) / np.sum(h_valid))
                   if np.sum(h_valid) > 0 else np.nan)
        else:
            hcpv = np.nan
            bvw = np.nan

    return ZoneSummary(
        zone_name=zone_name,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        gross_thickness=gross_thickness,
        net_rock=net_rock_thick,
        net_reservoir=net_res_thick,
        net_pay=net_pay_thick,
        ntg=ntg,
        avg_phie=avg_phie,
        avg_sw=avg_sw,
        avg_vcl=avg_vcl,
        avg_perm=avg_perm,
        hcpv=hcpv,
        bvw=bvw,
        n_samples=n_samples,
        n_pay_samples=n_pay_samples,
        phie_curve=None if reproducing_benchmark() else phie_curve,
        perm_curve=None if reproducing_benchmark() else perm_curve,
        phie_plain_samples=0 if reproducing_benchmark() else plain_samples["PHIE"],
        perm_plain_samples=0 if reproducing_benchmark() else plain_samples["PERM"],
    )


def _pay_data(well: "Well", names: List[str]) -> np.ndarray:
    """Data of the named curve (NaN where absent); several names compose per sample."""
    data, _, _ = get_curve_data_depthwise(well, names)
    if data is None:
        return np.full(well.n_samples, np.nan, dtype=float)
    return np.asarray(data, dtype=float)


def _get_average(well: "Well", curve_name: str, mask: np.ndarray) -> float:
    """Get average value over mask."""
    curve = well.get(curve_name)
    if curve is None:
        return np.nan

    data = curve.data[mask]
    valid = data[~np.isnan(data)]

    if len(valid) == 0:
        return np.nan

    return float(np.mean(valid))


def _get_geometric_average(well: "Well", curve_name: str, mask: np.ndarray) -> float:
    """Get geometric average (for permeability)."""
    curve = well.get(curve_name)
    if curve is None:
        return np.nan

    data = curve.data[mask]
    valid = data[(~np.isnan(data)) & (data > 0)]

    if len(valid) == 0:
        return np.nan

    return float(np.exp(np.mean(np.log(valid))))


def summarize_zones(
    well: "Well",
    zones: List[Dict[str, Union[str, float]]]
) -> List[ZoneSummary]:
    """
    Calculate summaries for multiple zones.

    Parameters
    ----------
    well : Well
        Well object
    zones : list
        List of zone definitions, each with:
        - name: str
        - top: float
        - bottom: float

    Returns
    -------
    list
        List of ZoneSummary objects
    """
    summaries = []
    tops = {float(zone["top"]) for zone in zones}

    with _one_pay_gap_warning():
        for zone in zones:
            summary = calculate_zone_summary(
                well,
                top_depth=zone["top"],
                bottom_depth=zone["bottom"],
                zone_name=zone.get("name", "Zone"),
                # A boundary sample shared with the next zone is counted once, in that zone.
                include_bottom=float(zone["bottom"]) not in tops,
            )
            summaries.append(summary)

    return summaries
