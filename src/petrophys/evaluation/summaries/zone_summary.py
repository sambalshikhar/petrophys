"""
Zone summary calculations.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Union
from dataclasses import dataclass, field
import warnings
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

    # Average properties (over net pay)
    avg_phie: float = np.nan
    avg_sw: float = np.nan
    avg_vcl: float = np.nan
    avg_perm: float = np.nan

    # Hydrocarbon properties
    # NOTE: hcpv here is a hydrocarbon pore THICKNESS, the sum of
    # PHIE*(1-SW)*thickness over the net-pay samples (units of depth/length,
    # per unit area), NOT a volume. For volumetric OOIP/OGIP in acre-ft/bbl/m3
    # use volumetrics.hydrocarbon_pore_volume.calculate_hcpv. The benchmark
    # reproduction keeps the index avg_phie*(1-avg_sw)*net_pay.
    hcpv: float = np.nan  # Hydrocarbon pore thickness (length, per unit area)
    # Bulk volume water (fraction): thickness-weighted mean of PHIE*SW over
    # net pay (avg_phie*avg_sw in the benchmark reproduction)
    bvw: float = np.nan

    # Statistics
    n_samples: int = 0
    n_pay_samples: int = 0

    #: Curves the porosity (avg_phie, hcpv, bvw) and permeability averages
    #: are taken from; None when the well has neither.
    phie_curve: Optional[str] = None
    perm_curve: Optional[str] = None

    @property
    def reservoir_ntg(self) -> float:
        """Net reservoir thickness / gross thickness (the net-to-gross of the net reservoir)."""
        if not self.gross_thickness or not np.isfinite(self.net_reservoir):
            return np.nan
        return self.net_reservoir / self.gross_thickness


def _written_by_workflow_step(curve) -> bool:
    """Whether a custom :class:`~petrophys.workflow.Workflow` step wrote ``curve``."""
    return "workflow_step" in (getattr(curve.metadata, "custom", None) or {})


def pay_average_curves(well: "Well") -> Dict[str, Optional[str]]:
    """The one porosity and one permeability curve the pay averages are taken from.

    Each average is taken from a single curve over the whole zone, never
    filled sample by sample from another model. The core-corrected curve
    (``PHIE_CC``, ``PERM_CC``) is used when the interpretation produced or
    used it, else the plain one (``PHIE``, ``PERM``):

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
        those of :func:`pay_average_curves`. Each average, HCPV and BVW come
        from that one curve over the net-pay samples where it is defined.

    Returns
    -------
    ZoneSummary
        Zone summary results; a net thickness (and pay/gross ``ntg``) whose
        flag curve is not on the well is NaN. ``avg_perm`` is the geometric
        mean of the positive permeability samples.
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
            ntg=0,
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
        # One porosity and one permeability curve for the whole zone: the
        # core-corrected one where the interpretation produced or used it.
        if phie_curve is None or perm_curve is None:
            chosen = pay_average_curves(well)
            phie_curve = phie_curve or chosen["PHIE"]
            perm_curve = perm_curve or chosen["PERM"]
        phie = _pay_data(well, [phie_curve] if phie_curve else [])
        sw = _pay_data(well, ["SW"])
        perm = _pay_data(well, [perm_curve] if perm_curve else [])
        avg_phie = _mean(phie[pay_mask])
        avg_sw = _mean(sw[pay_mask])
        avg_vcl = _get_average(well, "VCL", pay_mask)
        avg_perm = _geometric_mean(perm[pay_mask])
        for name, data in ((phie_curve, phie), (perm_curve, perm)):
            missing = int(np.count_nonzero(pay_mask & ~np.isfinite(data)))
            if name and n_pay_samples and missing:
                warnings.warn(
                    f"{zone_name}: {name} is undefined at {missing} of {n_pay_samples} net-pay "
                    f"samples; its average covers the other {n_pay_samples - missing}.",
                    UserWarning, stacklevel=2,
                )

        # HCPV sums PHIE*(1-SW)*thickness over pay; BVW is the thickness-weighted
        # mean of PHIE*SW over pay, so hcpv = (avg_phie - bvw)*net_pay when the
        # pay samples share one thickness.
        hc_valid = pay_mask & np.isfinite(phie) & np.isfinite(sw)
        if np.any(hc_valid):
            h = sample_thickness[hc_valid]
            hcpv = float(np.sum(phie[hc_valid] * (1.0 - sw[hc_valid]) * h))
            bvw = float(np.sum(phie[hc_valid] * sw[hc_valid] * h) / np.sum(h)) if np.sum(h) > 0 else np.nan
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
    )


def _pay_data(well: "Well", names: List[str]) -> np.ndarray:
    """Data of the named curve (NaN where absent); several names compose per sample."""
    data, _, _ = get_curve_data_depthwise(well, names)
    if data is None:
        return np.full(well.n_samples, np.nan, dtype=float)
    return np.asarray(data, dtype=float)


def _mean(data: np.ndarray) -> float:
    valid = data[np.isfinite(data)]
    return float(np.mean(valid)) if len(valid) else np.nan


def _geometric_mean(data: np.ndarray) -> float:
    valid = data[np.isfinite(data) & (data > 0)]
    return float(np.exp(np.mean(np.log(valid)))) if len(valid) else np.nan


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
