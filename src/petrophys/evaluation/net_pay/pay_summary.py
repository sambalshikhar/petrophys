"""Net-pay and hydrocarbon-pore-thickness summary for one depth window.

:func:`summarize_pay` applies the three net-pay cutoffs to clay volume,
effective porosity and water saturation, and integrates the flags over depth.
It is the building block of the sensitivity tools in
:mod:`petrophys.evaluation.sensitivity`, and of any well-by-well screening
table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from petrophys._internal.advice import Advice
from petrophys._internal.analysis_helpers import cell_thickness, get_curve_data_depthwise
from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.evaluation.net_pay.cutoffs import cutoff_fraction, standard_cutoffs
from petrophys.evaluation.net_pay.flags import net_flags

VCL_CURVES = ("VCL", "VCLAY", "VSH")
PHIE_CURVES = ("PHIE_CC", "PHIE", "PHIT")
SW_CURVES = ("SW", "SWA", "SWT-LOG")


@dataclass(repr=False)
class PaySummary:
    """Net-pay summary of one depth window.

    Thicknesses are in the well depth unit (``depth_unit``); porosity and
    saturation averages are fractions (v/v). Over the net-pay samples,
    porosity is weighted by thickness and water saturation by pore volume
    (porosity times thickness), so ``avg_phie_pay * (1 - avg_sw_pay) *
    net_pay_thickness`` is ``hcpt``.

    Attributes
    ----------
    gross_thickness : float
        Thickness of the window.
    net_rock_thickness, net_reservoir_thickness, net_pay_thickness : float
        Thickness passing the VCL cutoff; VCL and PHIE cutoffs; all three.
    ntg_reservoir : float
        Net reservoir / gross (the usual net-to-gross).
    ntg_pay : float
        Net pay / gross.
    avg_phie_pay, avg_sw_pay : float
        Average effective porosity and water saturation over net pay (NaN
        without pay).
    hcpt : float
        Hydrocarbon pore thickness, the sum of ``PHIE * (1 - Sw) * dz`` over
        net pay. Multiplied by an area it gives the hydrocarbon pore volume.
    n_samples, n_pay_samples : int
        Samples in the window and in net pay.
    vcl_cutoff, phie_cutoff, sw_cutoff : float
        Cutoffs applied (net rock: VCL < cutoff; net reservoir: also
        PHIE > cutoff; net pay: also Sw < cutoff). A sample exactly at a
        cutoff fails that test.
    depth_unit : str
        Unit of the thicknesses.
    net_reservoir, net_pay : numpy.ndarray of bool
        Per-sample net-reservoir and net-pay flags inside the window.
    """

    gross_thickness: float
    net_rock_thickness: float
    net_reservoir_thickness: float
    net_pay_thickness: float
    ntg_reservoir: float
    ntg_pay: float
    avg_phie_pay: float
    avg_sw_pay: float
    hcpt: float
    n_samples: int
    n_pay_samples: int
    vcl_cutoff: float
    phie_cutoff: float
    sw_cutoff: float
    depth_unit: str = ""
    net_reservoir: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    net_pay: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def to_dict(self) -> dict:
        """Return the scalar results as a plain dictionary (without the flags)."""
        values = asdict(self)
        values.pop("net_reservoir")
        values.pop("net_pay")
        return values

    def __repr__(self) -> str:
        unit = f" {self.depth_unit}" if self.depth_unit else ""
        text = (
            f"PaySummary(gross {self.gross_thickness:.2f}{unit}, net reservoir "
            f"{self.net_reservoir_thickness:.2f}{unit}, net pay {self.net_pay_thickness:.2f}{unit} "
            f"(NTG pay {self.ntg_pay:.3f}), HCPT {self.hcpt:.3f}{unit}"
        )
        if self.n_pay_samples:
            text += f", avg PHIE {self.avg_phie_pay:.3f}, avg Sw {self.avg_sw_pay:.3f} over pay"
        return text + ")"


def _resolve_array(well, values, names, label, n_samples):
    if values is None:
        if well is None:
            raise ValueError(f"Pass {label}= when no well is given")
        values, _, _ = get_curve_data_depthwise(well, list(names))
        if values is None:
            raise ValueError(
                f"No {label.upper()} curve ({'/'.join(names)}) with finite samples on the "
                f"well; pass {label}= explicitly"
            )
    data = getattr(values, "data", values)
    array = np.asarray(data, dtype=float).reshape(-1)
    if array.size == 1 and n_samples != 1:
        array = np.full(n_samples, float(array[0]))
    if array.size != n_samples:
        raise ValueError(
            f"{label} has {array.size} samples but the depth has {n_samples}"
        )
    return array


def resolve_windowed_inputs(well, depth, arrays, top=None, bottom=None, depth_unit=None):
    """Resolve depth and named arrays, then cut them to a window (internal helper).

    ``arrays`` maps a label to ``(values_or_None, curve_names)``. Returns
    ``(depth, {label: array}, dz, depth_unit, window_mask)`` with every array
    restricted to the window; ``dz`` holds each sample's cell thickness.
    """
    if depth is None:
        if well is None or well.depth is None:
            raise ValueError(Advice("Pass a well with depth data, or depth=", ("depth",)))
        depth = well.depth
    depth = np.asarray(getattr(depth, "data", depth), dtype=float).reshape(-1)
    if depth.size == 0:
        raise ValueError("depth is empty")
    n_samples = depth.size
    resolved = {
        label: _resolve_array(well, values, names, label, n_samples)
        for label, (values, names) in arrays.items()
    }
    if top is not None and bottom is not None and float(top) > float(bottom):
        raise ValueError(f"top ({top}) must not be below bottom ({bottom})")
    dz = cell_thickness(depth)
    window = np.isfinite(depth)
    if top is not None:
        window &= depth >= float(top)
    if bottom is not None:
        window &= depth <= float(bottom)
    if not np.any(window):
        raise ValueError("No samples in the requested depth window")
    if depth_unit is None:
        depth_unit = str(getattr(well, "depth_unit", "") or "") if well is not None else ""
    windowed = {label: values[window] for label, values in resolved.items()}
    return depth[window], windowed, dz[window], depth_unit, window


def resolve_cutoffs(vcl_cutoff, phie_cutoff, sw_cutoff, lithology):
    """Fill unset cutoffs from the standard set for ``lithology`` (internal helper).

    The cutoffs come back as fractions: a value above 1 and up to 100 is
    read as percent
    (:func:`~petrophys.evaluation.net_pay.cutoffs.cutoff_fraction`).
    """
    standard = standard_cutoffs(lithology)
    resolved = {
        "vcl_cutoff": standard["vcl_cutoff"] if vcl_cutoff is None else vcl_cutoff,
        "phie_cutoff": standard["phie_cutoff"] if phie_cutoff is None else phie_cutoff,
        "sw_cutoff": standard["sw_cutoff"] if sw_cutoff is None else sw_cutoff,
    }
    return {name: cutoff_fraction(value, name) for name, value in resolved.items()}


def pay_summary_from_arrays(vcl, phie, sw, dz, cutoffs, depth_unit="") -> PaySummary:
    """Summarize already-windowed arrays (internal helper)."""
    with np.errstate(invalid="ignore"):
        flags = net_flags(
            vcl, phie, sw,
            cutoffs["vcl_cutoff"], cutoffs["phie_cutoff"], cutoffs["sw_cutoff"],
        )
    net_rock = flags["net_rock"]
    net_reservoir = flags["net_reservoir"]
    net_pay = flags["net_pay"]
    gross = float(np.sum(dz))
    pay_dz = dz[net_pay]
    pay_thickness = float(np.sum(pay_dz))
    reservoir_thickness = float(np.sum(dz[net_reservoir]))
    if pay_thickness > 0:
        avg_phie = float(np.sum(phie[net_pay] * pay_dz) / pay_thickness)
        pore_thickness = float(np.sum(phie[net_pay] * pay_dz))
        if reproducing_benchmark() or pore_thickness <= 0:
            avg_sw = float(np.sum(sw[net_pay] * pay_dz) / pay_thickness)
        else:
            avg_sw = float(np.sum(sw[net_pay] * phie[net_pay] * pay_dz) / pore_thickness)
        hcpt = float(np.sum(phie[net_pay] * (1.0 - sw[net_pay]) * pay_dz))
    else:
        avg_phie = avg_sw = np.nan
        hcpt = 0.0
    return PaySummary(
        gross_thickness=gross,
        net_rock_thickness=float(np.sum(dz[net_rock])),
        net_reservoir_thickness=reservoir_thickness,
        net_pay_thickness=pay_thickness,
        ntg_reservoir=reservoir_thickness / gross if gross > 0 else np.nan,
        ntg_pay=pay_thickness / gross if gross > 0 else np.nan,
        avg_phie_pay=avg_phie,
        avg_sw_pay=avg_sw,
        hcpt=hcpt,
        n_samples=int(dz.size),
        n_pay_samples=int(np.count_nonzero(net_pay)),
        vcl_cutoff=cutoffs["vcl_cutoff"],
        phie_cutoff=cutoffs["phie_cutoff"],
        sw_cutoff=cutoffs["sw_cutoff"],
        depth_unit=depth_unit,
        net_reservoir=net_reservoir,
        net_pay=net_pay,
    )


def summarize_pay(
    well=None,
    *,
    vcl=None,
    phie=None,
    sw=None,
    depth=None,
    vcl_cutoff: Optional[float] = None,
    phie_cutoff: Optional[float] = None,
    sw_cutoff: Optional[float] = None,
    lithology: str = "clastic",
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    depth_unit: Optional[str] = None,
) -> PaySummary:
    """Net pay, net-to-gross, pay averages and hydrocarbon pore thickness.

    Flags follow :func:`~petrophys.evaluation.net_pay.net_flags`:
    net rock is ``VCL < vcl_cutoff``, net reservoir adds
    ``PHIE > phie_cutoff`` and net pay adds ``Sw < sw_cutoff``. A sample
    exactly at a cutoff fails that test, and samples with a missing input
    fail the test that needs it. Each sample counts
    with its own depth increment, so irregular sampling and gaps are handled
    as in :func:`~petrophys.evaluation.summaries.calculate_net_gross`.

    The flags are recomputed from the curves with one cutoff set and no
    exclusions, so they are not an interpretation's ``NET_PAY``:
    :func:`petrophys.interpret` takes each sample's lithology-family cutoffs
    and leaves evaporite, coal, halite recognised on the logs and bad-hole
    samples out. Summarize an interpretation's own flags with
    ``result.summary(top, bottom)`` or ``result.formation_summary()``.

    Parameters
    ----------
    well : Well, optional
        Source of the depth and of any curve not passed as an array. Curves
        are looked up as in :func:`~petrophys.evaluation.calculate_net_pay`:
        VCL/VCLAY/VSH, PHIE_CC/PHIE/PHIT and SW/SWA/SWT-LOG.
    vcl, phie, sw : array-like or Curve, optional
        Clay volume, effective porosity and water saturation (fractions),
        one value per depth sample. Required when ``well`` is None.
    depth : array-like, optional
        Sample depths. Defaults to ``well.depth``.
    vcl_cutoff, phie_cutoff, sw_cutoff : float, optional
        Cutoffs, fractions (v/v); a value above 1 and up to 100 is read as
        percent (``sw_cutoff=60`` is 0.60). Unset ones come from the
        standard set for ``lithology``
        (:data:`~petrophys.evaluation.STANDARD_CUTOFFS`).
    lithology : str, default "clastic"
        Lithology family for the default cutoffs.
    top, bottom : float, optional
        Depth window (inclusive), in depth units. Defaults to all samples.
    depth_unit : str, optional
        Unit label for the thicknesses. Defaults to ``well.depth_unit``.

    Returns
    -------
    PaySummary
        Thicknesses, ratios, pay averages and HCPT. ``net_reservoir`` and
        ``net_pay`` hold the per-sample flags inside the window.

    Raises
    ------
    ValueError
        If inputs are missing, lengths differ, the window holds no samples,
        or a cutoff is not a finite number between 0 and 100.

    Examples
    --------
    >>> import numpy as np
    >>> depth = np.arange(2000.0, 2010.0, 1.0)
    >>> vcl = np.array([0.1, 0.1, 0.2, 0.6, 0.6, 0.1, 0.1, 0.3, 0.1, 0.1])
    >>> phie = np.array([0.25, 0.22, 0.18, 0.05, 0.04, 0.20, 0.21, 0.12, 0.06, 0.24])
    >>> sw = np.array([0.20, 0.30, 0.45, 1.00, 1.00, 0.35, 0.80, 0.50, 0.90, 0.25])
    >>> pay = summarize_pay(vcl=vcl, phie=phie, sw=sw, depth=depth, depth_unit="m",
    ...                     vcl_cutoff=0.4, phie_cutoff=0.08, sw_cutoff=0.6)
    >>> pay.net_pay_thickness, round(pay.hcpt, 4)
    (6.0, 0.823)
    """
    _, arrays, dz, unit, _ = resolve_windowed_inputs(
        well, depth,
        {"vcl": (vcl, VCL_CURVES), "phie": (phie, PHIE_CURVES), "sw": (sw, SW_CURVES)},
        top=top, bottom=bottom, depth_unit=depth_unit,
    )
    cutoffs = resolve_cutoffs(vcl_cutoff, phie_cutoff, sw_cutoff, lithology)
    return pay_summary_from_arrays(arrays["vcl"], arrays["phie"], arrays["sw"], dz, cutoffs, unit)


__all__ = ["PaySummary", "summarize_pay"]
