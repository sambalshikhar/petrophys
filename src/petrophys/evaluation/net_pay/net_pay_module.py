"""
Main Net Pay calculation module.
"""

from typing import TYPE_CHECKING, Optional, Union
from dataclasses import dataclass
import warnings
import numpy as np

from petrophys.model.constants import DefaultParameters
from petrophys.evaluation.net_pay.cutoffs import cutoff_family, cutoff_fraction
from petrophys.evaluation.net_pay.flags import net_flags
from petrophys._internal.analysis_helpers import cell_thickness, get_curve_data_depthwise
from petrophys._internal.benchmark import reproducing_benchmark

if TYPE_CHECKING:
    from petrophys.model.well import Well


@dataclass
class NetPayResult:
    """Net pay calculation results."""
    net_rock: np.ndarray
    net_reservoir: np.ndarray
    net_pay: np.ndarray
    net_rock_thickness: float
    net_reservoir_thickness: float
    net_pay_thickness: float
    gross_thickness: float


class NetPayModule:
    """
    Net pay calculator with lithology-aware cutoffs.

    Unset cutoffs resolve to the generic lithology standards
    (``STANDARD_CUTOFFS`` in :mod:`petrophys.evaluation.net_pay.cutoffs`)::

        Net Rock:      VCL < 0.40 (clastics) / < 0.30 (carbonates)
        Net Reservoir: PHIE > 0.08 (clastics) / > 0.045 (carbonates)
        Net Pay:       Sw < 0.60 (clastics) / < 0.70 (carbonates)

    Lithology can be a single string or a per-depth numpy array.
    When an array is provided, the VCL and PHIE cutoffs vary by depth
    with the local lithology.

    The carbonate cutoffs apply to the carbonate family of
    :func:`~petrophys.evaluation.net_pay.cutoffs.cutoff_family` (carbonate,
    limestone, chalk, dolomite, marl); every other label is clastic. The
    benchmark reproduction keeps the exact label ``"carbonate"`` alone.

    The flags come from the cutoffs alone, with no samples left out unless a
    ``quality_mask`` is given. :func:`petrophys.interpret` also takes each
    sample's lithology-family cutoffs and leaves evaporite, coal, halite
    recognised on the logs and bad-hole samples out of net reservoir and net
    pay, so on an interpreted well these flags differ from its ``NET_PAY``;
    summarize those with ``result.summary(top, bottom)``.
    """

    def __init__(
        self,
        vcl_cutoff: Optional[float] = None,
        phie_cutoff: Optional[float] = None,
        sw_cutoff: Optional[float] = None,
        lithology: Union[str, np.ndarray] = "clastic",
    ):
        self.lithology = lithology
        self._explicit_phie_cutoff = phie_cutoff
        self._explicit_vcl_cutoff = vcl_cutoff
        self._explicit_sw_cutoff = sw_cutoff

        # For single-string lithology, resolve cutoffs immediately
        if isinstance(lithology, str):
            carbonate = self._is_carbonate(lithology)
            if phie_cutoff is not None:
                self.phie_cutoff = phie_cutoff
            elif carbonate:
                self.phie_cutoff = DefaultParameters.PHIE_CUTOFF_CARBONATE
            else:
                self.phie_cutoff = DefaultParameters.PHIE_CUTOFF_CLASTIC
            if vcl_cutoff is not None:
                self.vcl_cutoff = vcl_cutoff
            elif carbonate:
                self.vcl_cutoff = DefaultParameters.VCL_CUTOFF_CARBONATE
            else:
                self.vcl_cutoff = DefaultParameters.VCL_CUTOFF_CLASTIC
            if sw_cutoff is not None:
                self.sw_cutoff = sw_cutoff
            elif carbonate:
                self.sw_cutoff = DefaultParameters.SW_CUTOFF_CARBONATE
            else:
                self.sw_cutoff = DefaultParameters.SW_CUTOFF_CLASTIC
        else:
            # Per-depth lithology: cutoffs built at calculate() time
            self.phie_cutoff = phie_cutoff
            self.vcl_cutoff = vcl_cutoff
            self.sw_cutoff = sw_cutoff

    @staticmethod
    def _is_carbonate(label) -> bool:
        if reproducing_benchmark():
            return str(label).lower() == "carbonate"
        return cutoff_family(label) == "carbonate"

    def _carbonate_mask(self) -> np.ndarray:
        # Each distinct label is classified once.
        known: dict = {}
        mask = np.zeros(len(self.lithology), dtype=bool)
        for index, label in enumerate(self.lithology):
            try:
                if label not in known:
                    known[label] = self._is_carbonate(label)
                mask[index] = known[label]
            except TypeError:  # an unhashable label
                mask[index] = self._is_carbonate(label)
        return mask

    def _build_phie_cutoff(self, n_samples: int) -> Union[float, np.ndarray]:
        """Build per-depth PHIE cutoff array from lithology."""
        if self._explicit_phie_cutoff is not None:
            return self._explicit_phie_cutoff

        if isinstance(self.lithology, str):
            return self.phie_cutoff

        # Per-depth lithology array
        cutoff = np.full(n_samples, DefaultParameters.PHIE_CUTOFF_CLASTIC)
        cutoff[self._carbonate_mask()] = DefaultParameters.PHIE_CUTOFF_CARBONATE
        return cutoff

    def _build_vcl_cutoff(self, n_samples: int) -> Union[float, np.ndarray]:
        """Build per-depth VCL cutoff array from lithology."""
        if self._explicit_vcl_cutoff is not None:
            return self._explicit_vcl_cutoff

        if isinstance(self.lithology, str):
            return self.vcl_cutoff

        cutoff = np.full(n_samples, DefaultParameters.VCL_CUTOFF_CLASTIC)
        cutoff[self._carbonate_mask()] = DefaultParameters.VCL_CUTOFF_CARBONATE
        return cutoff

    def _build_sw_cutoff(self, n_samples: int) -> Union[float, np.ndarray]:
        """Build per-depth Sw cutoff array from lithology."""
        if self._explicit_sw_cutoff is not None:
            return self._explicit_sw_cutoff

        if isinstance(self.lithology, str):
            return self.sw_cutoff

        cutoff = np.full(n_samples, DefaultParameters.SW_CUTOFF_CLASTIC)
        cutoff[self._carbonate_mask()] = DefaultParameters.SW_CUTOFF_CARBONATE
        return cutoff

    def calculate(
        self,
        well: "Well",
        vcl: Optional[np.ndarray] = None,
        phie: Optional[np.ndarray] = None,
        sw: Optional[np.ndarray] = None,
        quality_mask: Optional[np.ndarray] = None,
        rock_mask: Optional[np.ndarray] = None,
    ) -> NetPayResult:
        """Calculate net pay.

        ``quality_mask`` (optional): boolean array, True where the log data are
        usable. Samples that are False (e.g. washed-out / bad-hole intervals,
        where RHOB and NPHI -- and therefore PHIE and Sw -- are unreliable) are
        excluded from net reservoir and net pay. This prevents borehole washout
        from being certified as hydrocarbon pay.

        ``rock_mask`` (optional): boolean array, False where the rock cannot be
        reservoir at any clay volume (salt, anhydrite, gypsum). Those samples
        are left out of net rock, and so of net reservoir and net pay.
        """
        if vcl is None:
            vcl, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY", "VSH"])
            if vcl is None:
                warnings.warn(
                    "No clay-volume curve (VCL/VCLAY/VSH) found; assuming VCL=0 for "
                    "net-pay flags. Every sample will pass the VCL test, so net rock "
                    "(and thus net reservoir/pay) may be over-estimated.",
                    stacklevel=2,
                )
                vcl = np.zeros(well.n_samples)

        if phie is None:
            phie, _, _ = get_curve_data_depthwise(well, ["PHIE_CC", "PHIE", "PHIT"])
            if phie is None:
                raise ValueError("PHIE_CC, PHIE, or PHIT with finite samples required")

        if sw is None:
            sw, _, _ = get_curve_data_depthwise(well, ["SW", "SWA", "SWT-LOG"])
            if sw is None:
                raise ValueError("SW with finite samples required")

        phie_cutoff = self._build_phie_cutoff(len(phie))
        vcl_cutoff = self._build_vcl_cutoff(len(phie))
        sw_cutoff = self._build_sw_cutoff(len(phie))

        flags = net_flags(
            vcl, phie, sw,
            vcl_cutoff, phie_cutoff, sw_cutoff,
        )

        if rock_mask is not None:
            rock = np.asarray(rock_mask, dtype=bool)
            if rock.shape != flags["net_rock"].shape:
                raise ValueError("rock_mask must match the well sample shape")
            for key in ("net_rock", "net_reservoir", "net_pay"):
                flags[key] = flags[key] & rock

        if quality_mask is not None:
            quality = np.asarray(quality_mask, dtype=float)
            if quality.shape != flags["net_reservoir"].shape:
                raise ValueError("quality_mask must match the well sample shape")
            usable = np.isfinite(quality) & (quality == 1)
            flags["net_reservoir"] = flags["net_reservoir"] & usable
            flags["net_pay"] = flags["net_pay"] & usable

        # Each sample counts with its midpoint cell from the actual depths, in
        # the well's depth unit, so mixed sampling rates add up correctly.
        n_flags = len(flags["net_pay"])
        depth = well.depth
        if depth is not None and len(depth) == n_flags:
            dz = cell_thickness(depth)
        else:
            dz = np.full(n_flags, np.nan)

        return NetPayResult(
            net_rock=flags["net_rock"],
            net_reservoir=flags["net_reservoir"],
            net_pay=flags["net_pay"],
            net_rock_thickness=float(np.sum(dz[flags["net_rock"]])),
            net_reservoir_thickness=float(np.sum(dz[flags["net_reservoir"]])),
            net_pay_thickness=float(np.sum(dz[flags["net_pay"]])),
            gross_thickness=float(np.sum(dz)),
        )

    def add_to_well(self, well: "Well", result: Optional[NetPayResult] = None) -> "Well":
        """Add net pay flags to well."""
        if result is None:
            result = self.calculate(well)

        well.add_curve("NET_ROCK", result.net_rock.astype(float), "", "Net Rock Flag")
        well.add_curve("NET_RES", result.net_reservoir.astype(float), "", "Net Reservoir Flag")
        well.add_curve("NET_PAY", result.net_pay.astype(float), "", "Net Pay Flag")

        return well


def calculate_net_pay(
    well: "Well",
    vcl_cutoff: Optional[float] = None,
    phie_cutoff: Optional[float] = None,
    sw_cutoff: Optional[float] = None,
    lithology: Union[str, np.ndarray] = "clastic",
    **kwargs
) -> NetPayResult:
    """
    Calculate net pay for a well.

    Parameters
    ----------
    well : Well
        Well object
    vcl_cutoff : float, optional
        Net rock is ``VCL < vcl_cutoff``. Default: the standard for
        ``lithology``, 0.40 clastic or 0.30 carbonate.
    phie_cutoff : float, optional
        Net reservoir adds ``PHIE > phie_cutoff``. Default: 0.08 clastic or
        0.045 carbonate.
    sw_cutoff : float, optional
        Net pay adds ``Sw < sw_cutoff``. Default: 0.60 clastic or 0.70
        carbonate.

        The three cutoffs are fractions (v/v); a value above 1 and up to
        100 is read as percent (``sw_cutoff=60`` is 0.60), and one that is
        negative, above 100 or not finite raises ValueError.
    lithology : str or np.ndarray
        'clastic' or 'carbonate', or per-depth array of lithology labels.
        The carbonate family (carbonate, limestone, chalk, dolomite, marl;
        :func:`~petrophys.evaluation.net_pay.cutoffs.cutoff_family`) selects
        the carbonate standards, every other label the clastic ones.

    Notes
    -----
    The flags are recomputed with one cutoff set and no exclusions; they are
    not an interpretation's ``NET_PAY``, which also leaves evaporite, coal,
    log-identified halite and bad-hole samples out (see
    :class:`NetPayModule`). Summarize those flags with
    ``result.summary(top, bottom)``.
    """
    given = {"vcl_cutoff": vcl_cutoff, "phie_cutoff": phie_cutoff, "sw_cutoff": sw_cutoff}
    cutoffs = {
        name: cutoff_fraction(value, name) if value is not None and np.ndim(value) == 0 else value
        for name, value in given.items()
    }
    module = NetPayModule(lithology=lithology, **cutoffs)
    return module.calculate(well, **kwargs)
