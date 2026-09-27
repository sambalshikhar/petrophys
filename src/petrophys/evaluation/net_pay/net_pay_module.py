"""
Main Net Pay calculation module.
"""

from typing import TYPE_CHECKING, Optional, Union
from dataclasses import dataclass
import warnings
import numpy as np

from petrophys.model.constants import DefaultParameters
from petrophys.evaluation.net_pay.flags import generate_net_flags
from petrophys.evaluation.net_pay.cutoffs import Cutoffs
from petrophys._internal.analysis_helpers import get_curve_data_depthwise

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
    """

    def __init__(
        self,
        vcl_cutoff: Optional[float] = None,
        phie_cutoff: Optional[float] = None,
        sw_cutoff: Optional[float] = None,
        lithology: Union[str, np.ndarray] = "clastic"
    ):
        self.lithology = lithology
        self._explicit_phie_cutoff = phie_cutoff
        self._explicit_vcl_cutoff = vcl_cutoff
        self._explicit_sw_cutoff = sw_cutoff

        # For single-string lithology, resolve cutoffs immediately
        if isinstance(lithology, str):
            carbonate = lithology.lower() == "carbonate"
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

    def _carbonate_mask(self) -> np.ndarray:
        return np.array([str(v).lower() == "carbonate" for v in self.lithology])

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
    ) -> NetPayResult:
        """Calculate net pay.

        ``quality_mask`` (optional): boolean array, True where the log data are
        usable. Samples that are False (e.g. washed-out / bad-hole intervals,
        where RHOB and NPHI -- and therefore PHIE and Sw -- are unreliable) are
        excluded from net reservoir and net pay. This prevents borehole washout
        from being certified as hydrocarbon pay.
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

        flags = generate_net_flags(
            vcl, phie, sw,
            vcl_cutoff, phie_cutoff, sw_cutoff
        )

        if quality_mask is not None:
            usable = np.asarray(quality_mask, dtype=bool)
            if usable.shape == flags["net_reservoir"].shape:
                flags["net_reservoir"] = flags["net_reservoir"] & usable
                flags["net_pay"] = flags["net_pay"] & usable

        step = well.step or 0.5
        gross = well.n_samples * step

        return NetPayResult(
            net_rock=flags["net_rock"],
            net_reservoir=flags["net_reservoir"],
            net_pay=flags["net_pay"],
            net_rock_thickness=float(np.sum(flags["net_rock"]) * step),
            net_reservoir_thickness=float(np.sum(flags["net_reservoir"]) * step),
            net_pay_thickness=float(np.sum(flags["net_pay"]) * step),
            gross_thickness=gross,
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
    vcl_cutoff : float
        VCL cutoff (default: 0.3)
    phie_cutoff : float, optional
        PHIE cutoff (auto-selects by lithology if not provided)
    sw_cutoff : float
        Sw cutoff (default: 0.7)
    lithology : str or np.ndarray
        'clastic' or 'carbonate', or per-depth array of these strings
    """
    module = NetPayModule(vcl_cutoff, phie_cutoff, sw_cutoff, lithology)
    return module.calculate(well, **kwargs)
