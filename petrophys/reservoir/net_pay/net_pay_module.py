"""
Main Net Pay calculation module.
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np

from petrophys.core.constants import DefaultParameters
from petrophys.reservoir.net_pay.flags import generate_net_flags
from petrophys.reservoir.net_pay.cutoffs import Cutoffs


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

    Default cutoffs from design doc:
        Net Rock:      VCL <= 0.3
        Net Reservoir: PHIE >= 0.08 (clastics) / >= 0.045 (carbonates)
        Net Pay:       Sw <= 0.7
    """

    def __init__(
        self,
        vcl_cutoff: float = DefaultParameters.VCL_CUTOFF,
        phie_cutoff: Optional[float] = None,
        sw_cutoff: float = DefaultParameters.SW_CUTOFF,
        lithology: str = "clastic"
    ):
        self.vcl_cutoff = vcl_cutoff
        self.sw_cutoff = sw_cutoff
        self.lithology = lithology.lower()

        if phie_cutoff is not None:
            self.phie_cutoff = phie_cutoff
        elif self.lithology == "carbonate":
            self.phie_cutoff = DefaultParameters.PHIE_CUTOFF_CARBONATE
        else:
            self.phie_cutoff = DefaultParameters.PHIE_CUTOFF_CLASTIC

    def calculate(
        self,
        well: "Well",
        vcl: Optional[np.ndarray] = None,
        phie: Optional[np.ndarray] = None,
        sw: Optional[np.ndarray] = None
    ) -> NetPayResult:
        """Calculate net pay."""
        if vcl is None:
            vcl_curve = well.get("VCL")
            vcl = vcl_curve.data if vcl_curve is not None else np.zeros(well.n_samples)

        if phie is None:
            phie_curve = well.get("PHIE")
            if phie_curve is None:
                raise ValueError("PHIE required")
            phie = phie_curve.data

        if sw is None:
            sw_curve = well.get("SW")
            if sw_curve is None:
                raise ValueError("SW required")
            sw = sw_curve.data

        flags = generate_net_flags(
            vcl, phie, sw,
            self.vcl_cutoff, self.phie_cutoff, self.sw_cutoff
        )

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
    vcl_cutoff: float = 0.3,
    phie_cutoff: Optional[float] = None,
    sw_cutoff: float = 0.7,
    lithology: str = "clastic",
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
    lithology : str
        'clastic' or 'carbonate' (default: 'clastic')
    """
    module = NetPayModule(vcl_cutoff, phie_cutoff, sw_cutoff, lithology)
    return module.calculate(well, **kwargs)
