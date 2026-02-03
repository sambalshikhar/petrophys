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
    """Net pay calculator."""

    def __init__(
        self,
        vcl_cutoff: float = DefaultParameters.VCL_CUTOFF,
        phie_cutoff: float = DefaultParameters.PHIE_CUTOFF_CLASTIC,
        sw_cutoff: float = DefaultParameters.SW_CUTOFF
    ):
        self.vcl_cutoff = vcl_cutoff
        self.phie_cutoff = phie_cutoff
        self.sw_cutoff = sw_cutoff

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
    phie_cutoff: float = 0.08,
    sw_cutoff: float = 0.7,
    **kwargs
) -> NetPayResult:
    """Calculate net pay for a well."""
    module = NetPayModule(vcl_cutoff, phie_cutoff, sw_cutoff)
    return module.calculate(well, **kwargs)
