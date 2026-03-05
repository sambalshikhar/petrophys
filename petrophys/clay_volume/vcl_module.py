"""
Main VCL calculator module.

This module provides the VclModule class that combines multiple
VCL methods and selects the minimum value.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import numpy as np

from petrophys.core.constants import DefaultParameters
from petrophys.clay_volume.gr_method import vcl_from_gr
from petrophys.clay_volume.sp_method import vcl_from_sp, should_use_sp
from petrophys.clay_volume.nd_method import vcl_from_nd
from petrophys.clay_volume.classification import classify_vcl


@dataclass
class VclResult:
    """
    Result container for VCL calculation.

    Attributes
    ----------
    vcl : np.ndarray
        Final VCL (minimum of all methods)
    vcl_gr : np.ndarray
        VCL from gamma ray
    vcl_sp : np.ndarray
        VCL from SP (if available)
    vcl_nd : np.ndarray
        VCL from neutron-density (if available)
    method_used : np.ndarray
        Which method provided the minimum at each depth
    """

    vcl: np.ndarray
    vcl_gr: Optional[np.ndarray] = None
    vcl_sp: Optional[np.ndarray] = None
    vcl_nd: Optional[np.ndarray] = None
    method_used: Optional[np.ndarray] = None


class VclModule:
    """
    Multi-method clay volume calculator.

    Calculates VCL using multiple methods and selects the minimum
    value at each depth point, as is standard practice.

    Parameters
    ----------
    gr_min : float, optional
        Clean sand GR value (default: 20 API)
    gr_max : float, optional
        Shale GR value (default: 100 API)
    gr_method : str, optional
        GR method (default: "linear")
    use_sp : bool, optional
        Whether to use SP method (default: True)
    use_nd : bool, optional
        Whether to use N-D method (default: True)

    Examples
    --------
    >>> vcl_calc = VclModule(gr_min=15, gr_max=120)
    >>> result = vcl_calc.calculate(well)
    >>> well["VCL"] = result.vcl
    """

    def __init__(
        self,
        gr_min: float = DefaultParameters.GR_MIN,
        gr_max: float = DefaultParameters.GR_MAX,
        gr_method: str = "linear",
        use_sp: bool = True,
        use_nd: bool = True,
        sp_max: Optional[float] = None,
        sp_min: Optional[float] = None
    ):
        self.gr_min = gr_min
        self.gr_max = gr_max
        self.gr_method = gr_method
        self.use_sp = use_sp
        self.use_nd = use_nd
        self.sp_max = sp_max
        self.sp_min = sp_min

    def calculate(
        self,
        well: "Well",
        curves: Optional[Dict[str, str]] = None
    ) -> VclResult:
        """
        Calculate VCL using all available methods.

        Parameters
        ----------
        well : Well
            Well object with required curves
        curves : dict, optional
            Curve name mapping (e.g., {"gr": "GR", "sp": "SP"})

        Returns
        -------
        VclResult
            VCL calculation results
        """
        n_samples = well.n_samples

        # Default curve names
        if curves is None:
            curves = {}

        gr_name = curves.get("gr", "GR")
        sp_name = curves.get("sp", "SP")
        nphi_name = curves.get("nphi", "NPHI")
        rhob_name = curves.get("rhob", "RHOB")

        # Initialize results
        vcl_methods = []
        method_names = []

        vcl_gr = None
        vcl_sp = None
        vcl_nd = None

        # GR method (primary)
        if gr_name in well:
            gr_data = well[gr_name].data
            vcl_gr = vcl_from_gr(
                gr_data, self.gr_min, self.gr_max, self.gr_method
            )
            vcl_methods.append(vcl_gr)
            method_names.append("GR")

        # SP method
        if self.use_sp and sp_name in well:
            sp_data = well[sp_name].data

            # Auto-detect SP range if not provided
            sp_max = self.sp_max
            sp_min = self.sp_min
            if sp_max is None:
                valid_sp = sp_data[~np.isnan(sp_data)]
                if len(valid_sp) > 0:
                    sp_max = np.percentile(valid_sp, 5)  # Most negative
                    sp_min = np.percentile(valid_sp, 95)  # Shale baseline

            if sp_max is not None and sp_min is not None:
                vcl_sp = vcl_from_sp(sp_data, sp_max, sp_min)
                vcl_methods.append(vcl_sp)
                method_names.append("SP")

        # N-D method
        if self.use_nd and nphi_name in well and rhob_name in well:
            nphi_data = well[nphi_name].data
            rhob_data = well[rhob_name].data
            vcl_nd = vcl_from_nd(nphi_data, rhob_data)
            vcl_methods.append(vcl_nd)
            method_names.append("ND")

        # Select minimum VCL at each point
        if len(vcl_methods) == 0:
            raise ValueError("No VCL methods could be applied. Check curve availability.")

        if len(vcl_methods) == 1:
            vcl_final = vcl_methods[0]
            method_used = np.full(n_samples, method_names[0])
        else:
            # Stack all methods and take minimum
            stacked = np.vstack(vcl_methods)
            vcl_final = np.nanmin(stacked, axis=0)

            # Track which method was used
            method_used = np.empty(n_samples, dtype=object)
            for i in range(n_samples):
                values = [m[i] for m in vcl_methods]
                min_idx = np.nanargmin(values)
                method_used[i] = method_names[min_idx]

        return VclResult(
            vcl=vcl_final,
            vcl_gr=vcl_gr,
            vcl_sp=vcl_sp,
            vcl_nd=vcl_nd,
            method_used=method_used
        )

    def add_to_well(
        self,
        well: "Well",
        result: Optional[VclResult] = None
    ) -> "Well":
        """
        Add VCL curves to well.

        Parameters
        ----------
        well : Well
            Well object
        result : VclResult, optional
            Pre-computed result

        Returns
        -------
        Well
            Well with VCL curves added
        """
        if result is None:
            result = self.calculate(well)

        well.add_curve("VCL", result.vcl, unit="v/v", description="Clay Volume")

        if result.vcl_gr is not None:
            well.add_curve("VCLGR", result.vcl_gr, unit="v/v", description="VCL from GR")
        if result.vcl_sp is not None:
            well.add_curve("VCLSP", result.vcl_sp, unit="v/v", description="VCL from SP")
        if result.vcl_nd is not None:
            well.add_curve("VCLND", result.vcl_nd, unit="v/v", description="VCL from N-D")

        return well


def calculate_vcl(
    well: "Well",
    gr_min: float = DefaultParameters.GR_MIN,
    gr_max: float = DefaultParameters.GR_MAX,
    gr_method: str = "linear",
    use_sp: bool = True,
    use_nd: bool = True,
    **kwargs
) -> VclResult:
    """
    Calculate clay volume for a well.

    Convenience function for VclModule.

    Parameters
    ----------
    well : Well
        Well object
    gr_min : float, optional
        Clean sand GR
    gr_max : float, optional
        Shale GR
    gr_method : str, optional
        GR method
    use_sp : bool, optional
        Use SP method
    use_nd : bool, optional
        Use N-D method
    **kwargs
        Additional arguments

    Returns
    -------
    VclResult
        VCL calculation results

    Examples
    --------
    >>> from petrophys.clay_volume import calculate_vcl
    >>> result = calculate_vcl(well, gr_min=15, gr_max=110)
    >>> well["VCL"] = result.vcl
    """
    module = VclModule(
        gr_min=gr_min,
        gr_max=gr_max,
        gr_method=gr_method,
        use_sp=use_sp,
        use_nd=use_nd,
        **kwargs
    )
    return module.calculate(well)
