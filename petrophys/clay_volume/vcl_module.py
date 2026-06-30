"""
Main VCL calculator module.

This module provides the VclModule class that combines multiple
VCL methods and selects the minimum value.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import numpy as np

from petrophys.base.well import Well
from petrophys.base.constants import DefaultParameters, MatrixProperties
from petrophys.clay_volume.gr_method import vcl_from_gr
from petrophys.clay_volume.sp_method import vcl_from_sp, should_use_sp
from petrophys.clay_volume.nd_method import vcl_from_nd, nd_clean_line
from petrophys.clay_volume.classification import classify_vcl
from petrophys.petrophysics.porosity.density_porosity import density_porosity
from petrophys.utils.analysis_helpers import get_curve_data_depthwise


@dataclass
class VclResult:
    """
    Result container for VCL calculation.

    Attributes
    ----------
    vcl : np.ndarray
        Final VCL (combined per ``combine`` rule)
    vcl_gr : np.ndarray
        VCL from gamma ray
    vcl_sp : np.ndarray
        VCL from SP (if available)
    vcl_nd : np.ndarray
        VCL from neutron-density (if available)
    method_used : np.ndarray
        Which method provided the final value at each depth
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
    >>> well["VCLAY"] = result.vcl
    """

    def __init__(
        self,
        gr_min: float = DefaultParameters.GR_MIN,
        gr_max: float = DefaultParameters.GR_MAX,
        gr_method: str = "linear",
        use_sp: bool = True,
        use_nd: bool = True,
        sp_max: Optional[float] = None,
        sp_min: Optional[float] = None,
        matrix: Optional[str] = None,
        combine: str = "min",
        rho_fl: float = DefaultParameters.FLUID_DENSITY,
    ):
        self.gr_min = gr_min
        self.gr_max = gr_max
        self.gr_method = gr_method
        self.use_sp = use_sp
        self.use_nd = use_nd
        # Matrix selects the N-D crossplot clean line (sandstone for clastic,
        # limestone for carbonate, dolomite). None -> limestone (back-compat).
        self.matrix = matrix
        self.sp_max = sp_max
        self.sp_min = sp_min
        # How to combine the per-method clay indicators:
        #   "min"  -- minimum of all methods (legacy default; under-reads
        #             non-radioactive / soft clay because a low SP or GR
        #             collapses the estimate below the neutron-density signal).
        #   "auto" -- neutron-density is the primary clay indicator (it responds
        #             to clay mineralogy, not radioactivity or SP suppression);
        #             GR is used only where gas crossover invalidates N-D or N-D
        #             is unavailable; SP is a reported cross-check that never
        #             pulls the final value down. Recommended for clastics.
        self.combine = str(combine or "min").strip().lower()
        self.rho_fl = rho_fl

    def _resolve_gr_baselines(self, gr_data: np.ndarray) -> tuple[float, float]:
        """
        Resolve GR baselines from the well when the module is using defaults.

        Default behavior:
        - use the log minimum/maximum when they extend beyond 20/100
        - force GRmin = 20 if the log minimum is higher than 20
        - force GRmax = 100 if the log maximum is lower than 100

        If the caller supplied non-default GR baselines, respect them.
        """
        valid_gr = np.asarray(gr_data, dtype=float)
        valid_gr = valid_gr[np.isfinite(valid_gr)]

        if len(valid_gr) == 0:
            return self.gr_min, self.gr_max

        actual_min = float(np.nanmin(valid_gr))
        actual_max = float(np.nanmax(valid_gr))

        if self.gr_min == DefaultParameters.GR_MIN:
            gr_min = min(actual_min, DefaultParameters.GR_MIN)
        else:
            gr_min = self.gr_min

        if self.gr_max == DefaultParameters.GR_MAX:
            gr_max = max(actual_max, DefaultParameters.GR_MAX)
        else:
            gr_max = self.gr_max

        return gr_min, gr_max

    def _combine_nd_primary(self, n_samples, vcl_gr, vcl_sp, vcl_nd, nphi_data, rhob_data):
        """Neutron-density-primary clay combination.

        The neutron-density crossplot responds to clay *mineralogy* and is the
        most reliable single clay indicator in water/oil-bearing clastics: GR
        under-reads non-radioactive (kaolinite-type) clay and over-reads
        feldspathic/micaceous sands, while SP is suppressed by fresh mud and
        thin beds. The legacy ``min`` rule therefore collapses VCL below the
        true clay content whenever GR or SP happens to read low.

        Rule:
          * N-D is the primary estimate where it is finite and the interval is
            NOT gas-crossover affected (gas suppresses the neutron, making N-D
            under-read clay).
          * GR is used where gas invalidates N-D, or where N-D is unavailable.
          * SP is only a fallback where neither GR nor N-D exists; it never
            pulls the final value down (it remains reported as VCL_SP).
        """
        vcl_final = np.full(n_samples, np.nan, dtype=float)
        method_used = np.full(n_samples, "NaN", dtype=object)

        gr = np.asarray(vcl_gr, dtype=float) if vcl_gr is not None else None
        nd = np.asarray(vcl_nd, dtype=float) if vcl_nd is not None else None
        sp = np.asarray(vcl_sp, dtype=float) if vcl_sp is not None else None

        # Gas crossover flag (density porosity exceeds neutron) -> N-D unreliable.
        gas = np.zeros(n_samples, dtype=bool)
        if nd is not None and nphi_data is not None and rhob_data is not None:
            try:
                rho_ma = MatrixProperties.get(self.matrix or "sandstone").density
            except (KeyError, AttributeError):
                rho_ma = 2.65
            phid = density_porosity(np.asarray(rhob_data, dtype=float), rho_ma, self.rho_fl)
            nphi = np.asarray(nphi_data, dtype=float)
            gas = np.isfinite(phid) & np.isfinite(nphi) & (phid > nphi + 0.05)

        # Start from GR (most universally present), then prefer N-D in non-gas.
        if gr is not None:
            ok = np.isfinite(gr)
            vcl_final[ok] = gr[ok]
            method_used[ok] = "GR"
        if nd is not None:
            use_nd = np.isfinite(nd) & ~gas
            vcl_final[use_nd] = nd[use_nd]
            method_used[use_nd] = "ND"
            # Where GR is missing, also use N-D in gas zones (best available).
            nd_fill = np.isfinite(nd) & ~np.isfinite(vcl_final)
            vcl_final[nd_fill] = nd[nd_fill]
            method_used[nd_fill] = "ND"
        if sp is not None:
            sp_fill = np.isfinite(sp) & ~np.isfinite(vcl_final)
            vcl_final[sp_fill] = sp[sp_fill]
            method_used[sp_fill] = "SP"

        return np.clip(vcl_final, 0.0, 1.0), method_used

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

        # GR method (primary). When the default "GR" is requested, fall back to the
        # GR family (computed/spectral) so a well that only carries CGR/SGR is still
        # usable -- alias_map keeps SGR/CGR as their own standards, so a plain "GR"
        # lookup would otherwise miss them.
        gr_candidates = [gr_name] if gr_name != "GR" else ["GR", "CGR", "SGR", "HSGR", "HCGR", "GAMMA"]
        gr_data, _, _ = get_curve_data_depthwise(well, gr_candidates)
        if gr_data is not None:
            gr_min, gr_max = self._resolve_gr_baselines(gr_data)
            vcl_gr = vcl_from_gr(
                gr_data, gr_min, gr_max, self.gr_method
            )
            vcl_methods.append(vcl_gr)
            method_names.append("GR")

        # SP method
        sp_data, _, _ = get_curve_data_depthwise(well, [sp_name])
        if self.use_sp and sp_data is not None:

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
        nphi_data, _, _ = get_curve_data_depthwise(well, [nphi_name, "PHIN"])
        rhob_data, _, _ = get_curve_data_depthwise(well, [rhob_name])
        if self.use_nd and nphi_data is not None and rhob_data is not None:
            den_cl1, neu_cl1, den_cl2, neu_cl2 = nd_clean_line(self.matrix)
            vcl_nd = vcl_from_nd(
                nphi_data, rhob_data,
                den_cl1=den_cl1, neu_cl1=neu_cl1,
                den_cl2=den_cl2, neu_cl2=neu_cl2,
            )
            vcl_methods.append(vcl_nd)
            method_names.append("ND")

        # Select VCL at each point
        if len(vcl_methods) == 0:
            raise ValueError("No VCL methods could be applied. Check curve availability.")

        if len(vcl_methods) == 1:
            vcl_final = vcl_methods[0]
            method_used = np.full(n_samples, method_names[0])
        elif self.combine in ("auto", "nd_primary", "robust"):
            vcl_final, method_used = self._combine_nd_primary(
                n_samples, vcl_gr, vcl_sp, vcl_nd, nphi_data, rhob_data
            )
        else:
            # Legacy: minimum of all methods at each depth.
            stacked = np.vstack(vcl_methods)
            valid_any = np.any(np.isfinite(stacked), axis=0)
            vcl_final = np.full(n_samples, np.nan, dtype=float)
            if np.any(valid_any):
                vcl_final[valid_any] = np.nanmin(stacked[:, valid_any], axis=0)

            # Track which method was used
            method_used = np.empty(n_samples, dtype=object)
            for i in range(n_samples):
                values = [m[i] for m in vcl_methods]
                valid = [(v, name) for v, name in zip(values, method_names) if not np.isnan(v)]
                if valid:
                    min_val, min_name = min(valid, key=lambda x: x[0])
                    method_used[i] = min_name
                else:
                    method_used[i] = "NaN"

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

        # Primary standardized mnemonics
        well.add_curve("VCL", result.vcl, unit="v/v", description="Clay Volume")
        well.add_curve("VCLAY", result.vcl, unit="v/v", description="Clay Volume")

        if result.vcl_gr is not None:
            well.add_curve("VCL_GR", result.vcl_gr, unit="v/v", description="VCL from GR")
            well.add_curve("VCLGR", result.vcl_gr, unit="v/v", description="VCL from GR")
        if result.vcl_sp is not None:
            well.add_curve("VCL_SP", result.vcl_sp, unit="v/v", description="VCL from SP")
            well.add_curve("VCLSP", result.vcl_sp, unit="v/v", description="VCL from SP")
        if result.vcl_nd is not None:
            well.add_curve("VCL_ND", result.vcl_nd, unit="v/v", description="VCL from N-D")
            well.add_curve("VCLND", result.vcl_nd, unit="v/v", description="VCL from N-D")

        return well


def calculate_vcl(
    well: "Well",
    gr_min: float = DefaultParameters.GR_MIN,
    gr_max: float = DefaultParameters.GR_MAX,
    gr_method: str = "linear",
    use_sp: bool = True,
    use_nd: bool = True,
    matrix: Optional[str] = None,
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
        matrix=matrix,
        **kwargs
    )
    return module.calculate(well)
