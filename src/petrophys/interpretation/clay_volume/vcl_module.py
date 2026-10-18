"""
Main VCL calculator module.

This module provides the VclModule class that combines multiple
VCL methods and selects the minimum value.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
import warnings
import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys.model.well import Well
from petrophys.model.constants import DefaultParameters, MatrixProperties
from petrophys.interpretation.clay_volume.gr_method import vcl_from_gr
from petrophys.interpretation.clay_volume.sp_method import vcl_from_sp, should_use_sp, sp_polarity
from petrophys.interpretation.clay_volume.nd_method import ND_CLAY_POINT, vcl_from_nd, nd_clean_line
from petrophys.interpretation.clay_volume.classification import classify_vcl
from petrophys.interpretation.porosity.density_porosity import density_porosity
from petrophys._internal.analysis_helpers import get_curve_data_depthwise


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
    gr_min_used: Optional[float] = None
    gr_max_used: Optional[float] = None


class VclModule:
    """
    Multi-method clay volume calculator.

    Calculates VCL using multiple methods and selects the minimum
    value at each depth point, as is standard practice.

    Parameters
    ----------
    gr_min : float, optional
        Clean sand GR value (default: 20 API). Left at the default, the log
        minimum is used where it is lower than 20 API.
    gr_max : float, optional
        Shale GR value (default: 100 API). Left at the default, the log
        maximum is used where it is higher than 100 API, so isolated GR
        spikes raise the shale baseline for the whole well; pass explicit
        baselines for a log with spikes.
    gr_method : str, optional
        GR method (default: "linear")
    use_sp : bool, optional
        Whether to use SP method (default: True)
    use_nd : bool, optional
        Whether to use N-D method (default: True)
    sp_max : float, optional
        Clean-sand SP line (static SP), mV: the line of greatest deflection.
    sp_min : float, optional
        Shale SP baseline, mV. With ``sp_max`` left as None both lines are
        picked from the log, P5 for ``sp_max`` and P95 for ``sp_min``; SP is
        not used when only ``sp_min`` is missing. Where the log has a GR and
        the SP falls with it clearly (rank correlation below -0.3 after
        de-trending, :func:`~petrophys.interpretation.clay_volume.sp_method.sp_polarity`),
        the SP is reversed and the two picks are swapped, with a
        ``UserWarning``.
    sp_fallback_only : bool, optional
        Keep VCL_SP out of the minimum (default: False). It is still computed
        and returned, and fills only depths where neither GR nor N-D gives a
        value. Use it when the SP baselines are picked over a drifting log.
    nd_clay : tuple of float, optional
        N-D clay point ``(RHOB, NPHI)``, e.g. from
        :func:`~petrophys.interpretation.clay_volume.nd_method.nd_shale_point`.
        Default: the :func:`~petrophys.interpretation.clay_volume.vcl_from_nd`
        point (2.62 g/cm3, 0.38).
    nd_triangle_only : bool, optional
        Leave VCL_ND undefined (NaN) where NPHI exceeds both the clean line's
        40 p.u. end point and the clay point (default: False). Such samples,
        e.g. undercompacted mudstones at NPHI 0.45-0.55, lie outside the
        triangle the three-point formula interpolates in and read as clean
        rock there; the minimum then takes the other indicators.

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
        sp_fallback_only: bool = False,
        nd_clay: Optional[Tuple[float, float]] = None,
        nd_triangle_only: bool = False,
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
        #   "min"  -- minimum valid method at each depth; this is the default
        #             convention used by the dynamic workflow.
        #   "auto" -- optional N-D-primary mode. GR is used where gas crossover
        #             invalidates N-D or N-D is unavailable; SP is a fallback.
        self.combine = str(combine or "min").strip().lower()
        self.rho_fl = rho_fl
        self.sp_fallback_only = bool(sp_fallback_only)
        self.nd_clay = nd_clay
        self.nd_triangle_only = bool(nd_triangle_only)

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

    @staticmethod
    def _sp_reversed(well: "Well", sp_data: np.ndarray, gr_data: Optional[np.ndarray]) -> bool:
        """Whether the SP and GR show a reversed SP (falls with GR) clearly."""
        if reproducing_benchmark() or gr_data is None or well.depth is None:
            return False
        unit = str(getattr(well, "depth_unit", "") or "").strip().lower()
        window = 100.0 if unit in {"ft", "feet", "foot", "f"} else 30.0
        return sp_polarity(sp_data, gr_data, well.depth, window=window, min_correlation=0.3) < 0

    def _combine_nd_primary(self, n_samples, vcl_gr, vcl_sp, vcl_nd, nphi_data, rhob_data):
        """Neutron-density-primary clay combination.

        The neutron-density crossplot responds to clay *mineralogy* and is the
        most reliable single clay indicator in water/oil-bearing clastics: GR
        under-reads non-radioactive (kaolinite-type) clay and over-reads
        feldspathic/micaceous sands, while SP is suppressed by fresh mud and
        thin beds. A plain minimum of the methods can therefore fall below the
        true clay content wherever GR or SP happens to read low.

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
                rho_ma = MatrixProperties.get_log_density(self.matrix or "sandstone")
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
        gr_min_used = None
        gr_max_used = None

        # GR method (primary). When the default "GR" is requested, fall back to the
        # GR family (computed/spectral) so a well that only carries CGR/SGR is still
        # usable -- alias_map keeps SGR/CGR as their own standards, so a plain "GR"
        # lookup would otherwise miss them.
        gr_candidates = [gr_name] if gr_name != "GR" else ["GR", "CGR", "SGR", "HSGR", "HCGR", "GAMMA"]
        gr_data, _, _ = get_curve_data_depthwise(well, gr_candidates)
        if gr_data is not None:
            gr_min, gr_max = self._resolve_gr_baselines(gr_data)
            gr_min_used = min(float(gr_min), 20.0)
            gr_max_used = max(float(gr_max), 100.0)
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
                    if self._sp_reversed(well, sp_data, gr_data):
                        # A reversed SP (Rmf < Rw) deflects positive in clean
                        # sand: the clean line is the P95 and the shale
                        # baseline the P5.
                        sp_max, sp_min = sp_min, sp_max
                        warnings.warn(
                            "SP polarity is reversed (SP falls with GR): the picked SP "
                            f"baselines are swapped, clean-sand line {sp_max:.1f} mV and "
                            f"shale baseline {sp_min:.1f} mV",
                            UserWarning,
                            stacklevel=2,
                        )

            if sp_max is not None and sp_min is not None:
                vcl_sp = vcl_from_sp(sp_data, sp_max, sp_min)
                vcl_methods.append(vcl_sp)
                method_names.append("SP")

        # N-D method
        nphi_data, _, _ = get_curve_data_depthwise(well, [nphi_name, "PHIN"])
        rhob_data, _, _ = get_curve_data_depthwise(well, [rhob_name])
        if self.use_nd and nphi_data is not None and rhob_data is not None:
            den_cl1, neu_cl1, den_cl2, neu_cl2 = nd_clean_line(self.matrix)
            clay_point = {}
            if self.nd_clay is not None:
                clay_point = {"den_clay": self.nd_clay[0], "neu_clay": self.nd_clay[1]}
            vcl_nd = vcl_from_nd(
                nphi_data, rhob_data,
                den_cl1=den_cl1, neu_cl1=neu_cl1,
                den_cl2=den_cl2, neu_cl2=neu_cl2,
                **clay_point,
            )
            if self.nd_triangle_only:
                neu_clay = self.nd_clay[1] if self.nd_clay is not None else ND_CLAY_POINT[1]
                with np.errstate(invalid="ignore"):
                    outside = np.asarray(nphi_data, dtype=float) > max(neu_cl2, neu_clay)
                vcl_nd = np.where(outside, np.nan, vcl_nd)
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
            # Minimum of the available methods at each depth (SP left out when
            # it is a fallback only).
            in_min = [
                (values, name) for values, name in zip(vcl_methods, method_names)
                if not (self.sp_fallback_only and name == "SP")
            ]
            stacked = np.vstack([values for values, _ in in_min])
            valid_any = np.any(np.isfinite(stacked), axis=0)
            vcl_final = np.full(n_samples, np.nan, dtype=float)
            if np.any(valid_any):
                vcl_final[valid_any] = np.nanmin(stacked[:, valid_any], axis=0)

            # Track which method was used
            method_used = np.empty(n_samples, dtype=object)
            for i in range(n_samples):
                valid = [(values[i], name) for values, name in in_min if not np.isnan(values[i])]
                if valid:
                    min_val, min_name = min(valid, key=lambda x: x[0])
                    method_used[i] = min_name
                else:
                    method_used[i] = "NaN"

            if self.sp_fallback_only and vcl_sp is not None:
                fill = ~np.isfinite(vcl_final) & np.isfinite(vcl_sp)
                vcl_final[fill] = vcl_sp[fill]
                method_used[fill] = "SP"

        return VclResult(
            vcl=vcl_final,
            vcl_gr=vcl_gr,
            vcl_sp=vcl_sp,
            vcl_nd=vcl_nd,
            method_used=method_used,
            gr_min_used=gr_min_used,
            gr_max_used=gr_max_used,
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
    >>> from petrophys.interpretation.clay_volume import calculate_vcl
    >>> result = calculate_vcl(well, gr_min=15, gr_max=110)
    >>> well["VCL"] = result.vcl
    """
    if gr_min is not None and gr_max is not None and float(gr_min) >= float(gr_max):
        raise ValueError(
            f"gr_min ({gr_min:g}) must be below gr_max ({gr_max:g}): gr_min is the "
            "clean-sand GR and gr_max the shale GR"
        )
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
