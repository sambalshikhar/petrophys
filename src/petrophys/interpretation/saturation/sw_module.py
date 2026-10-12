"""
Main water saturation module.

This module provides the SaturationModule class for automatic
model selection based on formation properties: VCL, lithology,
CEC availability, and salinity.

Decision flowchart::

    VCL < cutoff -> CLEAN -> Archie
    VCL >= cutoff -> SHALY:
        Carbonate with an Rwb -> Dual Water
        Clastic, or carbonate without an Rwb:
            CEC and B_COEFF available -> Waxman-Smits
            salinity >= 20 kppm -> Simandoux
            salinity < 20 kppm -> Indonesia

Lithology can be a single string (applied globally) or a per-depth
numpy array for interval-based lithology control.
"""

import warnings
from typing import TYPE_CHECKING, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from petrophys.model.constants import DefaultParameters
from petrophys.interpretation.saturation.archie import archie
from petrophys.interpretation.saturation.north_sea import north_sea_shaly_sand
from petrophys.interpretation.saturation.nigeria_schlumberger import nigeria_schlumberger
from petrophys.interpretation.saturation.simandoux import simandoux
from petrophys.interpretation.saturation.indonesia import indonesia
from petrophys.interpretation.saturation.dual_water import dual_water
from petrophys.interpretation.saturation.waxman_smits import waxman_smits
from petrophys.interpretation.saturation.sxo import calculate_sxo
from petrophys._internal.analysis_helpers import (
    curve_has_finite_data,
    get_curve,
    get_curve_data_depthwise,
)

if TYPE_CHECKING:
    from petrophys.model.well import Well


#: Lithology labels that auto routing treats as carbonate.
_CARBONATE_LITHOLOGIES = {"carbonate", "limestone", "dolomite", "calcite"}


class SwModel(Enum):
    """Water saturation model selection."""
    ARCHIE = "archie"
    NORTH_SEA = "north_sea"
    NIGERIA_SCHLUMBERGER = "nigeria_schlumberger"
    SIMANDOUX = "simandoux"
    INDONESIA = "indonesia"
    DUAL_WATER = "dual_water"
    WAXMAN_SMITS = "waxman_smits"


@dataclass
class SaturationResult:
    """
    Result container for saturation calculation.

    Attributes
    ----------
    sw : np.ndarray
        Water saturation (uninvaded zone)
    sxo : np.ndarray
        Water saturation (flushed zone)
    model : str
        Model used for calculation (or "mixed" if per-depth lithology)
    """

    sw: np.ndarray
    sxo: Optional[np.ndarray]
    model: str
    sxo_method: Optional[str] = None
    sxo_reason: Optional[str] = None
    model_counts: Optional[Dict[str, int]] = None
    sxo_method_counts: Optional[Dict[str, int]] = None
    sxo_input_sources: Optional[Dict[str, Dict[str, int]]] = None
    #: Samples that auto routing gave another model because the preferred one
    #: lacked an input, e.g. ``{"dual_water->indonesia": 12}`` for shaly
    #: carbonate samples that had no Rwb and took the clastic routing.
    fallback_counts: Optional[Dict[str, int]] = None
    #: Waxman-Smits samples set to SW = 0 by the effective-basis conversion
    #: (``effective_waxman_smits``) because their total-porosity Swt lies below
    #: the bound-water fraction (PHIT - PHIE) / PHIT.
    waxman_smits_clipped_to_zero: int = 0


class SaturationModule:
    """
    Water saturation calculator with automatic model selection.

    Model selection criteria (from design doc flowchart)::

        VCL < vcl_cutoff  -> CLEAN -> Archie
        VCL >= vcl_cutoff -> SHALY:
            Carbonate with an Rwb -> Dual Water
            Clastic, or carbonate without an Rwb:
                CEC and B_COEFF available -> Waxman-Smits
                salinity >= 20 kppm -> Simandoux
                salinity < 20 kppm -> Indonesia

    Lithology may be a single string or a per-depth numpy array.
    When auto-routing is active, the clean/shaly decision is applied
    sample-by-sample so a single well can use multiple Sw models across depth.

    Parameters
    ----------
    a : float, optional
        Tortuosity factor (default: 1.0)
    m : float, optional
        Cementation exponent (default: 2.0)
    n : float, optional
        Saturation exponent (default: 2.0)
    model : str, optional
        Force specific model (default: "auto")
    vcl_cutoff : float, optional
        VCL threshold for clean vs shaly (default: 0.3)
    rwb_fallback : float or None, optional
        Bound-water resistivity Dual Water takes when no ``rwb`` is given, as
        a multiple of Rw. Default None: Rwb is not estimated, so Dual Water
        runs only with a supplied ``rwb``. Auto routing then gives shaly
        carbonate samples without one the clastic routing (Waxman-Smits with
        CEC and B, Simandoux at a salinity of 20 kppm or more, otherwise
        Indonesia) and counts them in ``SaturationResult.fallback_counts``;
        ``model="dual_water"`` raises ``ValueError``. A number takes
        ``Rwb = rwb_fallback * Rw``, an assumption rather than a measurement;
        the benchmark calculation profile of :func:`petrophys.interpret`
        pins 0.3.
    effective_waxman_smits : bool, optional
        Report the Waxman-Smits saturation, which is solved on total
        porosity, on the effective-porosity basis of the other models,
        ``Swe = 1 - (PHIT / PHIE) * (1 - Swt)`` (Crain's Petrophysical
        Handbook), so that SW can be paired with PHIE downstream. The
        hydrocarbon pore volume ``PHIE * (1 - Swe) = PHIT * (1 - Swt)`` is
        kept wherever Swt is at least the bound-water fraction
        ``(PHIT - PHIE) / PHIT``. Below it the resistivity leaves less water
        than the clay-bound water, the hydrocarbon volume would exceed PHIE,
        and SW is clipped to 0; such samples are counted in
        ``SaturationResult.waxman_smits_clipped_to_zero``. Pass ``phit`` on
        the same basis as ``phie``. Default False (total-porosity Swt);
        :func:`petrophys.interpret` uses True in its default ``"current"``
        calculation profile.

    Examples
    --------
    >>> sw_calc = SaturationModule()
    >>> result = sw_calc.calculate(well, rw=0.05, lithology='carbonate')
    """

    def __init__(
        self,
        a: float = DefaultParameters.ARCHIE_A,
        m: float = DefaultParameters.ARCHIE_M,
        n: float = DefaultParameters.ARCHIE_N,
        model: str = "auto",
        vcl_cutoff: float = 0.3,
        sxo_quicklook_fallback: str = "per_sample",
        depthwise_curve_merge: bool = True,
        rwb_fallback: Optional[float] = None,
        effective_waxman_smits: bool = False,
    ):
        self.a = a
        self.m = m
        self.n = n
        self.model = model.lower()
        self.vcl_cutoff = vcl_cutoff
        #: How the quick-look ``SW**0.2`` relation stands in for the
        #: invaded-zone Archie SXO. ``"per_sample"`` fills every sample whose
        #: Archie SXO is not finite. ``"whole_curve"`` uses it only where the
        #: invaded-zone inputs are missing altogether, leaving individual
        #: non-finite Archie samples as NaN.
        self.sxo_quicklook_fallback = str(sxo_quicklook_fallback)
        #: Fill a curve's missing samples from the next candidate curve. When
        #: False the first candidate curve present is used as it stands.
        self.depthwise_curve_merge = bool(depthwise_curve_merge)
        #: Rwb as a multiple of Rw when Dual Water gets no ``rwb``; None
        #: requires a supplied Rwb.
        self.rwb_fallback = rwb_fallback
        #: Convert the total-porosity Waxman-Smits Sw to the effective basis.
        self.effective_waxman_smits = bool(effective_waxman_smits)

    def calculate(
        self,
        well: "Well",
        rw: Union[float, np.ndarray],
        rcl: Optional[float] = None,
        salinity: Optional[float] = None,
        rmf: Optional[Union[float, np.ndarray]] = None,
        vcl: Optional[np.ndarray] = None,
        phie: Optional[np.ndarray] = None,
        phit: Optional[np.ndarray] = None,
        lithology: Union[str, np.ndarray] = "clastic",
        cec: Optional[float] = None,
        b_coeff: Optional[float] = None,
        rho_ma: float = 2.65,
        m_star: float = 2.0,
        rwb: Optional[float] = None,
        phi_cl: float = DefaultParameters.CLAY_POROSITY,
        rtsh: Optional[float] = None,
        vsh_exp: float = 1.6,
        shale_exp: float = 1.6,
    ) -> SaturationResult:
        """
        Calculate water saturation.

        Parameters
        ----------
        well : Well
            Well object
        rw : float
            Formation water resistivity
        rcl : float, optional
            Clay resistivity (auto-estimated if not provided)
        salinity : float, optional
            Formation water salinity (ppm)
        rmf : float, optional
            Mud filtrate resistivity (for Sxo)
        vcl : np.ndarray, optional
            Clay volume
        phie : np.ndarray, optional
            Effective porosity
        phit : np.ndarray, optional
            Total porosity (required for Dual Water / Waxman-Smits), on the
            basis of ``phie``: PHIT - PHIE is read as clay-bound porosity.
            Default: the PHIT curve, else ``phie``.
        lithology : str or np.ndarray, optional
            'clastic' or 'carbonate' globally, or per-depth array
        cec : float, optional
            Cation exchange capacity (for Waxman-Smits, meq/g)
        b_coeff : float, optional
            Equivalent counter-ion conductance B (for Waxman-Smits),
            (S/m)/(meq/cm3)
        rho_ma : float, optional
            Matrix density (for Waxman-Smits, default: 2.65)
        m_star : float, optional
            Intrinsic cementation exponent (for Waxman-Smits, default: 2.0)
        rwb : float, optional
            Bound water resistivity (for Dual Water), ohm.m; see
            ``rwb_fallback`` for what happens without it
        phi_cl : float, optional
            Clay porosity (for Dual Water, default from constants)

        Returns
        -------
        SaturationResult
            Saturation calculation results
        """
        # Get required curves
        rt_data, rt_mnemonic, rt_meta = get_curve_data_depthwise(well, ["RDEP", "RT"])
        rxo_existing = get_curve(well, ["RSHA", "RXO", "RESS"], require_finite=False)
        if self.depthwise_curve_merge:
            rxo_data, rxo_mnemonic, rxo_meta = get_curve_data_depthwise(well, ["RSHA", "RXO", "RESS"])
        elif rxo_existing is None:
            rxo_data, rxo_mnemonic, rxo_meta = None, None, {}
        else:
            rxo_data = np.asarray(rxo_existing.data, dtype=float)
            rxo_mnemonic = rxo_existing.mnemonic
            rxo_meta = {"source_counts": {rxo_mnemonic: int(np.count_nonzero(np.isfinite(rxo_data)))}}
            if not np.any(np.isfinite(rxo_data)):
                rxo_data = None

        # Get VCL and effective porosity. PHIE_CC is preferred when present so
        # saturation honors same-well core-corrected porosity.
        if vcl is None:
            vcl_data, _, _ = get_curve_data_depthwise(well, ["VCL", "VCLAY"])
            vcl = vcl_data if vcl_data is not None else np.zeros(well.n_samples)

        if phie is None:
            phie_data, _, _ = get_curve_data_depthwise(well, ["PHIE_CC", "PHIE"])
            if phie_data is None:
                raise ValueError("Effective porosity (PHIE_CC or PHIE) with finite samples required")
            phie = phie_data

        if phit is None:
            phit_data, _, _ = get_curve_data_depthwise(well, ["PHIT"])
            phit = phit_data if phit_data is not None else phie

        fallback_counts: Dict[str, int] = {}
        clip_counts: Dict[str, int] = {}
        if rt_data is None or rw is None:
            sw = np.full(well.n_samples, np.nan, dtype=float)
            model_name = "not_computed_missing_rt" if rt_data is None else "not_computed_missing_rw"
            model_counts = {}
        else:
            # Estimate Rcl if not provided
            if rcl is None:
                rcl = self._estimate_rcl(rt_data, vcl)

            # Auto-routing is applied sample-by-sample so clean and shaly
            # intervals can coexist within the same well/zone.
            if self.model == "auto":
                if isinstance(lithology, np.ndarray):
                    lithology_array = np.asarray(lithology, dtype=object)
                else:
                    lithology_array = np.full(rt_data.shape, lithology, dtype=object)
                sw, model_name, model_counts, fallback_counts = self._calculate_depth_routed(
                    rt_data, rw, rcl, salinity, vcl, phie, phit,
                    lithology_array, cec, b_coeff, rho_ma, m_star, rwb, phi_cl, rtsh, vsh_exp, shale_exp,
                    clip_counts=clip_counts,
                )
            else:
                # Single model for entire well
                model = self._select_model(vcl, salinity,
                                           lithology if isinstance(lithology, str) else "clastic",
                                           cec, b_coeff, rwb)
                sw = self._run_model(model, rt_data, rw, rcl, vcl, phie, phit,
                                     cec, b_coeff, rho_ma, m_star, rwb, phi_cl, rtsh, vsh_exp, shale_exp,
                                     self.a, self.m, self.n, clip_counts=clip_counts)
                model_name = model.value
                model_counts = {model_name: int(rt_data.shape[0])}

        # Calculate Sxo from invaded-zone inputs where they are finite;
        # otherwise fall back to the quick-look empirical relation requested
        # by the user.
        sxo = None
        sxo_method = None
        sxo_reason = None
        sxo_method_counts: Dict[str, int] = {}
        sw_clipped = np.clip(np.asarray(sw, dtype=float), 0.0, 1.0) if sw is not None else None
        fallback_sxo = np.power(sw_clipped, 0.2) if sw_clipped is not None else None
        per_sample_fallback = self.sxo_quicklook_fallback != "whole_curve"
        finite_fallback = (
            np.isfinite(fallback_sxo)
            if fallback_sxo is not None
            else np.zeros(well.n_samples, dtype=bool)
        )
        finite_archie = np.zeros(well.n_samples, dtype=bool)

        if rxo_data is not None and rmf is not None:
            archie_sxo = calculate_sxo(rxo_data, rmf, phie, self.a, self.m, self.n)
            archie_sxo = np.clip(np.asarray(archie_sxo, dtype=float), 0.0, 1.0)
            finite_archie = np.isfinite(archie_sxo)
            if fallback_sxo is not None and per_sample_fallback:
                sxo = np.where(finite_archie, archie_sxo, fallback_sxo)
            else:
                sxo = np.where(finite_archie, archie_sxo, np.nan)
            finite_sw = np.isfinite(sw_clipped) if sw_clipped is not None else np.zeros(well.n_samples, dtype=bool)
            sxo = np.where(finite_sw & np.isfinite(sxo), np.maximum(sxo, sw_clipped), sxo)
            use_fallback = ~finite_archie & finite_fallback & per_sample_fallback
            if np.any(finite_archie):
                sxo_method_counts["archie_rxo_rmf"] = int(np.count_nonzero(finite_archie))
            if np.any(use_fallback):
                sxo_method_counts["fifth_root_quicklook"] = int(np.count_nonzero(use_fallback))
            if np.any(finite_archie) and np.any(use_fallback):
                sxo_method = "archie_rxo_rmf+fifth_root_quicklook"
                sxo_reason = "invaded-zone Archie used where finite; fifth-root fallback used where RSHA/RXO/RESS/RMF were non-finite"
            elif np.any(finite_archie):
                sxo_method = "archie_rxo_rmf"
                sxo_reason = "RSHA/RXO/RESS and RMF available"
            elif np.any(finite_fallback):
                sxo_method = "fifth_root_quicklook"
                sxo_reason = "invaded-zone Archie produced no finite samples"
        elif fallback_sxo is not None and np.any(finite_fallback):
            sxo = fallback_sxo
            sxo_method = "fifth_root_quicklook"
            sxo_method_counts["fifth_root_quicklook"] = int(np.count_nonzero(finite_fallback))
            if rxo_existing is None and rmf is None:
                sxo_reason = "RSHA/RXO/RESS and RMF unavailable"
            elif rxo_existing is None:
                sxo_reason = "RSHA/RXO/RESS unavailable"
            elif not curve_has_finite_data(rxo_existing):
                sxo_reason = "RSHA/RXO/RESS present but had no finite usable samples"
            else:
                sxo_reason = "RMF unavailable"

        return SaturationResult(
            sw=sw,
            sxo=sxo,
            model=model_name,
            sxo_method=sxo_method,
            sxo_reason=sxo_reason,
            model_counts=model_counts,
            sxo_method_counts=sxo_method_counts,
            sxo_input_sources={
                "rt": rt_meta.get("source_counts", {}),
                "rxo": rxo_meta.get("source_counts", {}),
            },
            fallback_counts=fallback_counts,
            waxman_smits_clipped_to_zero=clip_counts.get(SwModel.WAXMAN_SMITS.value, 0),
        )

    @staticmethod
    def _segment_value(value, mask: np.ndarray):
        """Return a masked view for per-depth arrays, else pass scalars through."""
        if isinstance(value, np.ndarray):
            if value.shape == mask.shape:
                return value[mask]
            return value
        return value

    def _calculate_depth_routed(
        self,
        rt: np.ndarray, rw: Union[float, np.ndarray], rcl: float,
        salinity: Optional[float],
        vcl: np.ndarray, phie: np.ndarray, phit: np.ndarray,
        lithology: np.ndarray,
        cec, b_coeff, rho_ma, m_star, rwb, phi_cl, rtsh, vsh_exp, shale_exp,
        clip_counts: Optional[Dict[str, int]] = None,
    ):
        """Calculate Sw with sample-by-sample routing across depth."""
        sw = np.full_like(rt, np.nan)
        model_per_sample = np.empty(rt.shape, dtype=object)
        fallback_counts: Dict[str, int] = {}

        for i in range(rt.shape[0]):
            lith_i = str(lithology[i]).lower() if lithology is not None else "clastic"
            model = self._select_model_scalar(
                float(vcl[i]) if np.isfinite(vcl[i]) else np.nan,
                salinity,
                lith_i,
                cec,
                b_coeff,
                rwb,
            )
            model_per_sample[i] = model.value
            # Shaly carbonate leaves Dual Water only when it has no Rwb.
            if lith_i in _CARBONATE_LITHOLOGIES and model not in (SwModel.ARCHIE, SwModel.DUAL_WATER):
                route = f"dual_water->{model.value}"
                fallback_counts[route] = fallback_counts.get(route, 0) + 1

        ordered_models = []
        for model_name in model_per_sample:
            if model_name not in ordered_models:
                ordered_models.append(model_name)

        for model_name in ordered_models:
            mask = model_per_sample == model_name
            if not np.any(mask):
                continue
            sw_segment = self._run_model(
                SwModel(model_name),
                rt[mask], self._segment_value(rw, mask), rcl,
                vcl[mask], phie[mask], phit[mask],
                self._segment_value(cec, mask),
                self._segment_value(b_coeff, mask),
                self._segment_value(rho_ma, mask),
                self._segment_value(m_star, mask),
                self._segment_value(rwb, mask),
                self._segment_value(phi_cl, mask),
                self._segment_value(rtsh, mask),
                self._segment_value(vsh_exp, mask),
                self._segment_value(shale_exp, mask),
                self._segment_value(self.a, mask),
                self._segment_value(self.m, mask),
                self._segment_value(self.n, mask),
                clip_counts=clip_counts,
            )
            sw[mask] = sw_segment

        model_name = "+".join(ordered_models) if len(ordered_models) > 1 else (
            ordered_models[0] if ordered_models else "archie"
        )
        model_counts = {
            model_name: int(np.count_nonzero(model_per_sample == model_name))
            for model_name in ordered_models
        }
        return sw, model_name, model_counts, fallback_counts

    def _select_model_scalar(
        self,
        vcl_value: float,
        salinity: Optional[float],
        lithology: str = "clastic",
        cec: Optional[float] = None,
        b_coeff: Optional[float] = None,
        rwb: Optional[float] = None,
    ) -> SwModel:
        """Select saturation model for a single depth sample."""
        if self.model != "auto":
            return SwModel(self.model)

        if not np.isfinite(vcl_value):
            vcl_value = 0.0

        if vcl_value < self.vcl_cutoff:
            return SwModel.ARCHIE

        lithology = lithology.lower()

        if lithology in _CARBONATE_LITHOLOGIES and (rwb is not None or self.rwb_fallback is not None):
            return SwModel.DUAL_WATER

        # Clastic, and carbonate without the Rwb that Dual Water needs.
        if cec is not None and b_coeff is not None:
            return SwModel.WAXMAN_SMITS

        if salinity is not None and salinity >= 20000:
            return SwModel.SIMANDOUX
        return SwModel.INDONESIA

    def _run_model(
        self, model: SwModel,
        rt, rw, rcl, vcl, phie, phit,
        cec, b_coeff, rho_ma, m_star, rwb, phi_cl, rtsh, vsh_exp, shale_exp,
        a_value=None, m_value=None, n_value=None,
        clip_counts: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """Run a specific Sw model.

        ``clip_counts``, when given, gets the number of Waxman-Smits samples
        whose effective-basis SW was clipped to 0 under the model's name.
        """
        a_value = self.a if a_value is None else a_value
        m_value = self.m if m_value is None else m_value
        n_value = self.n if n_value is None else n_value
        if model == SwModel.ARCHIE:
            return archie(rt, rw, phie, a_value, m_value, n_value)

        elif model == SwModel.NORTH_SEA:
            return north_sea_shaly_sand(
                rt, rw, phie, vcl, rtsh if rtsh is not None else rcl,
                a_value, m_value, n_value, vsh_exp
            )

        elif model == SwModel.NIGERIA_SCHLUMBERGER:
            return nigeria_schlumberger(
                rt, rw, phie, vcl, rtsh if rtsh is not None else rcl,
                a_value, m_value, n_value, shale_exp
            )

        elif model == SwModel.SIMANDOUX:
            return simandoux(rt, rw, phie, vcl, rcl, a_value, m_value, n_value)

        elif model == SwModel.INDONESIA:
            return indonesia(rt, rw, phie, vcl, rcl, a_value, m_value, n_value)

        elif model == SwModel.DUAL_WATER:
            if rwb is None:
                if self.rwb_fallback is None:
                    raise ValueError(
                        "Dual Water needs the bound-water resistivity rwb (ohm.m), from lab "
                        "data or the user; it is not estimated"
                    )
                rwb = rw * self.rwb_fallback
            return dual_water(
                phit, rt, rw, rwb, vcl, phi_cl,
                a_value, m_value, n_value
            )

        elif model == SwModel.WAXMAN_SMITS:
            sw_total = waxman_smits(
                phit, rt, rw, cec, rho_ma, b_coeff,
                m_star, n_value
            )
            if self.effective_waxman_smits:
                if clip_counts is not None:
                    below = _below_bound_water(sw_total, phit, phie)
                    clip_counts[model.value] = clip_counts.get(model.value, 0) + int(np.count_nonzero(below))
                return _effective_from_total_sw(sw_total, phit, phie)
            return sw_total

        else:
            return archie(rt, rw, phie, a_value, m_value, n_value)

    def _select_model(
        self,
        vcl: np.ndarray,
        salinity: Optional[float],
        lithology: str = "clastic",
        cec: Optional[float] = None,
        b_coeff: Optional[float] = None,
        rwb: Optional[float] = None,
    ) -> SwModel:
        """
        Select saturation model based on formation properties.

        Decision logic (from design doc flowchart):
        - VCL < cutoff -> Archie (clean)
        - VCL >= cutoff -> Shaly:
          - Carbonate with an Rwb -> Dual Water
          - Clastic, or carbonate without an Rwb:
            - CEC and B_COEFF -> Waxman-Smits
            - salinity >= 20 kppm -> Simandoux
            - salinity < 20 kppm -> Indonesia
        """
        if self.model != "auto":
            return SwModel(self.model)

        mean_vcl = float(np.nanmean(vcl))
        return self._select_model_scalar(mean_vcl, salinity, lithology, cec, b_coeff, rwb)

    def _estimate_rcl(self, rt: np.ndarray, vcl: np.ndarray) -> float:
        """Estimate clay resistivity from shale intervals."""
        shale_mask = vcl > 0.9
        if np.any(shale_mask):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                val = float(np.nanmean(rt[shale_mask]))
            # Shale samples may have no finite RT; a NaN Rcl would poison
            # every shaly-routed Sw in the well, so fall through to the default.
            if np.isfinite(val):
                return val
        return 5.0  # Default


def _below_bound_water(swt, phit, phie) -> np.ndarray:
    """True where Swt is below the bound-water fraction ``(PHIT - PHIE) / PHIT``.

    There the effective-basis Sw of :func:`_effective_from_total_sw` would be
    negative and is clipped to 0.
    """
    swt = np.asarray(swt, dtype=float)
    phit = np.asarray(phit, dtype=float)
    phie = np.asarray(phie, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        swe = 1.0 - (phit / phie) * (1.0 - swt)
    return np.isfinite(swe) & (phie > 0.0) & (swe < 0.0)


def _effective_from_total_sw(swt, phit, phie) -> np.ndarray:
    """Effective-porosity Sw from a total-porosity Sw.

    ``Swe = 1 - (PHIT / PHIE) * (1 - Swt)`` (Crain's Petrophysical Handbook):
    the clay-bound water fills ``PHIT - PHIE``, so the hydrocarbon pore volume
    ``PHIE * (1 - Swe) = PHIT * (1 - Swt)`` is the same on both bases wherever
    Swt is at least the bound-water fraction ``(PHIT - PHIE) / PHIT``. Where
    Swt is below it, the hydrocarbon volume would exceed PHIE and Swe is
    clipped to 0 (see :func:`_below_bound_water`). Where PHIE is zero or
    negative there is no effective pore space and Swe is 1; a missing Swt
    (e.g. no deep resistivity) stays missing.
    """
    swt = np.asarray(swt, dtype=float)
    phit = np.asarray(phit, dtype=float)
    phie = np.asarray(phie, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        swe = 1.0 - (phit / phie) * (1.0 - swt)
    no_effective_pores = np.isfinite(swt) & np.isfinite(phie) & (phie <= 0.0)
    swe = np.where(no_effective_pores, 1.0, swe)
    return np.clip(swe, 0.0, 1.0)


def calculate_sw(
    well: "Well",
    rw: float,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    model: str = "auto",
    **kwargs
) -> np.ndarray:
    """
    Calculate water saturation.

    Convenience function for SaturationModule.

    Parameters
    ----------
    well : Well
        Well object
    rw : float
        Formation water resistivity
    a, m, n : float
        Archie parameters
    model : str
        Saturation model
    **kwargs
        Additional arguments passed to SaturationModule.calculate()

    Returns
    -------
    np.ndarray
        Water saturation

    Notes
    -----
    Dual Water needs the bound-water resistivity ``rwb=`` (ohm.m), which is
    not estimated: ``model="dual_water"`` without it raises ``ValueError``,
    and auto routing gives shaly carbonate samples without it the clastic
    routing, with a ``UserWarning`` that counts them.
    """
    module = SaturationModule(a=a, m=m, n=n, model=model)
    result = module.calculate(well, rw, **kwargs)
    for route, count in (result.fallback_counts or {}).items():
        warnings.warn(
            f"{count} shaly carbonate samples used {route.split('->', 1)[1]} instead of "
            "Dual Water, which needs rwb= (it is not estimated)",
            UserWarning,
            stacklevel=2,
        )
    return result.sw
