"""
Hydrocarbon analysis classifications and derived indicators.
"""

from __future__ import annotations

import numpy as np

from petrophys._internal.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    format_curve_source,
    get_curve_data,
)
from petrophys.model.classifications import MhiClassification
from petrophys.evaluation.net_pay.net_pay_module import NetPayModule
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from petrophys.model.well import Well


class HydrocarbonAnalysis:
    """Integrated hydrocarbon classification analysis."""

    CATEGORY_KEY = "HYDROCARBON_ANALYSIS"

    def __init__(self, porosity_curves=None):
        #: Porosity curve preference, in order. The default prefers the
        #: core-corrected PHIE_CC and falls back to PHIT.
        self.porosity_curves = list(porosity_curves) if porosity_curves else ["PHIE_CC", "PHIE", "PHIT"]

    CROSSOVER_ND_SEPARATION = 0.05
    CROSSOVER_VCL_MAX = 0.35
    CROSSOVER_PHIE_MIN = 0.05
    CROSSOVER_BAD_HOLE_MAX = 0.5

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
    ) -> CategoryAnalysisResult:
        n_samples = well.n_samples
        sw, _ = get_curve_data(well, ["SWA", "SW", "SWT-LOG"])
        mhi, _ = get_curve_data(well, ["MHI"])
        vcl, vcl_source = get_curve_data(well, ["VCL", "VCLAY", "VSH"])
        phie, phie_source = get_curve_data(well, self.porosity_curves)
        phie_effective, phie_effective_source = get_curve_data(
            well, [name for name in self.porosity_curves if name != "PHIT"] or ["PHIE"]
        )
        phid, _ = get_curve_data(well, ["PHID"])
        phin, _ = get_curve_data(well, ["PHIN", "NPHIC", "NPHI_HC", "NPHI"])
        bad_hole, bad_hole_source = get_curve_data(well, ["BAD_HOLE", "BAD_HOLE_FLAG"])
        dtc, _ = get_curve_data(well, ["DTC", "DT"])
        dts, _ = get_curve_data(well, ["DTS"])
        perm, perm_source = get_curve_data(well, ["PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"])
        phie_source_text = format_curve_source(phie_source)
        perm_source_text = format_curve_source(perm_source)

        pay_flag = self._resolve_pay_flag(well, n_samples)

        crossover_flag = np.full(n_samples, np.nan, dtype=float)
        if phid is not None and phin is not None and vcl is not None and phie_effective is not None and bad_hole is not None:
            valid = (
                np.isfinite(phid)
                & np.isfinite(phin)
                & np.isfinite(vcl)
                & np.isfinite(phie_effective)
                & np.isfinite(bad_hole)
            )
            gas_crossover = (
                (phid - phin > self.CROSSOVER_ND_SEPARATION)
                & (vcl < self.CROSSOVER_VCL_MAX)
                & (phie_effective > self.CROSSOVER_PHIE_MIN)
                & (bad_hole < self.CROSSOVER_BAD_HOLE_MAX)
            )
            crossover_flag[valid] = gas_crossover[valid].astype(float)

        vp_vs_ratio = np.full(n_samples, np.nan, dtype=float)
        if dts is not None and dtc is not None:
            valid = (
                np.isfinite(dts)
                & np.isfinite(dtc)
                & (dtc >= 20.0)
                & (dtc <= 300.0)
                & (dts >= 20.0)
                & (dts <= 600.0)
            )
            ratio = np.full(n_samples, np.nan, dtype=float)
            ratio[valid] = dts[valid] / dtc[valid]
            plausible = valid & (ratio >= 0.5) & (ratio <= 5.0)
            vp_vs_ratio[plausible] = ratio[plausible]

        petro_labels = np.full(n_samples, "NO_INDICATION", dtype=object)
        if sw is not None:
            low_sw = np.isfinite(sw) & (sw < 0.50)
            petro_labels[low_sw] = "LOW_SW"

        if sw is not None and mhi is not None:
            mhi_classes = MhiClassification.classify(mhi)
            movable = np.isfinite(sw) & (sw < 0.60) & (
                (mhi_classes == MhiClassification.HIGH_MOVEABLE_HC)
                | (mhi_classes == MhiClassification.MODERATE_MOVEABLE_HC)
            )
            petro_labels[movable] = "MOVABLE_HC"

        petro_labels[np.isfinite(pay_flag) & (pay_flag > 0.5)] = "POTENTIAL_PAY"

        geophysical_labels = np.full(n_samples, "NO_INDICATION", dtype=object)
        if np.any(np.isfinite(vp_vs_ratio) & (vp_vs_ratio < 1.80)):
            geophysical_labels[np.isfinite(vp_vs_ratio) & (vp_vs_ratio < 1.80)] = "LOW_VPVS"
        if np.any(np.isfinite(crossover_flag) & (crossover_flag > 0.5)):
            geophysical_labels[np.isfinite(crossover_flag) & (crossover_flag > 0.5)] = "GAS_INDICATION"
        strong_gas = (
            np.isfinite(crossover_flag)
            & (crossover_flag > 0.5)
            & np.isfinite(vp_vs_ratio)
            & (vp_vs_ratio < 1.75)
        )
        geophysical_labels[strong_gas] = "STRONG_GAS_INDICATION"

        reservoir_labels = np.full(n_samples, "UNDETERMINED", dtype=object)
        if phie is not None and perm is not None:
            valid = np.isfinite(phie) & np.isfinite(perm)
            reservoir_labels[valid] = "NON_RESERVOIR"
            reservoir_labels[valid & (perm > 0.1) & (phie > 0.08)] = "FAIR_RESERVOIR"
            reservoir_labels[valid & (perm > 1.0) & (phie > 0.10)] = "GOOD_RESERVOIR"
            reservoir_labels[valid & (perm > 10.0) & (phie > 0.15)] = "EXCELLENT_RESERVOIR"

        integrated_labels = np.full(n_samples, "MARGINAL", dtype=object)
        gas_show = np.char.find(geophysical_labels.astype(str), "GAS") >= 0
        integrated_labels[gas_show] = "GAS_SHOW"
        integrated_labels[reservoir_labels == "NON_RESERVOIR"] = "NON_RESERVOIR"
        integrated_labels[petro_labels == "POTENTIAL_PAY"] = "POTENTIAL_PAY"
        integrated_labels[
            (petro_labels == "POTENTIAL_PAY")
            & (reservoir_labels == "EXCELLENT_RESERVOIR")
        ] = "PRIME_OIL_TARGET"
        integrated_labels[
            (geophysical_labels == "STRONG_GAS_INDICATION")
            & np.isin(reservoir_labels, ["EXCELLENT_RESERVOIR", "GOOD_RESERVOIR"])
        ] = "PRIME_GAS_TARGET"
        prime_gas_target = np.full(n_samples, np.nan, dtype=float)
        if (
            np.any(np.isfinite(crossover_flag))
            and np.any(np.isfinite(vp_vs_ratio))
            and phie is not None
            and perm is not None
        ):
            prime_gas_target = self._indicator_curve(integrated_labels, "PRIME_GAS_TARGET")

        prime_oil_target = np.full(n_samples, np.nan, dtype=float)
        if np.any(np.isfinite(pay_flag)) and phie is not None and perm is not None:
            prime_oil_target = self._indicator_curve(integrated_labels, "PRIME_OIL_TARGET")

        result = CategoryAnalysisResult(
            category_key=self.CATEGORY_KEY,
            sections={
                "petrophysical_hydrocarbon_indicators": build_section(
                    well,
                    petro_labels,
                    description=(
                        "Petrophysical assessment of hydrocarbon presence using water saturation, "
                        "moveable hydrocarbon index, and pay-flag behavior from the current workflow."
                    ),
                    methodology=(
                        "Uses SW as the primary saturation indicator, existing MHI when available, "
                        "and NET_PAY as the workflow pay flag. MHI is not synthesized in this analysis."
                    ),
                    calculation_thresholds=(
                        "POTENTIAL_PAY: existing pay flag > 0.5. "
                        "MOVABLE_HC: MHI < 0.7 and SW < 0.60. "
                        "LOW_SW: SW < 0.50. NO_INDICATION: no petrophysical hydrocarbon criterion met."
                    ),
                ),
                "geophysical_hydrocarbon_indicators": build_section(
                    well,
                    geophysical_labels,
                    description=(
                        "Geophysical indicators of hydrocarbon presence using neutron-density crossover "
                        "and Vp/Vs-style screening from available sonic curves."
                    ),
                    methodology=(
                        "CROSSOVER_FLAG is calculated from PHID - PHIN > 0.05, VCL < 0.35, "
                        f"selected effective porosity > 0.05, and BAD_HOLE < 0.5 using VCL={format_curve_source(vcl_source)}, "
                        f"effective porosity={format_curve_source(phie_effective_source)}, and "
                        f"bad-hole={format_curve_source(bad_hole_source)}. VP_VS_RATIO is calculated only when both DTS and DTC "
                        "are present in plausible slowness ranges; no artificial estimate is added "
                        "when shear sonic is missing or implausible."
                    ),
                    calculation_thresholds=(
                        "STRONG_GAS_INDICATION: CROSSOVER_FLAG = 1 and VP_VS_RATIO < 1.75. "
                        "GAS_INDICATION: CROSSOVER_FLAG = 1. "
                        "LOW_VPVS: VP_VS_RATIO < 1.80. "
                        "NO_INDICATION: no significant geophysical gas indicator."
                    ),
                ),
                "reservoir_quality_for_hydrocarbons": build_section(
                    well,
                    reservoir_labels,
                    description=(
                        "Reservoir quality assessment in the context of hydrocarbon production potential "
                        "using effective porosity and permeability thresholds."
                    ),
                    methodology=(
                        f"Uses effective porosity={phie_source_text} and permeability={perm_source_text}. "
                        "Priorities are PHIE_CC, PHIE, PHIT for porosity and PERM_CC, PERM, PERM_CORE, PERM_TIMUR for permeability."
                    ),
                    calculation_thresholds=(
                        "EXCELLENT_RESERVOIR: selected permeability > 10 mD and selected effective porosity > 0.15. "
                        "GOOD_RESERVOIR: selected permeability > 1 mD and selected effective porosity > 0.10. "
                        "FAIR_RESERVOIR: selected permeability > 0.1 mD and selected effective porosity > 0.08. "
                        "NON_RESERVOIR: below fair-reservoir cutoffs."
                    ),
                ),
                "integrated_hydrocarbon_potential": build_section(
                    well,
                    integrated_labels,
                    description=(
                        "Combined hydrocarbon potential assessment integrating petrophysical indicators, "
                        "geophysical signals, and reservoir quality in a simple hierarchy."
                    ),
                    methodology=(
                        "Starts from gas-show, non-reservoir, and pay labels, then promotes prime oil "
                        "and prime gas target flags where their stricter reservoir criteria are met."
                    ),
                    calculation_thresholds=(
                        "PRIME_GAS_TARGET: STRONG_GAS_INDICATION with GOOD/EXCELLENT_RESERVOIR. "
                        "PRIME_OIL_TARGET: POTENTIAL_PAY with EXCELLENT_RESERVOIR. "
                        "POTENTIAL_PAY: pay-flag positive but not prime oil. "
                        "GAS_SHOW: any gas indication short of PRIME_GAS_TARGET. "
                        "NON_RESERVOIR: explicitly poor reservoir quality. "
                        "MARGINAL: all other combinations."
                    ),
                ),
            },
            derived_curves={
                "CROSSOVER_FLAG": DerivedCurve(
                    crossover_flag,
                    "",
                    "Neutron-density crossover flag",
                ),
                "VP_VS_RATIO": DerivedCurve(
                    vp_vs_ratio,
                    "",
                    "Vp/Vs ratio proxy from DTS / DTC",
                ),
                "PRIME_GAS_TARGET": DerivedCurve(
                    prime_gas_target,
                    "",
                    "Integrated prime gas target flag",
                ),
                "PRIME_OIL_TARGET": DerivedCurve(
                    prime_oil_target,
                    "",
                    "Integrated prime oil target flag",
                ),
            },
        )

        if add_curves:
            result.add_curves_to_well(well)

        return result

    @staticmethod
    def _resolve_pay_flag(well: "Well", n_samples: int) -> np.ndarray:
        existing, _ = get_curve_data(well, ["NET_PAY"])
        if existing is not None:
            return np.where(np.isfinite(existing), (existing > 0.5).astype(float), np.nan)

        try:
            flags = NetPayModule().calculate(well)
            return flags.net_pay.astype(float)
        except Exception:
            return np.full(n_samples, np.nan, dtype=float)

    @staticmethod
    def _indicator_curve(labels: np.ndarray, target: str) -> np.ndarray:
        label_array = np.asarray(labels, dtype=object)
        return (label_array == target).astype(float)
