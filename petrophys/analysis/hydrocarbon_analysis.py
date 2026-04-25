"""
Hydrocarbon analysis classifications and derived indicators.
"""

from __future__ import annotations

import numpy as np

from petrophys.utils.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    get_curve_data,
)
from petrophys.base.classifications import MhiClassification
from petrophys.reservoir.net_pay.net_pay_module import NetPayModule


class HydrocarbonAnalysis:
    """Integrated hydrocarbon classification analysis."""

    CATEGORY_KEY = "HYDROCARBON_ANALYSIS"
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
        vcl, _ = get_curve_data(well, ["VCL", "VCLAY", "VSH"])
        phie, _ = get_curve_data(well, ["PHIE", "PHIT"])
        phie_effective, _ = get_curve_data(well, ["PHIE"])
        phid, _ = get_curve_data(well, ["PHID"])
        phin, _ = get_curve_data(well, ["PHIN", "NPHIC", "NPHI_HC", "NPHI"])
        bad_hole, _ = get_curve_data(well, ["BAD_HOLE", "BADHOLEFLAG", "BAD_HOLE_FLAG"])
        dtc, _ = get_curve_data(well, ["DTC", "DT"])
        dts, _ = get_curve_data(well, ["DTS"])
        perm, _ = get_curve_data(well, ["PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"])

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
                        "PHIE > 0.05, and BAD_HOLE < 0.5 using the current porosity, clay, and "
                        "hole-quality curves. VP_VS_RATIO is calculated only when both DTS and DTC "
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
                        "Uses PHIE as the porosity basis and permeability as the flow-capacity basis, "
                        "preferring PERM_CC when a core-calibrated permeability curve is available and "
                        "otherwise using PERM. This keeps empirical permeability as the default export "
                        "while still ranking reservoir quality with the best available permeability input."
                    ),
                    calculation_thresholds=(
                        "EXCELLENT_RESERVOIR: PERM > 10 mD and PHIE > 0.15. "
                        "GOOD_RESERVOIR: PERM > 1 mD and PHIE > 0.10. "
                        "FAIR_RESERVOIR: PERM > 0.1 mD and PHIE > 0.08. "
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
                    self._indicator_curve(integrated_labels, "PRIME_GAS_TARGET"),
                    "",
                    "Integrated prime gas target flag",
                ),
                "PRIME_OIL_TARGET": DerivedCurve(
                    self._indicator_curve(integrated_labels, "PRIME_OIL_TARGET"),
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
