"""
Saturation and fluid-distribution analysis.
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
from petrophys.evaluation.diagnostics.bands import Bands
from petrophys.evaluation.mhi.mhi_module import calculate_mhi
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from petrophys.model.well import Well


class SaturationAnalysis:
    """Saturation, fluid-contact, and gas-indicator analysis."""

    CATEGORY_KEY = "SATURATION_AND_FLUID_DISTRIBUTION"
    # The one-curve classes of the sections below (Bands: base label, then each step in order).
    WATER_SATURATION_BANDS = Bands("WATER_BEARING", (("<", 0.60, "FAIR_HC_SATURATION"), ("<", 0.40, "GOOD_HC_SATURATION"),
                                                     ("<", 0.20, "EXCELLENT_HC_SATURATION")))
    BVW_BANDS = Bands("UNDETERMINED", (("<", 0.10, "FAIR_MOVEABLE_HC"), ("<", 0.06, "GOOD_MOVEABLE_HC"),
                                       ("<", 0.03, "EXCELLENT_MOVEABLE_HC")))

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
    ) -> CategoryAnalysisResult:
        n_samples = well.n_samples
        sw, sw_source = get_curve_data(well, ["SWA", "SW", "SWT-LOG"])
        sxo, sxo_source = get_curve_data(well, ["SWXOA", "SXO", "SXO-LOG"])
        if sxo is None:
            sxo = self._first_sxo_curve(well)
            if sxo is not None:
                sxo_source = "SXO-*"
        phit, phit_source = get_curve_data(well, ["PHIE_CC", "PHIT"])
        phie, phie_source = get_curve_data(well, ["PHIE_CC", "PHIE", "PHIT"])
        bvw, bvw_source = get_curve_data(well, ["BVW"])
        mhi, mhi_source = get_curve_data(well, ["MHI"])
        rxo, rxo_source = get_curve_data(well, ["RXO", "RSHA", "RESS"])
        rdep, rdep_source = get_curve_data(well, ["RDEP", "RT"])
        phid, _ = get_curve_data(well, ["PHID"])
        phin, _ = get_curve_data(well, ["PHIN", "NPHIC", "NPHI_HC", "NPHI"])
        drho, _ = get_curve_data(well, ["DRHO"])
        dtc, _ = get_curve_data(well, ["DTC", "DT"])
        sw_source_text = format_curve_source(sw_source)
        phie_source_text = format_curve_source(phie_source)
        phit_source_text = format_curve_source(phit_source)

        derived_curves: dict[str, DerivedCurve] = {}

        if bvw is None and phie is not None and sw is not None:
            bvw = phie * sw
            derived_curves["BVW"] = DerivedCurve(bvw, "v/v", "Bulk Volume Water")

        if mhi is None and sw is not None and sxo is not None:
            mhi = calculate_mhi(sxo, sw)
            derived_curves["MHI"] = DerivedCurve(mhi, "", "Moveable Hydrocarbon Index")

        water_sat_labels = np.full(n_samples, "NO_SATURATION_DATA", dtype=object)
        if sw is not None:
            water_sat_labels = self.WATER_SATURATION_BANDS.classify(sw, missing="NO_SATURATION_DATA")

        bvw_labels = np.full(n_samples, "NO_BVW_DATA", dtype=object)
        if bvw is not None:
            bvw_labels = self.BVW_BANDS.classify(bvw, missing="NO_BVW_DATA")

        invasion_ratio = np.full(n_samples, np.nan, dtype=float)
        movability_labels = np.full(n_samples, "NO_INVASION_DATA", dtype=object)
        if mhi is not None:
            mhi_classes = MhiClassification.classify(mhi)
            movability_labels[:] = "MINIMAL_MOVEABLE_HC"
            movability_labels[mhi_classes == MhiClassification.LOW_MOVEABLE_HC] = "LOW_MOVEABLE_HC"
            movability_labels[mhi_classes == MhiClassification.MODERATE_MOVEABLE_HC] = "MODERATE_MOVEABLE_HC"
            movability_labels[mhi_classes == MhiClassification.HIGH_MOVEABLE_HC] = "HIGH_MOVEABLE_HC"
        elif rxo is not None and rdep is not None:
            valid = np.isfinite(rxo) & np.isfinite(rdep) & (rdep > 0.1)
            invasion_ratio[valid] = rxo[valid] / rdep[valid]
            movability_labels[valid] = "MINIMAL_INVASION"
            movability_labels[valid & (invasion_ratio < 0.7)] = "CONDUCTIVE_INVASION"
            movability_labels[valid & (invasion_ratio > 1.3)] = "RESISTIVE_INVASION"
            derived_curves["SEPARATION_RATIO"] = DerivedCurve(
                invasion_ratio,
                "",
                "Invasion separation ratio RXO / RDEP",
            )

        contact_labels = np.full(n_samples, "NO_CONTACT_INDICATORS", dtype=object)
        if sw is not None and phit is not None and n_samples > 0:
            valid = np.isfinite(sw) & np.isfinite(phit)
            prev_sw = np.roll(sw, 1)
            prev_phi = np.roll(phit, 1)
            sw_change = sw - prev_sw
            phit_change = np.abs(phit - prev_phi)

            stable_hc = valid & (sw < 0.4)
            stable_water = valid & (sw > 0.6)
            contact_labels[stable_hc] = "STABLE_HC_ZONE"
            contact_labels[stable_water] = "STABLE_WATER_ZONE"
            contact_labels[valid & (sw_change > 0.1) & (phit_change < 0.05)] = "TRANSITION_ZONE"
            contact_labels[valid & (sw_change > 0.2) & (phit_change < 0.05)] = "POTENTIAL_CONTACT_ZONE"

            if valid[0]:
                if sw[0] < 0.4:
                    contact_labels[0] = "STABLE_HC_ZONE"
                elif sw[0] > 0.6:
                    contact_labels[0] = "STABLE_WATER_ZONE"
                else:
                    contact_labels[0] = "TRANSITION_ZONE"

        nd_separation = np.full(n_samples, np.nan, dtype=float)
        phit_gas_corr = np.full(n_samples, np.nan, dtype=float)
        gas_labels = np.full(n_samples, "NO_GAS_INDICATORS", dtype=object)
        if phid is not None and phin is not None:
            valid = np.isfinite(phid) & np.isfinite(phin)
            nd_separation[valid] = phid[valid] - phin[valid]
            gas_labels[valid] = "NO_GAS_EFFECT"
            gas_labels[valid & (nd_separation >= 0.03)] = "WEAK_GAS_EFFECT"
            gas_labels[valid & (nd_separation >= 0.08)] = "MODERATE_GAS_EFFECT"
            gas_labels[valid & (nd_separation > 0.15)] = "STRONG_GAS_EFFECT"
            derived_curves["ND_SEPARATION"] = DerivedCurve(
                nd_separation,
                "v/v",
                "Neutron-density separation for gas indication",
            )

            phit_gas_corr[valid] = np.clip(
                np.sqrt((phin[valid] ** 2 + phid[valid] ** 2) / 2.0),
                0.0,
                0.45,
            )
            derived_curves["PHIT_GAS_CORR"] = DerivedCurve(
                phit_gas_corr,
                "v/v",
                "Gas-corrected total porosity proxy",
            )
        elif drho is not None:
            valid = np.isfinite(drho) & (drho < -0.05)
            gas_labels[valid] = "DRHO_GAS_EFFECT"
        elif dtc is not None and sw is not None:
            valid = np.isfinite(dtc) & np.isfinite(sw) & (dtc > 120.0) & (sw < 0.5)
            gas_labels[valid] = "SONIC_GAS_EFFECT"

        if phid is not None and np.any(np.isfinite(phit_gas_corr)):
            gas_corr_effect = np.isfinite(phid) & np.isfinite(phit_gas_corr) & ((phid - phit_gas_corr) > 0.03)
            fallback_mask = gas_labels == "NO_GAS_INDICATORS"
            gas_labels[fallback_mask & gas_corr_effect] = "GAS_CORRECTION_EFFECT"
        elif drho is not None:
            fallback_mask = gas_labels == "NO_GAS_INDICATORS"
            gas_labels[fallback_mask & np.isfinite(drho) & (drho < -0.05)] = "DRHO_GAS_EFFECT"
        elif dtc is not None and sw is not None:
            fallback_mask = gas_labels == "NO_GAS_INDICATORS"
            gas_labels[fallback_mask & np.isfinite(dtc) & np.isfinite(sw) & (dtc > 120.0) & (sw < 0.5)] = "SONIC_GAS_EFFECT"

        integrated_labels = np.full(n_samples, "UNDETERMINED", dtype=object)
        integrated_labels[water_sat_labels == "WATER_BEARING"] = "WATER_ZONE"
        integrated_labels[water_sat_labels == "FAIR_HC_SATURATION"] = "MARGINAL_HC_ZONE"
        integrated_labels[np.isin(water_sat_labels, ["GOOD_HC_SATURATION", "EXCELLENT_HC_SATURATION"])] = "GOOD_HC_ZONE"
        integrated_labels[
            (water_sat_labels == "EXCELLENT_HC_SATURATION")
            & np.isin(bvw_labels, ["EXCELLENT_MOVEABLE_HC", "GOOD_MOVEABLE_HC"])
        ] = "EXCELLENT_HC_ZONE"
        integrated_labels[gas_labels == "MODERATE_GAS_EFFECT"] = "POSSIBLE_GAS_ZONE"
        integrated_labels[gas_labels == "STRONG_GAS_EFFECT"] = "GAS_ZONE"

        result = CategoryAnalysisResult(
            category_key=self.CATEGORY_KEY,
            sections={
                "water_saturation_estimation_analysis": build_section(
                    well,
                    water_sat_labels,
                    description="Hydrocarbon saturation classes from water saturation.",
                    methodology=f"Uses {sw_source_text}; priority is SWA, then SW, SWT-LOG.",
                    calculation_thresholds=(
                        "EXCELLENT < 0.20; GOOD 0.20-0.40; FAIR 0.40-0.60; WATER_BEARING >= 0.60."
                    ),
                ),
                "bulk_volume_water_porosity_integration": build_section(
                    well,
                    bvw_labels,
                    description="Moveable hydrocarbon interpretation from BVW.",
                    methodology=(
                        f"Uses {format_curve_source(bvw_source, 'computed BVW')} when present; otherwise BVW = "
                        f"selected effective porosity ({phie_source_text}) * water saturation ({sw_source_text})."
                    ),
                    calculation_thresholds=(
                        "EXCELLENT < 0.03; GOOD 0.03-0.06; FAIR 0.06-0.10; "
                        "UNDETERMINED >= 0.10."
                    ),
                ),
                "hydrocarbon_movability_invasion_characteristics": build_section(
                    well,
                    movability_labels,
                    description="Movability from MHI or invasion fallback.",
                    methodology=(
                        f"Prefers MHI={format_curve_source(mhi_source)}. If MHI is unavailable, uses "
                        f"invasion ratio {format_curve_source(rxo_source)} / {format_curve_source(rdep_source)}."
                    ),
                    calculation_thresholds=(
                        "MHI bands: HIGH_MOVEABLE_HC <0.5, MODERATE 0.5-0.7, LOW 0.7-0.9, MINIMAL >=0.9. "
                        "Fallback invasion bands: conductive <0.7, minimal 0.7-1.3, resistive >1.3."
                    ),
                ),
                "fluid_contact_detection_indicators": build_section(
                    well,
                    contact_labels,
                    description="Fluid-contact and transition-zone indicators.",
                    methodology=(
                        f"Compares per-sample changes in SW ({sw_source_text}) and selected porosity "
                        f"({phit_source_text}) relative to the previous depth."
                    ),
                    calculation_thresholds=(
                        "POTENTIAL_CONTACT_ZONE: dSw > 0.2 and dPHIT < 0.05; "
                        "TRANSITION_ZONE: dSw > 0.1 and dPHIT < 0.05; "
                        "STABLE_HC_ZONE: SW < 0.4; STABLE_WATER_ZONE: SW > 0.6."
                    ),
                ),
                "gas_indicator_analysis_methods": build_section(
                    well,
                    gas_labels,
                    description="Gas-indicator hierarchy from neutron-density, DRHO, sonic, and gas correction.",
                    methodology="Primary method is ND separation; DRHO, sonic, and PHIT_GAS_CORR are fallbacks.",
                    calculation_thresholds=(
                        "STRONG_GAS_EFFECT > 0.15; MODERATE 0.08-0.15; WEAK 0.03-0.08; "
                        "DRHO_GAS_EFFECT if DRHO < -0.05; SONIC_GAS_EFFECT if DTC > 120 and Sw < 0.5; "
                        "GAS_CORRECTION_EFFECT if PHID - PHIT_GAS_CORR > 0.03."
                    ),
                ),
                "integrated_saturation_interpretation": build_section(
                    well,
                    integrated_labels,
                    description="Integrated fluid-distribution interpretation.",
                    methodology="Combines water saturation, BVW, and gas-indicator sections.",
                    calculation_thresholds=(
                        "GAS_ZONE, POSSIBLE_GAS_ZONE, EXCELLENT_HC_ZONE, GOOD_HC_ZONE, "
                        "MARGINAL_HC_ZONE, WATER_ZONE, otherwise UNDETERMINED."
                    ),
                ),
            },
            derived_curves=derived_curves,
        )

        if add_curves:
            result.add_curves_to_well(well)

        return result

    @staticmethod
    def _first_sxo_curve(well: "Well") -> np.ndarray | None:
        for name in well.curves:
            if name.upper().startswith("SXO-"):
                curve = well.get(name)
                if curve is not None:
                    return np.asarray(curve.data, dtype=float)
        return None
