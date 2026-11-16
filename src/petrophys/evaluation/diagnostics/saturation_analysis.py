"""
Saturation and fluid-distribution analysis.
"""

from __future__ import annotations

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys._internal.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    format_curve_source,
    get_curve_data,
)
from petrophys.evaluation.diagnostics._tables import class_table_set, rxo_rt_ratio
from petrophys.evaluation.diagnostics.bands import Bands
from petrophys.evaluation.mhi.mhi_module import MhiModule, calculate_mhi
from petrophys.model.class_tables import CLASS_TABLES
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from petrophys.model.well import Well

# Water-saturation classes of a stable hydrocarbon zone, under the "current"
# class tables.
_STABLE_HC_CLASSES = ("EXCELLENT_HC_SATURATION", "GOOD_HC_SATURATION")
# BVW classes of an excellent hydrocarbon zone, under the "current" class tables.
_EXCELLENT_ZONE_BVW_CLASSES = ("VERY_LOW_BVW", "LOW_BVW")
#: Effective porosity (v/v) below which a sample's BVW is not graded
#: (NO_EFFECTIVE_POROSITY), under the "current" class tables.
MIN_BVW_POROSITY = 0.01


class SaturationAnalysis:
    """Saturation, fluid-contact, and gas-indicator analysis.

    Parameters
    ----------
    class_tables : {"current", "geolink"}, default "current"
        ``"current"`` grades water saturation, BVW, MHI and Rxo/Rt with the
        canonical tables of :data:`~petrophys.model.class_tables.CLASS_TABLES`
        (a sample without MHI is NO_MHI_DATA; the invasion fallback classifies
        the workflow's INVASION_RATIO when the well has one) and places the
        stable contact zones on the water-saturation classes. ``"geolink"``
        keeps the GeoLink eval package's classes and texts: the class
        attributes below, with every sample without MHI labelled
        MINIMAL_MOVEABLE_HC as the package was.
    """

    CATEGORY_KEY = "SATURATION_AND_FLUID_DISTRIBUTION"
    # The "geolink" profile's one-curve classes of the sections below (Bands: base label,
    # then each step in order). An analyzer with the "current" class tables holds
    # the canonical tables under the same names.
    WATER_SATURATION_BANDS = Bands("WATER_BEARING", (("<", 0.60, "FAIR_HC_SATURATION"), ("<", 0.40, "GOOD_HC_SATURATION"),
                                                     ("<", 0.20, "EXCELLENT_HC_SATURATION")))
    BVW_BANDS = Bands("UNDETERMINED", (("<", 0.10, "FAIR_MOVEABLE_HC"), ("<", 0.06, "GOOD_MOVEABLE_HC"),
                                       ("<", 0.03, "EXCELLENT_MOVEABLE_HC")))
    MHI_BANDS = Bands("HIGH_MOVEABLE_HC", ((">=", 0.5, "MODERATE_MOVEABLE_HC"), (">=", 0.7, "LOW_MOVEABLE_HC"),
                                           (">=", 0.9, "MINIMAL_MOVEABLE_HC")))
    # Fallback of the movability section without MHI, on RXO / RDEP where RDEP > 0.1.
    INVASION_BANDS = Bands("MINIMAL_INVASION", (("<", 0.7, "CONDUCTIVE_INVASION"), (">", 1.3, "RESISTIVE_INVASION")))

    def __init__(self, class_tables: str = "current"):
        self.class_tables = class_table_set(class_tables)
        if self.class_tables == "current":
            self.WATER_SATURATION_BANDS = CLASS_TABLES["SW"].bands
            self.BVW_BANDS = CLASS_TABLES["BVW"].bands
            self.MHI_BANDS = CLASS_TABLES["MHI"].bands
            self.INVASION_BANDS = CLASS_TABLES["RXO_RT"].bands

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
    ) -> CategoryAnalysisResult:
        """Classify each sample's saturation and fluid indicators and summarize them by interval.

        Parameters
        ----------
        well : Well
            Interpreted well. Water saturation (SW), flushed-zone saturation
            (SXO), porosity, BVW, MHI, the flushed-zone and deep
            resistivities, the density and neutron porosities, DRHO and the
            compressional slowness are read where present; a class that
            needs an absent curve is undetermined.
        add_curves : bool, default True
            Add the derived curves that hold values to ``well``, replacing
            curves of the same name: ``ND_SEPARATION`` (v/v) and
            ``PHIT_GAS_CORR`` (v/v), computed where the density and neutron
            porosities exist.

        Returns
        -------
        CategoryAnalysisResult
            Sections ``water_saturation_estimation_analysis``,
            ``bulk_volume_water_porosity_integration``,
            ``hydrocarbon_movability_invasion_characteristics``,
            ``fluid_contact_detection_indicators``,
            ``gas_indicator_analysis_methods`` and
            ``integrated_saturation_interpretation``, each with its
            description, method, thresholds and the depth intervals of every
            class, and the derived curves in ``derived_curves``.
        """
        benchmark = self.class_tables == "geolink"
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
            if not (benchmark or reproducing_benchmark()):
                # As MhiModule: no MHI where SXO lacks invaded-zone support (e.g.
                # the SW^(1/5) quick-look SXO, which makes MHI a function of SW alone).
                mhi = np.where(MhiModule.diagnostic_sxo_mask(well, sxo=sxo, sw=sw), mhi, np.nan)
                if not np.any(np.isfinite(mhi)):
                    mhi = None
            if mhi is not None:
                derived_curves["MHI"] = DerivedCurve(mhi, "", "Moveable Hydrocarbon Index")

        water_sat_labels = np.full(n_samples, "NO_SATURATION_DATA", dtype=object)
        if sw is not None:
            water_sat_labels = self.WATER_SATURATION_BANDS.classify(sw, missing="NO_SATURATION_DATA")

        bvw_labels = np.full(n_samples, "NO_BVW_DATA", dtype=object)
        if bvw is not None:
            bvw_labels = self.BVW_BANDS.classify(bvw, missing="NO_BVW_DATA")
            if phie is not None and not (benchmark or reproducing_benchmark()):
                # BVW grades the water held in pore space; rock without effective
                # porosity (shale, tight streaks) holds none to grade.
                with np.errstate(invalid="ignore"):
                    no_pores = np.isfinite(phie) & (phie < MIN_BVW_POROSITY)
                bvw_labels = np.where(no_pores, "NO_EFFECTIVE_POROSITY", bvw_labels).astype(object)

        invasion_ratio = np.full(n_samples, np.nan, dtype=float)
        movability_labels = np.full(n_samples, "NO_INVASION_DATA", dtype=object)
        movability_source_text = (
            f"invasion ratio {format_curve_source(rxo_source)} / {format_curve_source(rdep_source)}"
        )
        if mhi is not None:
            if benchmark:
                # As the package: a sample without MHI counts as MINIMAL_MOVEABLE_HC.
                movability_labels = self.MHI_BANDS.classify(mhi, ~np.isnan(mhi), missing="MINIMAL_MOVEABLE_HC")
            else:
                movability_labels = self.MHI_BANDS.classify(mhi, missing=CLASS_TABLES["MHI"].missing)
        elif benchmark:
            if rxo is not None and rdep is not None:
                valid = np.isfinite(rxo) & np.isfinite(rdep) & (rdep > 0.1)
                invasion_ratio[valid] = rxo[valid] / rdep[valid]
                movability_labels = self.INVASION_BANDS.classify(invasion_ratio, valid, missing="NO_INVASION_DATA")
                derived_curves["SEPARATION_RATIO"] = DerivedCurve(
                    invasion_ratio,
                    "",
                    "Invasion separation ratio RXO / RDEP",
                )
        else:
            ratio, ratio_source_text = rxo_rt_ratio(well)
            movability_source_text = f"Rxo/Rt ratio {ratio_source_text}"
            if ratio is not None:
                invasion_ratio = ratio
                movability_labels = self.INVASION_BANDS.classify(ratio, missing=CLASS_TABLES["RXO_RT"].missing)
                derived_curves["SEPARATION_RATIO"] = DerivedCurve(
                    invasion_ratio,
                    "",
                    f"Invasion separation ratio ({ratio_source_text})",
                )

        contact_labels = np.full(n_samples, "NO_CONTACT_INDICATORS", dtype=object)
        if sw is not None and phit is not None and n_samples > 0:
            valid = np.isfinite(sw) & np.isfinite(phit)
            prev_sw = np.roll(sw, 1)
            prev_phi = np.roll(phit, 1)
            sw_change = sw - prev_sw
            phit_change = np.abs(phit - prev_phi)

            if benchmark:
                hydrocarbon = sw < 0.4
                water = sw > 0.6
            else:
                hydrocarbon = np.isin(water_sat_labels, _STABLE_HC_CLASSES)
                water = water_sat_labels == "WATER_BEARING"
            stable_hc = valid & hydrocarbon
            stable_water = valid & water
            contact_labels[stable_hc] = "STABLE_HC_ZONE"
            contact_labels[stable_water] = "STABLE_WATER_ZONE"
            contact_labels[valid & (sw_change > 0.1) & (phit_change < 0.05)] = "TRANSITION_ZONE"
            contact_labels[valid & (sw_change > 0.2) & (phit_change < 0.05)] = "POTENTIAL_CONTACT_ZONE"

            if valid[0]:
                if hydrocarbon[0]:
                    contact_labels[0] = "STABLE_HC_ZONE"
                elif water[0]:
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

        excellent_zone_bvw = (
            ["EXCELLENT_MOVEABLE_HC", "GOOD_MOVEABLE_HC"] if benchmark else list(_EXCELLENT_ZONE_BVW_CLASSES)
        )
        integrated_labels = np.full(n_samples, "UNDETERMINED", dtype=object)
        integrated_labels[water_sat_labels == "WATER_BEARING"] = "WATER_ZONE"
        integrated_labels[water_sat_labels == "FAIR_HC_SATURATION"] = "MARGINAL_HC_ZONE"
        integrated_labels[np.isin(water_sat_labels, ["GOOD_HC_SATURATION", "EXCELLENT_HC_SATURATION"])] = "GOOD_HC_ZONE"
        integrated_labels[
            (water_sat_labels == "EXCELLENT_HC_SATURATION")
            & np.isin(bvw_labels, excellent_zone_bvw)
        ] = "EXCELLENT_HC_ZONE"
        integrated_labels[gas_labels == "MODERATE_GAS_EFFECT"] = "POSSIBLE_GAS_ZONE"
        integrated_labels[gas_labels == "STRONG_GAS_EFFECT"] = "GAS_ZONE"

        if benchmark:
            texts = {
                "water": "EXCELLENT < 0.20; GOOD 0.20-0.40; FAIR 0.40-0.60; WATER_BEARING >= 0.60.",
                "bvw_description": "Moveable hydrocarbon interpretation from BVW.",
                "bvw": "EXCELLENT < 0.03; GOOD 0.03-0.06; FAIR 0.06-0.10; UNDETERMINED >= 0.10.",
                "movability_methodology": (
                    f"Prefers MHI={format_curve_source(mhi_source)}. If MHI is unavailable, uses "
                    f"{movability_source_text}."
                ),
                "movability": (
                    "MHI bands: HIGH_MOVEABLE_HC <0.5, MODERATE 0.5-0.7, LOW 0.7-0.9, MINIMAL >=0.9. "
                    "Fallback invasion bands: conductive <0.7, minimal 0.7-1.3, resistive >1.3."
                ),
                "contacts": (
                    "POTENTIAL_CONTACT_ZONE: dSw > 0.2 and dPHIT < 0.05; "
                    "TRANSITION_ZONE: dSw > 0.1 and dPHIT < 0.05; "
                    "STABLE_HC_ZONE: SW < 0.4; STABLE_WATER_ZONE: SW > 0.6."
                ),
                "integrated": (
                    "GAS_ZONE, POSSIBLE_GAS_ZONE, EXCELLENT_HC_ZONE, GOOD_HC_ZONE, "
                    "MARGINAL_HC_ZONE, WATER_ZONE, otherwise UNDETERMINED."
                ),
            }
        else:
            sw_ranges = {label: (lower, closed) for label, lower, closed, _, _ in CLASS_TABLES["SW"].bands.ranges()}
            sw_upper = {label: (upper, closed) for label, _, _, upper, closed in CLASS_TABLES["SW"].bands.ranges()}
            hc_limit, hc_closed = sw_upper[_STABLE_HC_CLASSES[-1]]
            water_limit, water_closed = sw_ranges["WATER_BEARING"]
            texts = {
                "water": CLASS_TABLES["SW"].describe(),
                "bvw_description": "Bulk volume water (effective porosity x water saturation) classes.",
                "bvw": CLASS_TABLES["BVW"].describe() + (
                    "" if reproducing_benchmark() else
                    f" NO_EFFECTIVE_POROSITY where effective porosity < {MIN_BVW_POROSITY:g} "
                    "(no pore water to grade)."
                ),
                "movability_methodology": (
                    f"Prefers MHI={format_curve_source(mhi_source)}. If MHI is unavailable, uses the "
                    f"{movability_source_text}."
                ),
                "movability": (
                    f"MHI classes: {CLASS_TABLES['MHI'].describe()} "
                    f"Without MHI, Rxo/Rt classes: {CLASS_TABLES['RXO_RT'].describe()}"
                ),
                "contacts": (
                    "POTENTIAL_CONTACT_ZONE: dSw > 0.2 and dPHIT < 0.05; "
                    "TRANSITION_ZONE: dSw > 0.1 and dPHIT < 0.05; "
                    f"STABLE_HC_ZONE: SW class {' or '.join(_STABLE_HC_CLASSES)} "
                    f"(SW {'<=' if hc_closed else '<'} {hc_limit:g}); "
                    f"STABLE_WATER_ZONE: SW class WATER_BEARING (SW {'>=' if water_closed else '>'} {water_limit:g})."
                ),
                "integrated": (
                    "GAS_ZONE, POSSIBLE_GAS_ZONE, EXCELLENT_HC_ZONE (EXCELLENT_HC_SATURATION with "
                    f"{' or '.join(_EXCELLENT_ZONE_BVW_CLASSES)}), GOOD_HC_ZONE, "
                    "MARGINAL_HC_ZONE, WATER_ZONE, otherwise UNDETERMINED."
                ),
            }

        result = CategoryAnalysisResult(
            category_key=self.CATEGORY_KEY,
            sections={
                "water_saturation_estimation_analysis": build_section(
                    well,
                    water_sat_labels,
                    description="Hydrocarbon saturation classes from water saturation.",
                    methodology=f"Uses {sw_source_text}; priority is SWA, then SW, SWT-LOG.",
                    calculation_thresholds=texts["water"],
                ),
                "bulk_volume_water_porosity_integration": build_section(
                    well,
                    bvw_labels,
                    description=texts["bvw_description"],
                    methodology=(
                        f"Uses {format_curve_source(bvw_source, 'computed BVW')} when present; otherwise BVW = "
                        f"selected effective porosity ({phie_source_text}) * water saturation ({sw_source_text})."
                    ),
                    calculation_thresholds=texts["bvw"],
                ),
                "hydrocarbon_movability_invasion_characteristics": build_section(
                    well,
                    movability_labels,
                    description="Movability from MHI or invasion fallback.",
                    methodology=texts["movability_methodology"],
                    calculation_thresholds=texts["movability"],
                ),
                "fluid_contact_detection_indicators": build_section(
                    well,
                    contact_labels,
                    description="Fluid-contact and transition-zone indicators.",
                    methodology=(
                        f"Compares per-sample changes in SW ({sw_source_text}) and selected porosity "
                        f"({phit_source_text}) relative to the previous depth."
                    ),
                    calculation_thresholds=texts["contacts"],
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
                    calculation_thresholds=texts["integrated"],
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
