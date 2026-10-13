"""One registry of class tables: the canonical classes of each property, the model enums and helpers that
classify through it, and the diagnostics analyzers, whose "current" class tables give a property the same class in
every category while the "geolink" set keeps the GeoLink eval package's tables and texts."""
from types import SimpleNamespace

import numpy as np
import pytest

from petrophys import Well
from petrophys.evaluation.diagnostics import (
    Bands,
    HydrocarbonAnalysis,
    LithologyAnalysis,
    PermeabilityAnalysis,
    PorosityAnalysis,
    SaturationAnalysis,
    bands_of,
)
from petrophys.evaluation.net_pay.cutoffs import STANDARD_CUTOFFS
from petrophys.exceptions import DataQualityWarning
from petrophys.interpretation.clay_volume.classification import get_vcl_quality_code, is_clean_sand, is_shale
from petrophys.interpretation.electrical.invasion.invasion_classification import (
    InvasionType,
    get_invasion_type,
    interpret_invasion,
)
from petrophys.interpretation.electrical.invasion.invasion_module import InvasionModule
from petrophys.interpretation.porosity.classification import classify_porosity
from petrophys.interpretation.saturation.classification import classify_sw
from petrophys.model import classifications
from petrophys.evaluation.diagnostics._tables import lithology_family, sample_families
from petrophys.model.class_tables import CLASS_TABLES, ClassTable, ReservoirGrade, ReservoirQualityTable
from petrophys.model.classifications import (
    HcSaturationClassification,
    InvasionClassification,
    MhiClassification,
    PermeabilityClassification,
    PorosityClassification,
    RQIClassification,
    VclClassification,
)
from petrophys.workflow import DynamicWorkflowConfig, run_comprehensive_analysis

NAN = float("nan")


# -- The registry ------------------------------------------------------------------------------------------------

def test_one_table_per_property_with_unit_source_and_missing_label():
    assert list(CLASS_TABLES) == ["VCL", "PHI", "PERM", "SW", "BVW", "RQI", "FZI", "MHI", "RXO_RT", "RESERVOIR_QUALITY"]
    units = {"VCL": "v/v", "PHI": "v/v", "PERM": "mD", "SW": "v/v", "BVW": "v/v", "RQI": "um", "FZI": "um",
             "MHI": "", "RXO_RT": ""}
    for name, table in CLASS_TABLES.items():
        assert table.name == name and table.source
        assert table.missing.startswith("NO_") and table.missing.endswith("_DATA")
        assert table.missing not in table.labels                     # a missing value never gets a class
        if isinstance(table, ClassTable):
            assert table.unit == units[name]
            assert list(table.classify([NAN, np.inf])) == [table.missing, table.missing]
    quality = CLASS_TABLES["RESERVOIR_QUALITY"]
    assert isinstance(quality, ReservoirQualityTable) and quality.units == ("v/v", "mD", "v/v")
    assert quality.families == ("clastic", "carbonate")


@pytest.mark.parametrize("name, values, labels", [
    ("VCL", [0.2499, 0.25, 0.5, 0.5001],
     ["CLEAN_FORMATION", "MODERATE_CLAY", "MODERATE_CLAY", "SHALY_FORMATION"]),
    ("PHI", [0.06, 0.0601, 0.12, 0.1201, 0.20, 0.2001],
     ["POOR_POROSITY", "FAIR_POROSITY", "FAIR_POROSITY", "GOOD_POROSITY", "GOOD_POROSITY", "EXCELLENT_POROSITY"]),
    ("PERM", [0.1, 0.1001, 1.0, 1.0001, 10.0, 100.0, 1000.0, 1000.1],
     ["VERY_POOR_PERMEABILITY", "POOR_PERMEABILITY", "POOR_PERMEABILITY", "FAIR_PERMEABILITY", "FAIR_PERMEABILITY",
      "GOOD_PERMEABILITY", "VERY_GOOD_PERMEABILITY", "EXCELLENT_PERMEABILITY"]),
    ("SW", [0.1999, 0.2, 0.3999, 0.4, 0.5999, 0.6, 1.02],
     ["EXCELLENT_HC_SATURATION", "GOOD_HC_SATURATION", "GOOD_HC_SATURATION", "FAIR_HC_SATURATION",
      "FAIR_HC_SATURATION", "WATER_BEARING", "WATER_BEARING"]),
    ("BVW", [0.0299, 0.03, 0.0599, 0.06, 0.0999, 0.1],
     ["VERY_LOW_BVW", "LOW_BVW", "LOW_BVW", "MODERATE_BVW", "MODERATE_BVW", "HIGH_BVW"]),
    ("RQI", [0.4999, 0.5, 1.0, 1.0001], ["LOW_RQI", "MODERATE_RQI", "MODERATE_RQI", "HIGH_RQI"]),
    ("FZI", [0.09, 0.0938, 0.1875, 0.5, 2.0, 47.9, 48.0],
     ["BELOW_GHE_1", "GHE_1", "GHE_2", "GHE_3", "GHE_5", "GHE_9", "GHE_10"]),
    ("MHI", [0.4999, 0.5, 0.6999, 0.7, 0.8999, 0.9, 1.4],
     ["HIGH_MOVEABLE_HC", "MODERATE_MOVEABLE_HC", "MODERATE_MOVEABLE_HC", "LOW_MOVEABLE_HC", "LOW_MOVEABLE_HC",
      "MINIMAL_MOVEABLE_HC", "MINIMAL_MOVEABLE_HC"]),
    ("RXO_RT", [0.7999, 0.8, 1.2, 1.2001],
     ["CONDUCTIVE_INVASION", "MINIMAL_INVASION", "MINIMAL_INVASION", "RESISTIVE_INVASION"]),
])
def test_canonical_edges_and_sides(name, values, labels):
    assert list(CLASS_TABLES[name].classify(values)) == labels


def test_reservoir_quality_floor_is_each_familys_net_reservoir_rock():
    quality = CLASS_TABLES["RESERVOIR_QUALITY"]
    fair, good, excellent = quality.grades
    for family, cutoffs in STANDARD_CUTOFFS.items():
        # The floor is net reservoir rock as net pay counts it: PHIE > phie_cutoff and VCL < vcl_cutoff.
        assert fair.limits(family) == (cutoffs["phie_cutoff"], 0.1, cutoffs["vcl_cutoff"])
        assert good.limits(family) == (0.10, 1.0, cutoffs["vcl_cutoff"])
        assert excellent.limits(family) == (0.15, 10.0, 0.25)
    # Both net-rock clay cutoffs lie inside MODERATE_CLAY, and EXCELLENT needs CLEAN_FORMATION.
    assert CLASS_TABLES["VCL"].label_of(0.40) == CLASS_TABLES["VCL"].label_of(0.30) == "MODERATE_CLAY"
    assert CLASS_TABLES["VCL"].label_of(0.2499) == "CLEAN_FORMATION" != CLASS_TABLES["VCL"].label_of(0.25)

    #           PHIE    PERM     VCL     family       grade
    cases = [(0.08, 5.0, 0.10, "clastic", "NON_RESERVOIR"),            # at the clastic porosity cutoff
             (0.0801, 0.1, 0.10, "clastic", "NON_RESERVOIR"),          # at the 0.1 mD tight limit
             (0.0801, 0.1001, 0.3999, "clastic", "FAIR_RESERVOIR"),
             (0.0801, 0.1001, 0.40, "clastic", "NON_RESERVOIR"),       # at the clastic net-rock clay cutoff
             (0.06, 5.0, 0.10, "clastic", "NON_RESERVOIR"),
             (0.06, 5.0, 0.10, "carbonate", "FAIR_RESERVOIR"),         # carbonate net reservoir: PHIE > 0.045
             (0.045, 5.0, 0.10, "carbonate", "NON_RESERVOIR"),
             (0.0451, 0.1001, 0.2999, "carbonate", "FAIR_RESERVOIR"),
             (0.20, 50.0, 0.30, "carbonate", "NON_RESERVOIR"),         # at the carbonate net-rock clay cutoff
             (0.20, 50.0, 0.30, "clastic", "GOOD_RESERVOIR"),          # moderate clay caps the grade at GOOD
             (0.1001, 1.0001, 0.3999, "clastic", "GOOD_RESERVOIR"),
             (0.10, 1.0001, 0.10, "clastic", "FAIR_RESERVOIR"),
             (0.1501, 10.0001, 0.2499, "carbonate", "EXCELLENT_RESERVOIR"),
             (0.1501, 10.0001, 0.25, "carbonate", "GOOD_RESERVOIR"),   # VCL 0.25 is MODERATE_CLAY
             (0.30, 2000.0, 0.45, "clastic", "NON_RESERVOIR"),         # shaly rock is never graded
             (0.02, 5.0, NAN, "clastic", "NO_QUALITY_DATA"),           # no clay volume: no grade, not even NON
             (NAN, 5.0, 0.10, "carbonate", "NO_QUALITY_DATA")]
    phie, perm, vcl, families, grades = (list(column) for column in zip(*cases))
    assert list(quality.classify(phie, perm, vcl, families)) == grades
    assert list(quality.classify(0.06, 5.0, 0.1, "carbonate").ravel()) == ["FAIR_RESERVOIR"]
    with pytest.raises(ValueError, match="unknown lithology family"):
        quality.classify([0.2], [5.0], [0.1], ["limestone"])
    with pytest.raises(ValueError, match="must imply"):
        ReservoirQualityTable("Q", ("a", "b", "c"), ("", "", ""), "LOW",
                              (ReservoirGrade("MID", 0.1, 1.0, 0.25), ReservoirGrade("HIGH", 0.2, 10.0, 0.4)),
                              "NO_Q_DATA", ("clastic",), "test")


def test_threshold_texts_come_from_the_tables():
    assert CLASS_TABLES["VCL"].describe() == (
        "CLEAN_FORMATION < 0.25; MODERATE_CLAY >= 0.25 and <= 0.5; SHALY_FORMATION > 0.5 (VCL, v/v); "
        "NO_VCL_DATA where VCL is missing."
    )
    tail = (" NON_RESERVOIR: below FAIR_RESERVOIR; NO_QUALITY_DATA where effective porosity, permeability or "
            "clay volume is missing.")
    carbonate = (
        "Carbonate rock: EXCELLENT_RESERVOIR: effective porosity > 0.15 v/v, permeability > 10 mD and clay volume "
        "< 0.25 v/v; GOOD_RESERVOIR: effective porosity > 0.1 v/v, permeability > 1 mD and clay volume < 0.3 v/v; "
        "FAIR_RESERVOIR: effective porosity > 0.045 v/v, permeability > 0.1 mD and clay volume < 0.3 v/v."
    )
    clastic = (
        "Clastic rock: EXCELLENT_RESERVOIR: effective porosity > 0.15 v/v, permeability > 10 mD and clay volume "
        "< 0.25 v/v; GOOD_RESERVOIR: effective porosity > 0.1 v/v, permeability > 1 mD and clay volume < 0.4 v/v; "
        "FAIR_RESERVOIR: effective porosity > 0.08 v/v, permeability > 0.1 mD and clay volume < 0.4 v/v."
    )
    quality = CLASS_TABLES["RESERVOIR_QUALITY"]
    assert quality.describe() == f"{clastic} {carbonate}{tail}"
    assert quality.describe(["carbonate"]) == carbonate + tail


def test_bands_ranges_and_coarsening():
    perm = CLASS_TABLES["PERM"].bands
    assert perm.ranges()[:2] == (("VERY_POOR_PERMEABILITY", -np.inf, False, 0.1, True),
                                 ("POOR_PERMEABILITY", 0.1, False, 1.0, True))
    merged = perm.without_edges(0.1)
    assert merged.labels == ("POOR_PERMEABILITY", "FAIR_PERMEABILITY", "GOOD_PERMEABILITY",
                             "VERY_GOOD_PERMEABILITY", "EXCELLENT_PERMEABILITY")
    assert merged.label_of(0.05) == "POOR_PERMEABILITY"
    with pytest.raises(ValueError, match="no step at"):
        perm.without_edges(0.5)
    with pytest.raises(ValueError, match="more than one range"):
        Bands("A", ((">", 1.0, "B"), (">", 2.0, "A"))).ranges()


# -- The model enums and helpers -------------------------------------------------------------------------------------

ENUM_TABLES = [
    (VclClassification, "VCL", classifications._VCL_MEMBERS),
    (InvasionClassification, "RXO_RT", classifications._INVASION_MEMBERS),
    (PorosityClassification, "PHI", classifications._POROSITY_MEMBERS),
    (HcSaturationClassification, "SW", classifications._SATURATION_MEMBERS),
    (PermeabilityClassification, "PERM", classifications._PERMEABILITY_MEMBERS),
    (RQIClassification, "RQI", classifications._RQI_MEMBERS),
    (MhiClassification, "MHI", classifications._MHI_MEMBERS),
]


@pytest.mark.parametrize("enum, name, members", ENUM_TABLES)
def test_enums_classify_through_the_registry(enum, name, members):
    table = CLASS_TABLES[name]
    assert set(members) == set(table.labels) and set(members.values()) == set(enum)
    values = sorted({e + d for e in table.bands.edges for d in (-0.01, 0.0, 0.01)})
    expected = [members[label] for label in table.classify(values)]
    assert [enum.classify(v) for v in values] == expected                      # scalar path
    assert list(enum.classify(np.array(values))) == expected                   # array path
    assert list(enum.classify([NAN, values[0]])) == [None, expected[0]]        # NaN gives None, lists work
    assert enum.classify(NAN) is None


def test_saturation_and_porosity_take_explicit_units():
    # A fraction just above 1 is water-bearing (it was read as 1.02 %: excellent hydrocarbon saturation).
    assert classify_sw(1.02) is HcSaturationClassification.WATER_BEARING
    assert list(classify_sw(np.array([0.95, 1.0, 1.02, 1.3]))) == [HcSaturationClassification.WATER_BEARING] * 4
    assert classify_sw(35.0, percent=True) is HcSaturationClassification.GOOD_HC
    # 0.8 % porosity is poor once the unit is given (it was read as 0.8 v/v: excellent).
    assert list(classify_porosity(np.array([25.0, 0.8]), percent=True)) == [
        PorosityClassification.EXCELLENT, PorosityClassification.POOR]
    assert classify_porosity(0.15) is PorosityClassification.GOOD
    with pytest.warns(DataQualityWarning, match="percent=True"):
        assert classify_porosity(np.array([25.0, 0.18]))[1] is PorosityClassification.GOOD
    with pytest.warns(DataQualityWarning, match="percent=True"):
        classify_sw(np.array([35.0, 80.0]))


def test_vcl_helpers():
    assert list(get_vcl_quality_code(np.array([0.1, 0.25, 0.5, 0.6, NAN]))) == [1, 2, 2, 3, 0]
    assert get_vcl_quality_code(NAN) == 0 and get_vcl_quality_code(0.6) == 3   # NaN was 1 in arrays, 3 as a scalar
    assert list(is_clean_sand(np.array([0.2499, 0.25, NAN]))) == [True, False, False]
    assert list(is_shale(np.array([0.5, 0.5001, NAN]))) == [False, True, False]
    assert is_clean_sand(0.3, threshold=0.35) and not is_shale(0.6, threshold=0.7)


def test_invasion_helpers_and_module_statistics():
    assert get_invasion_type(NAN) is None                                       # NaN was MINIMAL
    assert interpret_invasion(NAN).startswith("No valid Rxo/Rt ratio")
    assert [get_invasion_type(v) for v in (0.7999, 0.8, 1.2, 1.2001)] == [
        InvasionType.CONDUCTIVE, InvasionType.MINIMAL, InvasionType.MINIMAL, InvasionType.RESISTIVE]

    well = Well("INVASION")
    well.depth = np.arange(5.0)
    well.add_curve("RDEP", np.full(5, 10.0), "ohm.m", "Deep")
    well.add_curve("RSHA", np.array([7.9, 8.0, 12.0, 12.1, NAN]), "ohm.m", "Shallow")
    stats = InvasionModule().analyze(well).statistics
    assert [stats[k] for k in ("conductive_percent", "minimal_percent", "resistive_percent")] == [25.0, 50.0, 25.0]
    # The statistics and classes follow the module's table.
    wide = Bands("MINIMAL_INVASION", (("<", 0.7, "CONDUCTIVE_INVASION"), (">", 1.3, "RESISTIVE_INVASION")))
    result = InvasionModule(class_bands=wide).analyze(well)
    assert result.statistics["minimal_percent"] == 100.0
    assert list(result.classification) == [InvasionClassification.MINIMAL_INVASION] * 4 + [None]


# -- The analyzers ---------------------------------------------------------------------------------------------------

def _well(mhi=True):
    """Six sandstone samples; each column is hand-picked to land on a class edge or a former contradiction."""
    well = Well("CLASS-TABLES")
    well.depth = np.arange(1000.0, 1003.0, 0.5)
    well.depth_unit = "m"
    phie = np.array([0.07, 0.20, 0.12, 0.05, 0.16, 0.30])
    sw = np.array([0.55, 0.10, 0.60, 0.90, 0.30, 0.50])
    well.add_curve("VCL", np.array([0.22, 0.45, 0.10, 0.60, 0.30, 0.05]), "v/v", "Clay volume")
    well.add_curve("PHIE", phie, "v/v", "Effective porosity")
    well.add_curve("PHIT", phie.copy(), "v/v", "Total porosity")
    well.add_curve("PERM", np.array([5.0, 50.0, 2.0, 0.05, 20.0, 2000.0]), "mD", "Permeability")
    well.add_curve("SW", sw, "v/v", "Water saturation")
    if mhi:
        well.add_curve("MHI", np.array([0.95, NAN, 0.60, 0.95, 0.45, 0.75]), "", "MHI")
    well.add_curve("RDEP", np.full(6, 10.0), "ohm.m", "Deep resistivity")
    well.add_curve("RXO", np.array([7.5, 5.0, 12.5, 7.9, NAN, 12.1]), "ohm.m", "Flushed-zone resistivity")
    well.add_curve("INVASION_RATIO", np.array([0.75, 1.0, 1.25, 0.79, NAN, 1.21]), "", "Workflow Rxo/Rt")
    well.add_curve("FZI", np.array([0.5, 2.0, 0.09, 48.0, 3.0, 12.0]), "um", "Flow zone indicator")
    well.add_curve("RQI", np.array([0.3, 0.8, 0.2, 0.1, 0.5, 2.0]), "um", "Reservoir quality index")
    well.add_curve("NET_PAY", np.zeros(6), "", "Net pay")
    well.add_curve("LITHOLOGY_GEOLINK", np.ones(6), "", "GeoLink lithology")
    return well


def _labels(result, section):
    return [str(label) for label in result.sections[section].raw_labels]


def _analyses(class_tables, well=None):
    well = well or _well()
    return {
        "porosity": PorosityAnalysis(class_tables=class_tables).analyze(well, add_curves=False),
        "permeability": PermeabilityAnalysis(class_tables=class_tables).analyze(well, add_curves=False),
        "saturation": SaturationAnalysis(class_tables=class_tables).analyze(well, add_curves=False),
        "hydrocarbon": HydrocarbonAnalysis(class_tables=class_tables).analyze(well, add_curves=False),
        "lithology": LithologyAnalysis(class_tables=class_tables).analyze(well, add_curves=False),
    }


NON, FAIR, GOOD, EXC = "NON_RESERVOIR", "FAIR_RESERVOIR", "GOOD_RESERVOIR", "EXCELLENT_RESERVOIR"


def test_current_tables_give_a_property_one_class_in_every_category():
    a = _analyses("current")
    clean, moderate, shaly = "CLEAN_FORMATION", "MODERATE_CLAY", "SHALY_FORMATION"
    vcl = [clean, moderate, clean, shaly, moderate, clean]                     # VCL 0.22 is clean in both
    assert _labels(a["porosity"], "clay_content_impact_on_porosity_variation") == vcl
    assert _labels(a["permeability"], "permeability_variation_controls") == vcl

    # One meaning of each class. Clastic rock: the second sample (PHIE 0.20, 50 mD) has VCL 0.45, above the
    # 0.40 net-rock cutoff, so it is no reservoir; the fifth (PHIE 0.16, 20 mD) has VCL 0.30, moderate clay,
    # so it is GOOD, not EXCELLENT.
    quality = [NON, NON, GOOD, NON, GOOD, EXC]
    assert _labels(a["porosity"], "porosity_quality_assessment_characteristics") == quality
    assert _labels(a["porosity"], "integrated_porosity_reservoir_assessment") == quality
    assert _labels(a["permeability"], "overall_permeability_interpretation") == quality
    assert _labels(a["hydrocarbon"], "reservoir_quality_for_hydrocarbons") == quality
    assert _labels(a["lithology"], "overall_interpretation") == quality
    assert _labels(a["lithology"], "lithology_petrophysics_relationships") == [
        "TIGHT_SANDSTONE", "TIGHT_SANDSTONE", "GOOD_SANDSTONE_RESERVOIR", "TIGHT_SANDSTONE",
        "GOOD_SANDSTONE_RESERVOIR", "EXCELLENT_SANDSTONE_RESERVOIR"]
    thresholds = {a[category].sections[section].calculation_thresholds for category, section in (
        ("permeability", "overall_permeability_interpretation"), ("hydrocarbon", "reservoir_quality_for_hydrocarbons"))}
    assert thresholds == {CLASS_TABLES["RESERVOIR_QUALITY"].describe(["clastic"])}

    # The workflow's INVASION_RATIO (1.0 at the second sample, where RXO/RDEP is 0.5) and 0.8/1.2 throughout.
    assert _labels(a["permeability"], "invasion_profile_permeability_indicators") == [
        "CONDUCTIVE_INVASION", "MINIMAL_INVASION", "RESISTIVE_INVASION", "CONDUCTIVE_INVASION",
        "NO_INVASION_DATA", "RESISTIVE_INVASION"]
    no_mhi = SaturationAnalysis().analyze(_well(mhi=False), add_curves=False)
    assert _labels(no_mhi, "hydrocarbon_movability_invasion_characteristics") == _labels(
        a["permeability"], "invasion_profile_permeability_indicators")

    assert _labels(a["permeability"], "flow_unit_classification_system") == [
        "GHE_3", "GHE_5", "BELOW_GHE_1", "GHE_10", "GHE_6", "GHE_8"]
    assert _labels(a["permeability"], "permeability_distribution_analysis") == [
        "FAIR_PERMEABILITY", "GOOD_PERMEABILITY", "FAIR_PERMEABILITY", "POOR_PERMEABILITY", "GOOD_PERMEABILITY",
        "EXCELLENT_PERMEABILITY"]

    # Sw 0.55 is a hydrocarbon class in saturation, so LOW_SW in hydrocarbon; Sw 0.60 is water-bearing in both.
    assert _labels(a["saturation"], "water_saturation_estimation_analysis") == [
        "FAIR_HC_SATURATION", "EXCELLENT_HC_SATURATION", "WATER_BEARING", "WATER_BEARING", "GOOD_HC_SATURATION",
        "FAIR_HC_SATURATION"]
    assert _labels(a["hydrocarbon"], "petrophysical_hydrocarbon_indicators") == [
        "LOW_SW", "LOW_SW", "NO_INDICATION", "NO_INDICATION", "MOVABLE_HC", "LOW_SW"]
    assert _labels(a["saturation"], "fluid_contact_detection_indicators") == [
        "TRANSITION_ZONE", "STABLE_HC_ZONE", "STABLE_WATER_ZONE", "STABLE_WATER_ZONE", "STABLE_HC_ZONE",
        "NO_CONTACT_INDICATORS"]

    # A sample without MHI is missing, not MINIMAL_MOVEABLE_HC.
    assert _labels(a["saturation"], "hydrocarbon_movability_invasion_characteristics") == [
        "MINIMAL_MOVEABLE_HC", "NO_MHI_DATA", "MODERATE_MOVEABLE_HC", "MINIMAL_MOVEABLE_HC", "HIGH_MOVEABLE_HC",
        "LOW_MOVEABLE_HC"]
    # BVW = PHIE * SW = 0.0385, 0.02, 0.072, 0.045, 0.048, 0.15: classes that describe BVW.
    assert _labels(a["saturation"], "bulk_volume_water_porosity_integration") == [
        "LOW_BVW", "VERY_LOW_BVW", "MODERATE_BVW", "LOW_BVW", "LOW_BVW", "HIGH_BVW"]
    assert a["permeability"].metadata["mhi_confirmation"] == {
        "HIGH_MOVEABLE_HC": 1, "MODERATE_MOVEABLE_HC": 1, "LOW_MOVEABLE_HC": 1, "MINIMAL_MOVEABLE_HC": 2}

    variation = a["permeability"].sections["permeability_variation_controls"]
    assert variation.calculation_thresholds == CLASS_TABLES["VCL"].describe()


def test_benchmark_tables_keep_the_eval_package_classes_and_texts():
    a = _analyses("geolink")
    assert _labels(a["permeability"], "permeability_variation_controls") == [
        "MIXED_LITHOLOGY", "MIXED_LITHOLOGY", "CLEAN_RESERVOIR", "SHALE_DOMINATED", "MIXED_LITHOLOGY",
        "CLEAN_RESERVOIR"]
    assert _labels(a["porosity"], "clay_content_impact_on_porosity_variation") == [
        "CLEAN_FORMATION", "MODERATE_CLAY_IMPACT", "CLEAN_FORMATION", "SHALY_FORMATION", "MODERATE_CLAY_IMPACT",
        "CLEAN_FORMATION"]
    # Four meanings of a reservoir class at the second sample (PHIE 0.20, 50 mD, VCL 0.45).
    assert _labels(a["porosity"], "integrated_porosity_reservoir_assessment")[1] == "GOOD_RESERVOIR"
    assert _labels(a["hydrocarbon"], "reservoir_quality_for_hydrocarbons")[1] == "EXCELLENT_RESERVOIR"
    assert _labels(a["lithology"], "overall_interpretation")[1] == "TIGHT_FORMATION"
    assert _labels(a["permeability"], "overall_permeability_interpretation") == [
        "MARGINAL_RESERVOIR", "GOOD_RESERVOIR", "MARGINAL_RESERVOIR", "TIGHT_FORMATION", "GOOD_RESERVOIR",
        "EXCELLENT_RESERVOIR"]
    assert _labels(a["porosity"], "porosity_quality_assessment_characteristics")[0] == "FAIR_FLOW_UNIT"
    assert _labels(a["permeability"], "flow_unit_classification_system") == [
        "MESO_PORES", "MACRO_PORES", "MICRO_PORES", "MEGA_PORES", "MACRO_PORES", "MEGA_PORES"]
    # RXO / RDEP, not INVASION_RATIO: 0.5 is conductive at the second sample.
    assert _labels(a["permeability"], "invasion_profile_permeability_indicators")[:2] == [
        "CONDUCTIVE_INVASION", "CONDUCTIVE_INVASION"]
    no_mhi = SaturationAnalysis(class_tables="geolink").analyze(_well(mhi=False), add_curves=False)
    assert _labels(no_mhi, "hydrocarbon_movability_invasion_characteristics") == [   # 0.7 / 1.3 fallback
        "MINIMAL_INVASION", "CONDUCTIVE_INVASION", "MINIMAL_INVASION", "MINIMAL_INVASION", "NO_INVASION_DATA",
        "MINIMAL_INVASION"]
    assert _labels(a["saturation"], "hydrocarbon_movability_invasion_characteristics")[1] == "MINIMAL_MOVEABLE_HC"
    assert _labels(a["hydrocarbon"], "petrophysical_hydrocarbon_indicators") == [
        "NO_INDICATION", "LOW_SW", "NO_INDICATION", "NO_INDICATION", "MOVABLE_HC", "NO_INDICATION"]
    assert _labels(a["saturation"], "fluid_contact_detection_indicators")[2] == "NO_CONTACT_INDICATORS"
    assert _labels(a["saturation"], "bulk_volume_water_porosity_integration") == [
        "GOOD_MOVEABLE_HC", "EXCELLENT_MOVEABLE_HC", "FAIR_MOVEABLE_HC", "GOOD_MOVEABLE_HC", "GOOD_MOVEABLE_HC",
        "UNDETERMINED"]
    assert a["permeability"].metadata["mhi_confirmation"] == {"high_mhi_samples": 5}
    assert a["permeability"].sections["permeability_variation_controls"].calculation_thresholds == (
        "CLEAN_RESERVOIR < 0.20; MIXED_LITHOLOGY 0.20-0.50; SHALE_DOMINATED > 0.50.")
    assert a["saturation"].sections["hydrocarbon_movability_invasion_characteristics"].calculation_thresholds == (
        "MHI bands: HIGH_MOVEABLE_HC <0.5, MODERATE 0.5-0.7, LOW 0.7-0.9, MINIMAL >=0.9. "
        "Fallback invasion bands: conductive <0.7, minimal 0.7-1.3, resistive >1.3.")


def test_missing_curves_get_missing_labels_under_current():
    well = Well("NO-PERM")
    well.depth = np.arange(3.0)
    well.depth_unit = "m"
    well.add_curve("PHIE", np.array([0.1, 0.2, 0.3]), "v/v", "Effective porosity")
    well.add_curve("SW", np.array([0.3, 0.5, 0.8]), "v/v", "Water saturation")
    well.add_curve("NET_PAY", np.zeros(3), "", "Net pay")
    perm = PermeabilityAnalysis().analyze(well, add_curves=False)
    assert _labels(perm, "permeability_distribution_analysis") == ["NO_PERMEABILITY_DATA"] * 3
    assert _labels(perm, "overall_permeability_interpretation") == ["NO_QUALITY_DATA"] * 3
    hydrocarbon = HydrocarbonAnalysis().analyze(well, add_curves=False)
    assert _labels(hydrocarbon, "reservoir_quality_for_hydrocarbons") == ["NO_QUALITY_DATA"] * 3
    old = PermeabilityAnalysis(class_tables="geolink").analyze(well, add_curves=False)
    assert _labels(old, "overall_permeability_interpretation") == ["TIGHT_FORMATION"] * 3


def test_lithology_families_are_read_from_rock_names():
    names = ["CHALK", "SANDY LIMESTONE", "POROUS LIMESTONE", "DOLOMITE2", "Marl", "ls", "carbonate",
             "CALCAREOUS SHALE", "CALCAREOUS SANDSTONE", "SAND", "clastic", ""]
    assert [lithology_family(name) for name in names] == [
        "carbonate", "carbonate", "carbonate", "carbonate", "carbonate", "carbonate", "carbonate",
        "clastic", "clastic", "clastic", "clastic", None]
    assert lithology_family(None) is None and lithology_family(NAN) is None
    assert list(sample_families(None, 2)) == ["clastic", "clastic"]
    assert list(sample_families("Chalk", 2)) == ["carbonate", "carbonate"]
    assert list(sample_families(["limestone", None, NAN, "shale"], 4)) == [
        "carbonate", "clastic", "clastic", "clastic"]
    with pytest.raises(ValueError, match="2 entries for 3 samples"):
        sample_families(["clastic", "carbonate"], 3)


def _quality_views(well, lithology_family):
    graded = {"class_tables": "current", "lithology_family": lithology_family}
    porosity = PorosityAnalysis(**graded).analyze(well, add_curves=False)
    views = {
        "porosity": _labels(porosity, "porosity_quality_assessment_characteristics"),
        "porosity_integrated": _labels(porosity, "integrated_porosity_reservoir_assessment"),
        "permeability": _labels(PermeabilityAnalysis(**graded).analyze(well, add_curves=False),
                                "overall_permeability_interpretation"),
        "hydrocarbon": _labels(HydrocarbonAnalysis(**graded).analyze(well, add_curves=False),
                               "reservoir_quality_for_hydrocarbons"),
        "lithology": _labels(LithologyAnalysis(**graded).analyze(well, add_curves=False), "overall_interpretation"),
    }
    texts = {porosity.sections["porosity_quality_assessment_characteristics"].calculation_thresholds.split(" RQI")[0]}
    return views, texts


def test_every_category_grades_each_sample_with_its_familys_limits():
    # PHIE 0.07, 0.20, 0.12, 0.05, 0.16, 0.30; PERM 5, 50, 2, 0.05, 20, 2000 mD; VCL 0.22, 0.45, 0.10, 0.60,
    # 0.30, 0.05. Carbonate rock: 0.07 clears the 0.045 floor (FAIR); VCL 0.30 is not below the 0.30 net-rock
    # cutoff, so the fifth sample is no reservoir.
    views, texts = _quality_views(_well(), "CHALK")
    assert all(labels == [FAIR, NON, GOOD, NON, NON, EXC] for labels in views.values()), views
    assert texts == {CLASS_TABLES["RESERVOIR_QUALITY"].describe(["carbonate"])}
    # One family per sample: chalk at the first and last samples, sandstone between.
    views, texts = _quality_views(_well(), ["chalk", "sand", "sand", "sand", "sand", "chalk"])
    assert all(labels == [FAIR, NON, GOOD, NON, GOOD, EXC] for labels in views.values()), views
    assert texts == {CLASS_TABLES["RESERVOIR_QUALITY"].describe()}
    # No clay volume: no grade anywhere. The reservoir-quality sections say NO_QUALITY_DATA, lithology falls back
    # to the lithology group, porosity-integrated to porosity alone, and no prime target is flagged.
    well = _well()
    well.remove_curve("VCL")
    views, _ = _quality_views(well, None)
    for view in ("porosity", "permeability", "hydrocarbon"):
        assert set(views[view]) == {"NO_QUALITY_DATA"}
    assert set(views["lithology"]) == {"GT_SANDSTONE"}
    assert views["porosity_integrated"] == [
        "MARGINAL_RESERVOIR", "UNDETERMINED", "MARGINAL_RESERVOIR", "TIGHT_FORMATION", "UNDETERMINED", "UNDETERMINED"]
    lithology = LithologyAnalysis().analyze(well, add_curves=False)
    assert set(_labels(lithology, "lithology_petrophysics_relationships")) == {"SANDSTONE_NO_QUALITY_DATA"}
    hydrocarbon = HydrocarbonAnalysis().analyze(well, add_curves=False)
    assert np.isnan(hydrocarbon.derived_curves["PRIME_OIL_TARGET"].data).all()


def test_context_samples_take_the_family_of_their_lithology():
    from petrophys.reporting import reference_export as ref

    def run(lithology, zones=(), zone_lithology="clastic"):
        base = SimpleNamespace(config=DynamicWorkflowConfig(lithology_domain=lithology), steps_executed=[],
                               steps_skipped=[])
        zone_runs = [SimpleNamespace(config=DynamicWorkflowConfig(lithology_domain=zone_lithology)) for _ in zones]
        return SimpleNamespace(base_result=base, zone_results=zone_runs, zones=list(zones))

    well = Well("CONTEXT-FAMILIES")
    well.depth = np.arange(1000.0, 1007.0)
    well.depth_unit = "m"
    # GeoLink 9 Limestone, 15 Dolomite, 16 Sandy Limestone, 1 Sand, 14 Calcareous Shale, then no lithology.
    well.add_curve("LITHOLOGY_GEOLINK", np.array([9, 15, 16, 1, 14, NAN, NAN]), "", "GeoLink lithology")
    well.add_curve("PHIE", np.full(7, 0.06), "v/v", "Effective porosity")
    well.add_curve("PERM", np.full(7, 5.0), "mD", "Permeability")
    well.add_curve("VCL", np.full(7, 0.10), "v/v", "Clay volume")
    carbonate, clastic = "carbonate", "clastic"
    assert list(ref._context_lithology_families(well, run("carbonate"))) == [
        carbonate, carbonate, carbonate, clastic, clastic, carbonate, carbonate]
    # The codebook's names (9 CHALK, 15 DOLOMITE2) name the same rocks.
    kingdom = {9: {"lithology_attribute": "Chalk"}, 15: {"lithology_attribute": "Dolomite2"}}
    # Without lithology a sample takes the family of the run that interpreted it: a zone's re-run in its zone.
    zone = SimpleNamespace(top=1006.0, bottom=1006.0)
    assert list(ref._context_lithology_families(well, run("clastic", [zone], "carbonate"), kingdom)) == [
        carbonate, carbonate, carbonate, clastic, clastic, clastic, carbonate]
    # PHIE 0.06 is FAIR in carbonate rock (PHIE > 0.045) and NON in clastic rock (PHIE > 0.08).
    result = ref._category_analysis_result("hydrocarbon", well, run("carbonate"))
    assert _labels(result, "reservoir_quality_for_hydrocarbons") == [FAIR, FAIR, FAIR, NON, NON, FAIR, FAIR]
    group = ref._category_analysis_result("permeability", well, run("carbonate"), lithology_family="CHALK")
    assert set(_labels(group, "overall_permeability_interpretation")) == {FAIR}


def test_workflow_grades_reservoir_quality_with_each_samples_family(monkeypatch):
    import petrophys.evaluation.diagnostics as diagnostics
    from petrophys.workflow import DynamicWorkflow

    seen = []

    class SpyHydrocarbon(diagnostics.HydrocarbonAnalysis):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("lithology_family"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "HydrocarbonAnalysis", SpyHydrocarbon)
    for profile in ("geolink", "current"):
        n = 60
        well = Well("FAMILY-WORKFLOW")
        well.depth = np.arange(2000.0, 2000.0 + 0.5 * n, 0.5)
        well.depth_unit = "m"
        well.add_curve("GR", np.linspace(20.0, 130.0, n), "API", "Gamma Ray")
        well.add_curve("RHOB", np.linspace(2.20, 2.62, n), "g/cc", "Bulk Density")
        well.add_curve("NPHI", np.linspace(0.30, 0.05, n), "v/v", "Neutron")
        well.add_curve("PEF", np.linspace(2.0, 4.5, n), "b/e", "PEF")
        well.add_curve("RDEP", np.linspace(40.0, 2.0, n), "ohm.m", "Deep Resistivity")
        well.add_curve("CALI", np.full(n, 8.7), "in", "Caliper")
        # GeoLink sandstone (1) over limestone (11).
        well.add_curve("LITHOLOGY_GEOLINK", np.where(np.arange(n) < 30, 1.0, 11.0), "", "Lithology")
        config = DynamicWorkflowConfig(rw=0.05, skip_steps=["temperature", "rmf"], calculation_profile=profile)
        assert DynamicWorkflow(config).run(well).success
    assert seen[0] is None                                   # the benchmark tables take no family
    assert list(seen[1]) == ["clastic"] * 30 + ["carbonate"] * 30


def _is_coarsening(section: Bands, canonical: Bands) -> bool:
    """Every section class is a union of canonical classes and named after one of them; no edge is added."""
    if not set(section.edges) <= set(canonical.edges):
        return False
    edges = canonical.edges
    probes = [edges[0] - 1.0, edges[-1] + 1.0] + [e + d for e in edges for d in (-1e-6, 0.0, 1e-6)]
    section_of = {}
    for value in probes:
        section_of.setdefault(canonical.label_of(value), set()).add(section.label_of(value))
    if any(len(labels) != 1 for labels in section_of.values()):
        return False
    covered = {}
    for canonical_label, (section_label,) in section_of.items():
        covered.setdefault(section_label, set()).add(canonical_label)
    return all(label in labels for label, labels in covered.items())


TABLE_OF_BANDS = {
    PorosityAnalysis: {"TOTAL_POROSITY_BANDS": "PHI", "CLAY_IMPACT_BANDS": "VCL", "RQI_BANDS": "RQI",
                       "FZI_BANDS": "FZI"},
    PermeabilityAnalysis: {"DISTRIBUTION_BANDS": "PERM", "FLOW_UNIT_BANDS": "FZI", "VARIATION_BANDS": "VCL",
                           "INVASION_BANDS": "RXO_RT"},
    SaturationAnalysis: {"WATER_SATURATION_BANDS": "SW", "BVW_BANDS": "BVW", "MHI_BANDS": "MHI",
                         "INVASION_BANDS": "RXO_RT"},
    HydrocarbonAnalysis: {"MHI_BANDS": "MHI"},
}


@pytest.mark.parametrize("analyzer", list(TABLE_OF_BANDS))
def test_every_current_section_table_coarsens_its_canonical_table(analyzer):
    published = {name for name, value in vars(analyzer).items() if isinstance(value, Bands)}
    assert published == set(TABLE_OF_BANDS[analyzer])                         # no section escapes the check
    current = analyzer()
    for name, table in TABLE_OF_BANDS[analyzer].items():
        assert _is_coarsening(bands_of(current, name), CLASS_TABLES[table].bands), name
        # The class attribute stays the benchmark's frozen table, which a "geolink" analyzer uses.
        assert bands_of(analyzer(class_tables="geolink"), name) is bands_of(analyzer, name)
    assert not _is_coarsening(PermeabilityAnalysis.VARIATION_BANDS, CLASS_TABLES["VCL"].bands)
    assert not _is_coarsening(SaturationAnalysis.INVASION_BANDS, CLASS_TABLES["RXO_RT"].bands)


def test_class_tables_argument_is_validated():
    with pytest.raises(ValueError, match="class_tables"):
        PorosityAnalysis(class_tables="may2026")


# -- Where the profile selects the tables ---------------------------------------------------------------------------

def test_context_exporter_uses_the_profile_tables():
    from petrophys.reporting import reference_export as ref

    def run(profile):
        base = SimpleNamespace(config=DynamicWorkflowConfig(calculation_profile=profile), steps_executed=[],
                               steps_skipped=[])
        return SimpleNamespace(base_result=base, zone_results=[])

    first = {}
    for key, workflow_result in (("geolink", run("geolink")), ("current", run("current")), ("none", None)):
        result = ref._category_analysis_result("permeability", _well(), workflow_result)
        first[key] = _labels(result, "permeability_variation_controls")[0]
    assert first == {"geolink": "MIXED_LITHOLOGY", "current": "CLEAN_FORMATION", "none": "CLEAN_FORMATION"}
    summary = ref._build_analysis_summary(_well(), run("geolink"))["HYDROCARBON_ANALYSIS"]
    assert summary["petrophysical_hydrocarbon_indicators"]["calculation_thresholds"].endswith(
        "LOW_SW: SW < 0.50. NO_INDICATION: no petrophysical hydrocarbon criterion met.")


def test_run_comprehensive_analysis_selects_the_tables():
    current = run_comprehensive_analysis(_well(), add_curves=False, store_results=False)
    benchmark = run_comprehensive_analysis(_well(), add_curves=False, store_results=False, class_tables="geolink")
    key = "PERMEABILITY_AND_FLOW_UNITS"
    assert _labels(current.categories[key], "permeability_variation_controls")[0] == "CLEAN_FORMATION"
    assert _labels(benchmark.categories[key], "permeability_variation_controls")[0] == "MIXED_LITHOLOGY"


def test_workflow_passes_the_profile_tables(monkeypatch):
    import petrophys.evaluation.diagnostics as diagnostics
    from petrophys.interpretation.electrical.invasion import invasion_module
    from petrophys.workflow import DynamicWorkflow
    from petrophys.workflow import dynamic_workflow

    seen = {"hydrocarbon": [], "invasion": []}

    class SpyHydrocarbon(diagnostics.HydrocarbonAnalysis):
        def __init__(self, *args, **kwargs):
            seen["hydrocarbon"].append(kwargs.get("class_tables"))
            super().__init__(*args, **kwargs)

    class SpyInvasion(invasion_module.InvasionModule):
        def __init__(self, *args, **kwargs):
            seen["invasion"].append(kwargs.get("class_bands"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "HydrocarbonAnalysis", SpyHydrocarbon)
    monkeypatch.setattr(invasion_module, "InvasionModule", SpyInvasion)

    def well():
        n = 60
        result = Well("PROFILE-TABLES")
        result.depth = np.arange(2000.0, 2000.0 + 0.5 * n, 0.5)
        result.depth_unit = "m"
        result.add_curve("GR", np.linspace(20.0, 130.0, n), "API", "Gamma Ray")
        result.add_curve("RHOB", np.linspace(2.20, 2.62, n), "g/cc", "Bulk Density")
        result.add_curve("NPHI", np.linspace(0.30, 0.05, n), "v/v", "Neutron")
        result.add_curve("PEF", np.linspace(2.0, 4.5, n), "b/e", "PEF")
        result.add_curve("RDEP", np.linspace(40.0, 2.0, n), "ohm.m", "Deep Resistivity")
        result.add_curve("RSHA", np.linspace(30.0, 2.5, n), "ohm.m", "Shallow Resistivity")
        result.add_curve("CALI", np.full(n, 8.7), "in", "Caliper")
        return result

    for profile in ("geolink", "current"):
        config = DynamicWorkflowConfig(rw=0.05, skip_steps=["temperature", "rmf"], calculation_profile=profile)
        result = DynamicWorkflow(config).run(well())
        assert result.success, result.errors
    assert seen["hydrocarbon"] == ["geolink", "current"]
    assert seen["invasion"][0] is dynamic_workflow._BENCHMARK_INVASION_BANDS and seen["invasion"][1] is None
    assert dynamic_workflow._BENCHMARK_INVASION_BANDS.edges == (0.8, 1.2)
