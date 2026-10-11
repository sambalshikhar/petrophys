"""Bands: the one-curve classes the analyzers apply, published as data, including which side of an edge a value
that equals it falls on."""
import numpy as np
import pytest

from petrophys.evaluation.diagnostics import Bands, PermeabilityAnalysis, PorosityAnalysis, SaturationAnalysis, bands_of


def test_classify_edges_labels():
    b = Bands("POOR", ((">", 1.0, "FAIR"), (">", 10.0, "GOOD")))
    assert list(b.classify([np.nan, 0.5, 1.0, 1.01, 10.0, 50.0])) == ["NO_DATA", "POOR", "POOR", "FAIR", "FAIR", "GOOD"]
    assert b.edges == (1.0, 10.0) and b.labels == ("POOR", "FAIR", "GOOD") and b.label_of(np.nan) is None
    with pytest.raises(ValueError, match="unknown comparison"):
        Bands("A", (("=>", 1.0, "B"),))


def test_mixed_edge_sides_and_descending_labels():
    vcl = PermeabilityAnalysis.VARIATION_BANDS                      # < 0.20 clean, <= 0.50 mixed
    assert [vcl.label_of(v) for v in (0.19, 0.2, 0.5, 0.51)] == ["CLEAN_RESERVOIR", "MIXED_LITHOLOGY", "MIXED_LITHOLOGY",
                                                                "SHALE_DOMINATED"]
    inv = PermeabilityAnalysis.INVASION_BANDS                       # the middle class is closed on both sides
    assert [inv.label_of(v) for v in (0.79, 0.8, 1.2, 1.21)] == ["CONDUCTIVE_INVASION", "MINIMAL_INVASION",
                                                                "MINIMAL_INVASION", "RESISTIVE_INVASION"]
    assert SaturationAnalysis.WATER_SATURATION_BANDS.labels == ("EXCELLENT_HC_SATURATION", "GOOD_HC_SATURATION",
                                                                "FAIR_HC_SATURATION", "WATER_BEARING")
    assert PermeabilityAnalysis.FLOW_UNIT_BANDS.label_of(2.0) == "MACRO_PORES"          # >= : the edge goes up
    assert PorosityAnalysis.TOTAL_POROSITY_BANDS.label_of(0.12) == "FAIR_POROSITY"       # > : the edge stays down


def test_bands_of():
    assert bands_of(SaturationAnalysis, "BVW_BANDS").edges == (0.03, 0.06, 0.10)
    with pytest.raises(ValueError, match="no Bands named"):
        bands_of(SaturationAnalysis, "NOPE")
