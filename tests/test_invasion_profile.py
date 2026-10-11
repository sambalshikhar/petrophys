"""The resistivity order a bed should show from Rmf/Rw and its fluid (Asquith and Krygowski 2004, ch. 1)."""
import pytest

from petrophys.interpretation.electrical.invasion import (
    EXPECTED_ORDER, MUD_REGIMES, expected_resistivity_order, mud_regime, shallowest_and_deepest,
)


@pytest.mark.parametrize("rmf,rw,want", [(0.3, 0.05, "fresh"), (0.02, 0.05, "saline"), (0.06, 0.05, "similar"),
                                         (0.0625, 0.05, "similar"), (0.0626, 0.05, "fresh")])
def test_mud_regime(rmf, rw, want):
    assert mud_regime(rmf, rw) == want


def test_mud_regime_needs_positive_resistivities():
    with pytest.raises(ValueError, match="must be positive"):
        mud_regime(0.0, 0.05)


def test_expected_order_table_and_unknown_fluid():
    assert expected_resistivity_order("fresh", "water") == "shallow"            # resistive invasion
    assert expected_resistivity_order("saline", "water") == "deep"              # conductive invasion
    assert expected_resistivity_order("similar", "hydrocarbon") == "deep"
    assert expected_resistivity_order("saline", None) == "deep"                 # water and hydrocarbon agree
    assert expected_resistivity_order("fresh", None) == "either"                # they do not
    assert set(EXPECTED_ORDER) == set(MUD_REGIMES)
    assert all(set(v) == {"water", "hydrocarbon"} for v in EXPECTED_ORDER.values())


def test_shallowest_and_deepest():
    assert shallowest_and_deepest(["RDEP", "RXO", "RMED"]) == ("RXO", "RDEP")
    assert shallowest_and_deepest(["RDEP"]) is None
    assert shallowest_and_deepest(["RDEP", "GR"]) is None                     # not all resistivity readings
    assert shallowest_and_deepest(["A", "B"], ranking=["A", "B"]) == ("A", "B")
