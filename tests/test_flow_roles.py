"""Flow roles of zones from mean permeability: strongest, capacity, baffles and qualifiers."""
import numpy as np
import pytest

from petrophys.evaluation.flow_units import flow_roles


def test_roles_strongest_capacity_and_baffles():
    r = flow_roles([150.0, 0.5, 40.0, 1.2, None], [2.0, 5.0, 30.0, 4.0, 3.0])
    assert [z["role"] for z in r["zones"]] == ["strongest", "baffle", "flow", "baffle", None]
    assert r["strongest"] == 0 and r["capacity"] == 2                     # 40 mD x 30 m beats 150 mD x 2 m
    assert r["zones"][2]["kh"] == pytest.approx(1200.0) and r["zones"][4]["kh"] is None
    # 1.2 mD is above poor_perm but two decades below 150 mD: a baffle; at 1 decade it is not
    assert flow_roles([150.0, 1.2], [1, 1], baffle_decades=3.0)["zones"][1]["role"] == "flow"


def test_seal_dominated_zone_is_a_baffle():
    r = flow_roles([100.0, 80.0], [1.0, 1.0], seal=[0.0, 0.6])
    assert [z["role"] for z in r["zones"]] == ["strongest", "baffle"]


def test_qualifiers():
    r = flow_roles([50.0, 20.0, 5.0], [1, 1, 1], sw=[0.2, 0.85, np.nan], vcl=[0.1, 0.45, 0.5],
                   bad_hole=[0.8, 0.1, None], at_limit=[False, True, False], sw_max=0.7, vcl_max=0.3)
    q = [dict(z["qualifiers"]) for z in r["zones"]]
    assert q[0] == {"bad_hole": 0.8}
    assert set(q[1]) == {"at_model_limit", "shaly", "water_bearing"} and q[1]["shaly"] == 0.45
    assert q[2] == {}                                  # shaly only with good modelled permeability (>= 10 mD)


def test_empty_and_mismatched():
    assert flow_roles([], []) == {"zones": [], "strongest": None, "capacity": None}
    with pytest.raises(ValueError, match="zones but thickness"):
        flow_roles([1.0], [1.0, 2.0])
