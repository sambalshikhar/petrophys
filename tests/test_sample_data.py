"""Tests for the bundled sample-data loader (petrophys.sample / list_samples)."""

import pytest

import petrophys
from petrophys.datasets import (
    DATA_DIR,
    DAYDREAM_DATA_DIR,
    integration_fixture,
    list_integration_fixtures,
)
from petrophys.io import read_las


def test_sample_returns_existing_path():
    p = petrophys.sample("HW-3")
    assert p.exists()
    assert p.suffix.lower() == ".las"


def test_sample_extension_optional_and_case_insensitive():
    assert petrophys.sample("hw-3") == petrophys.sample("HW-3.las")


def test_sample_loads_a_well():
    well = read_las(petrophys.sample("30_6-5"))
    assert len(well.curves) > 3


def test_list_samples_kinds():
    costa = petrophys.list_samples("costa")
    assert len(costa) == 17 and "HW-3" in costa
    assert set(petrophys.list_samples("geolink")) == {
        "30_6-5",
        "16_7-1",
        "25_5-1",
        "15_9-14",
    }
    assert set(petrophys.list_samples("daydream")) == {
        "daydream1_cmr",
        "daydream1_cmr_shallow",
    }
    assert set(petrophys.list_samples("kgs")) == {
        "1046123714",
        "1046123714_formation_tops",
    }


def test_kgs_legacy_sample_loads_with_feet_depth():
    well = read_las(petrophys.sample("1046123714"))

    assert well.name == "Morgan No. 1-17"
    assert well.depth_unit == "ft"
    assert well.n_samples == 9396
    assert {"GR", "RHOB", "CNLS", "DT", "RILD"}.issubset(well.curves)


def test_sample_unknown_raises_helpful_error():
    with pytest.raises(FileNotFoundError):
        petrophys.sample("does-not-exist")


def test_repository_datasets_share_one_root():
    assert DAYDREAM_DATA_DIR.parent == DATA_DIR
    assert (
        integration_fixture("daydream1_shallow_aps_pex_hngs.dlis").parent
        == DAYDREAM_DATA_DIR
    )


def test_integration_fixtures_are_not_bundled_samples():
    fixture_names = list_integration_fixtures("daydream")
    assert "daydream1_deep_curated_scalar_logs.las" in fixture_names
    assert "daydream1_fmi_processed_dynamic_image.dlis" in fixture_names
    assert "daydream1_shallow_aps_pex_hngs.las" in fixture_names
    assert "daydream1_shallow_aps_pex_hngs" not in petrophys.list_samples()
