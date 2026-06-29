"""Tests for the bundled sample-data loader (petrophys.sample / list_samples)."""

import pytest

import petrophys
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
    assert set(petrophys.list_samples("geolink")) == {"30_6-5", "16_7-1"}


def test_sample_unknown_raises_helpful_error():
    with pytest.raises(FileNotFoundError):
        petrophys.sample("does-not-exist")
