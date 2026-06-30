import os
from pathlib import Path

import numpy as np
import pytest

from petrophys.io import read_dlis, read_dlis_merged
from petrophys.io.dlis_reader import (
    HAS_DLISIO,
    _channel_requested,
    _channel_field_names,
    _convert_depth,
)


# Optional integration fixtures: point these env vars at local DLIS files to run
# the reader integration tests; otherwise they skip (see _skip_without_dlisio_and_file).
DEEP_DAYDREAM_DLIS = Path(os.environ.get("PETROPHYS_DLIS_DEEP", "_unset_deep_daydream.dlis"))
SHALLOW_DAYDREAM_DLIS = Path(os.environ.get("PETROPHYS_DLIS_SHALLOW", "_unset_shallow_daydream.dlis"))


class _FakeChannel:
    def __init__(self, name):
        self.name = name


class _FakeFrame:
    def __init__(self, names):
        self.channels = [_FakeChannel(name) for name in names]


def _skip_without_dlisio_and_file(path):
    if not HAS_DLISIO:
        pytest.skip("dlisio is not installed")
    if not path.exists():
        pytest.skip(f"DLIS test file is not available: {path}")


def test_dlis_tenth_inch_depth_is_converted_to_metres_by_default():
    depth, unit = _convert_depth(np.array([0.0, 10.0, 1000.0]), "0.1 in")

    assert unit == "m"
    np.testing.assert_allclose(depth, np.array([0.0, 0.0254, 2.54]))


def test_dlis_channel_fields_follow_structured_array_order_for_duplicate_mnemonics():
    frame = _FakeFrame(["TDEP", "HMAS", "HMAS"])
    data = np.zeros(
        2,
        dtype=[
            ("FRAMENO", "i4"),
            ("TDEP", "f8"),
            ("HMAS.78.0", "f8"),
            ("HMAS.78.1", "f8"),
        ],
    )

    assert _channel_field_names(frame, data) == ["TDEP", "HMAS.78.0", "HMAS.78.1"]


def test_dlis_channel_filter_accepts_standard_names_for_vendor_aliases():
    assert _channel_requested("HGR", {"GR"})
    assert _channel_requested("PHIE_HILT", {"PHIE"})
    assert not _channel_requested("RVDR", {"GR"})


def test_daydream_deep_dlis_single_frame_handles_duplicate_fields_when_available():
    _skip_without_dlisio_and_file(DEEP_DAYDREAM_DLIS)

    well = read_dlis(
        DEEP_DAYDREAM_DLIS,
        merge_frames=False,
        include_vector_channels=False,
    )

    assert well.depth_unit == "m"
    assert well.top is not None and 2700.0 < well.top < 2800.0
    assert well.bottom is not None and 4100.0 < well.bottom < 4200.0
    assert well.n_samples > 0
    assert well.n_curves > 0


def test_daydream_deep_dlis_merged_exposes_interpretation_curves_when_available():
    _skip_without_dlisio_and_file(DEEP_DAYDREAM_DLIS)

    well = read_dlis_merged(DEEP_DAYDREAM_DLIS)

    assert well.depth_unit == "m"
    assert well.n_samples > 9000
    assert well.n_curves > 100

    interval = (well.depth >= 3000.0) & (well.depth <= 3010.0)
    assert np.count_nonzero(interval) > 0
    for mnemonic in ["GR", "RDEP", "RXO", "RHOB", "NPHI", "PHIE", "VCL", "BS"]:
        assert mnemonic in well
        assert np.any(np.isfinite(well[mnemonic].data[interval]))

    assert well["PHIE"].unit == "v/v"
    assert well["VCL"].unit == "v/v"
    assert well["NPHI"].unit == "v/v"


def test_read_dlis_defaults_to_merged_scalar_log_view_when_available():
    _skip_without_dlisio_and_file(DEEP_DAYDREAM_DLIS)

    well = read_dlis(DEEP_DAYDREAM_DLIS)

    assert "dlis_merge" in well._parameters
    assert well.depth_unit == "m"
    assert "GR" in well
    assert "PHIE" in well


def test_daydream_shallow_dlis_merged_depth_range_when_available():
    _skip_without_dlisio_and_file(SHALLOW_DAYDREAM_DLIS)

    well = read_dlis_merged(SHALLOW_DAYDREAM_DLIS)

    assert well.depth_unit == "m"
    assert well.top is not None and 800.0 < well.top < 900.0
    assert well.bottom is not None and 2800.0 < well.bottom < 2900.0
    assert "GR" in well
    assert "RDEP" in well
    assert not np.any((well.depth >= 3000.0) & (well.depth <= 3010.0))
