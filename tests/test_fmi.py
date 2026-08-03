import os
from pathlib import Path

import numpy as np
import pytest

from petrophys.datasets import integration_fixture
from petrophys.interpretation.borehole_imaging import (
    BoreholeImageData,
    azimuthal_coverage,
    equalize_dynamic,
    equalize_static,
    get_navigation,
    orient_to_azimuth,
    read_fmi_dlis,
    safe_per_depth_statistics,
    with_navigation,
)

FMI_FIXTURE = integration_fixture("daydream1_fmi_processed_dynamic_image.dlis")


def test_navigation_is_preserved_when_filtering_by_depth():
    image = BoreholeImageData.from_arrays(
        depth=[100.0, 101.0, 102.0],
        image=np.ones((3, 1)),
        channel_names=["FMPA"],
        samples_per_channel=[1],
    )
    image = with_navigation(image, {"P1AZ": [0.0, 45.0, 90.0], "RB": [5.0, 10.0, 15.0]})

    filtered = image.filter_by_depth(101.0, 102.0)
    navigation = get_navigation(filtered)

    np.testing.assert_allclose(filtered.depth, [101.0, 102.0])
    np.testing.assert_allclose(navigation["P1AZ"], [45.0, 90.0])
    np.testing.assert_allclose(navigation["RB"], [10.0, 15.0])


def test_fmi_orientation_maps_pad_arrays_to_north_azimuth_bins():
    image = BoreholeImageData.from_arrays(
        depth=[0.0, 1.0],
        image=np.array([[10.0, 20.0], [30.0, 40.0]]),
        channel_names=["FMPA", "FMPB"],
        samples_per_channel=[1, 1],
    )
    image = with_navigation(image, {"P1AZ": [0.0, 90.0]})

    oriented = orient_to_azimuth(image, n_azimuth=8)

    assert oriented.metadata["oriented"] is True
    assert oriented.channel_names == ["AZIMUTH"]
    assert oriented.samples_per_channel == [8]
    assert oriented.image.shape == (2, 8)
    assert oriented.image[0, 0] == 10.0
    assert oriented.image[0, 2] == 20.0
    assert oriented.image[1, 2] == 30.0
    assert oriented.image[1, 4] == 40.0
    np.testing.assert_allclose(azimuthal_coverage(oriented), [2 / 8, 2 / 8])


def test_fmi_orientation_can_use_highside_reference():
    image = BoreholeImageData.from_arrays(
        depth=[0.0],
        image=np.array([[7.0]]),
        channel_names=["FMPA"],
        samples_per_channel=[1],
    )
    image = with_navigation(image, {"RB": [90.0]})

    oriented = orient_to_azimuth(image, n_azimuth=8, reference="highside")

    assert oriented.image[0, 2] == 7.0
    assert oriented.metadata["orientation_key"] == "RB"


def test_fmi_orientation_averages_colliding_azimuth_bins():
    image = BoreholeImageData.from_arrays(
        depth=[0.0],
        image=np.array([[10.0, 30.0]]),
        channel_names=["FMPA", "FMPA"],
        samples_per_channel=[1, 1],
    )
    image = with_navigation(image, {"P1AZ": [0.0]})

    oriented = orient_to_azimuth(image, n_azimuth=8)

    assert oriented.image[0, 0] == 20.0


def test_fmi_orientation_interpolates_within_pad_arc_by_default():
    image = BoreholeImageData.from_arrays(
        depth=[0.0],
        image=np.array([[10.0, 20.0]]),
        channel_names=["FMPA"],
        samples_per_channel=[2],
    )
    image = with_navigation(image, {"P1AZ": [90.0]})

    oriented = orient_to_azimuth(image, n_azimuth=360)
    finite_bins = np.flatnonzero(np.isfinite(oriented.image[0]))

    assert finite_bins.size > 2
    np.testing.assert_array_equal(np.diff(finite_bins), np.ones(finite_bins.size - 1))
    assert oriented.image[0, finite_bins[0]] < oriented.image[0, finite_bins[-1]]
    assert oriented.metadata["interpolate_within_pads"] is True


def test_fmi_equalization_preserves_nan_and_normalizes_to_unit_range():
    image = np.array([[0.0, 10.0], [20.0, np.nan], [30.0, 40.0]])

    static = equalize_static(image, clip=(0.0, 100.0))
    dynamic = equalize_dynamic(image, np.array([0.0, 0.5, 1.0]), window_m=1.0)

    assert np.isnan(static[1, 1])
    assert np.nanmin(static) >= 0.0
    assert np.nanmax(static) <= 1.0
    assert dynamic.shape == image.shape
    assert np.nanmin(dynamic) >= 0.0
    assert np.nanmax(dynamic) <= 1.0


def test_safe_per_depth_statistics_handles_all_nan_rows():
    image = BoreholeImageData.from_arrays(
        depth=[0.0, 1.0],
        image=np.array([[np.nan, np.nan], [1.0, 3.0]]),
    )

    stats = safe_per_depth_statistics(image)

    assert np.isnan(stats["mean"][0])
    assert stats["coverage"][0] == 0.0
    assert stats["mean"][1] == 2.0
    assert stats["coverage"][1] == 1.0


def test_daydream_processed_fmi_dlis_can_be_read_and_oriented_when_available():
    path = Path(os.environ.get("PETROPHYS_FMI_DLIS", FMI_FIXTURE))
    if not path.exists():
        pytest.skip(
            "FMI DLIS not available; run 'git lfs pull' or set PETROPHYS_FMI_DLIS"
        )
    if path.stat().st_size < 1024:
        pytest.skip("FMI DLIS is only a Git LFS pointer; run 'git lfs pull'")

    data = read_fmi_dlis(path)
    navigation = get_navigation(data)
    assert data.image.shape[1] == 192
    assert {"P1AZ", "RB"}.issubset(navigation)

    zoom = data.filter_by_depth(3000.0, 3002.0)
    oriented = orient_to_azimuth(zoom, n_azimuth=180)

    assert oriented.image.shape[1] == 180
    assert np.nanmean(azimuthal_coverage(oriented)) > 0.0
