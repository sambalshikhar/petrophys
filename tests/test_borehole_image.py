import numpy as np

from petrophys import Well
from petrophys.imaging import BoreholeImageData
from petrophys.io import read_borehole_image_las


def test_borehole_image_from_arrays_filters_descending_depth_and_normalizes():
    image = BoreholeImageData.from_arrays(
        depth=np.array([100.0, 99.5, 99.0]),
        image=np.array(
            [
                [0, 10, 20, 30],
                [5, 15, 25, 35],
                [10, 20, 30, 40],
            ],
            dtype=np.int16,
        ),
        channel_names=["PAD_A", "PAD_B"],
        samples_per_channel=[2, 2],
        depth_unit="m",
    )

    assert image.n_samples == 3
    assert image.n_borehole_samples == 4
    assert image.top == 99.0
    assert image.bottom == 100.0

    filtered = image.filter_by_depth(99.5, 100.0)

    np.testing.assert_allclose(filtered.depth, [100.0, 99.5])
    assert filtered.image.shape == (2, 4)
    np.testing.assert_allclose(image.normalized_image()[0], [0.0, 0.25, 0.5, 0.75])


def test_borehole_image_summary_curves_can_be_added_to_well():
    image = BoreholeImageData.from_arrays(
        depth=np.array([10.0, 10.5]),
        image=np.array([[1.0, 3.0, np.nan], [2.0, 4.0, 6.0]]),
        channel_names=["IMAGE"],
        samples_per_channel=[3],
        metadata={"well": "IMG-1"},
    )

    well = image.to_well_curves(prefix="FMI")

    assert well.name == "IMG-1"
    assert well.depth_unit == "m"
    np.testing.assert_allclose(well["FMI_MEAN"].data, [2.0, 4.0])
    np.testing.assert_allclose(well["FMI_COVERAGE"].data, [2.0 / 3.0, 1.0])


def test_flattened_fmi_las_is_reassembled_into_image(tmp_path):
    path = tmp_path / "flattened_fmi.las"
    path.write_text(
        "\n".join(
            [
                "~VERSION INFORMATION",
                "VERS. 2.0 : CWLS Log ASCII Standard",
                "WRAP. NO",
                "~WELL INFORMATION",
                "STRT.M 100.0000 : START DEPTH",
                "STOP.M 99.5000 : STOP DEPTH",
                "STEP.M -0.2500 : STEP",
                "NULL. -999.25 : NULL VALUE",
                "WELL. FMI-LAS-1 : WELL",
                "~CURVE INFORMATION",
                "DEPT.M : DEPTH",
                "FMPA_0. : FMI Pad A button 0",
                "FMPA_1. : FMI Pad A button 1",
                "FMFA_0. : FMI Flap A button 0",
                "FMFA_1. : FMI Flap A button 1",
                "~A",
                "100.0000 1 2 3 4",
                " 99.7500 5 6 7 8",
                " 99.5000 9 10 11 12",
            ]
        ),
        encoding="utf-8",
    )

    image = read_borehole_image_las(path)

    assert image.metadata["source_format"] == "LAS"
    assert image.metadata["well"] == "FMI-LAS-1"
    assert image.channel_names == ["FMPA", "FMFA"]
    assert image.samples_per_channel == [2, 2]
    np.testing.assert_allclose(image.depth, [100.0, 99.75, 99.5])
    np.testing.assert_allclose(
        image.image,
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
    )


def test_flattened_fmi_curves_in_well_are_reassembled():
    well = Well("FMI-WELL")
    well.depth = np.array([10.0, 10.5])
    well.depth_unit = "m"
    well.add_curve("FMPA_1", np.array([1.0, 3.0]), "", "Pad A button 1")
    well.add_curve("FMPA_2", np.array([2.0, 4.0]), "", "Pad A button 2")
    well.add_curve("FMFA_1", np.array([5.0, 7.0]), "", "Flap A button 1")
    well.add_curve("FMFA_2", np.array([6.0, 8.0]), "", "Flap A button 2")

    image = BoreholeImageData.from_well(well)

    assert image.metadata["well"] == "FMI-WELL"
    assert image.channel_names == ["FMPA", "FMFA"]
    assert image.samples_per_channel == [2, 2]
    np.testing.assert_allclose(image.image, [[1, 2, 5, 6], [3, 4, 7, 8]])
