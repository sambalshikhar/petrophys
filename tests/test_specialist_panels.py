from pathlib import Path

import numpy as np

from petrophys import Well
from petrophys.interpretation.borehole_imaging import BoreholeImageData
from petrophys.visualization import (
    create_fmi_panel,
    create_nmr_panel,
    has_nmr_curves,
)


def _nmr_well():
    depth = np.linspace(1000.0, 1004.0, 5)
    well = Well("NMR-PANEL")
    well.depth = depth
    well.depth_unit = "m"
    well.add_curve("GR", np.linspace(40.0, 120.0, 5), "API", "Gamma Ray")
    well.add_curve("BS", np.full(5, 8.5), "in", "Bit Size")
    well.add_curve("HD1_PPC1", np.linspace(8.6, 9.0, 5), "in", "Caliper 1")
    well.add_curve("HD2_PPC1", np.linspace(8.7, 9.1, 5), "in", "Caliper 2")
    well.add_curve("CDF", np.linspace(1000.0, 5000.0, 5), "N", "Downhole Force")
    well.add_curve("TENS", np.linspace(3000.0, 6000.0, 5), "lbf", "Cable Tension")
    well.add_curve("TCMR", np.full(5, 0.30), "v/v", "Total CMR Porosity")
    well.add_curve("CMFF", np.full(5, 0.21), "v/v", "CMR Free Fluid")
    well.add_curve("KSDR", np.geomspace(100.0, 1.0, 5), "mD", "SDR Permeability")
    well.add_curve("KTIM", np.geomspace(80.0, 0.8, 5), "mD", "Timur Permeability")
    well.add_curve("T2CUTOFF", np.full(5, 33.0), "ms", "T2 Cutoff")
    return well


def test_nmr_specialist_panel_saves(tmp_path):
    well = _nmr_well()

    assert has_nmr_curves(well)
    fig = create_nmr_panel(well)
    output = tmp_path / "nmr_panel.png"
    fig.savefig(output, dpi=40)

    assert output.exists()
    np.testing.assert_allclose(fig.axes[0].get_xlim(), (0.0, 150.0))
    assert fig.axes[1].get_title() == "Borehole"
    assert {line.get_label() for line in fig.axes[1].lines} == {"HD1", "HD2", "BS"}
    all_labels = {
        line.get_label()
        for axis in fig.axes
        for line in axis.lines
    }
    assert {"GR", "CDF", "TENS"}.issubset(all_labels)


def test_fmi_specialist_panel_saves_without_orientation(tmp_path):
    image = BoreholeImageData.from_arrays(
        depth=np.linspace(1000.0, 1004.0, 5),
        image=np.arange(40, dtype=float).reshape(5, 8),
        channel_names=["IMAGE"],
        samples_per_channel=[8],
    )

    fig = create_fmi_panel(image, orient=False)
    output = Path(tmp_path) / "fmi_panel.png"
    fig.savefig(output, dpi=40)

    assert output.exists()
