from pathlib import Path

import numpy as np

from petrophys import Well
from petrophys.imaging import BoreholeImageData
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
