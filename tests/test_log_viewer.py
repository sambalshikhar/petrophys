import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from petrophys import Well
from petrophys.visualization import (
    LogViewer,
    create_log_viewer,
    load_log_viewer_template,
)


def _plot_well() -> Well:
    depth = np.arange(1000.0, 1010.0, 0.5)
    well = Well("PLOT-1")
    well.depth = depth
    well.depth_unit = "ft"
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("GR", np.linspace(35.0, 120.0, depth.size), "API", "Gamma Ray")
    well.add_curve("CALI", np.full(depth.size, 8.5), "in", "Caliper")
    well.add_curve("BS", np.full(depth.size, 8.5), "in", "Bit Size")
    well.add_curve("RDEP", np.geomspace(2.0, 300.0, depth.size), "ohm.m", "Deep Resistivity")
    well.add_curve("RSHAL", np.geomspace(1.0, 120.0, depth.size), "ohm.m", "Shallow Resistivity")
    well.add_curve("RHOB", np.linspace(2.15, 2.55, depth.size), "g/cc", "Density")
    well.add_curve("NPHI", np.linspace(0.28, 0.08, depth.size), "v/v", "Neutron")
    well.add_curve("VCL", np.linspace(0.05, 0.55, depth.size), "v/v", "Clay Volume")
    well.add_curve("PHIT", np.linspace(0.26, 0.08, depth.size), "v/v", "Total Porosity")
    well.add_curve("PHIE_CC", np.linspace(0.23, 0.05, depth.size), "v/v", "Core Corrected Effective Porosity")
    well.add_curve("PHIE", np.linspace(0.22, 0.04, depth.size), "v/v", "Effective Porosity")
    well.add_curve("SW", np.linspace(0.25, 0.88, depth.size), "v/v", "Water Saturation")
    well.add_curve("SXO", np.linspace(0.55, 0.98, depth.size), "v/v", "Flushed Water Saturation")
    well.add_curve("PERM_CC", np.geomspace(0.05, 500.0, depth.size), "mD", "Core Corrected Permeability")
    well.add_curve("PERM", np.geomspace(0.01, 300.0, depth.size), "mD", "Permeability")
    well.add_curve("VQUAR", np.full(depth.size, 0.50), "v/v", "Quartz")
    well.add_curve("VCALC", np.full(depth.size, 0.15), "v/v", "Calcite")
    well.add_curve("VDOLO", np.full(depth.size, 0.05), "v/v", "Dolomite")
    return well


def test_log_viewer_builtin_template_creates_and_saves_plot(tmp_path: Path):
    viewer = create_log_viewer(
        _plot_well(),
        template_defaults="petrophysics",
        depth_range=(1000.0, 1009.5),
    )

    assert isinstance(viewer, LogViewer)
    assert viewer.fig is not None
    assert len(viewer.axes) == 6
    assert not any("PHIE_CC" in warning for warning in viewer.warnings)

    output_file = tmp_path / "petrophysics.png"
    viewer.save(output_file)
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_log_viewer_custom_xml_supports_aliases_and_cumulative_track(tmp_path: Path):
    xml_path = tmp_path / "template.xml"
    xml_path.write_text(
        """
<template name="custom">
  <track display_name="Depth" depth_track="true" width="0.5" tick_spacing="2" number_spacing="4"/>
  <track display_name="Resistivity" scale="log" width="1.0">
    <curve curve_name="RSHA" display_name="Shallow" left="0.2" right="2000" color="green"/>
  </track>
  <track display_name="Minerals" cumulative="true" left="0" right="1" width="1.0">
    <curve curve_name="VCL" display_name="Clay" color="brown"/>
    <curve curve_name="VQUAR" display_name="Quartz" color="gold"/>
    <curve curve_name="VCALC" display_name="Calcite" color="gray"/>
  </track>
</template>
""",
        encoding="utf-8",
    )

    template = load_log_viewer_template(xml_path)
    assert template.name == "custom"
    assert len(template.tracks) == 3

    viewer = create_log_viewer(_plot_well(), template_xml_path=xml_path, strict=True)
    assert viewer.fig is not None
    assert len(viewer.axes) == 3
    assert viewer.warnings == []


def test_log_viewer_strict_mode_raises_for_missing_curves(tmp_path: Path):
    xml_path = tmp_path / "missing.xml"
    xml_path.write_text(
        """
<template name="missing">
  <track display_name="Missing">
    <curve curve_name="NOT_A_CURVE" left="0" right="1"/>
  </track>
</template>
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="NOT_A_CURVE"):
        create_log_viewer(_plot_well(), template_xml_path=xml_path, strict=True)

