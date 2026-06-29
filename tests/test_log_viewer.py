import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from petrophys import Well
from petrophys.visualization import (
    LogViewer,
    create_adaptive_log_viewer_template,
    create_log_viewer,
    load_log_viewer_template,
)
import petrophys.visualization.log_viewer as log_viewer_module


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
    well.add_curve("LITHOLOGY_GEOLINK", np.where(depth < 1005, 1.0, 7.0), "", "Lithology")
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
    well.add_curve("NET_ROCK", np.ones(depth.size), "", "Net Rock")
    well.add_curve("NET_RES", np.where(depth < 1006, 1.0, 0.0), "", "Net Reservoir")
    well.add_curve("NET_PAY", np.where(depth < 1004, 1.0, 0.0), "", "Net Pay")
    return well


def _mineral_well() -> Well:
    depth = np.arange(1000.0, 1010.0, 0.5)
    well = Well("MINERAL-PLOT")
    well.depth = depth
    well.depth_unit = "ft"
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("VCL", np.linspace(0.05, 0.20, depth.size), "v/v", "Clay Volume")
    well.add_curve("VQUAR", np.full(depth.size, 0.50), "v/v", "Quartz")
    well.add_curve("VCALC", np.full(depth.size, 0.15), "v/v", "Calcite")
    well.add_curve("VDOLO", np.full(depth.size, 0.05), "v/v", "Dolomite")
    well.add_curve("PHIE", np.linspace(0.22, 0.04, depth.size), "v/v", "Effective Porosity")
    return well


def _viewer_texts(viewer: LogViewer):
    texts = [text.get_text() for ax in viewer.header_axes + viewer.axes for text in ax.texts]
    for ax in viewer.axes:
        legend = ax.get_legend()
        if legend is not None:
            texts.extend(text.get_text() for text in legend.get_texts())
    if viewer.fig is not None:
        for legend in viewer.fig.legends:
            texts.extend(text.get_text() for text in legend.get_texts())
    return texts


def _header_texts_for_track(viewer: LogViewer, track_name: str):
    for ax in viewer.header_axes:
        texts = [text.get_text() for text in ax.texts]
        if track_name in texts:
            return texts
    return []


def test_log_viewer_builtin_template_creates_and_saves_plot(tmp_path: Path):
    viewer = create_log_viewer(
        _plot_well(),
        template_defaults="petrophysics",
        depth_range=(1000.0, 1009.5),
    )

    assert isinstance(viewer, LogViewer)
    assert viewer.fig is not None
    assert len(viewer.axes) == 8
    assert not any("PHIE_CC" in warning for warning in viewer.warnings)

    output_file = tmp_path / "petrophysics.png"
    viewer.save(output_file)
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_log_viewer_lithology_track_uses_names_not_codes(monkeypatch):
    monkeypatch.setattr(log_viewer_module, "GEOLINK_LITHOLOGY_MAPPING_FILENAMES", ())
    log_viewer_module._default_geolink_lithology_mapping.cache_clear()
    try:
        viewer = create_log_viewer(
            _plot_well(),
            template_defaults="auto",
            depth_range=(1000.0, 1009.5),
        )
        texts = _viewer_texts(viewer)
    finally:
        log_viewer_module._default_geolink_lithology_mapping.cache_clear()

    assert "Sandstone" in texts
    assert "Shale/Claystone" in texts
    assert not any(text.startswith("Code ") for text in texts)


def test_log_viewer_geolink_lithology_defaults_use_attribute_table(
    tmp_path: Path,
    monkeypatch,
):
    pd = pytest.importorskip("pandas")
    mapping_path = tmp_path / "Litho_Nomenclature_Kingdom.xlsx"
    pd.DataFrame(
        {
            "Abbreviation": [1, 7],
            "Lithology Attribute": ["Workbook Sand", "Workbook Shale"],
        }
    ).to_excel(mapping_path, index=False)
    monkeypatch.setattr(
        log_viewer_module,
        "GEOLINK_LITHOLOGY_MAPPING_FILENAMES",
        (str(mapping_path),),
    )
    log_viewer_module._default_geolink_lithology_mapping.cache_clear()
    try:
        viewer = create_log_viewer(
            _plot_well(),
            template_defaults="auto",
            depth_range=(1000.0, 1009.5),
        )
        texts = _viewer_texts(viewer)
    finally:
        log_viewer_module._default_geolink_lithology_mapping.cache_clear()

    assert "Workbook Sand" in texts
    assert "Workbook Shale" in texts
    assert "Sandstone" not in texts
    assert "Silty Shale" not in texts


def test_log_viewer_lithology_track_states_when_not_available():
    depth = np.arange(1000.0, 1010.0, 0.5)
    well = Well("NO-LITHOLOGY")
    well.depth = depth
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("GR", np.linspace(35.0, 120.0, depth.size), "API", "Gamma Ray")

    viewer = create_log_viewer(well, template_defaults="auto", depth_range=(1000.0, 1009.5))
    texts = _viewer_texts(viewer)

    assert "No lithology available" in texts


def test_log_viewer_lithology_uses_industry_style_pattern_symbols():
    viewer = create_log_viewer(
        _plot_well(),
        template_defaults="auto",
        depth_range=(1000.0, 1009.5),
    )

    assert viewer._lithology_hatch("Calcareous Sandstone") == ".."
    assert viewer._lithology_hatch("Sandy Shale") == "----"
    assert viewer._lithology_hatch("Shale") == "----"
    assert viewer._lithology_hatch("Breccia") == "xx"
    assert viewer._lithology_hatch("Halite") == "++"
    assert viewer._lithology_hatch("Dolomite") == "//"


def test_log_viewer_lithology_styles_group_families_and_distinguish_patterns():
    viewer = create_log_viewer(
        _plot_well(),
        template_defaults="auto",
        depth_range=(1000.0, 1009.5),
    )
    cmap = log_viewer_module.plt.get_cmap("tab20")
    sandstone = viewer._lithology_color(0.0, "Sandstone", cmap, 0, 6)
    shale = viewer._lithology_color(0.0, "Shale", cmap, 1, 6)
    silt = viewer._lithology_color(0.0, "Siltstone/Silt", cmap, 2, 6)
    limestone = viewer._lithology_color(0.0, "Limestone", cmap, 3, 6)
    dolomite = viewer._lithology_color(0.0, "Dolomite", cmap, 4, 6)
    halite = viewer._lithology_color(0.0, "Halite", cmap, 5, 6)
    anhydrite = viewer._lithology_color(0.0, "Anhydrite", cmap, 5, 6)

    assert sandstone != shale
    assert sandstone != silt
    # Every code is visually distinguishable: members share a colour FAMILY but
    # are stepped to distinct shades (e.g. limestone vs dolomite teal, halite vs
    # anhydrite pink) on top of their distinct patterns.
    assert limestone != dolomite
    assert halite != anhydrite
    assert viewer._lithology_pattern("Sandstone") == "dots"
    assert viewer._lithology_pattern("Limestone") == "block_grid"
    assert viewer._lithology_pattern("Dolomite") == "diagonal"
    assert viewer._lithology_pattern("Halite") == "grid"
    assert viewer._lithology_pattern("Anhydrite") == "carets"


def test_log_viewer_uses_tiles_for_carbonate_lithology_patterns():
    viewer = create_log_viewer(
        _plot_well(),
        template_defaults="auto",
        depth_range=(1000.0, 1009.5),
    )

    assert log_viewer_module.LITHOLOGY_TILE_PATTERNS[viewer._lithology_pattern("Chalk")] == "chalk"
    assert log_viewer_module.LITHOLOGY_TILE_PATTERNS[viewer._lithology_pattern("Limestone")] == "limestone"
    assert log_viewer_module.LITHOLOGY_TILE_PATTERNS[viewer._lithology_pattern("Dolomite")] == "dolomite"
    assert log_viewer_module.LITHOLOGY_TILE_PATTERNS[viewer._lithology_pattern("Calcareous Shale")] == "calcareous_clay"
    assert log_viewer_module.LITHOLOGY_TILE_PATTERNS[viewer._lithology_pattern("Marlstone")] == "marl"

    for tile_name in ("limestone", "calcareous_clay", "marl"):
        tile = viewer._lithology_tile_rgba(tile_name)
        assert tile.shape[-1] == 4
        assert tile[..., 3].max() > 0
        assert tile[..., 3].min() == 0

    dolomite = viewer._lithology_tile_rgba("dolomite")
    upper = np.argwhere(dolomite[: dolomite.shape[0] // 2, :, 3] > 0)
    lower = np.argwhere(dolomite[dolomite.shape[0] // 2 :, :, 3] > 0)
    assert float(np.mean(upper[:, 1])) > float(np.mean(lower[:, 1]))


def test_log_viewer_lithology_legend_deduplicates_same_display_label():
    depth = np.arange(1000.0, 1004.0, 0.5)
    well = Well("DUPLICATE-LITHOLOGY-LABELS")
    well.depth = depth
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("LITHOLOGY_GEOLINK", np.where(depth < 1002.0, 5.0, 6.0), "", "Lithology")

    viewer = create_log_viewer(
        well,
        template_defaults="auto",
        depth_range=(1000.0, 1003.5),
        lithology_mapping={5: "Silt", 6: "Silt"},
    )
    texts = _viewer_texts(viewer)

    assert texts.count("Siltstone/Silt") == 1


def test_log_viewer_groups_builtin_lithology_legend_to_common_families():
    viewer = create_log_viewer(
        _plot_well(),
        template_defaults="auto",
        depth_range=(1000.0, 1009.5),
    )

    common_labels = {
        viewer._lithology_legend_label(label)
        for label in log_viewer_module.GEOLINK_LITHOLOGY_CODE_LABELS.values()
    }

    assert "Sandstone" in common_labels
    assert "Shale/Claystone" in common_labels
    assert "Limestone/Chalk" in common_labels
    assert "Marl/Calcareous Shale" in common_labels
    assert "Evaporite" in common_labels
    assert len(common_labels) < len(log_viewer_module.GEOLINK_LITHOLOGY_CODE_LABELS)
    assert viewer._lithology_legend_label("Workbook Shale") == "Workbook Shale"


def test_log_viewer_all_geolink_codes_plot_common_lithology_legend():
    depth = np.arange(1000.0, 1000.0 + 36.0, 1.0)
    well = Well("ALL-GEOLINK-LITHOLOGY")
    well.depth = depth
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("LITHOLOGY_GEOLINK", np.arange(1.0, 37.0), "", "Lithology")

    viewer = create_log_viewer(
        well,
        template_defaults="auto",
        depth_range=(float(depth.min()), float(depth.max())),
    )
    texts = _viewer_texts(viewer)

    assert "Sandstone" in texts
    assert "Limestone/Chalk" in texts
    assert "Evaporite" in texts
    assert "Igneous/Volcanic" in texts
    assert "Chert/Siliceous" in texts
    assert "Metamorphic" in texts
    assert "Silty Sand" not in texts
    assert "Volcanic Tuff" not in texts


def test_log_viewer_lithology_defaults_keep_labels_and_use_pattern_styles(monkeypatch):
    monkeypatch.setattr(log_viewer_module, "GEOLINK_LITHOLOGY_MAPPING_FILENAMES", ())
    log_viewer_module._default_geolink_lithology_mapping.cache_clear()
    try:
        viewer = create_log_viewer(_plot_well(), template_defaults="auto", depth_range=(1000.0, 1009.5))
        assert viewer._lithology_label(1.0, "LITHOLOGY_GEOLINK") == "Sandstone"
        assert viewer._lithology_label(7.0, "LITHOLOGY_GEOLINK") == "Silty Shale"
        assert viewer._lithology_label(21.0, "LITHOLOGY_GEOLINK") == "Halite"
        assert viewer._lithology_label(22.0, "LITHOLOGY_GEOLINK") == "Anhydrite"
        assert viewer._lithology_label(32.0, "LITHOLOGY_GEOLINK") == "Polyhalite"
        assert viewer._lithology_hatch("Sandstone") == ".."
        assert viewer._lithology_hatch("Silty Shale") == "----"
        assert viewer._lithology_hatch("Halite") == "++"
        assert viewer._lithology_hatch("Polyhalite") == "xx"
    finally:
        log_viewer_module._default_geolink_lithology_mapping.cache_clear()


def test_log_viewer_adds_units_and_formation_tops():
    viewer = create_log_viewer(
        _plot_well(),
        template_defaults="petrophysics",
        depth_range=(1000.0, 1009.5),
        formation_tops={"Reservoir A Top": 1002.0, "Reservoir B Top": 1007.0},
        formation_name="Reservoir A",
    )
    texts = _viewer_texts(viewer)

    assert "GR" in texts
    assert "[API]" in texts
    assert "Clean Rock" in texts
    assert "RSHAL" in texts
    assert "Shallow" not in texts
    assert "Reservoir A" in texts
    assert "Reservoir B" in texts
    assert viewer.fig is not None
    assert viewer.fig._suptitle is not None
    assert viewer.fig._suptitle.get_text() == "PLOT-1 | Reservoir A"


def test_log_viewer_bad_hole_shading_has_legend():
    well = _plot_well()
    active = np.where((well.depth >= 1002.0) & (well.depth <= 1004.0), 1.0, 0.0)
    well.add_curve("BADHOLEFLAG", active, "", "Bad Hole")

    viewer = create_log_viewer(well, template_defaults="auto", depth_range=(1000.0, 1009.5))
    texts = _viewer_texts(viewer)

    assert "BADHOLE" in texts


def test_log_viewer_custom_xml_supports_aliases_and_cumulative_track(tmp_path: Path):
    xml_path = tmp_path / "template.xml"
    xml_path.write_text(
        """
<graph name="custom">
  <track display_name="Depth" depth_track="true" width="0.5" tick_spacing="2" number_spacing="4"/>
  <track display_name="Lithology" track_type="lithology" width="0.8">
    <curve curve_name="LITHOLOGY_GEOLINK" display_name="Lithology" fill_color_map="tab20"/>
  </track>
  <track display_name="Resistivity" scale="log" width="1.0">
    <curve curve_name="RSHA" display_name="Shallow" left="0.2" right="2000" color="green"/>
  </track>
  <track display_name="Minerals" cumulative="true" left="0" right="1" width="1.0">
    <curve curve_name="VCL" display_name="Clay" color="brown"/>
    <curve curve_name="VQUAR" display_name="Quartz" color="gold"/>
    <curve curve_name="VCALC" display_name="Calcite" color="gray"/>
  </track>
</graph>
""",
        encoding="utf-8",
    )

    template = load_log_viewer_template(xml_path)
    assert template.name == "custom"
    assert len(template.tracks) == 4

    viewer = create_log_viewer(_plot_well(), template_xml_path=xml_path, strict=True)
    assert viewer.fig is not None
    assert len(viewer.axes) == 3
    assert viewer.warnings == []


def test_log_viewer_comprehensive_template_accepts_derived_curve_aliases(tmp_path: Path):
    depth = np.arange(1000.0, 1010.0, 0.5)
    well = Well("DERIVED-PLOT")
    well.depth = depth
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("LITHO", np.where(depth < 1005, 1.0, 4.0), "", "Lithology")
    well.add_curve("SGR", np.linspace(40.0, 115.0, depth.size), "API", "Spectral GR")
    well.add_curve("RDEP", np.geomspace(2.0, 300.0, depth.size), "ohm.m", "Deep Resistivity")
    well.add_curve("RHOB", np.linspace(2.15, 2.55, depth.size), "g/cc", "Density")
    well.add_curve("NPHI", np.linspace(0.28, 0.08, depth.size), "v/v", "Neutron")
    well.add_curve("DTC", np.linspace(120.0, 65.0, depth.size), "us/ft", "Compressional Sonic")
    well.add_curve("VSHALE_GR_LINEAR", np.linspace(0.05, 0.65, depth.size), "v/v", "Shale Volume")
    well.add_curve("PHIT_GAS_CORR", np.linspace(0.24, 0.07, depth.size), "v/v", "Gas-corrected Porosity")
    well.add_curve("PHIE", np.linspace(0.22, 0.04, depth.size), "v/v", "Effective Porosity")
    well.add_curve("SW_ARCHIE", np.linspace(0.28, 0.88, depth.size), "v/v", "Archie Sw")
    well.add_curve("PERM_TIMUR", np.geomspace(0.04, 400.0, depth.size), "mD", "Timur Permeability")
    well.add_curve("RQI", np.geomspace(0.01, 4.0, depth.size), "", "RQI")

    viewer = create_log_viewer(
        well,
        template_defaults="comprehensive",
        depth_range=(1000.0, 1009.5),
    )

    assert viewer.fig is not None
    assert len(viewer.axes) == 11
    assert viewer.curve_lookup["GR"] == "SGR"
    assert viewer.curve_lookup["VCL"] == "VSHALE_GR_LINEAR"
    assert viewer.curve_lookup["SW"] == "SW_ARCHIE"
    assert viewer.curve_lookup["PERM"] == "PERM_TIMUR"

    output_file = tmp_path / "geolink-full.png"
    viewer.save(output_file)
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_log_viewer_auto_template_adapts_to_available_and_unfamiliar_curves(tmp_path: Path):
    depth = np.arange(1000.0, 1012.0, 0.5)
    well = Well("ADAPTIVE-PLOT")
    well.depth = depth
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("GR", np.linspace(40.0, 120.0, depth.size), "API", "Gamma Ray")
    well.add_curve("RDEP", np.geomspace(2.0, 300.0, depth.size), "ohm.m", "Deep Resistivity")
    well.add_curve("PHIC", np.linspace(0.21, 0.09, depth.size), "v/v", "Core Porosity")
    well.add_curve("KH_CORE", np.geomspace(0.1, 200.0, depth.size), "mD", "Core Kh")
    well.add_curve("KV_CORE", np.geomspace(0.05, 80.0, depth.size), "mD", "Core Kv")
    well.add_curve("CUSTOM_PRESSURE", np.linspace(3100.0, 3300.0, depth.size), "psi", "Pressure")

    template = create_adaptive_log_viewer_template(well)
    track_names = [track.display_name for track in template.tracks]
    assert "GAMMA / HOLE" in track_names
    assert "POROSITY" in track_names
    assert "PERMEABILITY" in track_names
    assert not any(name.startswith("OTHER") for name in track_names)

    viewer = create_log_viewer(well, depth_range=(1000.0, 1011.5))
    assert viewer.fig is not None
    assert viewer.curve_lookup["PHIC"] == "PHIC"
    assert viewer.curve_lookup["KH_CORE"] == "KH_CORE"
    assert "KV_CORE" not in viewer.curve_lookup
    assert "CUSTOM_PRESSURE" not in viewer.curve_lookup

    output_file = tmp_path / "adaptive.png"
    viewer.save(output_file)
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_log_viewer_auto_template_shades_sw_from_zero_side():
    template = create_adaptive_log_viewer_template(_plot_well())
    sw_spec = next(
        spec
        for track in template.tracks
        for spec in track.curves
        if spec.curve_name == "SW"
    )

    assert sw_spec.left == 1
    assert sw_spec.right == 0
    assert sw_spec.fill == "right"


def test_log_viewer_auto_template_uses_distinct_dtc_and_pef_colors():
    depth = np.arange(1000.0, 1010.0, 0.5)
    well = Well("DTC-PEF-PLOT")
    well.depth = depth
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("DTC", np.linspace(110.0, 70.0, depth.size), "us/ft", "Compressional Sonic")
    well.add_curve("PEF", np.linspace(1.8, 5.0, depth.size), "b/e", "Photoelectric Factor")

    template = create_adaptive_log_viewer_template(well)
    density_track = next(track for track in template.tracks if track.display_name == "DENSITY / NEUTRON")
    colors = {spec.curve_name: spec.color for spec in density_track.curves}

    assert colors["DTC"] != colors["PEF"]


def test_log_viewer_normalizes_common_display_units():
    depth = np.arange(1000.0, 1010.0, 0.5)
    well = Well("UNIT-PLOT")
    well.depth = depth
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("RDEP", np.geomspace(1.0, 100.0, depth.size), "", "Deep Resistivity")
    well.add_curve("DTC", np.linspace(110.0, 70.0, depth.size), "", "Compressional Sonic")
    well.add_curve("RHOB", np.linspace(2.2, 2.6, depth.size), "", "Bulk Density")
    well.add_curve("SW", np.linspace(0.2, 0.8, depth.size), "v/v", "Water Saturation")
    well.add_curve("BVW", np.linspace(0.05, 0.15, depth.size), "v/v", "Bulk Volume Water")

    viewer = create_log_viewer(well, template_defaults="auto", depth_range=(1000.0, 1009.5))
    texts = _viewer_texts(viewer)

    assert "[ohm.m], log" in texts
    assert "[us/ft]" in texts
    assert "[g/cc]" in texts
    assert "[frac]" in texts
    assert viewer._curve_unit(well.get("SW")) == "frac"
    assert viewer._curve_unit(well.get("BVW")) == "v/v"


def test_log_viewer_marks_apparent_porosity_labels():
    well = _plot_well()
    well.get("PHIT").metadata.description = "Apparent total porosity, density-derived from PHID"
    well.get("PHIT").metadata.custom["porosity_apparent"] = True
    well.get("PHIE").metadata.description = "Apparent effective porosity from density-derived PHIT"
    well.get("PHIE").metadata.custom["porosity_apparent"] = True

    viewer = create_log_viewer(well, template_defaults="auto", depth_range=(1000.0, 1009.5))
    texts = "\n".join(_viewer_texts(viewer))

    assert "PHIT app" in texts
    assert "PHIE app" in texts


def test_log_viewer_mineral_volumes_close_stack_with_phie_porosity_wedge():
    template = create_adaptive_log_viewer_template(_mineral_well())
    track = next(track for track in template.tracks if track.display_name == "MINERAL VOLUMES")

    # The template still lists only the solid mineral curves; the porosity wedge is
    # added at render time (the actual PHIE curve, not a synthetic 1 - sum remainder).
    assert [spec.display_name for spec in track.curves] == ["VCLAY", "VQUAR", "VCALC", "VDOLO"]

    viewer = create_log_viewer(_mineral_well(), template_defaults="auto", depth_range=(1000.0, 1009.5))
    texts = _viewer_texts(viewer)

    assert "MINERAL VOLUMES" in texts
    assert "VCLAY" in texts
    mineral_header = _header_texts_for_track(viewer, "MINERAL VOLUMES")
    # Effective porosity (PHIE) closes the wet-clay volumetric stack to 1.0.
    assert "PHIE" in mineral_header
    assert mineral_header.count("[v/v]") == 1
    assert mineral_header.count("0") == 1
    assert mineral_header.count("0.5") == 1
    assert mineral_header.count("1") == 1
    assert "VCLAY [v/v]" not in mineral_header


def test_log_viewer_drops_mineral_volumes_for_geolink_auto_plots():
    template = create_adaptive_log_viewer_template(_plot_well())
    track_names = [track.display_name for track in template.tracks]

    assert "MINERAL VOLUMES" not in track_names

    viewer = create_log_viewer(_plot_well(), template_defaults="auto", depth_range=(1000.0, 1009.5))
    texts = _viewer_texts(viewer)

    assert "MINERAL VOLUMES" not in texts
    assert "VCLAY" not in texts


def test_log_viewer_mineral_volumes_accept_lower_level_solver_aliases():
    depth = np.arange(1000.0, 1002.0, 0.5)
    well = Well("LOWER-SOLVER-NAMES")
    well.depth = depth
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("VCL", np.full(depth.size, 0.20), "v/v", "Clay Volume")
    well.add_curve("VQTZ", np.full(depth.size, 0.40), "v/v", "Volume quartz")
    well.add_curve("VCAL", np.full(depth.size, 0.10), "v/v", "Volume calcite")
    well.add_curve("VDOL", np.full(depth.size, 0.05), "v/v", "Volume dolomite")
    well.add_curve("VANH", np.full(depth.size, 0.05), "v/v", "Volume anhydrite")
    well.add_curve("PHIE", np.full(depth.size, 0.20), "v/v", "Effective Porosity")

    viewer = create_log_viewer(well, template_defaults="auto", depth_range=(1000.0, 1001.5))
    texts = _viewer_texts(viewer)

    assert viewer.curve_lookup["VCLAY"] == "VCL"
    assert viewer.curve_lookup["VQUAR"] == "VQTZ"
    assert viewer.curve_lookup["VCALC"] == "VCAL"
    assert viewer.curve_lookup["VDOLO"] == "VDOL"
    assert viewer.curve_lookup["VANHY"] == "VANH"
    assert "PHIE (white)" not in texts


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
