"""Regenerate the GeoLink lithology pattern-check figures.

Outputs (overwritten in place):
  outputs/lithology_pattern_check/geolink_lithology_pattern_key_all_codes.png
  outputs/lithology_pattern_check/geolink_many_lithologies.png

Both figures are driven entirely by the library's lithology styling
(``LITHOLOGY_LABEL_COLORS`` / ``_lithology_color`` / ``_lithology_pattern``),
so they always reflect the official GeoLink / Kingdom palette defined in
``petrophys/visualization/log_viewer.py``.

Run:  python scripts/generate_lithology_pattern_key.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from petrophys import Well
from petrophys.visualization import create_log_viewer, use_plot_style, save_figure
from petrophys.visualization.log_viewer import (
    GEOLINK_LITHOLOGY_CODE_LABELS,
    LogViewer,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "lithology_pattern_check"


def _styling_viewer() -> LogViewer:
    """A minimal viewer instance used only for its lithology-styling methods."""
    depth = np.arange(1000.0, 1010.0, 0.5)
    well = Well("PATTERN-KEY")
    well.depth = depth
    well.depth_unit = "m"
    well.add_curve("DEPT", depth, "m", "Depth")
    well.add_curve("GR", np.linspace(40.0, 110.0, depth.size), "API", "Gamma Ray")
    well.add_curve("LITHOLOGY_GEOLINK", np.ones(depth.size), "", "Lithology")
    return LogViewer(well, template_defaults="auto")


def _draw_swatch(fig, rect, viewer, cmap, code: int, label: str) -> None:
    ax = fig.add_axes(rect)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    color = viewer._lithology_color(float(code), label, cmap, 0, 1)
    pattern = viewer._lithology_pattern(label)
    hatch = "" if pattern else viewer._lithology_hatch(label)
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            facecolor=color,
            edgecolor="0.25",
            linewidth=0.9,
            hatch=hatch,
        )
    )
    if pattern:
        # Legend swatch: render at the reference density (0.05) rather than the
        # full-tile fraction (~1.0), which would cram the pattern into a barcode.
        viewer._draw_lithology_pattern(ax, label, 0.0, 1.0, density_fraction=0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("0.25")
        spine.set_linewidth(0.9)


def generate_pattern_key(out_path: Path) -> None:
    viewer = _styling_viewer()
    cmap = plt.get_cmap("tab20")
    codes = sorted(GEOLINK_LITHOLOGY_CODE_LABELS)

    n_cols, n_rows = 3, 12
    fig = plt.figure(figsize=(12.0, 6.6))
    fig.suptitle(
        "GeoLink Lithology Pattern Key - Codes 1-36",
        fontsize=14,
        fontweight="bold",
        y=0.975,
    )

    top, bottom = 0.915, 0.02
    row_h = (top - bottom) / n_rows
    col_x = [0.025, 0.355, 0.685]
    sw_w, sw_h = 0.085, row_h * 0.66

    for code in codes:
        label = GEOLINK_LITHOLOGY_CODE_LABELS[code]
        row = (code - 1) // n_cols
        col = (code - 1) % n_cols
        y = top - (row + 1) * row_h + (row_h - sw_h) / 2.0
        _draw_swatch(fig, [col_x[col], y, sw_w, sw_h], viewer, cmap, code, label)
        fig.text(
            col_x[col] + sw_w + 0.012,
            y + sw_h / 2.0,
            f"{code:02d}  {label}",
            fontsize=9.0,
            va="center",
            ha="left",
        )

    save_figure(fig, out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def generate_many_lithologies(out_path: Path) -> None:
    # A representative sequence spanning the major families. The adaptive
    # template renders a Lithology track + a Gamma/Hole track for this well.
    sequence = [1, 3, 5, 8, 11, 9, 13, 15, 16, 18, 28, 22, 21, 26, 29, 30, 17, 1]
    depth = np.arange(1000.0, 1210.0, 0.5)
    band = np.linspace(0, len(sequence), depth.size, endpoint=False).astype(int)
    band = np.clip(band, 0, len(sequence) - 1)
    litho = np.array([sequence[b] for b in band], dtype=float)

    gr = 75.0 + 55.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, depth.size))

    well = Well("Synthetic Lithology")
    well.depth = depth
    well.depth_unit = "m"
    well.add_curve("DEPT", depth, "m", "Depth")
    well.add_curve("LITHOLOGY_GEOLINK", litho, "", "Lithology")
    well.add_curve("GR", gr, "API", "Gamma Ray")
    well.add_curve("CALI", np.full(depth.size, 8.5), "in", "Caliper")
    well.add_curve("BS", np.full(depth.size, 8.5), "in", "Bit Size")

    viewer = create_log_viewer(
        well,
        template_defaults="auto",
        depth_range=(float(depth[0]), float(depth[-1])),
    )
    fig = viewer.fig
    fig.suptitle("GeoLink Synthetic Lithology", fontsize=13, fontweight="bold")
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate the GeoLink lithology pattern-check figures "
        "(existing PNGs in the output directory are overwritten)."
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR,
                        help=f"Directory for the two PNGs (default: {OUT_DIR}).")
    out_dir = parser.parse_args(argv).output_dir
    use_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_pattern_key(out_dir / "geolink_lithology_pattern_key_all_codes.png")
    generate_many_lithologies(out_dir / "geolink_many_lithologies.png")


if __name__ == "__main__":
    main()
