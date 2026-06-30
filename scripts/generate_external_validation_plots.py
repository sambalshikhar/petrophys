"""Regenerate the external-validation plots (Taranaki Kapuni-15 + Teapot Dome).

Re-plots the already-computed LAS files in
``outputs/external_validation_20260616/`` into ``.../plots/`` using the current
library styling (so they pick up the official GeoLink / Kingdom lithology
palette). The computed curves themselves are not recomputed.

Per well it writes:  <well>_auto_log_viewer.png, <well>_triple_combo.png,
<well>_interpretation.png  and refreshes plots/plot_summary.json.

Run:  python scripts/generate_external_validation_plots.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from petrophys.io import read_las
from petrophys.visualization import (
    create_log_viewer,
    create_triple_combo,
)

BASE = Path(__file__).resolve().parents[1] / "outputs" / "external_validation_20260616"
PLOTS = BASE / "plots"

WELLS = {
    "taranaki": {
        "las": "taranaki_Kapuni-15_computed.las",
        "well": "Kapuni-15",
        "depth_unit": "m",
        "depth_range": (2250.6554, 2750.6554),
    },
    "teapot": {
        "las": "teapot_490252319500_AIT_TLD_MCFL_CNL_040PUP_computed.las",
        "well": "48X-28",
        "depth_unit": "ft",
        "depth_range": (2142.5, 2642.5),
    },
}


def _plot_well(tag: str, cfg: dict) -> dict:
    well = read_las(BASE / cfg["las"])
    depth_range = tuple(cfg["depth_range"])
    plots = []

    viewer = create_log_viewer(well, template_defaults="auto", depth_range=depth_range)
    viewer.save(PLOTS / f"{tag}_auto_log_viewer.png")
    plots.append({"name": "auto_log_viewer",
                  "path": str((PLOTS / f"{tag}_auto_log_viewer.png").relative_to(BASE.parents[1])),
                  "warnings": list(getattr(viewer, "warnings", []))})
    plt.close(viewer.fig)

    combo = create_triple_combo(well, depth_range=depth_range)
    combo.save(str(PLOTS / f"{tag}_triple_combo.png"))
    plots.append({"name": "triple_combo",
                  "path": str((PLOTS / f"{tag}_triple_combo.png").relative_to(BASE.parents[1])),
                  "warnings": []})
    plt.close(combo.fig)

    interp = create_log_viewer(well, template_defaults="interpretation", depth_range=depth_range)
    interp.save(str(PLOTS / f"{tag}_interpretation.png"))
    plots.append({"name": "interpretation",
                  "path": str((PLOTS / f"{tag}_interpretation.png").relative_to(BASE.parents[1])),
                  "warnings": []})
    plt.close(interp.fig)

    print(f"{tag}: regenerated {len(plots)} plots ({cfg['well']})")
    return {
        "well": cfg["well"],
        "depth_unit": cfg["depth_unit"],
        "depth_range": [depth_range[0], depth_range[1]],
        "curve_count": len(well.curves),
        "workflow_warnings": [],
        "plots": plots,
    }


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary = {tag: _plot_well(tag, cfg) for tag, cfg in WELLS.items()}
    (PLOTS / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {PLOTS / 'plot_summary.json'}")


if __name__ == "__main__":
    main()
