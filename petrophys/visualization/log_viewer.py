"""
Template-driven well log plotting.

This module provides a PetroPy-style log viewer built around reusable XML
templates while using the local :class:`petrophys.base.well.Well` data model.
The implementation is intentionally lightweight: templates describe tracks,
curves, ranges, fills, log scaling, and cumulative mineral/volume tracks, while
the plotting code handles missing curves gracefully unless strict mode is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import xml.etree.ElementTree as ET

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - exercised by import environment
    HAS_MATPLOTLIB = False


CURVE_ALIAS_CANDIDATES: Dict[str, List[str]] = {
    "DEPT": ["DEPT", "DEPTH", "MD"],
    "DEPTH": ["DEPT", "DEPTH", "MD"],
    "GR": ["GR", "GRC", "GR_CAL", "CGR"],
    "CALI": ["CALI", "CAL", "HCAL", "DCAL"],
    "BS": ["BS", "BIT", "BITSIZE"],
    "RDEP": ["RDEP", "RT", "LLD", "ILD", "AT90", "RD"],
    "RT": ["RDEP", "RT", "LLD", "ILD", "AT90", "RD"],
    "RMED": ["RMED", "ILM", "LLS", "AT60", "RESM"],
    "RSHA": ["RSHA", "RSHAL", "RXO", "RESS", "MSFL", "SFL", "AT10"],
    "RXO": ["RXO", "RSHA", "RSHAL", "RESS", "MSFL", "SFL"],
    "RHOB": ["RHOB", "DEN", "ZDEN"],
    "NPHI": ["NPHI", "TNPH", "NEUT"],
    "DTC": ["DTC", "DT", "AC"],
    "DTS": ["DTS", "DTSM"],
    "PEF": ["PEF", "PE"],
    "VCL": ["VCL", "VSH", "VSH_GR", "VCLAY"],
    "PHIT": ["PHIT", "PHI_T"],
    "PHIE": ["PHIE_CC", "PHIE", "PHI_E"],
    "PHIE_CC": ["PHIE_CC", "PHIE"],
    "SW": ["SW", "SWA", "SWT", "SWT-LOG"],
    "SXO": ["SXO", "SWXO", "SWXOA"],
    "PERM": ["PERM_CC", "PERM", "K", "KLOG"],
    "PERM_CC": ["PERM_CC", "PERM"],
    "VQUAR": ["VQUAR", "VSAND", "VQTZ"],
    "VCALC": ["VCALC", "VLIM", "VCAL"],
    "VDOLO": ["VDOLO", "VDOL"],
    "VANHY": ["VANHY"],
}


DEFAULT_LOG_VIEWER_TEMPLATES: Dict[str, str] = {
    "raw": """
<template name="raw">
  <track display_name="Depth" width="0.55" depth_track="true" tick_spacing="50" number_spacing="250"/>
  <track display_name="GR / CALI" width="1.15">
    <curve curve_name="GR" display_name="GR" left="0" right="150" color="forestgreen" fill="left" fill_color="#d8efd2"/>
    <curve curve_name="CALI" display_name="CALI" left="6" right="16" color="black"/>
    <curve curve_name="BS" display_name="BS" left="6" right="16" color="dimgray" line_style="--"/>
  </track>
  <track display_name="Resistivity" width="1.15" scale="log" major_lines="4">
    <curve curve_name="RDEP" display_name="Deep" left="0.2" right="2000" color="red"/>
    <curve curve_name="RMED" display_name="Medium" left="0.2" right="2000" color="blue"/>
    <curve curve_name="RSHA" display_name="Shallow" left="0.2" right="2000" color="green"/>
  </track>
  <track display_name="Density / Neutron" width="1.15">
    <curve curve_name="RHOB" display_name="RHOB" left="1.95" right="2.95" color="red"/>
    <curve curve_name="NPHI" display_name="NPHI" left="0.45" right="-0.15" color="blue"/>
  </track>
</template>
""",
    "triple_combo": """
<template name="triple_combo">
  <track display_name="Depth" width="0.55" depth_track="true" tick_spacing="50" number_spacing="250"/>
  <track display_name="GR / CALI" width="1.15">
    <curve curve_name="GR" display_name="GR" left="0" right="150" color="forestgreen" fill="left" fill_color="#d8efd2"/>
    <curve curve_name="CALI" display_name="CALI" left="6" right="16" color="black"/>
    <curve curve_name="BS" display_name="BS" left="6" right="16" color="dimgray" line_style="--"/>
  </track>
  <track display_name="Resistivity" width="1.15" scale="log" major_lines="4">
    <curve curve_name="RDEP" display_name="Deep" left="0.2" right="2000" color="red"/>
    <curve curve_name="RMED" display_name="Medium" left="0.2" right="2000" color="blue"/>
    <curve curve_name="RSHA" display_name="Shallow" left="0.2" right="2000" color="green"/>
  </track>
  <track display_name="Density / Neutron" width="1.15">
    <curve curve_name="RHOB" display_name="RHOB" left="1.95" right="2.95" color="red"/>
    <curve curve_name="NPHI" display_name="NPHI" left="0.45" right="-0.15" color="blue"/>
  </track>
</template>
""",
    "petrophysics": """
<template name="petrophysics">
  <track display_name="Depth" width="0.55" depth_track="true" tick_spacing="50" number_spacing="250"/>
  <track display_name="GR / VCL" width="1.10">
    <curve curve_name="GR" display_name="GR" left="0" right="150" color="forestgreen" fill="left" fill_color="#d8efd2"/>
    <curve curve_name="VCL" display_name="VCL" left="0" right="1" color="saddlebrown" fill="left" fill_color="#c49a6c"/>
  </track>
  <track display_name="Resistivity" width="1.10" scale="log" major_lines="4">
    <curve curve_name="RDEP" display_name="Deep" left="0.2" right="2000" color="red"/>
    <curve curve_name="RMED" display_name="Medium" left="0.2" right="2000" color="blue"/>
    <curve curve_name="RSHA" display_name="Shallow" left="0.2" right="2000" color="green"/>
  </track>
  <track display_name="Porosity" width="1.10">
    <curve curve_name="PHIT" display_name="PHIT" left="0.45" right="0" color="navy"/>
    <curve curve_name="PHIE_CC" display_name="PHIE_CC" left="0.45" right="0" color="darkmagenta" line_style="--" fill="right" fill_color="#f3d35b"/>
    <curve curve_name="PHIE" display_name="PHIE" left="0.45" right="0" color="crimson"/>
  </track>
  <track display_name="Water Saturation" width="1.10">
    <curve curve_name="SW" display_name="SW" left="1" right="0" color="blue" fill="left" fill_color="#9ecae1"/>
    <curve curve_name="SXO" display_name="SXO" left="1" right="0" color="deepskyblue" line_style="--"/>
  </track>
  <track display_name="Permeability" width="1.10" scale="log" major_lines="4">
    <curve curve_name="PERM_CC" display_name="PERM_CC" left="0.001" right="10000" color="darkmagenta" line_style="--"/>
    <curve curve_name="PERM" display_name="PERM" left="0.001" right="10000" color="black"/>
  </track>
</template>
""",
    "minerals": """
<template name="minerals">
  <track display_name="Depth" width="0.55" depth_track="true" tick_spacing="50" number_spacing="250"/>
  <track display_name="Lithology Volumes" width="1.35" cumulative="true" left="0" right="1">
    <curve curve_name="VCL" display_name="Clay" color="#8c6d31" fill_color="#8c6d31"/>
    <curve curve_name="VQUAR" display_name="Quartz" color="#f2d16b" fill_color="#f2d16b"/>
    <curve curve_name="VCALC" display_name="Calcite" color="#bdbdbd" fill_color="#bdbdbd"/>
    <curve curve_name="VDOLO" display_name="Dolomite" color="#969696" fill_color="#969696"/>
    <curve curve_name="VANHY" display_name="Anhydrite" color="#9ecae1" fill_color="#9ecae1"/>
  </track>
  <track display_name="Porosity / SW" width="1.10">
    <curve curve_name="PHIE_CC" display_name="PHIE_CC" left="0.45" right="0" color="darkmagenta" line_style="--"/>
    <curve curve_name="PHIE" display_name="PHIE" left="0.45" right="0" color="crimson"/>
    <curve curve_name="SW" display_name="SW" left="1" right="0" color="blue"/>
  </track>
</template>
""",
}


@dataclass
class LogCurveSpec:
    curve_name: str
    display_name: Optional[str] = None
    left: Optional[float] = None
    right: Optional[float] = None
    color: str = "black"
    line_width: float = 0.65
    line_style: str = "-"
    alpha: float = 1.0
    fill: Optional[str] = None
    fill_color: Optional[str] = None
    fill_alpha: float = 0.28
    left_cutoff_fill: Optional[float] = None
    right_cutoff_fill: Optional[float] = None
    left_crossover: Optional[str] = None
    right_crossover: Optional[str] = None


@dataclass
class LogTrackSpec:
    display_name: str = ""
    width: float = 1.0
    scale: str = "linear"
    left: float = 0.0
    right: float = 1.0
    cumulative: bool = False
    depth_track: bool = False
    major_lines: int = 5
    tick_spacing: Optional[float] = None
    number_spacing: Optional[float] = None
    curves: List[LogCurveSpec] = field(default_factory=list)


@dataclass
class LogViewerTemplate:
    name: str = "custom"
    top: Optional[float] = None
    height: Optional[float] = None
    tracks: List[LogTrackSpec] = field(default_factory=list)


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Optional[str], default: int) -> int:
    parsed = _as_float(value)
    if parsed is None:
        return default
    return int(parsed)


def _first_attr(element: ET.Element, names: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    for name in names:
        if name in element.attrib:
            return element.attrib[name]
    return default


def _parse_curve(element: ET.Element) -> LogCurveSpec:
    name = _first_attr(element, ("curve_name", "mnemonic", "name"))
    if not name:
        raise ValueError("Template curve is missing curve_name/mnemonic/name")

    return LogCurveSpec(
        curve_name=str(name).upper(),
        display_name=_first_attr(element, ("display_name", "label")),
        left=_as_float(_first_attr(element, ("left", "min"))),
        right=_as_float(_first_attr(element, ("right", "max"))),
        color=_first_attr(element, ("color", "line_color"), "black") or "black",
        line_width=_as_float(_first_attr(element, ("line_width", "width", "linewidth")), 0.65) or 0.65,
        line_style=_first_attr(element, ("line_style", "linestyle"), "-") or "-",
        alpha=_as_float(element.attrib.get("alpha"), 1.0) or 1.0,
        fill=_first_attr(element, ("fill", "fill_direction")),
        fill_color=_first_attr(element, ("fill_color", "fillcolor")),
        fill_alpha=_as_float(_first_attr(element, ("fill_alpha", "fillalpha")), 0.28) or 0.28,
        left_cutoff_fill=_as_float(element.attrib.get("left_cutoff_fill")),
        right_cutoff_fill=_as_float(element.attrib.get("right_cutoff_fill")),
        left_crossover=_first_attr(element, ("left_crossover", "left_crossover_curve")),
        right_crossover=_first_attr(element, ("right_crossover", "right_crossover_curve")),
    )


def _parse_track(element: ET.Element) -> LogTrackSpec:
    track = LogTrackSpec(
        display_name=_first_attr(element, ("display_name", "name", "label"), "") or "",
        width=_as_float(element.attrib.get("width"), 1.0) or 1.0,
        scale=(_first_attr(element, ("scale", "xscale"), "linear") or "linear").lower(),
        left=_as_float(_first_attr(element, ("left", "min")), 0.0) or 0.0,
        right=_as_float(_first_attr(element, ("right", "max")), 1.0) or 1.0,
        cumulative=_as_bool(element.attrib.get("cumulative")),
        depth_track=_as_bool(element.attrib.get("depth_track")),
        major_lines=max(0, _as_int(element.attrib.get("major_lines"), 5)),
        tick_spacing=_as_float(element.attrib.get("tick_spacing")),
        number_spacing=_as_float(element.attrib.get("number_spacing")),
    )
    for child in element:
        if child.tag.lower() == "curve":
            track.curves.append(_parse_curve(child))
    return track


def load_log_viewer_template(
    template_xml_path: Optional[str | Path] = None,
    template_defaults: str = "raw",
) -> LogViewerTemplate:
    """
    Load a log-viewer template from XML or one of the built-in defaults.

    Parameters
    ----------
    template_xml_path : str or Path, optional
        XML file containing ``<template><track><curve .../></track></template>``.
    template_defaults : str
        Built-in template name. Available: ``raw``, ``triple_combo``,
        ``petrophysics``, and ``minerals``.
    """
    if template_xml_path is not None:
        root = ET.parse(template_xml_path).getroot()
    else:
        key = str(template_defaults or "raw").lower()
        if key not in DEFAULT_LOG_VIEWER_TEMPLATES:
            valid = ", ".join(sorted(DEFAULT_LOG_VIEWER_TEMPLATES))
            raise ValueError(f"Unknown log viewer template '{template_defaults}'. Valid templates: {valid}")
        root = ET.fromstring(DEFAULT_LOG_VIEWER_TEMPLATES[key])

    if root.tag.lower() != "template":
        raise ValueError("Log viewer XML root must be <template>")

    template = LogViewerTemplate(
        name=root.attrib.get("name", str(template_defaults or "custom")),
        top=_as_float(root.attrib.get("top")),
        height=_as_float(root.attrib.get("height")),
    )
    for child in root:
        if child.tag.lower() == "track":
            template.tracks.append(_parse_track(child))

    if not template.tracks:
        raise ValueError("Log viewer template must contain at least one track")
    return template


class LogViewer:
    """
    PetroPy-style template-driven well log plotter.

    Parameters
    ----------
    well : Well
        Local ``Well`` object containing a depth array and curves.
    template_xml_path : str or Path, optional
        XML template. When omitted, ``template_defaults`` is used.
    template_defaults : str
        Built-in template name: ``raw``, ``triple_combo``, ``petrophysics``, or
        ``minerals``.
    strict : bool
        If True, missing curves listed by the template raise ``ValueError``.
        If False, they are skipped and recorded in ``warnings``.
    """

    def __init__(
        self,
        well: "Well",
        template_xml_path: Optional[str | Path] = None,
        template_defaults: str = "raw",
        strict: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")
        self.well = well
        self.template = load_log_viewer_template(template_xml_path, template_defaults)
        self.strict = strict
        self.figsize = figsize
        self.fig: Optional[Figure] = None
        self.axes: List[Axes] = []
        self.warnings: List[str] = []

    def create(
        self,
        depth_range: Optional[Tuple[float, float]] = None,
        top: Optional[float] = None,
        height: Optional[float] = None,
    ) -> Tuple[Figure, List[Axes]]:
        """Create and return ``(figure, axes)``."""
        depth = self._depth()
        plot_top, plot_bottom = self._resolve_depth_window(depth, depth_range, top, height)
        width_ratios = [max(track.width, 0.1) for track in self.template.tracks]
        figsize = self.figsize or (sum(width_ratios) * 2.2, 11.0)

        self.fig, axes = plt.subplots(
            1,
            len(self.template.tracks),
            figsize=figsize,
            sharey=True,
            gridspec_kw={"width_ratios": width_ratios},
        )
        self.axes = list(np.atleast_1d(axes))

        for ax, track in zip(self.axes, self.template.tracks):
            self._format_track_axis(ax, track, plot_top, plot_bottom)
            if track.depth_track:
                self._plot_depth_track(ax, track, plot_top, plot_bottom)
            elif track.cumulative:
                self._plot_cumulative_track(ax, track, depth)
            else:
                self._plot_curve_track(ax, track, depth)

        title = self.well.name or getattr(getattr(self.well, "header", None), "well_name", "") or "Unknown Well"
        self.fig.suptitle(title, fontsize=13)
        self.fig.tight_layout(rect=(0, 0, 1, 0.97))
        return self.fig, self.axes

    def show(self) -> None:
        """Display the figure."""
        if self.fig is None:
            self.create()
        plt.show()

    def save(self, filename: str | Path, dpi: int = 150) -> None:
        """Save the plot to disk."""
        if self.fig is None:
            self.create()
        assert self.fig is not None
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    def _depth(self) -> np.ndarray:
        if getattr(self.well, "depth", None) is not None:
            return np.asarray(self.well.depth, dtype=float)
        for name in CURVE_ALIAS_CANDIDATES["DEPT"]:
            curve = self.well.get(name)
            if curve is not None:
                return np.asarray(curve.data, dtype=float)
        raise ValueError("Well has no depth array or DEPT/DEPTH/MD curve")

    def _resolve_depth_window(
        self,
        depth: np.ndarray,
        depth_range: Optional[Tuple[float, float]],
        top: Optional[float],
        height: Optional[float],
    ) -> Tuple[float, float]:
        if depth_range is not None:
            return float(depth_range[0]), float(depth_range[1])
        top_value = top if top is not None else self.template.top
        height_value = height if height is not None else self.template.height
        if top_value is None:
            top_value = float(np.nanmin(depth))
        if height_value is not None:
            return float(top_value), float(top_value + height_value)
        return float(top_value), float(np.nanmax(depth))

    def _format_track_axis(
        self,
        ax: Axes,
        track: LogTrackSpec,
        top: float,
        bottom: float,
    ) -> None:
        ax.set_ylim(bottom, top)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([])
        ax.grid(axis="y", color="0.88", linewidth=0.5)
        ax.set_title(track.display_name, fontsize=9, pad=16)
        if ax is not self.axes[0]:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel(f"Depth [{getattr(self.well, 'depth_unit', '')}]".strip())

        if track.major_lines:
            for x in np.linspace(0.0, 1.0, track.major_lines + 1):
                ax.axvline(x, color="0.90", linewidth=0.45, zorder=0)

    def _plot_depth_track(
        self,
        ax: Axes,
        track: LogTrackSpec,
        top: float,
        bottom: float,
    ) -> None:
        tick_spacing = track.tick_spacing or self._nice_depth_spacing(top, bottom, target=16)
        number_spacing = track.number_spacing or tick_spacing * 5

        ticks = self._depth_ticks(top, bottom, tick_spacing)
        for tick in ticks:
            ax.axhline(tick, color="0.82", linewidth=0.45, zorder=0)

        for tick in self._depth_ticks(top, bottom, number_spacing):
            ax.text(0.50, tick, f"{tick:g}", ha="center", va="center", fontsize=7)
            ax.axhline(tick, color="0.70", linewidth=0.6, zorder=0)

    def _plot_curve_track(self, ax: Axes, track: LogTrackSpec, depth: np.ndarray) -> None:
        label_rows = 0
        for spec in track.curves:
            curve = self._get_curve(spec.curve_name)
            if curve is None:
                self._warn_missing(spec.curve_name)
                continue
            data = np.asarray(curve.data, dtype=float)
            left = spec.left if spec.left is not None else track.left
            right = spec.right if spec.right is not None else track.right
            x = self._normalize(data, left, right, track.scale)
            finite = np.isfinite(x) & np.isfinite(depth)
            if not np.any(finite):
                self._warn_missing(f"{spec.curve_name} (no finite values after scaling)")
                continue

            label = spec.display_name or curve.mnemonic
            fill_color = spec.fill_color or spec.color
            if spec.fill:
                baseline = 0.0 if str(spec.fill).lower() == "left" else 1.0
                ax.fill_betweenx(
                    depth,
                    baseline,
                    x,
                    where=finite,
                    color=fill_color,
                    alpha=spec.fill_alpha,
                    linewidth=0.0,
                    zorder=1,
                )

            self._plot_cutoff_fills(ax, depth, data, x, finite, left, right, track, spec)
            self._plot_crossover_fills(ax, depth, x, finite, left, right, track, spec)
            ax.plot(
                x,
                depth,
                color=spec.color,
                linewidth=spec.line_width,
                linestyle=spec.line_style,
                alpha=spec.alpha,
                label=label,
                zorder=3,
            )
            self._add_curve_header(ax, label_rows, label, left, right, spec.color, track.scale)
            label_rows += 1

    def _plot_cumulative_track(self, ax: Axes, track: LogTrackSpec, depth: np.ndarray) -> None:
        cumulative = np.zeros_like(depth, dtype=float)
        label_rows = 0
        for spec in track.curves:
            curve = self._get_curve(spec.curve_name)
            if curve is None:
                self._warn_missing(spec.curve_name)
                continue
            data = np.asarray(curve.data, dtype=float)
            finite = np.isfinite(data) & np.isfinite(depth)
            if not np.any(finite):
                self._warn_missing(f"{spec.curve_name} (no finite values)")
                continue

            cleaned = np.where(finite, np.clip(data, 0.0, None), 0.0)
            next_cumulative = cumulative + cleaned
            x0 = self._normalize(cumulative, track.left, track.right, track.scale)
            x1 = self._normalize(next_cumulative, track.left, track.right, track.scale)
            fill_color = spec.fill_color or spec.color
            ax.fill_betweenx(
                depth,
                x0,
                x1,
                where=np.isfinite(x0) & np.isfinite(x1),
                color=fill_color,
                alpha=spec.fill_alpha,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(x1, depth, color=spec.color, linewidth=0.45, zorder=2)
            label = spec.display_name or curve.mnemonic
            self._add_curve_header(ax, label_rows, label, track.left, track.right, spec.color, track.scale)
            label_rows += 1
            cumulative = next_cumulative

    def _plot_cutoff_fills(
        self,
        ax: Axes,
        depth: np.ndarray,
        data: np.ndarray,
        x: np.ndarray,
        finite: np.ndarray,
        left: float,
        right: float,
        track: LogTrackSpec,
        spec: LogCurveSpec,
    ) -> None:
        fill_color = spec.fill_color or spec.color
        if spec.left_cutoff_fill is not None:
            cutoff_x = self._normalize_scalar(spec.left_cutoff_fill, left, right, track.scale)
            ax.fill_betweenx(
                depth,
                cutoff_x,
                x,
                where=finite & (data < spec.left_cutoff_fill),
                color=fill_color,
                alpha=spec.fill_alpha,
                linewidth=0.0,
                zorder=1,
            )
        if spec.right_cutoff_fill is not None:
            cutoff_x = self._normalize_scalar(spec.right_cutoff_fill, left, right, track.scale)
            ax.fill_betweenx(
                depth,
                cutoff_x,
                x,
                where=finite & (data > spec.right_cutoff_fill),
                color=fill_color,
                alpha=spec.fill_alpha,
                linewidth=0.0,
                zorder=1,
            )

    def _plot_crossover_fills(
        self,
        ax: Axes,
        depth: np.ndarray,
        x: np.ndarray,
        finite: np.ndarray,
        left: float,
        right: float,
        track: LogTrackSpec,
        spec: LogCurveSpec,
    ) -> None:
        fill_color = spec.fill_color or spec.color
        for attr, direction in ((spec.left_crossover, "left"), (spec.right_crossover, "right")):
            if not attr:
                continue
            other_curve = self._get_curve(attr)
            if other_curve is None:
                self._warn_missing(str(attr))
                continue
            other = np.asarray(other_curve.data, dtype=float)
            other_x = self._normalize(other, left, right, track.scale)
            valid = finite & np.isfinite(other_x)
            if direction == "left":
                where = valid & (other_x < x)
            else:
                where = valid & (other_x > x)
            ax.fill_betweenx(
                depth,
                other_x,
                x,
                where=where,
                color=fill_color,
                alpha=spec.fill_alpha,
                linewidth=0.0,
                zorder=1,
            )

    def _add_curve_header(
        self,
        ax: Axes,
        row: int,
        label: str,
        left: float,
        right: float,
        color: str,
        scale: str,
    ) -> None:
        y = 1.025 - row * 0.045
        text = f"{label}  {left:g}-{right:g}"
        if str(scale).lower() == "log":
            text += " log"
        ax.text(
            0.5,
            y,
            text,
            color=color,
            fontsize=6.5,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            clip_on=False,
        )

    def _get_curve(self, curve_name: str):
        candidates = [curve_name.upper()] + CURVE_ALIAS_CANDIDATES.get(curve_name.upper(), [])
        seen = set()
        for candidate in candidates:
            key = str(candidate).upper()
            if key in seen:
                continue
            seen.add(key)
            curve = self.well.get(key)
            if curve is not None:
                return curve
        return None

    def _warn_missing(self, curve_name: str) -> None:
        message = f"Curve '{curve_name}' not available for log viewer template"
        if self.strict:
            raise ValueError(message)
        if message not in self.warnings:
            self.warnings.append(message)

    def _normalize(self, values: np.ndarray, left: float, right: float, scale: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if math.isclose(float(left), float(right)):
            return np.full(values.shape, np.nan, dtype=float)
        if str(scale).lower() == "log":
            out = np.full(values.shape, np.nan, dtype=float)
            if left <= 0 or right <= 0:
                return out
            finite = np.isfinite(values) & (values > 0)
            left_log = math.log10(left)
            right_log = math.log10(right)
            out[finite] = (np.log10(values[finite]) - left_log) / (right_log - left_log)
            return out
        return (values - left) / (right - left)

    def _normalize_scalar(self, value: float, left: float, right: float, scale: str) -> float:
        arr = self._normalize(np.asarray([value], dtype=float), left, right, scale)
        return float(arr[0]) if np.isfinite(arr[0]) else np.nan

    def _depth_ticks(self, top: float, bottom: float, spacing: float) -> Iterable[float]:
        if spacing <= 0:
            return []
        first = math.ceil(min(top, bottom) / spacing) * spacing
        last = math.floor(max(top, bottom) / spacing) * spacing
        if last < first:
            return []
        return np.arange(first, last + spacing * 0.5, spacing)

    def _nice_depth_spacing(self, top: float, bottom: float, target: int = 12) -> float:
        span = abs(bottom - top)
        if not np.isfinite(span) or span <= 0:
            return 10.0
        raw = span / max(target, 1)
        exponent = math.floor(math.log10(raw))
        fraction = raw / (10**exponent)
        if fraction <= 1:
            nice = 1
        elif fraction <= 2:
            nice = 2
        elif fraction <= 5:
            nice = 5
        else:
            nice = 10
        return nice * (10**exponent)


def create_log_viewer(
    well: "Well",
    template_xml_path: Optional[str | Path] = None,
    template_defaults: str = "raw",
    depth_range: Optional[Tuple[float, float]] = None,
    top: Optional[float] = None,
    height: Optional[float] = None,
    strict: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
) -> LogViewer:
    """
    Create a template-driven well log viewer and immediately build the figure.
    """
    viewer = LogViewer(
        well,
        template_xml_path=template_xml_path,
        template_defaults=template_defaults,
        strict=strict,
        figsize=figsize,
    )
    viewer.create(depth_range=depth_range, top=top, height=height)
    return viewer

