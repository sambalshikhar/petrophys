"""
Specialist visualization panels for logs outside the normal triple-combo view.

The regular visualization stack already provides triple-combo, petrophysical
interpretation, crossplots, Pickett plots, zone flags, and template-driven log
viewer pages. This module only adds specialist panels for NMR and FMI data.
"""

from __future__ import annotations

from petrophys._internal.advice import Advice
from petrophys.visualization.style import _styled_plot

import warnings
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from petrophys.model.well import Well

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    plt = None
    HAS_MATPLOTLIB = False

from petrophys.interpretation.borehole_imaging import BoreholeImageData, get_navigation, orient_to_azimuth, plot_fmi


NMR_TOTAL_ALIASES = ("TCMR", "MPHI", "NMR_PHIT", "CMRPHI", "NMRP", "PHINMR")
NMR_FFI_ALIASES = ("CMFF", "MFFI", "NMR_FFI", "CMR_FFI", "FFI")
NMR_BVI_ALIASES = ("MBVI", "CBVI", "NMR_BVI", "CMR_BVI", "BVI")
NMR_SWIRR_ALIASES = ("SWIRR_NMR", "NMR_SWIRR", "SWIRR_CMR", "CMR_SWIRR")


def has_nmr_curves(well: "Well") -> bool:
    """Return True when a well contains NMR summary curves."""
    return any(
        _curve(well, aliases) is not None
        for aliases in (
            NMR_TOTAL_ALIASES,
            NMR_FFI_ALIASES,
            NMR_BVI_ALIASES,
            NMR_SWIRR_ALIASES,
            ("KSDR", "NMR_KSDR"),
            ("KTIM", "NMR_KTIM", "K_COATES"),
            ("T2CUTOFF", "T2_CUTOFF", "T2CUT"),
            ("CMRP_3MS", "NMR_3MS"),
        )
    )


@_styled_plot
def create_nmr_panel(
    well: "Well",
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> "plt.Figure":
    """Create an NMR specialist panel for processed NMR summary logs.

    The panel is drawn as :func:`petrophys.plot` draws a well: gamma ray and
    calipers, cable tension, NMR porosity with bound fluid (brown) and free
    fluid (yellow) shaded, NMR permeability, and irreducible water
    saturation (from the file, else bound fluid over total porosity). Bound
    fluid is total porosity minus free fluid when the file has no bound-fluid
    curve.

    ``depth_range`` is the (top, bottom) window drawn, in the well's depth
    unit (default: the whole log). The permeability track spans the whole
    decades of the permeabilities inside that window.
    """
    _require_matplotlib()
    from copy import deepcopy

    from petrophys.visualization.log_viewer import (
        LogCurveSpec,
        LogTrackSpec,
        LogViewer,
        LogViewerTemplate,
    )

    depth = _depth(well)
    top, bottom = _resolve_depth_range(depth, depth_range)
    shown = deepcopy(well)
    total = _curve(shown, NMR_TOTAL_ALIASES)
    ffi = _curve(shown, NMR_FFI_ALIASES)
    bvi = _curve(shown, NMR_BVI_ALIASES)
    bound_label = None if bvi is not None else "Bound fluid"
    if bvi is None and total is not None and ffi is not None:
        bound = np.asarray(total.data, dtype=float) - np.asarray(ffi.data, dtype=float)
        shown.add_curve("NMR_BVI", np.where(bound >= 0.0, bound, np.nan), unit=total.unit,
                        description="Bound fluid: total minus free-fluid porosity")
        bvi = shown.get("NMR_BVI")
    swirr, _ = _nmr_swirr_for_plot(well)
    swirr_curve = _curve(shown, NMR_SWIRR_ALIASES)
    if swirr is not None:
        # As a fraction, from the file's curve or as bound fluid over total porosity.
        name = swirr_curve.mnemonic if swirr_curve is not None else "SWIRR_NMR"
        shown.add_curve(name, swirr, unit="v/v", description="Irreducible water saturation from NMR")
        swirr_curve = shown.get(name)

    def first(aliases: Sequence[str]) -> Optional[str]:
        curve = _curve(shown, aliases)
        return None if curve is None else curve.mnemonic

    def spec(aliases, label, left, right, color, **style) -> Optional[LogCurveSpec]:
        name = aliases if isinstance(aliases, str) else first(aliases)
        if name is None or shown.get(name) is None:
            return None
        return LogCurveSpec(curve_name=name, display_name=label, left=left, right=right, color=color, **style)

    def track(name, specs, **options) -> Optional[LogTrackSpec]:
        curves = [item for item in specs if item is not None]
        return LogTrackSpec(display_name=name, curves=curves, **options) if curves else None

    names = {key: first(aliases) for key, aliases in (
        ("total", NMR_TOTAL_ALIASES), ("bound", NMR_BVI_ALIASES),
    )}
    porosity = [
        # Free fluid, between total porosity and bound fluid; bound fluid, from it to zero.
        spec(NMR_TOTAL_ALIASES, None, 0.45, 0, "#000000", fill_color="#ffd92f", fill_alpha=0.55,
             right_crossover=names["bound"]),
        spec(NMR_FFI_ALIASES, None, 0.45, 0, "#1f78b4", line_style=":"),
        spec(NMR_BVI_ALIASES, bound_label, 0.45, 0, "#8c510a", fill="right", fill_color="#bf812d", fill_alpha=0.6),
    ]
    permeability = [
        spec(("KSDR", "NMR_KSDR"), None, 0.01, 10000, "#e7298a", line_style="-."),
        spec(("KTIM", "NMR_KTIM", "K_COATES"), None, 0.01, 10000, "#66a61e"),
    ]
    low, high = _decades([shown.get(item.curve_name) for item in permeability if item is not None],
                         depth, top, bottom)
    permeability = [None if item is None else LogCurveSpec(**{**item.__dict__, "left": low, "right": high})
                    for item in permeability]

    cutoff = _curve(shown, ("T2CUTOFF", "T2_CUTOFF", "T2CUT"))
    values = np.asarray(cutoff.data, dtype=float) if cutoff is not None else np.array([])
    values = values[np.isfinite(values)]
    one_cutoff = values.size and np.ptp(values) == 0
    saturation = [spec(swirr_curve.mnemonic if swirr_curve is not None else (), "Swirr", 1, 0, "#762a83")]
    if cutoff is not None and values.size and not one_cutoff:
        saturation.append(spec(cutoff.mnemonic, "T2 cutoff", float(np.min(values)), float(np.max(values)),
                               "#d73027", line_style="--"))

    tension = []
    for aliases, color in ((("TENS",), "#000000"), (("CDF",), "#e7298a")):
        curve = _curve(shown, aliases)
        data = np.asarray(curve.data, dtype=float) if curve is not None else np.array([])
        data = data[np.isfinite(data)]
        if data.size:
            # Round scale ends: steps of half the decade of the range.
            step = 10.0 ** np.floor(np.log10(max(float(np.ptp(data)), 1e-9))) / 2.0
            tension.append(spec(curve.mnemonic, None, float(np.floor(np.min(data) / step) * step),
                                float(np.ceil(np.max(data) / step) * step), color))

    tracks = [
        LogTrackSpec(display_name="DEPTH", width=0.55, depth_track=True, track_type="depth"),
        track("GAMMA / HOLE", [
            spec(("GR_EDTC", "GR"), None, 0, 150, "#1b7837"),
            spec(("HD1_PPC1", "HD1", "CALI"), None, 6, 16, "#d73027"),
            spec(("HD2_PPC1", "HD2"), None, 6, 16, "#0571b0"),
            spec(("BS",), None, 6, 16, "#000000", line_style="--"),
        ]),
        # Cable tension and its downhole force flag tool sticking, not NMR quality.
        track("CABLE TENSION", tension),
        track("NMR POROSITY", porosity, width=1.35),
        track("NMR PERMEABILITY", permeability, scale="log", left=low, right=high, width=1.25),
        # One cutoff for the whole log is stated in the title rather than drawn as a flat line.
        track(f"SWIRR, T2 CUTOFF {values[0]:g} ms" if one_cutoff else "SWIRR", saturation, width=1.4),
    ]
    viewer = LogViewer(shown, template_defaults="auto", figsize=figsize)
    viewer.template = LogViewerTemplate(name="nmr", tracks=[item for item in tracks if item is not None])
    viewer.optional_tracks, viewer._optional_track_specs = [], []
    fig, _ = viewer.create(depth_range=(top, bottom))
    return fig


def _decades(curves, depth: np.ndarray, top: float, bottom: float) -> Tuple[float, float]:
    """Whole decades spanning the 1-99 percentiles of the curves in the window.

    The scale reaches 0.01 mD and spans three decades at least, so a tight
    well is not drawn off the scale.
    """
    inside = (depth >= min(top, bottom)) & (depth <= max(top, bottom))
    parts = [np.asarray(curve.data, dtype=float)[inside] for curve in curves if curve is not None]
    values = np.concatenate(parts) if parts else np.array([])
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return 0.01, 10000.0
    low, high = np.percentile(values, [1, 99])
    low, high = 10.0 ** np.floor(np.log10(low)), 10.0 ** np.ceil(np.log10(high))
    return float(min(low, 0.01)), float(max(high, low * 1000))


@_styled_plot
def create_fmi_panel(
    fmi_image: BoreholeImageData,
    depth_range: Optional[Tuple[float, float]] = None,
    orient: bool = True,
    equalize: str = "dynamic",
    figsize: Tuple[float, float] = (5, 11),
) -> "plt.Figure":
    """Create a specialist FMI image panel."""
    _require_matplotlib()
    image = _prepared_fmi_image(fmi_image, depth_range, orient)
    fig, ax = plt.subplots(figsize=figsize)
    plot_fmi(image, ax=ax, equalize=equalize, interpolation="nearest")
    fig.suptitle("FMI Image", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _prepared_fmi_image(
    image: BoreholeImageData,
    depth_range: Optional[Tuple[float, float]],
    orient: bool,
) -> BoreholeImageData:
    data = image.filter_by_depth(*depth_range) if depth_range is not None else image
    if orient and not data.metadata.get("oriented"):
        if get_navigation(data):
            return orient_to_azimuth(data)
        warnings.warn(
            Advice("create_fmi_panel: orientation requested but no navigation curve "
            "(P1AZ/HAZI/RB) was found; plotting the raw un-oriented image "
            "(x-axis is borehole sample, not azimuth). Provide navigation or "
            "pass orient=False to silence this.", ('orient',)),
            stacklevel=3,
        )
    return data


def _curve(well: "Well", aliases: Sequence[str]):
    for name in aliases:
        curve = well.get(name)
        if curve is not None:
            return curve
    return None


def _curve_data(
    well: "Well",
    aliases: Sequence[str],
) -> tuple[Optional[np.ndarray], Optional[str]]:
    curve = _curve(well, aliases)
    if curve is None:
        return None, None
    try:
        return np.asarray(curve.data, dtype=float), curve.mnemonic
    except (TypeError, ValueError):
        return None, curve.mnemonic


def _depth(well: "Well") -> np.ndarray:
    if getattr(well, "depth", None) is None:
        raise ValueError("Well depth is required for plotting")
    return np.asarray(well.depth, dtype=float)


def _resolve_depth_range(
    depth: np.ndarray,
    depth_range: Optional[Tuple[float, float]],
) -> tuple[float, float]:
    if depth_range is not None:
        return float(depth_range[0]), float(depth_range[1])
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("Cannot plot without finite depth values")
    return float(np.nanmin(finite)), float(np.nanmax(finite))














def _nmr_swirr_for_plot(well: "Well") -> tuple[Optional[np.ndarray], Optional[str]]:
    swirr, swirr_name = _curve_data(well, NMR_SWIRR_ALIASES)
    if swirr is not None:
        return _fraction_if_percent(swirr), f"Swirr ({swirr_name})"
    total, total_name = _curve_data(well, NMR_TOTAL_ALIASES)
    if total is None:
        return None, None
    bvi, _ = _curve_data(well, NMR_BVI_ALIASES)
    ffi, _ = _curve_data(well, NMR_FFI_ALIASES)
    if bvi is None and ffi is not None:
        bvi = np.where(np.isfinite(total) & np.isfinite(ffi), total - ffi, np.nan)
    if bvi is None:
        return None, None
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.isfinite(total) & (total > 0), bvi / total, np.nan)
    result = np.where((result > 0.0) & (result <= 1.0), result, np.nan)
    return result, f"Swirr (NMR-derived from {total_name})"


def _fraction_if_percent(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(np.abs(finite)) > 1.5 and np.nanmax(np.abs(finite)) <= 100.0:
        arr = arr / 100.0
    return arr




def _require_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")


__all__ = [
    "create_fmi_panel",
    "create_nmr_panel",
    "has_nmr_curves",
]
