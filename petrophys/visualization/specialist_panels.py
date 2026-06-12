"""
Specialist visualization panels for logs outside the normal triple-combo view.

The regular visualization stack already provides triple-combo, petrophysical
interpretation, crossplots, Pickett plots, zone flags, and template-driven log
viewer pages. This module only adds specialist panels for NMR and FMI data.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    plt = None
    HAS_MATPLOTLIB = False

from petrophys.imaging import BoreholeImageData, get_navigation, orient_to_azimuth, plot_fmi


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


def create_nmr_panel(
    well: "Well",
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (15, 11),
) -> "plt.Figure":
    """Create an NMR specialist panel for processed NMR summary logs."""
    _require_matplotlib()
    depth = _depth(well)
    top, bottom = _resolve_depth_range(depth, depth_range)
    fig, axes = plt.subplots(1, 5, figsize=figsize, sharey=True)
    axes = list(np.atleast_1d(axes))
    _format_axes(axes, top, bottom, well)

    _plot_first(axes[0], well, ("GR_EDTC", "GR"), depth, "forestgreen", "GR", xlim=(0, 150))
    _plot_first(axes[0], well, ("CDF",), depth, "magenta", "CDF")
    _plot_first(axes[0], well, ("TENS",), depth, "black", "TENS")
    axes[0].set_title("GR / CMR QC")

    _plot_first(axes[1], well, ("BS",), depth, "dimgray", "BS", linestyle="--")
    _plot_first(axes[1], well, ("HD1_PPC1", "HD1"), depth, "dodgerblue", "HD1")
    _plot_first(axes[1], well, ("HD2_PPC1", "HD2"), depth, "darkorange", "HD2")
    _plot_first(axes[1], well, ("CALI", "CAL"), depth, "black", "CALI")
    axes[1].set_title("Borehole")

    total, total_name = _curve_data(well, NMR_TOTAL_ALIASES)
    ffi, ffi_name = _curve_data(well, NMR_FFI_ALIASES)
    bvi, bvi_name = _curve_data(well, NMR_BVI_ALIASES)
    if bvi is None and total is not None and ffi is not None:
        bvi = np.where(np.isfinite(total) & np.isfinite(ffi), total - ffi, np.nan)
        bvi = np.where(bvi >= 0.0, bvi, np.nan)
        bvi_name = "MBVI*"
    poro_xlim = _porosity_xlim([arr for arr in (total, ffi, bvi) if arr is not None])
    if total is not None:
        axes[2].plot(total, depth, color="black", linewidth=0.7, label=total_name)
    if ffi is not None:
        axes[2].plot(ffi, depth, color="magenta", linewidth=0.7, label=ffi_name)
    if bvi is not None:
        axes[2].plot(bvi, depth, color="saddlebrown", linewidth=0.7, label=bvi_name)
    axes[2].set_xlim(poro_xlim)
    axes[2].set_title("CMR Porosity")

    _plot_first(axes[3], well, ("KSDR", "NMR_KSDR"), depth, "royalblue", "KSDR", log=True)
    _plot_first(axes[3], well, ("KTIM", "NMR_KTIM", "K_COATES"), depth, "red", "KTIM", log=True)
    axes[3].set_xlim(0.01, 10000)
    axes[3].set_title("NMR Perm")

    swirr, swirr_name = _nmr_swirr_for_plot(well)
    if swirr is not None:
        axes[4].plot(swirr, depth, color="navy", linewidth=0.7, label=swirr_name)
    axes[4].set_xlim(0, 1)
    axes[4].set_title("Swirr / T2")
    cutoff, cutoff_name = _curve_data(well, ("T2CUTOFF", "T2_CUTOFF", "T2CUT"))
    if cutoff is not None:
        ax_t2 = axes[4].twiny()
        ax_t2.plot(cutoff, depth, color="red", linewidth=0.6, label=cutoff_name)
        ax_t2.set_xlabel("T2 cutoff [ms]", fontsize=8)
        _set_data_xlim(ax_t2, cutoff)

    _finalize_log_figure(fig, axes, well, "NMR Panel")
    return fig


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
    if orient and not data.metadata.get("oriented") and get_navigation(data):
        return orient_to_azimuth(data)
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


def _format_axes(axes: Sequence["plt.Axes"], top: float, bottom: float, well: "Well") -> None:
    for index, ax in enumerate(axes):
        ax.set_ylim(bottom, top)
        ax.grid(True, alpha=0.28)
        if index == 0:
            ax.set_ylabel(f"Depth [{getattr(well, 'depth_unit', '')}]".strip())
        else:
            ax.tick_params(axis="y", labelleft=False)


def _plot_first(
    ax: "plt.Axes",
    well: "Well",
    aliases: Sequence[str],
    depth: np.ndarray,
    color: str,
    label: str,
    xlim: Optional[Tuple[float, float]] = None,
    log: bool = False,
    linestyle: str = "-",
) -> bool:
    data, source = _curve_data(well, aliases)
    if data is None:
        return False
    finite = np.isfinite(data) & np.isfinite(depth)
    if log:
        finite &= data > 0
    if not np.any(finite):
        return False
    ax.plot(
        data[finite],
        depth[finite],
        color=color,
        linewidth=0.7,
        linestyle=linestyle,
        label=label or source,
    )
    if log:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(xlim)
    elif log:
        _set_log_data_xlim(ax, data)
    else:
        _set_data_xlim(ax, data)
    return True


def _set_data_xlim(ax: "plt.Axes", data: np.ndarray) -> None:
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if high <= low:
        pad = abs(high) * 0.1 if high else 1.0
        ax.set_xlim(low - pad, high + pad)
    else:
        pad = (high - low) * 0.08
        ax.set_xlim(low - pad, high + pad)


def _set_log_data_xlim(ax: "plt.Axes", data: np.ndarray) -> None:
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if high <= low:
        ax.set_xlim(max(low / 10.0, 1e-6), high * 10.0)
    else:
        ax.set_xlim(low / 1.5, high * 1.5)


def _porosity_xlim(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    finite_parts = [np.asarray(arr, dtype=float)[np.isfinite(arr)] for arr in arrays]
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        return 0.0, 0.45
    values = np.concatenate(finite_parts)
    return (0.0, 80.0) if np.nanmax(values) > 1.5 else (0.0, 0.8)


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


def _finalize_log_figure(fig: "plt.Figure", axes: Sequence["plt.Axes"], well: "Well", title: str) -> None:
    for ax in axes:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=7)
    well_name = well.name or "Unknown Well"
    fig.suptitle(f"{title} - {well_name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))


def _require_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")


__all__ = [
    "create_fmi_panel",
    "create_nmr_panel",
    "has_nmr_curves",
]
