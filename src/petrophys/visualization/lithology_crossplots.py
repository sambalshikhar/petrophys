"""Industry lithology crossplots: density-neutron (with matrix lines), M-N, MID.

These overlay the standard mineral reference lines/points so lithology can be read
straight off the plot, the way Techlog / Interactive Petrophysics / Geolog show them.
"""

from __future__ import annotations

from petrophys.visualization.style import _styled_plot

from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

from petrophys._internal.analysis_helpers import (
    curve_source_mnemonic,
    get_curve_data_depthwise,
)
from petrophys.preprocessing.standardization.unit_converter import get_conversion_factor


# zero-porosity matrix points (neutron v/v, density g/cc, sonic us/ft)
_MATRIX = {
    "Sandstone": {"nphi": -0.02, "rhob": 2.65, "dt": 55.5, "color": "goldenrod"},
    "Limestone": {"nphi": 0.00, "rhob": 2.71, "dt": 47.6, "color": "royalblue"},
    "Dolomite": {"nphi": 0.02, "rhob": 2.87, "dt": 43.5, "color": "seagreen"},
}
_RHO_FL, _NPHI_FL, _DT_FL = 1.0, 1.0, 189.0


def _xy(well, names, depth_range, unit=None):
    """Depth-wise curve data; with ``unit`` each source curve is converted to it.

    The chart constants assume v/v porosity, g/cc density and us/ft slowness,
    so curves read in %, kg/m3 or us/m are converted (unknown labels are kept).
    """
    data, _, meta = get_curve_data_depthwise(well, names)
    if data is None:
        return None, None
    arr = np.array(data, dtype=float)
    if unit is not None:
        sources = meta["source_per_sample"]
        seen = set()
        for name in names:
            curve = well.get(name)
            source = (curve_source_mnemonic(curve) or curve.mnemonic) if curve is not None else None
            if curve is None or source in seen or not curve.unit:
                continue
            seen.add(source)
            factor = get_conversion_factor(str(curve.unit), unit)
            if factor is not None and factor != 1.0:
                arr[sources == source] *= factor
    depth = np.asarray(getattr(well, "depth", None), dtype=float)
    if depth_range is not None and depth.shape == arr.shape:
        lo, hi = depth_range
        m = (depth >= lo) & (depth <= hi)
        arr = np.where(m, arr, np.nan)
    return arr, depth


def _color(well, color_curve, depth_range):
    if not color_curve:
        return None, None
    c, _ = _xy(well, [color_curve], depth_range)
    curve = well.get(color_curve)
    unit = str(getattr(curve, "unit", "") or "").strip() if curve is not None else ""
    return c, f"{color_curve} [{unit}]" if unit else color_curve


def _note_outside_axes(ax, x, y):
    """State how many plottable points fall outside the fixed chart limits."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    outside = finite & ((x < x0) | (x > x1) | (y < y0) | (y > y1))
    count = int(np.count_nonzero(outside))
    if count:
        ax.text(
            0.01, 0.01,
            f"{count} of {int(np.count_nonzero(finite))} points outside the plotted range",
            transform=ax.transAxes, fontsize=7, color="0.35", ha="left", va="bottom",
        )


def _scatter(ax, x, y, c, clabel, fig, cmap="viridis", norm=None):
    if c is not None and np.any(np.isfinite(c)):
        # Points without a colour value are drawn grey, not dropped.
        cmap = plt.get_cmap(cmap).with_extremes(bad="0.6")
        sc = ax.scatter(x, y, c=c, s=6, alpha=0.6, cmap=cmap, norm=norm, plotnonfinite=True)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(clabel, fontsize=8)
    else:
        ax.scatter(x, y, s=6, alpha=0.6, color="0.3")


@_styled_plot
def create_nd_crossplot(
    well, color_curve: str = "GR", depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
):
    """Density-neutron crossplot with sandstone/limestone/dolomite matrix lines
    (porosity ticked 0-40%). RHOB axis is inverted (high density at the bottom)."""
    nphi, _ = _xy(well, ["NPHI", "PHIN"], depth_range, "v/v")
    rhob, _ = _xy(well, ["RHOB"], depth_range, "g/cm3")
    if nphi is None or rhob is None:
        raise ValueError("N-D crossplot needs NPHI and RHOB")
    c, clabel = _color(well, color_curve, depth_range)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter(ax, nphi, rhob, c, clabel, fig)

    phis = np.linspace(0.0, 0.40, 5)
    for name, p in _MATRIX.items():
        # straight line from the matrix point (phi=0) toward the fluid point (phi=1)
        line_n = p["nphi"] + phis * (_NPHI_FL - p["nphi"])
        line_r = p["rhob"] + phis * (_RHO_FL - p["rhob"])
        ax.plot(line_n, line_r, color=p["color"], lw=1.6, label=name, zorder=5)
        ax.scatter(line_n, line_r, color=p["color"], s=14, zorder=6)
        for phi, xn, yr in zip(phis, line_n, line_r):
            ax.annotate(f"{int(phi*100)}", (xn, yr), fontsize=6,
                        color=p["color"], xytext=(2, 2), textcoords="offset points")

    ax.set_xlabel("NPHI (v/v, limestone)"); ax.set_ylabel("RHOB (g/cc)")
    ax.set_xlim(-0.05, 0.45); ax.set_ylim(3.0, 1.8)   # density inverted
    _note_outside_axes(ax, nphi, rhob)
    ax.set_title(f"Density-Neutron — {well.header.well_name or ''}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


@_styled_plot
def create_mn_plot(
    well, color_curve: str = "GR", depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
):
    """M-N crossplot (porosity-independent lithology), with the quartz/calcite/
    dolomite triangle (fresh mud: fluid 1.0 g/cc, 189 us/ft, neutron 1.0)."""
    nphi, _ = _xy(well, ["NPHI", "PHIN"], depth_range, "v/v")
    rhob, _ = _xy(well, ["RHOB"], depth_range, "g/cm3")
    dt, _ = _xy(well, ["DTC", "DT"], depth_range, "us/ft")
    if nphi is None or rhob is None or dt is None:
        raise ValueError("M-N plot needs NPHI, RHOB and DTC")
    c, clabel = _color(well, color_curve, depth_range)

    denom = rhob - _RHO_FL
    with np.errstate(divide="ignore", invalid="ignore"):
        M = (_DT_FL - dt) / denom * 0.01
        N = (_NPHI_FL - nphi) / denom
    M = np.where(np.isfinite(M), M, np.nan)
    N = np.where(np.isfinite(N), N, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter(ax, N, M, c, clabel, fig)

    pts = {"Quartz": (0.618, 0.810), "Calcite": (0.585, 0.827), "Dolomite": (0.524, 0.778)}
    xs = [v[0] for v in pts.values()] + [pts["Quartz"][0]]
    ys = [v[1] for v in pts.values()] + [pts["Quartz"][1]]
    ax.plot(xs, ys, "k--", lw=1, alpha=0.6, zorder=5)
    for name, (n, m) in pts.items():
        ax.scatter(n, m, s=80, marker="^", color="firebrick", zorder=6)
        ax.annotate(name, (n, m), fontsize=9, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("N  =  (1 - NPHI) / (RHOB - 1)")
    ax.set_ylabel("M  =  (189 - DT) / (RHOB - 1) x 0.01")
    ax.set_title(f"M-N Plot — {well.header.well_name or ''}")
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig


@_styled_plot
def create_mid_plot(
    well, color_curve: str = "GR", depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
):
    """MID plot: apparent matrix density vs apparent matrix sonic, with the
    quartz/calcite/dolomite points (porosity-independent lithology)."""
    nphi, _ = _xy(well, ["NPHI", "PHIN"], depth_range, "v/v")
    rhob, _ = _xy(well, ["RHOB"], depth_range, "g/cm3")
    dt, _ = _xy(well, ["DTC", "DT"], depth_range, "us/ft")
    if nphi is None or rhob is None or dt is None:
        raise ValueError("MID plot needs NPHI, RHOB and DTC")
    c, clabel = _color(well, color_curve, depth_range)

    # density porosity (limestone matrix) and neutron averaged for apparent matrix
    phi_d = (2.71 - rhob) / (2.71 - _RHO_FL)
    phi = np.clip((phi_d + nphi) / 2.0, 0.001, 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_maa = (rhob - phi * _RHO_FL) / (1.0 - phi)
        dt_maa = (dt - phi * _DT_FL) / (1.0 - phi)
    rho_maa = np.where(np.isfinite(rho_maa), rho_maa, np.nan)
    dt_maa = np.where(np.isfinite(dt_maa), dt_maa, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter(ax, dt_maa, rho_maa, c, clabel, fig)
    for name, p in _MATRIX.items():
        ax.scatter(p["dt"], p["rhob"], s=90, marker="s", color=p["color"],
                   edgecolor="k", zorder=6)
        ax.annotate(name, (p["dt"], p["rhob"]), fontsize=9, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Apparent matrix sonic  DTmaa (us/ft)")
    ax.set_ylabel("Apparent matrix density  RHOmaa (g/cc)")
    ax.set_xlim(35, 65); ax.set_ylim(3.0, 2.5)
    _note_outside_axes(ax, dt_maa, rho_maa)
    ax.set_title(f"MID Plot — {well.header.well_name or ''}")
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig


@_styled_plot
def create_buckles_plot(
    well, phie_curve: str = "PHIE", sw_curve: str = "SW",
    color_curve: str = "PERM", depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
):
    """Buckles plot: porosity vs water saturation with constant-BVW hyperbolas.
    Points on the same BVW = phi*Sw hyperbola are at irreducible water (a single
    well-defined hyperbola implies uniform grain size / Swirr)."""
    phie, _ = _xy(well, [phie_curve, "PHIT"], depth_range, "v/v")
    sw, _ = _xy(well, [sw_curve, "SWA", "SWT-LOG"], depth_range, "v/v")
    if phie is None or sw is None:
        raise ValueError("Buckles plot needs PHIE and SW")
    c, clabel = _color(well, color_curve, depth_range)

    norm = None
    if c is not None and color_curve.upper().startswith("PERM"):
        positive = c[np.isfinite(c) & (c > 0)]
        if positive.size:
            # Permeability spans decades; colour it on a log scale.
            norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
            c = np.where(c > 0, c, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter(ax, phie, sw, c, clabel, fig, cmap="viridis", norm=norm)

    phi_line = np.linspace(0.01, 0.40, 200)
    for bvw in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
        sw_line = np.clip(bvw / phi_line, 0, 1)
        ax.plot(phi_line, sw_line, color="0.5", lw=0.8, alpha=0.7)
        # Label each hyperbola where it meets the right edge (Sw = BVW / 0.40),
        # so the labels are spread out instead of stacked near the top.
        ax.annotate(f"BVW={bvw:g}", (0.40, bvw / 0.40), xytext=(-3, 2),
                    textcoords="offset points", ha="right", fontsize=7, color="0.4")

    ax.set_xlabel("Effective porosity (v/v)"); ax.set_ylabel("Water saturation (v/v)")
    ax.set_xlim(0, 0.40); ax.set_ylim(0, 1.0)
    ax.set_title(f"Buckles Plot — {well.header.well_name or ''}")
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig


@_styled_plot
def create_hingle_plot(
    well, rw: float = 0.05, m: float = 2.0, n: float = 2.0,
    phie_curve: str = "PHIE", rt_curve: str = "RDEP", color_curve: str = "VCL",
    a: float = 1.0, depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
):
    """Hingle plot: porosity vs Rt^(-1/m) on linear axes, with iso-Sw lines through
    the origin (the linear-axis counterpart of the Pickett plot). Water-bearing
    points fall on the Sw=1 line, whose slope is (a*Rw)^(-1/m), so
    a*Rw = slope^(-m)."""
    if not m:
        raise ValueError("Hingle plot requires a non-zero cementation exponent m")
    phie, _ = _xy(well, [phie_curve, "PHIT"], depth_range, "v/v")
    rt, _ = _xy(well, [rt_curve, "RT", "RDEP"], depth_range)
    if phie is None or rt is None:
        raise ValueError("Hingle plot needs PHIE and Rt")
    c, clabel = _color(well, color_curve, depth_range)

    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.power(rt, -1.0 / m)        # Rt^(-1/m); water line is linear in phi
    y = np.where(np.isfinite(y) & (rt > 0), y, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter(ax, phie, y, c, clabel, fig, cmap="viridis")

    phi_line = np.linspace(0.0, 0.40, 50)
    for sw in (1.0, 0.75, 0.5, 0.25):
        # Sw^n = a*Rw / (phi^m * Rt)  ->  Rt^(-1/m) = phi * Sw^(n/m) / (a*Rw)^(1/m)
        y_line = phi_line * sw ** (n / m) / (a * rw) ** (1.0 / m)
        ax.plot(phi_line, y_line, color="0.5", lw=0.9, alpha=0.7)
        ax.annotate(f"Sw={int(sw*100)}%", (0.40, y_line[-1]), xytext=(-3, 2),
                    textcoords="offset points", ha="right", fontsize=7, color="0.4")

    ax.set_xlabel("Effective porosity (v/v)")
    ax.set_ylabel(f"Rt^(-1/{m:g})")
    ax.set_xlim(0, 0.40)
    ax.set_title(f"Hingle Plot — {well.header.well_name or ''}  (Rw={rw})")
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig
