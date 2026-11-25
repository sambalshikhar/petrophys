"""Industry lithology crossplots: density-neutron (with matrix lines), M-N, MID.

The mineral overlays use a linear fresh-water response model and
limestone-referenced neutron endpoints. Tool-specific charts require their
own calibration; the overlays are not vendor response charts.
"""

from __future__ import annotations

from petrophys.visualization.style import _styled_plot, display_unit

from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

from petrophys.visualization._data import (
    add_note, depth_downward, curve_values, porosity_label, note_outside_axes, archie_parameters, chart_limits,
    clean_rock_only, formation_rw,
)
from petrophys._internal.advice import Advice
from petrophys.interpretation.porosity.neutron_porosity import CNL_CHART_LINES
from petrophys._internal.validation import warn_invalid


# zero-porosity matrix points (neutron v/v, density g/cc, sonic us/ft)
_MATRIX = {
    "Sandstone": {"nphi": -0.02, "rhob": 2.65, "dt": 55.5, "color": "goldenrod"},
    "Limestone": {"nphi": 0.00, "rhob": 2.71, "dt": 47.6, "color": "royalblue"},
    "Dolomite": {"nphi": 0.02, "rhob": 2.87, "dt": 43.5, "color": "seagreen"},
}
_RHO_FL, _NPHI_FL, _DT_FL = 1.0, 1.0, 189.0

#: The density-neutron chart's lithology lines: apparent limestone neutron
#: porosity (v/v) at true porosity 0, 0.05, ..., 0.40, with the matrix density
#: each line is drawn for. Read from the vector lines of Schlumberger's Por-11
#: (former CP-1e: CNL and Litho-Density, fresh water in the invaded zone,
#: Log Interpretation Charts, 2013), where the sandstone and dolomite lines
#: curve; bulk density follows from rho_ma (1 - phi) + phi. Dolomite ends at 0.35.
_ND_CHART_LINES = {
    name.title(): {**CNL_CHART_LINES[name], "color": color}
    for name, color in (("sandstone", "goldenrod"), ("limestone", "royalblue"), ("dolomite", "seagreen"))
}


def _xy(well, names, depth_range, unit=None):
    """Depth-wise curve data; with ``unit`` each source curve is converted to it.

    The chart constants assume v/v porosity, g/cc density and us/ft slowness,
    so curves read in %, kg/m3 or us/m are converted. Incompatible units
    are rejected rather than labelled with the chart's expected unit.
    """
    values, depth, _ = curve_values(well, names, depth_range, unit)
    return values, depth


def _color(well, color_curve, depth_range):
    if not color_curve:
        return None, None
    c, _ = _xy(well, [color_curve], depth_range)
    curve = well.get(color_curve)
    unit = display_unit(getattr(curve, "unit", "")) if curve is not None else ""
    return c, f"{color_curve} [{unit}]" if unit else color_curve


def _note_outside_axes(ax, x, y):
    """State how many plottable points fall outside the fixed chart limits."""
    note_outside_axes(ax, x, y)


def _scatter(ax, x, y, c, clabel, fig, cmap="viridis", norm=None):
    from petrophys.visualization.style import draw_categories

    if c is not None and norm is None and draw_categories(ax, x, y, c, clabel, size=6, alpha=0.6):
        return
    if c is not None and np.any(np.isfinite(c)):
        # Points without a colour value are drawn grey, not dropped.
        cmap = plt.get_cmap(cmap).with_extremes(bad="0.6")
        sc = ax.scatter(x, y, c=c, s=6, alpha=0.6, cmap=cmap, norm=norm, plotnonfinite=True)
        extend = "neither"
        if norm is not None and norm.vmin is not None and norm.vmax is not None:
            finite = c[np.isfinite(c)]
            below, above = bool(np.any(finite < norm.vmin)), bool(np.any(finite > norm.vmax))
            extend = {(True, True): "both", (True, False): "min", (False, True): "max"}.get((below, above), "neither")
        cb = fig.colorbar(sc, ax=ax, pad=0.02, extend=extend)
        if cb.solids is not None:  # opaque, whatever the points' transparency
            cb.solids.set_alpha(1.0)
        cb.set_label(clabel, fontsize=8)
        depth_downward(cb, clabel)
    else:
        ax.scatter(x, y, s=6, alpha=0.6, color="0.3")


@_styled_plot
def _nd_chart(
    well, color_curve: str = "GR", depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
):
    """Density-neutron crossplot with the sandstone, limestone and dolomite lines
    of the chartbook (Schlumberger Por-11, CNL, fresh water), porosity ticked every
    10 %. RHOB axis is inverted (high density at the bottom)."""
    nphi, _ = _xy(well, ["NPHI", "PHIN"], depth_range, "v/v")
    rhob, _ = _xy(well, ["RHOB"], depth_range, "g/cm3")
    if nphi is None or rhob is None:
        raise ValueError("N-D crossplot needs NPHI and RHOB")
    c, clabel = _color(well, color_curve, depth_range)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter(ax, nphi, rhob, c, clabel, fig)

    for name, p in _ND_CHART_LINES.items():
        # The chartbook line, curved where the mineral's neutron response is (Por-11).
        line_n = np.asarray(p["nphi"])
        phis = np.arange(line_n.size) * 0.05
        line_r = p["rhob"] + phis * (_RHO_FL - p["rhob"])
        ax.plot(line_n, line_r, color=p["color"], lw=1.6, label=name, zorder=5)
        ticks = phis <= 0.401
        ax.scatter(line_n[::2], line_r[::2], color=p["color"], s=14, zorder=6)
        for phi, xn, yr in zip(phis[::2][ticks[::2]], line_n[::2], line_r[::2]):
            ax.annotate(f"{int(round(phi * 100))}", (xn, yr), fontsize=6,
                        color=p["color"], xytext=(2, 2), textcoords="offset points")

    ax.set_xlabel("NPHI [v/v, limestone]"); ax.set_ylabel("RHOB [g/cc]")
    ax.set_xlim(-0.05, 0.45); ax.set_ylim(3.0, 1.8)   # density inverted
    _note_outside_axes(ax, nphi, rhob)
    ax.set_title(f"Density-Neutron - {well.header.well_name or ''}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


# Photoelectric matrix points (Schlumberger log interpretation charts): bulk
# density (g/cc), electron density (g/cc) and volumetric cross section
# U = PEF x electron density (barns/cc).
_PEF_MATRIX = {
    "Quartz": {"rhob": 2.65, "rhoe": 2.650, "u": 4.78, "color": "goldenrod"},
    "Calcite": {"rhob": 2.71, "rhoe": 2.708, "u": 13.77, "color": "royalblue"},
    "Dolomite": {"rhob": 2.87, "rhoe": 2.863, "u": 9.00, "color": "seagreen"},
}
_PEF_POINTS = {"Anhydrite": {"rhob": 2.98, "pef": 5.05, "color": "mediumpurple"}}
_WATER_RHOE, _WATER_U = 1.110, 0.398


@_styled_plot
def _pef_chart(
    well, color_curve: str = "GR", depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
):
    """Density-PEF crossplot with quartz, calcite and dolomite porosity lines
    (ticked 0-40 %, fresh water in the pores) and the anhydrite point.

    PEF hardly changes with porosity, so the minerals fall in columns while
    density moves with porosity. A line mixes the matrix and water by volume
    in the volumetric cross section U = PEF x electron density, as the
    charts do, so it bends slightly toward low PEF. RHOB is inverted (high
    density at the bottom); PEF runs 0-8 b/e, and points beyond, such as
    barite spikes, are counted in a note.
    """
    pef, _ = _xy(well, ["PEF", "PE"], depth_range)
    rhob, _ = _xy(well, ["RHOB"], depth_range, "g/cm3")
    if pef is None or rhob is None:
        raise ValueError("Density-PEF crossplot needs PEF and RHOB")
    c, clabel = _color(well, color_curve, depth_range)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter(ax, pef, rhob, c, clabel, fig)

    phis = np.linspace(0.0, 0.40, 5)
    fine = np.linspace(0.0, 0.40, 41)
    for name, m in _PEF_MATRIX.items():
        def line(phi):
            u = (1 - phi) * m["u"] + phi * _WATER_U
            rhoe = (1 - phi) * m["rhoe"] + phi * _WATER_RHOE
            return u / rhoe, (1 - phi) * m["rhob"] + phi * _RHO_FL
        x, y = line(fine)
        ax.plot(x, y, color=m["color"], lw=1.6, label=name, zorder=5)
        x, y = line(phis)
        ax.scatter(x, y, color=m["color"], s=14, zorder=6)
        for phi, xp, yp in zip(phis, x, y):
            ax.annotate(f"{int(round(phi * 100))}", (xp, yp), fontsize=6,
                        color=m["color"], xytext=(3, 0), textcoords="offset points", va="center")
    for name, m in _PEF_POINTS.items():
        ax.scatter([m["pef"]], [m["rhob"]], color=m["color"], marker="s", s=30, zorder=6, label=name)

    ax.set_xlabel("PEF [b/e]"); ax.set_ylabel("RHOB [g/cc]")
    ax.set_xlim(0, 8); ax.set_ylim(3.0, 1.8)   # density inverted
    _note_outside_axes(ax, pef, rhob)
    ax.set_title(f"Density-PEF - {well.header.well_name or ''}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


@_styled_plot
def _mn_chart(
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
                    xytext=(-5 if name == "Quartz" else 5, 5), textcoords="offset points",
                    ha="right" if name == "Quartz" else "left")

    ax.set_xlabel("N  =  (1 - NPHI) / (RHOB - 1)")
    ax.set_ylabel("M  =  (189 - DT) / (RHOB - 1) x 0.01")
    ax.set_title(f"M-N Plot - {well.header.well_name or ''}")
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig


@_styled_plot
def _mid_chart(
    well, color_curve: str = "GR", depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
    *, porosity_curve: Optional[str] = None,
):
    """Apparent matrix density vs sonic with clean mineral reference points.

    Uses PHIT by default, or ``porosity_curve`` when given (total porosity,
    converted to v/v). Without PHIT, a linear clean binary-mineral N-D model
    estimates porosity and flags unsupported samples. Fluid endpoints are
    fresh water, and the neutron reference is limestone.
    """
    from petrophys.interpretation.lithology.apparent_matrix import (
        nd_crossplot_porosity, apparent_matrix_density, apparent_matrix_slowness,
    )
    rhob, _ = _xy(well, ["RHOB"], depth_range, "g/cm3")
    dt, _ = _xy(well, ["DTC", "DT"], depth_range, "us/ft")
    if rhob is None or dt is None:
        raise ValueError("MID plot needs RHOB and DTC")
    c, clabel = _color(well, color_curve, depth_range)
    phi, _, sources = curve_values(well, [porosity_curve or "PHIT"], depth_range, "v/v")
    method = "total porosity: " + "+".join(sources)
    if phi is None or not np.isfinite(phi).any():
        if porosity_curve is not None:
            raise ValueError(f"MID total-porosity curve {porosity_curve!r} has no finite data in the requested window")
        nphi, _ = _xy(well, ["NPHI", "PHIN"], depth_range, "v/v")
        if nphi is None:
            raise ValueError(Advice("MID needs PHIT, porosity_curve=, or NPHI for the clean neutron-density estimate",
                                    ("porosity_curve",)))
        phi = nd_crossplot_porosity(rhob, nphi)
        method = "linear clean neutron-density porosity estimate"
    warn_invalid(
        np.isinf(phi) | (phi < 0) | (phi >= .95), "MID total porosity",
        "must be finite and in [0,0.95) for matrix removal",
        "Invalid porosity samples give NaN matrix coordinates.",
    )
    rho_maa = apparent_matrix_density(rhob, phi, _RHO_FL)
    dt_maa = apparent_matrix_slowness(dt, phi, _DT_FL)
    rho_maa = np.where(np.isfinite(rho_maa), rho_maa, np.nan)
    dt_maa = np.where(np.isfinite(dt_maa), dt_maa, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter(ax, dt_maa, rho_maa, c, clabel, fig)
    for name, p in _MATRIX.items():
        ax.scatter(p["dt"], p["rhob"], s=90, marker="s", color=p["color"],
                   edgecolor="k", zorder=6)
        ax.annotate(name, (p["dt"], p["rhob"]), fontsize=9, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Apparent matrix slowness DTmaa [us/ft]")
    ax.set_ylabel("Apparent matrix density RHOmaa [g/cc]")
    ax.set_xlim(35, 100); ax.set_ylim(3.0, 2.3)
    _note_outside_axes(ax, dt_maa, rho_maa)
    ax.set_title(f"MID Plot - {well.header.well_name or ''}\n{method}", fontsize=10)
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig


@_styled_plot
def _buckles_chart(
    well, phie_curve: str = "PHIE", sw_curve: str = "SW",
    color_curve: str = "PERM", depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
    *, xlim: Tuple[float, float] = (0, .40), ylim: Tuple[float, float] = (0, 1),
):
    """Buckles plot: porosity vs water saturation with constant-BVW hyperbolas.
    A constant-BVW trend is a diagnostic of consistent pore properties;
    irreducible saturation additionally needs geological and fluid evidence.
    PHIT is the fallback when the requested effective porosity is missing.
    """
    xlim, ylim = chart_limits(xlim, "xlim"), chart_limits(ylim, "ylim")
    phie, _, sources = curve_values(well, [phie_curve, "PHIT"], depth_range, "v/v")
    sw, _ = _xy(well, [sw_curve, "SWA", "SWT-LOG"], depth_range, "v/v")
    if phie is None or sw is None:
        raise ValueError("Buckles plot needs PHIE and SW")
    c, clabel = _color(well, color_curve, depth_range)

    norm = None
    if c is not None and color_curve.upper().startswith("PERM"):
        positive = c[np.isfinite(c) & (c > 0)]
        if positive.size:
            # Permeability spans decades; colour it on a log scale between its
            # 1st and 99th percentiles (at least 0.001 mD), so a few near-zero
            # floor values do not crowd the real range into one colour.
            low, high = np.percentile(positive, [1, 99])
            low = max(float(low), 1e-3)
            high = max(float(high), low * 10.0)
            norm = LogNorm(vmin=low, vmax=high, clip=False)
            c = np.where(c > 0, c, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter(ax, phie, sw, c, clabel, fig, cmap="viridis", norm=norm)

    phi_range = np.linspace(max(.0001, xlim[0]), xlim[1], 300)
    for bvw in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
        phi_line = phi_range[phi_range >= bvw]
        if not phi_line.size:
            continue
        sw_line = bvw / phi_line
        ax.plot(phi_line, sw_line, color="0.5", lw=0.8, alpha=0.7)
        # Label each hyperbola where it meets the right edge (Sw = BVW / 0.40),
        # so the labels are spread out instead of stacked near the top.
        ax.annotate(f"BVW={bvw:g}", (phi_line[-1], sw_line[-1]), xytext=(-3, 2),
                    textcoords="offset points", ha="right", fontsize=7, color="0.4")

    ax.set_xlabel(porosity_label(sources)); ax.set_ylabel("Water saturation [v/v]")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    _note_outside_axes(ax, phie, sw)
    ax.set_title(f"Buckles Plot - {well.header.well_name or ''}")
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig


@_styled_plot
def _hingle_chart(
    well, rw: Optional[float] = None, m: float = 2.0, n: float = 2.0,
    phie_curve: str = "PHIE", rt_curve: str = "RDEP", color_curve: str = "VCL",
    a: float = 1.0, depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 8),
    *, xlim: Tuple[float, float] = (0, .40), ylim: Optional[Tuple[float, float]] = None,
    vcl_max: Optional[float] = 0.15,
):
    """Hingle plot: porosity vs Rt^(-1/m) on linear axes, with iso-Sw lines through
    the origin (the linear-axis counterpart of the Pickett plot). Water-bearing
    points fall on the Sw=1 line, whose slope is (a*Rw)^(-1/m), so
    a*Rw = slope^(-m).

    As the Pickett plot: ``rw`` (ohm.m, at the temperature of the plotted
    samples) defaults to the median of the well's ``RW`` curve over them, and
    only samples with ``VCL`` at or below ``vcl_max`` are plotted (default
    0.15, clean rock; ``None`` plots every sample), the number left out
    noted on the plot."""
    xlim = chart_limits(xlim, "xlim")
    if ylim is not None:
        ylim = chart_limits(ylim, "ylim")
    phie, _, sources = curve_values(well, [phie_curve, "PHIT"], depth_range, "v/v")
    rt, _ = _xy(well, [rt_curve, "RT", "RDEP"], depth_range, "ohm.m")
    if phie is None or rt is None:
        raise ValueError("Hingle plot needs PHIE and Rt")
    phie, rt, shaly_left_out = clean_rock_only(well, phie, rt, depth_range, vcl_max)
    if rw is None:
        rw = formation_rw(well, phie, rt, depth_range)
    rw, a, m, n = archie_parameters(rw, a, m, n)
    c, clabel = _color(well, color_curve, depth_range)

    invalid = (np.isfinite(phie) & ((phie <= 0) | (phie > 1))) | (np.isfinite(rt) & (rt <= 0))
    warn_invalid(invalid, "Hingle input", "require porosity in (0,1] and positive resistivity",
                 "Invalid points are omitted from this plot.")
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.power(rt, -1.0 / m)        # Rt^(-1/m); water line is linear in phi
    y = np.where(np.isfinite(y) & np.isfinite(rt) & ~invalid, y, np.nan)
    if c is not None:  # the colour scale spans the plotted samples only
        c = np.where(np.isfinite(phie) & np.isfinite(y), c, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter(ax, phie, y, c, clabel, fig, cmap="viridis")

    phi_line = np.linspace(max(0., xlim[0]), xlim[1], 50)
    for sw in (1.0, 0.8, 0.6, 0.4, 0.2):  # the Pickett plot's steps
        # Sw^n = a*Rw / (phi^m * Rt)  ->  Rt^(-1/m) = phi * Sw^(n/m) / (a*Rw)^(1/m)
        y_line = phi_line * sw ** (n / m) / (a * rw) ** (1.0 / m)
        ax.plot(phi_line, y_line, color="0.5", lw=0.9, alpha=0.7)
        ax.annotate(f"Sw={int(sw*100)}%", (phi_line[-1], y_line[-1]), xytext=(-3, 2),
                    textcoords="offset points", ha="right", fontsize=7, color="0.4")

    ax.set_xlabel(porosity_label(sources))
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # The chart's nonlinear resistivity scale: ticks at Rt values, spaced as Rt^(-1/m).
    # Its foot is Rt = infinity (0), where the Sw lines meet at the matrix point.
    if ylim is None:
        ax.set_ylim(0.0, max(ax.get_ylim()))
    low, high = sorted(ax.get_ylim())
    ticks = []
    for rt_value in (0.5, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000):
        position = rt_value ** (-1.0 / m)
        # Kept apart by at least 4 % of the axis: the high resistivities crowd near zero.
        if low <= position <= high and all(abs(position - kept ** (-1.0 / m)) > 0.04 * (high - low)
                                           for kept in ticks):
            ticks.append(rt_value)
    if len(ticks) >= 2:
        ax.set_yticks([rt_value ** (-1.0 / m) for rt_value in ticks])
        ax.set_yticklabels([f"{rt_value:g}" for rt_value in ticks])
        ax.set_ylim(low, high)
        ax.set_ylabel(f"Rt [ohm.m]  (scaled as Rt^(-1/{m:g}))")
    else:
        ax.set_ylabel(f"Rt^(-1/{m:g})  [(ohm.m)^(-1/{m:g})]")
    _note_outside_axes(ax, phie, y)
    if shaly_left_out:
        add_note(ax, f"{shaly_left_out} samples with VCL > {vcl_max:g} left out")
    ax.set_title(f"Hingle Plot - {well.header.well_name or ''}  (Rw={rw:.3g} ohm.m)")
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig
