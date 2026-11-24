"""Formation-pressure stations, fitted fluid gradients and their intersection."""

from petrophys.visualization.style import _styled_plot
import numpy as np


@_styled_plot
def pressure_plot(data, result=None, *, hydrocarbon="oil",
                  figsize=(7, 8), title="Formation pressures"):
    """Plot FormationPressureData; return a Matplotlib Figure.

    If no result is provided, call data.fit_fwl(hydrocarbon). All finite
    excluded stations remain visible for QC. No report contacts are used.
    ``result`` is the fluid-labelled fit of ``data.fit_fwl``; the unlabelled
    inference of ``data.infer_contacts()`` is drawn by
    :func:`contact_plot`.
    """
    import matplotlib.pyplot as plt

    if result is None:
        result = data.fit_fwl(hydrocarbon)
    if not hasattr(result, "hydrocarbon_fluid"):
        raise TypeError(
            f"pressure_plot draws a fluid-labelled fit (data.fit_fwl(...)), "
            f"not a {type(result).__name__}: the stations need fluid labels for it. "
            "Plot an infer_contacts() result with contact_plot(data, result)."
        )
    if (result.depth_unit, result.pressure_unit, result.depth_reference) != (
            data.depth_unit, data.pressure_unit, data.depth_reference):
        raise ValueError("Fit and stations must use the same units and TVD datum")
    hydrocarbon = result.hydrocarbon_fluid
    p = data.points
    selected = data.fit_mask(hydrocarbon)
    fig, ax = plt.subplots(figsize=figsize)
    for fluid, color, line in [(hydrocarbon, "forestgreen", result.hydrocarbon),
                                ("water", "royalblue", result.water)]:
        pts = p.loc[selected & (p.fluid == fluid)]
        ax.scatter(pts.pressure, pts.depth, color=color, label=f"{fluid}: selected")
        z = np.linspace(min(pts.depth.min(), result.fwl_depth),
                        max(pts.depth.max(), result.fwl_depth), 100)
        ax.plot(line.pressure_at(z), z, color=color, label=f"{fluid}: fitted gradient")
    excluded = p.loc[~selected & np.isfinite(p.depth) & np.isfinite(p.pressure)]
    if len(excluded):
        ax.scatter(excluded.pressure, excluded.depth, marker="x", color="0.55",
                   label="Excluded stations")
    ax.axhline(result.fwl_depth, color="black",
               label=f"FWL {result.fwl_depth:.2f} {data.depth_unit}")
    ax.invert_yaxis()
    ax.set(xlabel=f"Formation pressure [{data.pressure_unit}]",
           ylabel=f"{data.depth_reference}, positive down [{data.depth_unit}]", title=title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


@_styled_plot
def contact_plot(data, result=None, *, figsize=(7, 8),
                 title="Pressure contact interpretation", zoom="fit"):
    """Plot unlabeled inference, including unresolved contacts and suspect points.

    Returns a Figure. Call ``data.infer_contacts(**settings)`` first to control
    assumptions. Ambiguous intersections are explicitly marked as candidates.

    ``zoom="fit"`` frames the stations assigned to a gradient and the
    contact intervals, with a margin of a quarter of their depth span (at
    least 10 m or 30 ft); excluded stations inside that window stay visible
    and the number outside it is noted on the plot. ``zoom="all"`` frames
    every station. Without fitted stations both show everything.
    """
    import matplotlib.pyplot as plt

    if result is None:
        result = data.infer_contacts()
    if (result.depth_unit, result.pressure_unit, result.depth_reference) != (
            data.depth_unit, data.pressure_unit, data.depth_reference):
        raise ValueError("Interpretation and stations must use the same units and TVD datum")
    if len(result.station_branch) != len(data.points):
        raise ValueError("Interpretation and station table must have matching rows")
    p = data.points
    fig, ax = plt.subplots(figsize=figsize)
    for i, branch in enumerate(result.branches):
        pts = p.iloc[np.flatnonzero(result.station_branch == i)]
        label = "/".join(result.fluid_sequences[0][i:i+1]) if len(result.fluid_sequences) == 1 else "/".join(branch.fluid_candidates)
        color = plt.get_cmap("tab10")(i)
        ax.scatter(pts.pressure, pts.depth, color=color,
                   label=f"Branch {i+1}: {label or 'unresolved'} ({branch.density_g_cc:.2f} g/cc)")
        ends = [branch.depth_min, branch.depth_max]
        ends.extend(c.depth for c in result.contacts[max(0, i-1):i+1])
        z = np.array([min(ends), max(ends)])
        ax.plot(branch.line.pressure_at(z), z, color=color)
    excluded = (result.station_branch < 0) | result.suspect
    pts = p.iloc[np.flatnonzero(excluded & np.isfinite(p.pressure) & np.isfinite(p.depth))]
    if len(pts):
        ax.scatter(pts.pressure, pts.depth, color="0.45", marker="x", label="Excluded / suspect")
    for contact in result.contacts:
        prefix = "Candidate " if result.status != "interpreted" else ""
        ax.axhline(contact.depth, color="black", linestyle="--",
                   label=f"{prefix}{contact.kind}: {contact.depth:.2f} {data.depth_unit}")
        ax.axhspan(*contact.depth_interval, color="0.5", alpha=0.10)
    if zoom not in ("fit", "all"):
        raise ValueError("zoom must be 'fit' or 'all'")
    depth = np.asarray(p.depth, dtype=float)
    pressure = np.asarray(p.pressure, dtype=float)
    finite = np.isfinite(depth) & np.isfinite(pressure)
    fitted = finite & (np.asarray(result.station_branch) >= 0)
    if zoom == "fit" and fitted.any():
        ends = list(depth[fitted])
        for contact in result.contacts:
            ends.extend(v for v in (contact.depth, *contact.depth_interval) if np.isfinite(v))
        margin = max(0.25*(max(ends)-min(ends)), 10.0 if data.depth_unit == "m" else 30.0)
        top, base = min(ends)-margin, max(ends)+margin
        shown = finite & (depth >= top) & (depth <= base)
        hidden = int(np.sum(finite & ~shown))
        if hidden:
            pad = max(0.05*np.ptp(pressure[shown]), 0.5)
            ax.set_ylim(base, top)
            ax.set_xlim(pressure[shown].min()-pad, pressure[shown].max()+pad)
            ax.text(0.02, 0.02, f"{hidden} station(s) outside the fitted interval not shown (zoom='all')",
                    transform=ax.transAxes, fontsize=8, color="0.3", va="bottom")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set(xlabel=f"Formation pressure [{data.pressure_unit}]",
           ylabel=f"{data.depth_reference}, positive down [{data.depth_unit}]",
           title=f"{title}\nStatus: {result.status}")
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
