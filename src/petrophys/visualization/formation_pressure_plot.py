"""Formation-pressure stations, fitted fluid gradients and their intersection."""
import numpy as np


def create_formation_pressure_plot(data, result=None, *, hydrocarbon="oil",
                                   figsize=(7, 8), title="Formation pressures"):
    """Plot FormationPressureData; return a Matplotlib Figure.

    If no result is provided, call data.fit_fwl(hydrocarbon). All finite
    excluded stations remain visible for QC. No report contacts are used.
    """
    import matplotlib.pyplot as plt

    if result is None:
        result = data.fit_fwl(hydrocarbon)
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
    ax.set(xlabel=f"Formation pressure ({data.pressure_unit})",
           ylabel=f"{data.depth_reference}, positive down ({data.depth_unit})", title=title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
