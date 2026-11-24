"""Show raw MDT pressure and the library's selected/review buildup windows."""

from petrophys.visualization.style import _styled_plot
import numpy as np


@_styled_plot
def pretest_plot(trace, result=None, *, figsize=(10, 4), title=None):
    """Return a Matplotlib Figure; review candidates are never labeled picks.

    Pass the result from this trace's pick_pressure() to preserve custom QC
    settings. Times are displayed as elapsed seconds; pressures retain units.
    No report pressure or image is used. The mud-baseline windows at both
    ends are shaded with their median and 10th-90th percentile range; a
    window whose range exceeds the picker's limit (``min_drawdown_bar``) is
    drawn in red. Review flags in the title wrap to the figure width.
    """
    import textwrap

    import matplotlib.pyplot as plt

    from petrophys.interpretation.saturation.height.mdt_pretest import _PBAR

    if result is None:
        result = trace.pick_pressure()
    if result.pressure_unit != trace.pressure_unit:
        raise ValueError("Trace and pressure pick must have the same pressure unit")
    t = np.asarray(trace.time)*{"s": 1.0, "ms": 0.001, "min": 60.0}[trace.time_unit]
    pressure = np.asarray(trace.pressure, dtype=float)
    origin = t[0] if len(t) else 0.0
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t-origin, pressure, color="0.35", linewidth=1, label="Raw quartz pressure")
    unit = trace.pressure_unit
    if result.mud_before is not None and len(t):
        # The picker's baseline windows and stability rule, shown so a failed check can be read off the plot.
        edge = result.config.edge_seconds
        limit = result.config.min_drawdown_bar/_PBAR[unit]
        for name, window in (("before", t <= t[0]+edge), ("after", t >= t[-1]-edge)):
            values = pressure[window]
            if not values.size:
                continue
            spread = float(np.ptp(np.quantile(values, [.1, .9])))
            bad = spread > limit
            color = "crimson" if bad else "steelblue"
            start, end = t[window][0]-origin, t[window][-1]-origin
            ax.axvspan(start, end, color=color, alpha=.12)
            ax.hlines(np.median(values), start, end, color=color, linewidth=2,
                      label=(f"Mud {name}: P10-P90 {spread:.2f} {unit}"
                             + (f" > limit {limit:.2f}" if bad else "")))
    for i, cycle in enumerate(result.buildups):
        selected = result.selected_buildup == i and result.status == "accepted"
        color = "forestgreen" if selected else "darkorange"
        label = "Selected plateau" if selected else f"Buildup {i+1}: {cycle.status}"
        ax.axvspan(cycle.window_start-origin, cycle.window_end-origin, color=color, alpha=.18)
        ax.hlines(cycle.candidate_pressure, cycle.window_start-origin,
                  cycle.window_end-origin, color=color, linewidth=2, label=label)
    if not result.buildups and result.flags:
        ax.text(0.5, 0.04, "Buildups not evaluated: the picker stopped at " + result.flags[0].replace("_", " "),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="0.25")
    if result.pressure is not None:
        ax.axhline(result.pressure, color="forestgreen", linestyle="--",
                   label=f"Automatic pick: {result.pressure:.3f} {result.pressure_unit}")
    heading = title or trace.source or "MDT pretest"
    status = result.status if not result.flags else "review: " + ", ".join(result.flags)
    width = max(40, int(fig.get_figwidth()*9))
    ax.set(xlabel="Elapsed acquisition time [s]", ylabel=f"Pressure [{trace.pressure_unit}]",
           title=heading + "\n" + textwrap.fill(status, width=width))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
