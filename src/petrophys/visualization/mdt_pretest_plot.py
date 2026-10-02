"""Show raw MDT pressure and the library's selected/review buildup windows."""
import numpy as np


def create_mdt_pretest_plot(trace, result=None, *, figsize=(10, 4), title=None):
    """Return a Matplotlib Figure; review candidates are never labeled picks.

    Pass the result from this trace's pick_pressure() to preserve custom QC
    settings. Times are displayed as elapsed seconds; pressures retain units.
    No report pressure or image is used.
    """
    import matplotlib.pyplot as plt

    if result is None:
        result = trace.pick_pressure()
    if result.pressure_unit != trace.pressure_unit:
        raise ValueError("Trace and pressure pick must have the same pressure unit")
    t = np.asarray(trace.time)*{"s": 1.0, "ms": 0.001, "min": 60.0}[trace.time_unit]
    origin = t[0] if len(t) else 0.0
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t-origin, trace.pressure, color="0.35", linewidth=1, label="Raw quartz pressure")
    for i, cycle in enumerate(result.buildups):
        selected = result.selected_buildup == i and result.status == "accepted"
        color = "forestgreen" if selected else "darkorange"
        label = "Selected plateau" if selected else f"Buildup {i+1}: {cycle.status}"
        ax.axvspan(cycle.window_start-origin, cycle.window_end-origin, color=color, alpha=.18)
        ax.hlines(cycle.candidate_pressure, cycle.window_start-origin,
                  cycle.window_end-origin, color=color, linewidth=2, label=label)
    if result.pressure is not None:
        ax.axhline(result.pressure, color="forestgreen", linestyle="--",
                   label=f"Automatic pick: {result.pressure:.3f} {result.pressure_unit}")
    heading = title or trace.source or "MDT pretest"
    status = result.status if not result.flags else "review: " + ", ".join(result.flags)
    ax.set(xlabel="Elapsed acquisition time (s)", ylabel=f"Pressure ({trace.pressure_unit})",
           title=f"{heading}\n{status}")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
